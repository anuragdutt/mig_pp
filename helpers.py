import os
import gc
import json
import time
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import hub

import model_family

# Module logger. NOTE: this was previously `from logging import log`, which
# bound the stdlib logging.log() FUNCTION — so every log.info()/log.warning()
# call below would have raised AttributeError.
log = logging.getLogger(__name__)


def get_wiki_sample(batch_size: int, seq_len: int, model_name: str) -> torch.Tensor:
    log.info(f"Loading WikiText... (SEQ_LEN={seq_len}, BATCH={batch_size})")
    try:
        dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test", streaming=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        text_sample = ""
        for item in dataset:
            if len(item["text"]) > 100:
                text_sample = item["text"]
                break

        inputs = tokenizer(
            text_sample,
            return_tensors="pt",
            max_length=seq_len,
            padding="max_length",
            truncation=True,
        )
        return inputs.input_ids.repeat(batch_size, 1)
    except Exception:
        log.warning("WikiText unavailable. Using random token IDs.")
        return torch.randint(0, 32000, (batch_size, seq_len))


def _find_checkpoint(model_name: str):
    """
    (format, [shard paths]) for model_name.

    Local directories and repo ids already in the HF cache go through
    model_family (safetensors first, then .bin; raises WeightFilesError when
    no complete checkpoint is there). Anything else falls back to the
    original lookup — transformers' hub.cached_file on the .bin index, which
    raises if that is missing too — so a standalone run on a repo id behaves
    as it always has.
    """
    try:
        wf = model_family.find_weight_files(
            model_family.resolve_model_path(model_name)
        )
        return wf.fmt, list(wf.files)
    except model_family.ModelPathError:
        pass

    cached_index = hub.cached_file(model_name, "pytorch_model.bin.index.json")
    folder_path = os.path.dirname(cached_index)
    with open(cached_index, "r") as f:
        weight_map = json.load(f)["weight_map"]
    return "bin", [os.path.join(folder_path, s) for s in sorted(set(weight_map.values()))]


def _iter_checkpoint(fmt: str, file_path: str, wanted):
    """
    Yield (key, tensor) for every key in one shard for which wanted(key) is
    true, reading only those tensors when the format allows it.

    safetensors: safe_open reads tensor by tensor, so a rank touches only its
    own layers. .bin: torch.load(mmap=True) maps the shard and pages in only
    what is copied out, instead of materialising the whole multi-GB shard per
    rank — the difference between ~10GB and ~0 of host RAM per rank once 32
    ranks load at the same time. Pre-zipfile .bin files cannot be mmapped;
    those fall back to the original full load.
    """
    if fmt == "safetensors":
        from safetensors import safe_open

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if wanted(key):
                    yield key, f.get_tensor(key)
        return

    try:
        state_dict = torch.load(
            file_path, map_location="cpu", mmap=True, weights_only=True
        )
    except Exception as e:
        log.info(f"mmap load of {os.path.basename(file_path)} failed ({e}); full load")
        state_dict = torch.load(file_path, map_location="cpu")
    try:
        for key, value in state_dict.items():
            if wanted(key):
                yield key, value
    finally:
        del state_dict


def load_specific_weights(
    rank: int,
    world_size: int,
    model_name: str,
    my_layers: nn.ModuleList,
    my_layer_indices: List[int],
    model_components: Dict[str, nn.Module],
    tie_word_embeddings: bool = False,
) -> Dict[str, object]:
    """
    Copy this rank's tensors out of the checkpoint into its modules.

    Returns (and logs as [B08]) what it loaded against what this rank's
    modules expect, so a model whose checkpoint does not match the classes
    it was built with is visible instead of silently running on random
    weights. Latency does not depend on the values, but the loaded /
    expected count is the check that the right model is being measured.

    tie_word_embeddings: the checkpoint has no lm_head.weight; the last rank
    fills lm_head from embed_tokens instead (Llama/Mistral/Qwen2.5-7B+ are
    untied, so this is False for every model the harness runs today).
    """
    log.info(f"[Rank {rank}] Loading weights...")
    t0 = time.perf_counter()
    last = rank == world_size - 1

    stats: Dict[str, object] = {
        "format": None,
        "files": 0,
        "expected": 0,
        "loaded": 0,
        "missing": [],
        "seconds": 0.0,
    }

    layer_to_local: Dict[int, int] = {idx: i for i, idx in enumerate(my_layer_indices)}

    # Parameter names this rank must fill, in checkpoint naming.
    expected = set()
    for global_idx, local_idx in layer_to_local.items():
        for pname, _ in my_layers[local_idx].named_parameters():
            expected.add(f"model.layers.{global_idx}.{pname}")
    if rank == 0 and "embed" in model_components:
        expected.add("model.embed_tokens.weight")
    if last and "norm" in model_components:
        expected.add("model.norm.weight")
    if last and "lm_head" in model_components:
        expected.add("lm_head.weight")
    stats["expected"] = len(expected)

    def _targets(key: str):
        """(name, destination tensor) pairs this rank copies `key` into."""
        out = []
        if "embed_tokens" in key:
            if rank == 0 and "embed" in model_components:
                out.append(("model.embed_tokens.weight", model_components["embed"].weight))
            if last and tie_word_embeddings and "lm_head" in model_components:
                out.append(("lm_head.weight", model_components["lm_head"].weight))
            return out

        if last:
            # endswith, not `in`: "norm.weight" also matches the per-layer
            # input_layernorm.weight / post_attention_layernorm.weight keys.
            # With `in`, every decoder layernorm on the last rank was copied
            # into the final norm and then `continue`d past the layer loader,
            # so the per-layer norms were never loaded at all and the final
            # norm held whichever layernorm the shard happened to yield last.
            if key.endswith("model.norm.weight") and "norm" in model_components:
                return [("model.norm.weight", model_components["norm"].weight)]
            if "lm_head.weight" in key and "lm_head" in model_components:
                return [("lm_head.weight", model_components["lm_head"].weight)]

        if "layers." in key:
            parts = key.split(".")
            try:
                layer_idx = int(parts[2])
            except (ValueError, IndexError):
                return out

            local_idx = layer_to_local.get(layer_idx)
            if local_idx is None:
                return out

            # Walk to the attribute. Keys with no counterpart in this
            # transformers version (e.g. old checkpoints' per-layer
            # rotary_emb.inv_freq) and absent biases (None) are skipped.
            sub_mod = my_layers[local_idx]
            sub_parts = parts[3:]
            try:
                for sp in sub_parts[:-1]:
                    sub_mod = getattr(sub_mod, sp)
                dest = getattr(sub_mod, sub_parts[-1])
            except AttributeError:
                return out
            if isinstance(dest, torch.Tensor):
                out.append((f"model.layers.{layer_idx}.{'.'.join(sub_parts)}", dest))
        return out

    try:
        fmt, shard_files = _find_checkpoint(model_name)
    except model_family.WeightFilesError as e:
        log.warning(f"[Rank {rank}] {e}. Skipping.")
        shard_files, fmt = [], None
    except Exception:
        log.warning(f"[Rank {rank}] Weight map not found. Skipping.")
        shard_files, fmt = [], None
    stats["format"] = fmt
    stats["files"] = len(shard_files)

    loaded = set()
    for file_path in tqdm(shard_files, desc=f"Rank {rank} shards", leave=False):
        for key, value in _iter_checkpoint(fmt, file_path, lambda k: bool(_targets(k))):
            for name, dest in _targets(key):
                dest.data.copy_(value)
                loaded.add(name)
            del value

        gc.collect()
        torch.cuda.empty_cache()

    missing = sorted(expected - loaded)
    stats["loaded"] = len(expected & loaded)
    stats["missing"] = missing
    stats["seconds"] = time.perf_counter() - t0

    # Greppable coverage line; parallel_plan.py's smoke verdict parses it.
    log.info(
        f"[B08][Rank {rank}] weights: loaded={stats['loaded']}/{stats['expected']} "
        f"missing={len(missing)} format={fmt} files={len(shard_files)}"
    )
    if missing:
        log.warning(
            f"[B08][Rank {rank}] missing tensors (first 10 of {len(missing)}): "
            f"{missing[:10]}"
        )

    log.info(f"[Rank {rank}] Weights loaded.")
    return stats


def _compute_slot_size_mb(hidden_size=5120, max_mb_size=32, max_seq_len=64):
    """Largest tensor: prefill activation (mb_size × seq_len × hidden × 2 bytes)

    NOTE: pass the mb_size the run ACTUALLY uses. Every slot is allocated at
    this size in BOTH host SHM and pinned RAM, once per slot per rank, so
    oversizing multiplies fast: 4 ranks x NUM_SLOTS x 2 (shm+pinned).
    """
    max_bytes = max_mb_size * max_seq_len * hidden_size * 2  # fp16
    mb = (max_bytes // (1024 * 1024)) + 1  # round up
    return mb
