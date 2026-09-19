import os
import gc
import json
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import hub

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


def load_specific_weights(
    rank: int,
    world_size: int,
    model_name: str,
    my_layers: nn.ModuleList,
    my_layer_indices: List[int],
    model_components: Dict[str, nn.Module],
) -> None:
    log.info(f"[Rank {rank}] Loading weights...")

    try:
        cached_index = hub.cached_file(model_name, "pytorch_model.bin.index.json")
        folder_path = os.path.dirname(cached_index)
        with open(cached_index, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    except Exception:
        log.warning(f"[Rank {rank}] Weight map not found. Skipping.")
        return

    layer_to_local: Dict[int, int] = {idx: i for i, idx in enumerate(my_layer_indices)}

    for shard_file in tqdm(shard_files, desc=f"Rank {rank} shards", leave=False):
        file_path = os.path.join(folder_path, shard_file)
        state_dict: Dict[str, torch.Tensor] = torch.load(file_path, map_location="cpu")

        for key, value in state_dict.items():
            if rank == 0 and "embed_tokens" in key and "embed" in model_components:
                model_components["embed"].weight.data.copy_(value)
                continue

            if rank == world_size - 1:
                if "norm.weight" in key and "norm" in model_components:
                    model_components["norm"].weight.data.copy_(value)
                    continue
                if "lm_head.weight" in key and "lm_head" in model_components:
                    model_components["lm_head"].weight.data.copy_(value)
                    continue

            if "layers." in key:
                parts = key.split(".")
                try:
                    layer_idx = int(parts[2])
                except ValueError:
                    continue

                local_idx = layer_to_local.get(layer_idx)
                if local_idx is None:
                    continue

                module = my_layers[local_idx]
                local_key = ".".join(parts[3:])

                try:
                    sub_mod = module
                    sub_parts = local_key.split(".")
                    for sp in sub_parts[:-1]:
                        sub_mod = getattr(sub_mod, sp)
                    getattr(sub_mod, sub_parts[-1]).data.copy_(value)
                except AttributeError:
                    pass

        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    log.info(f"[Rank {rank}] Weights loaded.")


def _compute_slot_size_mb(hidden_size=5120, max_mb_size=32, max_seq_len=64):
    """Largest tensor: prefill activation (mb_size × seq_len × hidden × 2 bytes)

    NOTE: pass the mb_size the run ACTUALLY uses. Every slot is allocated at
    this size in BOTH host SHM and pinned RAM, once per slot per rank, so
    oversizing multiplies fast: 4 ranks x NUM_SLOTS x 2 (shm+pinned).
    """
    max_bytes = max_mb_size * max_seq_len * hidden_size * 2  # fp16
    mb = (max_bytes // (1024 * 1024)) + 1  # round up
    return mb
