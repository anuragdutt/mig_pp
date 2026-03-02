import os
import logging
import traceback
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import pandas as pd
import queue
import gc
import json
from tqdm import tqdm

from transformers import (
    DynamicCache,
    LlamaConfig,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils import hub
from datasets import load_dataset

import mig_transport_pipeline_non_blocking as mig_transport
import dcgm_mem_monitor as monitor


# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------

LOG_FILE = "benchmark.log"


def setup_logging(log_file: str = LOG_FILE) -> None:
    """Configure root logger to write to file only."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="a")],
        force=True,
    )


log = logging.getLogger(__name__)


# --- CONFIGURATION ---
MODEL_NAME = "lmsys/vicuna-7b-v1.5"
TOTAL_LAYERS = 32
HIDDEN_SIZE = 4096
HEADS = 32

SEQ_LEN = 64
MAX_NEW_TOKENS = 512

BATCH_MB_PAIRS = [
    (32, 16),
    (64, 16),
    (64, 32),
    (128, 16),
    (128, 32),
]

MIG_UUIDS = [
    "MIG-c2cc1c36-c2ce-5fb7-b0b6-314533a3f0b4",  # Rank 0: 20GB
    "MIG-e2bdc502-5cb8-54ef-af62-fc9961c40f92",  # Rank 1: 10GB
    "MIG-55f42e36-5e0b-5895-be54-edd0181504fa",  # Rank 2: 5GB
]

LAYER_LIMITS = [24, 12, 6]


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------


def get_wiki_sample(batch_size: int) -> torch.Tensor:
    log.info(f"Loading WikiText... (SEQ_LEN={SEQ_LEN}, BATCH={batch_size})")
    try:
        dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test", streaming=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        text_sample = ""
        for item in dataset:
            if len(item["text"]) > 100:
                text_sample = item["text"]
                break

        inputs = tokenizer(
            text_sample,
            return_tensors="pt",
            max_length=SEQ_LEN,
            padding="max_length",
            truncation=True,
        )
        return inputs.input_ids.repeat(batch_size, 1)

    except Exception:
        log.warning("WikiText unavailable. Using random token IDs.")
        return torch.randint(0, 32000, (batch_size, SEQ_LEN))


# ---------------------------------------------------------------------------
# WEIGHT LOADING
# ---------------------------------------------------------------------------


def load_specific_weights(
    rank: int,
    my_layers: nn.ModuleList,
    my_layer_indices: List[int],
    model_components: Dict[str, nn.Module],
) -> None:
    log.info(f"[Rank {rank}] Loading weights...")

    try:
        cached_index = hub.cached_file(MODEL_NAME, "pytorch_model.bin.index.json")
        folder_path = os.path.dirname(cached_index)
        with open(cached_index, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    except Exception:
        log.warning(f"[Rank {rank}] Weight map not found. Skipping.")
        return

    # Speed: avoid repeated O(n) .index() calls
    layer_to_local: Dict[int, int] = {idx: i for i, idx in enumerate(my_layer_indices)}

    for shard_file in tqdm(shard_files, desc=f"Rank {rank} shards", leave=False):
        file_path = os.path.join(folder_path, shard_file)
        state_dict: Dict[str, torch.Tensor] = torch.load(file_path, map_location="cpu")

        for key, value in state_dict.items():
            if rank == 0 and "embed_tokens" in key and "embed" in model_components:
                model_components["embed"].weight.data.copy_(value)
                continue

            if rank == 2:
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
                    # In case HF changes internals; keep benchmark running
                    pass

        del state_dict
        gc.collect()
        # shard weights are CPU, but keep this (harmless) for safety in long runs
        torch.cuda.empty_cache()

    log.info(f"[Rank {rank}] Weights loaded.")


# ---------------------------------------------------------------------------
# FORWARD PASS HELPER
# ---------------------------------------------------------------------------


def forward_through_layers(
    layers: nn.ModuleList,
    current_hidden: torch.Tensor,
    position_embeddings,
    mask: Optional[torch.Tensor],
    cache: DynamicCache,
) -> torch.Tensor:
    for layer in layers:
        residual = current_hidden
        hidden_states = layer.input_layernorm(current_hidden)

        attn_outputs = layer.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
        )

        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        current_hidden = residual + hidden_states

    return current_hidden


# ---------------------------------------------------------------------------
# PIPELINE WORKER
# ---------------------------------------------------------------------------


def run_pipeline(
    rank: int,
    world_size: int,
    split_config: List[int],
    result_queue: mp.Queue,
    device_uuid: str,
    input_ids_seed: torch.Tensor,
    mb_size: int,
) -> None:
    setup_logging()

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # Patch dist send/recv to go through SHM transport
        mig_transport.register_hooks()

        config = LlamaConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "sdpa"

        start_layer = sum(split_config[:rank])
        end_layer = start_layer + split_config[rank]
        my_layer_indices = list(range(start_layer, end_layer))

        model_components: Dict[str, nn.Module] = {}

        if rank == 0:
            model_components["embed"] = (
                nn.Embedding(config.vocab_size, config.hidden_size).to(device).half()
            )

        layers = nn.ModuleList()
        for idx in my_layer_indices:
            layers.append(LlamaDecoderLayer(config, layer_idx=idx).half().to(device))
            torch.cuda.empty_cache()

        if rank == world_size - 1:
            model_components["norm"] = (
                LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                .to(device)
                .half()
            )
            model_components["lm_head"] = (
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                .to(device)
                .half()
            )

        # IMPORTANT: real inference mode (no dropout)
        for m in model_components.values():
            m.eval()
        layers.eval()

        rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

        load_specific_weights(rank, layers, my_layer_indices, model_components)

        gc.collect()
        torch.cuda.empty_cache()

        input_ids = input_ids_seed.to(device)
        batch_size, seq_length = input_ids.shape

        if batch_size % mb_size != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by microbatch size {mb_size}"
            )
        num_microbatches = batch_size // mb_size  # will be 1 for our test

        past_key_values_list = [DynamicCache() for _ in range(num_microbatches)]
        next_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        # Precompute mask once (full causal mask for prefill)
        prefill_mask = torch.full(
            (1, 1, seq_length, seq_length),
            torch.finfo(torch.float16).min,
            device=device,
        )
        prefill_mask = torch.triu(prefill_mask, diagonal=1).half()

        dist.barrier()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.no_grad():

            # -------------------------
            # PREFILL
            # -------------------------
            for mb_idx in range(num_microbatches):
                start_idx = mb_idx * mb_size
                end_idx = start_idx + mb_size

                if rank == 0:
                    current_hidden = model_components["embed"](
                        input_ids[start_idx:end_idx]
                    ).half()
                else:
                    recv_buf = torch.zeros(
                        (mb_size, seq_length, config.hidden_size),
                        dtype=torch.float16,
                        device=device,
                    )
                    dist.recv(recv_buf, src=rank - 1)
                    current_hidden = recv_buf

                position_ids = torch.arange(
                    0, seq_length, dtype=torch.long, device=device
                ).unsqueeze(0)

                position_embeddings = rotary_emb(current_hidden, position_ids)

                current_hidden = forward_through_layers(
                    layers,
                    current_hidden,
                    position_embeddings,
                    prefill_mask,
                    past_key_values_list[mb_idx],
                )

                if rank == world_size - 1:
                    normed = model_components["norm"](current_hidden)
                    logits = model_components["lm_head"](normed[:, -1:, :])
                    mb_next = torch.argmax(logits, dim=-1)
                    next_tokens[start_idx:end_idx] = mb_next

                if rank < world_size - 1:
                    # clone() keeps semantics safe if transport reads after return
                    dist.send(current_hidden.clone(), dst=rank + 1)

            # Share next tokens back to rank0
            if rank == world_size - 1:
                dist.send(next_tokens, dst=0)
            elif rank == 0:
                dist.recv(next_tokens, src=world_size - 1)

            dist.barrier()

            # -------------------------
            # DECODE (512 steps)
            # -------------------------
            for step in range(1, MAX_NEW_TOKENS + 1):
                # (mb_size, 1)
                position_ids = torch.tensor(
                    [[seq_length + step - 1]], dtype=torch.long, device=device
                ).expand(mb_size, -1)

                prev_send_handle = None

                for mb_idx in range(num_microbatches):
                    start_idx = mb_idx * mb_size
                    end_idx = start_idx + mb_size

                    if prev_send_handle is not None:
                        prev_send_handle.wait()
                        prev_send_handle = None

                    if rank == 0:
                        current_hidden = model_components["embed"](
                            next_tokens[start_idx:end_idx]
                        ).half()
                    else:
                        recv_buf = torch.zeros(
                            (mb_size, 1, config.hidden_size),
                            dtype=torch.float16,
                            device=device,
                        )
                        recv_handle = dist.irecv(recv_buf, src=rank - 1)
                        recv_handle.wait()
                        current_hidden = recv_buf

                    position_embeddings = rotary_emb(current_hidden, position_ids)

                    current_hidden = forward_through_layers(
                        layers,
                        current_hidden,
                        position_embeddings,
                        None,
                        past_key_values_list[mb_idx],
                    )

                    if rank == world_size - 1:
                        normed = model_components["norm"](current_hidden)
                        logits = model_components["lm_head"](normed)
                        mb_next = torch.argmax(logits, dim=-1)
                        next_tokens[start_idx:end_idx] = mb_next

                    if rank < world_size - 1:
                        prev_send_handle = dist.isend(
                            current_hidden.clone(), dst=rank + 1
                        )

                if prev_send_handle is not None:
                    prev_send_handle.wait()

                if step < MAX_NEW_TOKENS:
                    if rank == world_size - 1:
                        dist.send(next_tokens, dst=0)
                    elif rank == 0:
                        dist.recv(next_tokens, src=world_size - 1)

        dist.barrier()
        end_event.record()
        torch.cuda.synchronize()

        total_latency_ms = start_event.elapsed_time(end_event)
        log.info(f"[Rank {rank}] Finished. Latency: {total_latency_ms:.0f} ms")

        if rank == 0:
            result_queue.put(("latency", total_latency_ms))

        dist.destroy_process_group()

    except torch.cuda.OutOfMemoryError:
        log.error(f"[Rank {rank}] OOM — unexpected for this safe config.")
        result_queue.put(("oom", rank))
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    except Exception:
        log.error(f"[Rank {rank}] Unexpected exception:\n{traceback.format_exc()}")
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SPLIT GENERATION
# ---------------------------------------------------------------------------


def generate_layer_splits():
    valid_splits = []
    for l0 in range(1, LAYER_LIMITS[0] + 1):
        for l1 in range(1, LAYER_LIMITS[1] + 1):
            l2 = TOTAL_LAYERS - (l0 + l1)
            if 1 <= l2 <= LAYER_LIMITS[2]:
                valid_splits.append([l0, l1, l2])
    return valid_splits


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    setup_logging()

    # Monitor optional; keep as you had
    log.info("Setting up DCGM monitor group...")
    monitor.setup_dcgm_group()

    selected_splits = generate_layer_splits()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    total_runs = len(selected_splits) * len(BATCH_MB_PAIRS)
    results = []
    current_run = 1

    for split in selected_splits:
        for batch_size, mb_size in BATCH_MB_PAIRS:
            log.info(
                f"[{current_run}/{total_runs}] Split {split} | "
                f"Batch: {batch_size} | Microbatch: {mb_size} | "
                f"Microbatches: {batch_size // mb_size}"
            )

            input_ids_seed = get_wiki_sample(batch_size)

            label = f"s{'_'.join(map(str, split))}_b{batch_size}_mb{mb_size}"
            monitor.set_label(label)
            monitor._samples.clear()
            monitor.start()

            q = mp.Queue()
            procs: list[mp.Process] = []

            try:
                for rank in range(3):
                    p = mp.Process(
                        target=run_pipeline,
                        args=(
                            rank,
                            3,
                            split,
                            q,
                            MIG_UUIDS[rank],
                            input_ids_seed,
                            mb_size,
                        ),
                    )
                    p.start()
                    procs.append(p)

                # Wait up to N seconds for each rank; then kill any stragglers.
                JOIN_TIMEOUT_S = 1200

                for p in procs:
                    p.join(timeout=JOIN_TIMEOUT_S)

                hung = [p for p in procs if p.is_alive()]
                if hung:
                    log.error(
                        "Hang detected: ranks still alive after timeout: "
                        + ", ".join(str(procs.index(p)) for p in hung)
                    )
                    for p in hung:
                        p.terminate()
                    for p in hung:
                        p.join(timeout=10)

                monitor.stop()

                run_samples = list(monitor._samples)
                peak_gi5_mb = max((row[3] for row in run_samples), default=0)
                avg_gi5_mb = (
                    (sum(row[3] for row in run_samples) / len(run_samples))
                    if run_samples
                    else 0
                )

                exit_codes = [p.exitcode for p in procs]
                queue_items = {}
                try:
                    while True:
                        key, val = q.get(timeout=2.0)
                        queue_items[key] = val
                except queue.Empty:
                    pass

                base_row = {
                    "split": str(split),
                    "batch_size": batch_size,
                    "microbatch_size": mb_size,
                    "num_microbatches": batch_size // mb_size,
                    "peak_gi5gb_mb": peak_gi5_mb,
                    "avg_gi5gb_mb": round(avg_gi5_mb),
                    "total_latency_ms": None,
                    "status": None,
                }

                # IMPORTANT: hung must take priority and not be overwritten
                if hung:
                    base_row["status"] = "hang"

                elif "oom" in queue_items:
                    oom_rank = queue_items["oom"]
                    log.warning(f"OOM on Rank {oom_rank} — skipping.")
                    base_row["status"] = f"OOM_rank{oom_rank}"

                elif any((code is not None) and (code != 0) for code in exit_codes):
                    log.error(f"Crashed. Exit codes: {exit_codes}")
                    base_row["status"] = "crash"

                elif "latency" in queue_items:
                    latency = queue_items["latency"]
                    log.info(f"Total latency:     {latency:.0f} ms")
                    log.info(f"Peak 5GB memory:   {peak_gi5_mb} MB")
                    log.info(f"Avg  5GB memory:   {avg_gi5_mb:.0f} MB")
                    base_row["total_latency_ms"] = latency
                    base_row["status"] = "ok"

                else:
                    log.warning("Timeout — no results received.")
                    base_row["status"] = "timeout"

                results.append(base_row)

            finally:
                try:
                    monitor.stop()
                except Exception:
                    pass
                q.close()
                q.join_thread()

            for p in procs:
                if p.is_alive():
                    p.terminate()

            current_run += 1

    df = pd.DataFrame(results)
    df.to_csv("mig_benchmark_results.csv", index=False)

    # Save full DCGM memory trace across all runs — one row per 500ms sample,
    # labelled by configuration for post-run plotting
    monitor.save_csv("mig_memory_trace.csv")

    successful = df[df["status"] == "ok"]
    if not successful.empty:
        log.info("--- Best by Latency ---")
        best_lat = successful.loc[successful["total_latency_ms"].idxmin()]
        log.info(
            f"Split: {best_lat['split']} | Batch: {best_lat['batch_size']} "
            f"| MB: {best_lat['microbatch_size']} "
            f"| Latency: {best_lat['total_latency_ms']:.0f} ms "
            f"| Peak GI-2: {best_lat['peak_gi2_mb']} MB"
        )

        log.info("--- Most Memory Efficient (lowest peak GI-2 at batch=64) ---")
        b64 = successful[successful["batch_size"] == 64]
        if not b64.empty:
            best_mem = b64.loc[b64["peak_gi2_mb"].idxmin()]
            log.info(
                f"Split: {best_mem['split']} | MB: {best_mem['microbatch_size']} "
                f"| Latency: {best_mem['total_latency_ms']:.0f} ms "
                f"| Peak GI-2: {best_mem['peak_gi2_mb']} MB"
            )

        oom_count = len(df[df["status"].str.startswith("OOM", na=False)])
        log.info(f"{oom_count} configurations skipped due to OOM")

    log.info("Done.")
    log.info("Benchmark results → mig_benchmark_results.csv")
    log.info("Memory trace      → mig_memory_trace.csv")


if __name__ == "__main__":
    main()
