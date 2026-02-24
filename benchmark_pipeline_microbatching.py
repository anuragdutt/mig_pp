import os
from typing import Dict
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import pandas as pd
import queue
import gc
import json
from tqdm import tqdm
import pynvml

# Hugging Face Imports
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

# --- CONFIGURATION ---
MODEL_NAME = "lmsys/vicuna-7b-v1.5"
TOTAL_LAYERS = 32
HIDDEN_SIZE = 4096
HEADS = 32

# Fixed by mentor — do not change these
SEQ_LEN = 64  # Input (prefill) length
MAX_NEW_TOKENS = 512  # Output (decode) length

# Search dimensions — all three are swept in the benchmark
BATCH_SIZES = [16, 32, 64, 128]  # Total sequences processed simultaneously
MICROBATCH_SIZES = [16, 32, 64]  # How the batch is sliced for the pipeline

# How often to sample GPU utilization during decode (every N steps).
UTIL_SAMPLE_INTERVAL = 10

# --- MIG UUID MAPPING ---
MIG_UUIDS = [
    "MIG-1c4561cb-3b99-5158-bc81-3025c4d3022a",  # Rank 0: 20GB (Big Slice)
    "MIG-81816777-9b90-562b-9a63-9f3c039d6b0e",  # Rank 1: 10GB (Medium Slice)
    "MIG-8639c3a9-2760-5353-9e83-b5318a48d693",  # Rank 2: 5GB  (Small Slice)
]

LAYER_LIMITS = [24, 12, 6]


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------


def get_wiki_sample(batch_size: int) -> torch.Tensor:
    """
    Load a fixed-length batch of token IDs from WikiText-2.
    Takes batch_size as a parameter since we now sweep over batch sizes.
    """
    print(f"Loading WikiText... (SEQ_LEN={SEQ_LEN}, BATCH={batch_size})")
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
        print("Warning: WikiText unavailable. Using random token IDs.")
        return torch.randint(0, 32000, (batch_size, SEQ_LEN))


# ---------------------------------------------------------------------------
# WEIGHT LOADING
# ---------------------------------------------------------------------------


def load_specific_weights(rank, my_layers, my_layer_indices, model_components):
    print(f"[Rank {rank}] Loading weights...", flush=True)

    try:
        cached_index = hub.cached_file(MODEL_NAME, "pytorch_model.bin.index.json")
        folder_path = os.path.dirname(cached_index)
        with open(cached_index, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        shard_files = set(weight_map.values())
    except Exception:
        print(f"[Rank {rank}] Warning: weight map not found. Skipping.", flush=True)
        return

    for shard_file in tqdm(shard_files, desc=f"Rank {rank} shards", leave=False):
        file_path = os.path.join(folder_path, shard_file)
        state_dict: Dict[str, torch.Tensor] = torch.load(file_path, map_location="cpu")

        for key in list(state_dict.keys()):
            if rank == 0 and "embed_tokens" in key and "embed" in model_components:
                model_components["embed"].weight.data.copy_(state_dict[key])

            if rank == 2:
                if "norm.weight" in key and "norm" in model_components:
                    model_components["norm"].weight.data.copy_(state_dict[key])
                if "lm_head.weight" in key and "lm_head" in model_components:
                    model_components["lm_head"].weight.data.copy_(state_dict[key])

            if "layers." in key:
                parts = key.split(".")
                try:
                    layer_idx = int(parts[2])
                except ValueError:
                    continue

                if layer_idx in my_layer_indices:
                    local_idx = my_layer_indices.index(layer_idx)
                    module = my_layers[local_idx]
                    local_key = ".".join(parts[3:])
                    try:
                        sub_mod = module
                        sub_parts = local_key.split(".")
                        for sp in sub_parts[:-1]:
                            sub_mod = getattr(sub_mod, sp)
                        getattr(sub_mod, sub_parts[-1]).data.copy_(state_dict[key])
                    except AttributeError:
                        pass

        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[Rank {rank}] Weights loaded.", flush=True)


# ---------------------------------------------------------------------------
# FORWARD PASS HELPER
# ---------------------------------------------------------------------------


def forward_through_layers(layers, current_hidden, position_embeddings, mask, cache):
    """Run hidden states through this rank's decoder layers."""
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
    rank, world_size, split_config, result_queue, device_uuid, input_ids_seed, mb_size
):
    try:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        mig_transport.register_hooks()

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        config = LlamaConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "sdpa"

        start_layer = sum(split_config[:rank])
        end_layer = start_layer + split_config[rank]
        my_layer_indices = list(range(start_layer, end_layer))

        model_components = {}
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

        rotary_emb = LlamaRotaryEmbedding(config=config, device=device)
        load_specific_weights(rank, layers, my_layer_indices, model_components)

        gc.collect()
        torch.cuda.empty_cache()

        input_ids = input_ids_seed.to(device)
        batch_size, seq_length = input_ids.shape
        num_microbatches = batch_size // mb_size

        # One DynamicCache per microbatch — isolated K/V history per sequence group.
        # DynamicCache grows linearly across the 512 decode steps — if rank 2 OOMs
        # it will be caught by the except block and reported as OOM gracefully.
        past_key_values_list = [DynamicCache() for _ in range(num_microbatches)]

        # Full batch token tracking tensor — written by rank 2, read by rank 0
        next_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        # GPU utilization samples — only collected by rank 2 during decode
        util_samples = []

        #
        nvml_handle = None
        if rank == world_size - 1:
            try:
                pynvml.nvmlInit()
                nvml_handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid.encode())
            except Exception as e:
                print(f"[Rank 2] pynvml init failed: {e}. Util will be 0.", flush=True)

        dist.barrier()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.no_grad():

            # ==================================================================
            # PREFILL: process all SEQ_LEN=64 input tokens
            # Sequential microbatch processing here — prefill is one-time and
            # fast, async overhead isn't worth the complexity.
            # ==================================================================
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
                mask = torch.full(
                    (1, 1, seq_length, seq_length),
                    torch.finfo(torch.float16).min,
                    device=device,
                )
                mask = torch.triu(mask, diagonal=1).half()
                position_embeddings = rotary_emb(current_hidden, position_ids)

                current_hidden = forward_through_layers(
                    layers,
                    current_hidden,
                    position_embeddings,
                    mask,
                    past_key_values_list[mb_idx],
                )

                if rank == world_size - 1:
                    normed = model_components["norm"](current_hidden)
                    logits = model_components["lm_head"](normed[:, -1:, :])
                    mb_next = torch.argmax(logits, dim=-1)
                    next_tokens[start_idx:end_idx] = mb_next

                if rank < world_size - 1:
                    dist.send(current_hidden.clone(), dst=rank + 1)

            # Rank 2 sends first generated tokens back to rank 0
            if rank == world_size - 1:
                dist.send(next_tokens, dst=0)
            elif rank == 0:
                dist.recv(next_tokens, src=world_size - 1)

            dist.barrier()

            # ==================================================================
            # DECODE: generate MAX_NEW_TOKENS=512 tokens autoregressively
            #
            # Uses rolling prev_send_handle pattern:
            # - Issue isend for current microbatch
            # - Wait on it at the TOP of the NEXT microbatch iteration
            # - This means send and next microbatch's recv/compute overlap
            # - Only 1 slot occupied at a time — no slot exhaustion
            # ==================================================================
            for step in range(1, MAX_NEW_TOKENS + 1):

                # Sample rank 2 GPU utilization every UTIL_SAMPLE_INTERVAL steps
                if rank == world_size - 1 and step % UTIL_SAMPLE_INTERVAL == 0:
                    if nvml_handle is not None:
                        try:
                            rates = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                            util_samples.append(rates.gpu)
                        except Exception:
                            pass

                position_ids = torch.tensor(
                    [[seq_length + step - 1]], dtype=torch.long, device=device
                ).expand(mb_size, -1)

                prev_send_handle = None

                for mb_idx in range(num_microbatches):
                    start_idx = mb_idx * mb_size
                    end_idx = start_idx + mb_size

                    # Wait on the PREVIOUS microbatch's send before claiming
                    # a new slot. This keeps slot usage at 1 at a time while
                    # still overlapping the send with current microbatch compute.
                    if prev_send_handle is not None:
                        prev_send_handle.wait()
                        prev_send_handle = None

                    # --- RECEIVE ---
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

                    # --- COMPUTE ---
                    position_embeddings = rotary_emb(current_hidden, position_ids)
                    current_hidden = forward_through_layers(
                        layers,
                        current_hidden,
                        position_embeddings,
                        None,
                        past_key_values_list[mb_idx],
                    )

                    # --- TOKEN GENERATION (rank 2 only) ---
                    if rank == world_size - 1:
                        normed = model_components["norm"](current_hidden)
                        logits = model_components["lm_head"](normed)
                        mb_next = torch.argmax(logits, dim=-1)
                        next_tokens[start_idx:end_idx] = mb_next

                    # --- NON-BLOCKING SEND ---
                    # Fire and move on — wait happens at top of next iteration
                    if rank < world_size - 1:
                        prev_send_handle = dist.isend(
                            current_hidden.clone(), dst=rank + 1
                        )

                # Wait on the final microbatch's send before leaving the step
                if prev_send_handle is not None:
                    prev_send_handle.wait()

                # Global token feedback: rank 2 → rank 0 (once per step, blocking)
                if step < MAX_NEW_TOKENS:
                    if rank == world_size - 1:
                        dist.send(next_tokens, dst=0)
                    elif rank == 0:
                        dist.recv(next_tokens, src=world_size - 1)

        dist.barrier()
        end_event.record()
        torch.cuda.synchronize()

        total_latency_ms = start_event.elapsed_time(end_event)
        avg_util = sum(util_samples) / len(util_samples) if util_samples else 0.0

        if rank == 0:
            result_queue.put(("latency", total_latency_ms))
        if rank == world_size - 1:
            result_queue.put(("util", avg_util))

        dist.destroy_process_group()

    except torch.cuda.OutOfMemoryError:
        # OOM is expected for large batch + large microbatch + many layers on rank 2.
        # Report it clearly so main() can log it and move on to the next config.
        print(
            f"[Rank {rank}] OOM — batch/microbatch too large for this split.",
            flush=True,
        )
        result_queue.put(("oom", rank))
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    except Exception:
        import traceback

        traceback.print_exc()
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
    selected_splits = generate_layer_splits()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Build the full list of valid (batch_size, mb_size) combinations.
    # mb_size must divide evenly into batch_size and must be <= batch_size.
    valid_batch_mb_pairs = [
        (bs, mb)
        for bs in BATCH_SIZES
        for mb in MICROBATCH_SIZES
        if bs % mb == 0 and mb <= bs
    ]

    total_runs = len(selected_splits) * len(valid_batch_mb_pairs)
    results = []
    current_run = 1

    for split in selected_splits:
        for batch_size, mb_size in valid_batch_mb_pairs:
            print(
                f"\n[{current_run}/{total_runs}] "
                f"Split {split} | Batch: {batch_size} | Microbatch: {mb_size}"
            )

            # Load input data sized to this run's batch size
            input_ids_seed = get_wiki_sample(batch_size)

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

                for p in procs:
                    p.join()

                exit_codes = [p.exitcode for p in procs]

                # Collect all items the processes put in the queue
                queue_items = {}
                try:
                    while True:
                        key, val = q.get(timeout=2.0)
                        queue_items[key] = val
                except queue.Empty:
                    pass

                # Check for OOM first — if any rank reported OOM, skip this config
                if "oom" in queue_items:
                    oom_rank = queue_items["oom"]
                    print(f"  OOM on Rank {oom_rank} — skipping this configuration.")
                    results.append(
                        {
                            "split": str(split),
                            "batch_size": batch_size,
                            "microbatch_size": mb_size,
                            "num_microbatches": batch_size // mb_size,
                            "total_latency_ms": None,
                            "rank2_gpu_util_pct": None,
                            "status": f"OOM_rank{oom_rank}",
                        }
                    )

                elif any(code != 0 for code in exit_codes):
                    print(f"  Failed (Crash). Exit codes: {exit_codes}")
                    results.append(
                        {
                            "split": str(split),
                            "batch_size": batch_size,
                            "microbatch_size": mb_size,
                            "num_microbatches": batch_size // mb_size,
                            "total_latency_ms": None,
                            "rank2_gpu_util_pct": None,
                            "status": "crash",
                        }
                    )

                elif "latency" in queue_items and "util" in queue_items:
                    latency = queue_items["latency"]
                    util = queue_items["util"]
                    print(f"  Total latency:   {latency:.0f} ms")
                    print(f"  Rank 2 GPU util: {util:.1f}%")
                    results.append(
                        {
                            "split": str(split),
                            "batch_size": batch_size,
                            "microbatch_size": mb_size,
                            "num_microbatches": batch_size // mb_size,
                            "total_latency_ms": latency,
                            "rank2_gpu_util_pct": util,
                            "status": "ok",
                        }
                    )

                else:
                    print("  Timeout — no results received.")
                    results.append(
                        {
                            "split": str(split),
                            "batch_size": batch_size,
                            "microbatch_size": mb_size,
                            "num_microbatches": batch_size // mb_size,
                            "total_latency_ms": None,
                            "rank2_gpu_util_pct": None,
                            "status": "timeout",
                        }
                    )

            finally:
                q.close()
                q.join_thread()

            for p in procs:
                if p.is_alive():
                    p.terminate()

            current_run += 1

    df = pd.DataFrame(results)
    df.to_csv("mig_benchmark_results.csv", index=False)

    # Summary over successful runs only
    successful = df[df["status"] == "ok"]
    if not successful.empty:
        print("\n--- Best by Latency (successful runs) ---")
        best_lat = successful.loc[successful["total_latency_ms"].idxmin()]
        print(
            f"  Split: {best_lat['split']} | Batch: {best_lat['batch_size']} "
            f"| MB: {best_lat['microbatch_size']} "
            f"| Latency: {best_lat['total_latency_ms']:.0f} ms "
            f"| Rank2 Util: {best_lat['rank2_gpu_util_pct']:.1f}%"
        )

        print("\n--- Best by Rank 2 GPU Utilization (successful runs) ---")
        best_util = successful.loc[successful["rank2_gpu_util_pct"].idxmax()]
        print(
            f"  Split: {best_util['split']} | Batch: {best_util['batch_size']} "
            f"| MB: {best_util['microbatch_size']} "
            f"| Latency: {best_util['total_latency_ms']:.0f} ms "
            f"| Rank2 Util: {best_util['rank2_gpu_util_pct']:.1f}%"
        )

        oom_count = len(df[df["status"].str.startswith("OOM", na=False)])
        print(f"\n  ({oom_count} configurations skipped due to OOM)")

    print("\nDone. Results saved to mig_benchmark_results.csv")


if __name__ == "__main__":
    main()
