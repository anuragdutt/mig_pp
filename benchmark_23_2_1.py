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

SEQ_LEN = 64  # Input (prefill) length
MAX_NEW_TOKENS = 512  # Output (decode) length

GLOBAL_BATCH_SIZE = 128

# Microbatch sizes — must divide evenly into GLOBAL_BATCH_SIZE
MICROBATCH_SIZES = [16, 32, 64]

# How often to sample GPU utilization during decode (every N steps).
# Sampling every step would add overhead; every 10 steps is a good balance.
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


def get_wiki_sample() -> torch.Tensor:
    print(f"Loading WikiText... (SEQ_LEN={SEQ_LEN}, BATCH={GLOBAL_BATCH_SIZE})")
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
        return inputs.input_ids.repeat(GLOBAL_BATCH_SIZE, 1)

    except Exception:
        print("Warning: WikiText unavailable. Using random token IDs.")
        return torch.randint(0, 32000, (GLOBAL_BATCH_SIZE, SEQ_LEN))


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
    """
    Run hidden states through this rank's decoder layers.
    Separated into its own function so it's easy to call from both
    the prefill path and the async decode loop.
    """
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

        # Register async-capable transport hooks (replaces dist.send/recv/isend/irecv)
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
        batch_size, seq_length = input_ids.shape  # (128, 64)
        num_microbatches = batch_size // mb_size

        # One DynamicCache per microbatch — isolated K/V history per sequence group
        past_key_values_list = [DynamicCache() for _ in range(num_microbatches)]

        # Full batch token tracking tensor — written by rank 2, read by rank 0
        next_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        # GPU utilization samples — only collected by rank 2 during decode
        util_samples = []

        dist.barrier()

        # --- TIMING: measure total end-to-end latency ---
        # start_event fires before prefill, end_event fires after the last decode step.
        # elapsed_time() between them = total response latency.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.no_grad():

            # ==================================================================
            # STEP 0: PREFILL (process all 64 input tokens in one shot)
            # Microbatches are still processed sequentially here — prefill is
            # one-time and fast (64 tokens), so the overhead of async scheduling
            # isn't worth the complexity. Concurrency matters in the decode phase.
            # ==================================================================
            prefill_mb_buffers = []  # Store prefill outputs to kick off decode loop

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

                # Build causal mask for prefill
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
                    send_buf = current_hidden.clone()
                    dist.send(send_buf, dst=rank + 1)

            # After prefill: rank 2 sends first generated tokens to rank 0
            if rank == world_size - 1:
                dist.send(next_tokens, dst=0)
            elif rank == 0:
                dist.recv(next_tokens, src=world_size - 1)

            dist.barrier()  # All ranks aligned before decode loop starts

            # ==================================================================
            # STEPS 1..MAX_NEW_TOKENS: DECODE (concurrent async pipeline)
            #
            # This is the GPipe fill-drain schedule:
            # - Rank 0 embeds MB0, sends it async, immediately starts MB1
            # - Rank 1 receives MB0, processes it, sends async, receives MB1...
            # - Rank 2 receives MB0, generates token, while rank 1 already has MB1
            #
            # Each rank maintains a queue of outstanding async send handles.
            # It calls handle.wait() only when it needs to reuse that buffer slot,
            # not immediately after isend — this is what enables overlap.
            # ==================================================================

            for step in range(1, MAX_NEW_TOKENS + 1):
                # Sample rank 2 GPU utilization periodically during decode.
                # torch.cuda.utilization() returns SM utilization % (0-100).
                # We sample here (top of step) so we capture utilization while
                # the GPU is actively running decode kernels from the previous step.
                if rank == world_size - 1 and step % UTIL_SAMPLE_INTERVAL == 0:
                    util_samples.append(torch.cuda.utilization())

                position_ids = torch.tensor(
                    [[seq_length + step - 1]], dtype=torch.long, device=device
                ).expand(mb_size, -1)

                # Outstanding async send handles from this step — we collect them
                # all and wait at the end of the step to ensure all microbatch
                # data has been received before moving to next step.
                # This gives maximum overlap within a step while still maintaining
                # correct ordering between steps.
                send_handles = []

                for mb_idx in range(num_microbatches):
                    start_idx = mb_idx * mb_size
                    end_idx = start_idx + mb_size

                    # --- RECEIVE from previous rank (non-blocking) ---
                    if rank == 0:
                        # Rank 0 embeds the token predicted in previous step
                        current_hidden = model_components["embed"](
                            next_tokens[start_idx:end_idx]
                        ).half()
                    else:
                        recv_buf = torch.zeros(
                            (mb_size, 1, config.hidden_size),
                            dtype=torch.float16,
                            device=device,
                        )
                        # Non-blocking recv — start receiving while we process
                        # the previous microbatch's output on this rank
                        recv_handle = dist.irecv(recv_buf, src=rank - 1)
                        # Must wait before we can use recv_buf for compute
                        recv_handle.wait()
                        current_hidden = recv_buf

                    # --- COMPUTE: forward through this rank's layers ---
                    position_embeddings = rotary_emb(current_hidden, position_ids)
                    current_hidden = forward_through_layers(
                        layers,
                        current_hidden,
                        position_embeddings,
                        None,  # No causal mask needed for single-token decode
                        past_key_values_list[mb_idx],
                    )

                    # --- TOKEN GENERATION (rank 2 only) ---
                    if rank == world_size - 1:
                        normed = model_components["norm"](current_hidden)
                        logits = model_components["lm_head"](normed)
                        mb_next = torch.argmax(logits, dim=-1)
                        next_tokens[start_idx:end_idx] = mb_next

                    # --- NON-BLOCKING SEND to next rank ---
                    # isend fires immediately and returns a handle.
                    # We don't wait here — we let the next microbatch's compute
                    # overlap with this send completing in the background.
                    if rank < world_size - 1:
                        send_buf = current_hidden.clone()
                        handle = dist.isend(send_buf, dst=rank + 1)
                        send_handles.append(handle)

                # Wait for all sends from this step to complete before
                # moving to the next step. This ensures rank (n+1) has received
                # all microbatches before we overwrite the send buffers.
                for handle in send_handles:
                    handle.wait()

                # Global token feedback: rank 2 → rank 0 (blocking, once per step)
                # This is necessarily sequential — rank 0 needs all new tokens
                # before it can start the next decode step's embedding.
                if step < MAX_NEW_TOKENS:
                    if rank == world_size - 1:
                        dist.send(next_tokens, dst=0)
                    elif rank == 0:
                        dist.recv(next_tokens, src=world_size - 1)

        dist.barrier()
        end_event.record()
        torch.cuda.synchronize()

        # Total end-to-end latency: prefill + all 512 decode steps
        total_latency_ms = start_event.elapsed_time(end_event)

        # Average GPU utilization on rank 2 across all sampled decode steps
        avg_util = sum(util_samples) / len(util_samples) if util_samples else 0.0

        if rank == 0:
            # Send latency from rank 0 (it has the timing)
            result_queue.put(("latency", total_latency_ms))

        if rank == world_size - 1:
            # Send utilization from rank 2 (it has the util samples)
            result_queue.put(("util", avg_util))

        dist.destroy_process_group()

    except Exception:
        import traceback

        traceback.print_exc()
        return


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
    input_ids_seed = get_wiki_sample()
    selected_splits = generate_layer_splits()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    total_runs = len(selected_splits) * len(MICROBATCH_SIZES)
    results = []
    current_run = 1

    for split in selected_splits:
        for mb_size in MICROBATCH_SIZES:
            print(
                f"\n[{current_run}/{total_runs}] Split {split} | Microbatch: {mb_size}"
            )

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
                if any(code != 0 for code in exit_codes):
                    print(f"  Failed (OOM/Crash). Exit codes: {exit_codes}")
                else:
                    # Collect both metrics from the queue.
                    # rank 0 puts latency, rank 2 puts utilization.
                    metrics = {}
                    try:
                        for _ in range(2):  # Expecting exactly 2 items
                            key, val = q.get(timeout=180.0)
                            metrics[key] = val

                        latency = metrics.get("latency", -1)
                        util = metrics.get("util", -1)

                        print(f"  Total latency:    {latency:.0f} ms")
                        print(f"  Rank 2 GPU util:  {util:.1f}%")

                        results.append(
                            {
                                "split": str(split),
                                "batch_size": GLOBAL_BATCH_SIZE,
                                "microbatch_size": mb_size,
                                "total_latency_ms": latency,
                                "rank2_gpu_util_pct": util,
                            }
                        )
                    except queue.Empty:
                        print("  Timeout waiting for results.")

            finally:
                q.close()
                q.join_thread()

            for p in procs:
                if p.is_alive():
                    p.terminate()

            current_run += 1

    df = pd.DataFrame(results)
    df.to_csv("mig_benchmark_results.csv", index=False)

    if not df.empty:
        print("\n--- Best by Latency ---")
        best_lat = df.loc[df["total_latency_ms"].idxmin()]
        print(
            f"  Split: {best_lat['split']} | MB: {best_lat['microbatch_size']} "
            f"| Latency: {best_lat['total_latency_ms']:.0f} ms "
            f"| Rank2 Util: {best_lat['rank2_gpu_util_pct']:.1f}%"
        )

        print("\n--- Best by Rank 2 GPU Utilization ---")
        best_util = df.loc[df["rank2_gpu_util_pct"].idxmax()]
        print(
            f"  Split: {best_util['split']} | MB: {best_util['microbatch_size']} "
            f"| Latency: {best_util['total_latency_ms']:.0f} ms "
            f"| Rank2 Util: {best_util['rank2_gpu_util_pct']:.1f}%"
        )

    print("\nDone. Results saved to mig_benchmark_results.csv")


if __name__ == "__main__":
    main()
