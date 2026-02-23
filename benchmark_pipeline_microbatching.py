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
from transformers import DynamicCache, LlamaConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils import hub
from datasets import load_dataset

import mig_transport_pipeline as mig_transport

# --- CONFIGURATION ---
MODEL_NAME = "lmsys/vicuna-7b-v1.5"
TOTAL_LAYERS = 32
HIDDEN_SIZE = 4096
HEADS = 32
MAX_NEW_TOKENS = 5  # Number of autoregressive steps to perform

# FEEDBACK APPLIED: Fixed Input Length & Global Batch Size
SEQ_LEN = 64
GLOBAL_BATCH_SIZE = 128
MICROBATCH_SIZES = [4, 8, 16, 32, 64]  # Powers of 2 up to BATCH/2

# --- CORRECT UUID MAPPING ---
MIG_UUIDS = [
    "MIG-997bbe72-d0a5-53f4-ba78-f3b7b2a21687",  # Rank 0: 20GB (Big Slice)
    "MIG-93c2210e-e768-54a7-a5d7-bf8e2bb54d4d",  # Rank 1: 10GB (Medium Slice)
    "MIG-b3b486dd-a81a-5a46-910c-950d02628804",  # Rank 2: 5GB  (Small Slice)
]

LAYER_LIMITS = [24, 12, 6]


def get_wiki_sample():
    print(
        f"Loading WikiText dataset... (Fixed to Seq Len: {SEQ_LEN}, Batch: {GLOBAL_BATCH_SIZE})"
    )
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
        # Duplicate the single prompt to match the target GLOBAL_BATCH_SIZE
        input_ids = inputs.input_ids.repeat(GLOBAL_BATCH_SIZE, 1)
        return input_ids
    except Exception:
        print("Warning: Could not load WikiText. Using Random Data.")
        return torch.randint(0, 32000, (GLOBAL_BATCH_SIZE, SEQ_LEN))


def load_specific_weights(rank, my_layers, my_layer_indices, model_components):
    print(f"[Rank {rank}] Surgical weight loading started...", flush=True)

    try:
        cached_index = hub.cached_file(MODEL_NAME, "pytorch_model.bin.index.json")
        folder_path = os.path.dirname(cached_index)
        with open(cached_index, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        shard_files = set(weight_map.values())
    except Exception:
        print(
            f"[Rank {rank}] Warning: Could not find weight map. Skipping weight load."
        )
        return

    for shard_file in tqdm(
        shard_files, desc=f"Rank {rank} Loading Shards", leave=False
    ):
        file_path = os.path.join(folder_path, shard_file)
        state_dict: Dict[str, torch.Tensor] = torch.load(file_path, map_location="cpu")

        keys_to_process = list(state_dict.keys())
        for key in keys_to_process:
            # Embeddings (Rank 0)
            if rank == 0 and "embed_tokens" in key and "embed" in model_components:
                model_components["embed"].weight.data.copy_(state_dict[key])

            # Final Norm & LM Head (Rank 2)
            if rank == 2:
                if "norm.weight" in key and "norm" in model_components:
                    model_components["norm"].weight.data.copy_(state_dict[key])
                if "lm_head.weight" in key and "lm_head" in model_components:
                    model_components["lm_head"].weight.data.copy_(state_dict[key])

            # Layers
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

                        param_name = sub_parts[-1]
                        getattr(sub_mod, param_name).data.copy_(state_dict[key])
                    except AttributeError:
                        pass

        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[Rank {rank}] Weights Loaded.", flush=True)


def run_pipeline(
    rank, world_size, split_config, result_queue, device_uuid, input_ids_seed, mb_size
):
    try:
        # -----------------------------
        # 1️⃣ ENVIRONMENT CONFIGURATION
        # -----------------------------

        # Allow CUDA memory allocator to grow memory segments instead of preallocating.
        # Helps reduce fragmentation when loading large models.
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        # Restrict this process to a single MIG GPU slice.
        # Important: Each rank sees ONLY one GPU (mapped to cuda:0 locally).
        os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid

        # Distributed master node configuration.
        # All ranks communicate via this address and port.
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        # Initialize distributed communication backend.
        # "gloo" is CPU-based and works for point-to-point send/recv.
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Register custom hooks for MIG transport (likely custom optimized send/recv).
        mig_transport.register_hooks()

        # -----------------------------
        # 2️⃣ DEVICE SETUP
        # -----------------------------

        # Because CUDA_VISIBLE_DEVICES was set,
        # this rank only sees ONE GPU -> cuda:0.
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # -----------------------------
        # 3️⃣ LOAD MODEL CONFIG
        # -----------------------------

        # Load model configuration (no weights yet).
        config = LlamaConfig.from_pretrained(MODEL_NAME)

        # Use PyTorch’s optimized Scaled Dot Product Attention kernel.
        config._attn_implementation = "sdpa"

        # -----------------------------
        # 4️⃣ PIPELINE LAYER PARTITIONING
        # -----------------------------

        # split_config example: [10, 10, 12]
        # Means:
        #   rank 0 -> layers 0-9
        #   rank 1 -> layers 10-19
        #   rank 2 -> layers 20-31

        start_layer = sum(split_config[:rank])
        end_layer = start_layer + split_config[rank]

        # Store exact layer indices owned by this rank
        my_layer_indices = list(range(start_layer, end_layer))

        model_components = {}

        # Rank 0 owns embedding layer.
        # Embedding must only exist once in pipeline.
        if rank == 0:
            model_components["embed"] = (
                nn.Embedding(config.vocab_size, config.hidden_size)
                .to(device)
                .half()  # Use FP16 to reduce memory footprint
            )

        # -----------------------------
        # 5️⃣ CREATE TRANSFORMER LAYERS
        # -----------------------------

        layers = nn.ModuleList()

        # Each rank only constructs its assigned decoder layers.
        # This is model parallelism.
        for idx in my_layer_indices:
            layer = LlamaDecoderLayer(config, layer_idx=idx).half()
            layers.append(layer.to(device))

            # Free any temporary memory from layer creation
            torch.cuda.empty_cache()

        # Last rank owns final norm + LM head.
        # These must be on the final stage because logits are computed there.
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

        # Rotary positional embedding module (shared logic across layers)
        rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

        # Load only the weights relevant to this rank.
        load_specific_weights(rank, layers, my_layer_indices, model_components)

        # Clean up memory before starting inference.
        gc.collect()
        torch.cuda.empty_cache()

        # -----------------------------
        # 6️⃣ INPUT + MICROBATCHING SETUP
        # -----------------------------

        input_ids = input_ids_seed.to(device)
        batch_size, seq_length = input_ids.shape

        # Microbatching splits large batch into smaller chunks.
        # This enables pipeline parallel overlap and lower memory usage.
        num_microbatches = batch_size // mb_size

        # IMPORTANT:
        # Each microbatch needs its OWN KV cache.
        # Otherwise attention states from different microbatches would mix.
        past_key_values_list = [DynamicCache() for _ in range(num_microbatches)]

        # This tensor tracks generated tokens for the full global batch.
        next_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        # Synchronize all ranks before timing starts.
        dist.barrier()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Disable gradient tracking (pure inference mode).
        with torch.no_grad():

            # step 0 = PREFILL
            # step > 0 = AUTOREGRESSIVE DECODE
            for step in range(MAX_NEW_TOKENS + 1):

                is_prefill = step == 0

                # During prefill:
                #   full sequence processed
                # During decode:
                #   only 1 new token processed
                current_seq_len = seq_length if is_prefill else 1

                # Buffers used for inter-rank communication.
                buffer_shape = (mb_size, current_seq_len, config.hidden_size)

                recv_buffer = torch.zeros(
                    buffer_shape, dtype=torch.float16, device=device
                )
                send_buffer = torch.zeros(
                    buffer_shape, dtype=torch.float16, device=device
                )

                # -----------------------------
                # 7️⃣ PROCESS EACH MICROBATCH
                # -----------------------------

                for mb_idx in range(num_microbatches):

                    start_idx = mb_idx * mb_size
                    end_idx = start_idx + mb_size

                    # -----------------------------
                    # 7.1 INPUT STAGE
                    # -----------------------------

                    if rank == 0:
                        # Rank 0 converts tokens → embeddings.
                        current_input_ids = (
                            input_ids[start_idx:end_idx]
                            if is_prefill
                            else next_tokens[start_idx:end_idx]
                        )

                        current_hidden = model_components["embed"](
                            current_input_ids
                        ).half()
                    else:
                        # Other ranks receive activations from previous stage.
                        dist.recv(recv_buffer, src=rank - 1)
                        current_hidden = recv_buffer.clone()

                    # -----------------------------
                    # 7.2 POSITION + MASK LOGIC
                    # -----------------------------

                    if is_prefill:
                        # Create full causal mask for initial sequence.
                        position_ids = torch.arange(
                            0, current_seq_len, dtype=torch.long, device=device
                        ).unsqueeze(0)

                        mask = torch.full(
                            (1, 1, current_seq_len, current_seq_len),
                            torch.finfo(torch.float16).min,
                            device=device,
                        )

                        # Upper triangular mask blocks future tokens.
                        mask = torch.triu(mask, diagonal=1).half()
                    else:
                        # During decoding:
                        # No need for mask because we attend only to KV cache.
                        position_ids = torch.tensor(
                            [[seq_length + step - 1]],
                            dtype=torch.long,
                            device=device,
                        )
                        mask = None

                    position_embeddings = rotary_emb(current_hidden, position_ids)

                    # -----------------------------
                    # 7.3 TRANSFORMER LAYER FORWARD
                    # -----------------------------

                    for layer in layers:

                        residual = current_hidden

                        # Pre-attention layer norm
                        hidden_states = layer.input_layernorm(current_hidden)

                        # Self-attention with KV cache
                        attn_outputs = layer.self_attn(
                            hidden_states=hidden_states,
                            position_embeddings=position_embeddings,
                            attention_mask=mask,
                            past_key_values=past_key_values_list[mb_idx],
                            use_cache=True,  # Enables autoregressive KV reuse
                        )

                        hidden_states = attn_outputs[0]
                        hidden_states = residual + hidden_states  # Residual add

                        residual = hidden_states

                        # Feedforward block
                        hidden_states = layer.post_attention_layernorm(hidden_states)
                        hidden_states = layer.mlp(hidden_states)

                        current_hidden = residual + hidden_states  # Final residual

                    # -----------------------------
                    # 7.4 FINAL TOKEN GENERATION
                    # -----------------------------

                    if rank == world_size - 1:

                        current_hidden = model_components["norm"](current_hidden)

                        # Only last token is used for next token prediction.
                        last_token_hidden = (
                            current_hidden[:, -1:, :] if is_prefill else current_hidden
                        )

                        logits = model_components["lm_head"](last_token_hidden)

                        # Greedy decoding
                        mb_next_tokens = torch.argmax(logits, dim=-1)

                        # Store into global tensor
                        next_tokens[start_idx:end_idx] = mb_next_tokens

                    # -----------------------------
                    # 7.5 SEND TO NEXT PIPELINE STAGE
                    # -----------------------------

                    if rank < world_size - 1:
                        send_buffer.copy_(current_hidden)
                        dist.send(send_buffer, dst=rank + 1)

                # -----------------------------
                # 8️⃣ GLOBAL TOKEN FEEDBACK
                # -----------------------------

                # After all microbatches are processed,
                # last rank sends full batch next_tokens back to rank 0.
                if step < MAX_NEW_TOKENS:
                    if rank == world_size - 1:
                        dist.send(next_tokens, dst=0)
                    elif rank == 0:
                        dist.recv(next_tokens, src=world_size - 1)

        # Synchronize all ranks before stopping timer.
        dist.barrier()

        end_event.record()
        torch.cuda.synchronize()

        # Average time per token (including prefill).
        avg_ms = start_event.elapsed_time(end_event) / (MAX_NEW_TOKENS + 1)

        # Only rank 0 reports final latency.
        if rank == 0:
            result_queue.put(avg_ms)

        # Clean up distributed resources.
        dist.destroy_process_group()

    except Exception:
        import traceback

        traceback.print_exc()
        return


def generate_layer_splits():
    valid_splits = []
    for l0 in range(1, LAYER_LIMITS[0] + 1):
        for l1 in range(1, LAYER_LIMITS[1] + 1):
            l2 = TOTAL_LAYERS - (l0 + l1)
            if 1 <= l2 <= LAYER_LIMITS[2]:
                valid_splits.append([l0, l1, l2])
    return valid_splits


def main():
    input_ids_seed = get_wiki_sample()
    selected_splits = generate_layer_splits()

    # Calculate total iterations (Splits * Number of Microbatch Configurations)
    total_runs = len(selected_splits) * len(MICROBATCH_SIZES)

    print(f"Benchmarking {total_runs} configurations (Real Weights + Microbatching)...")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    results = []
    current_run = 1

    # Nested loop to test each split against each microbatch size
    for split in selected_splits:
        for mb_size in MICROBATCH_SIZES:
            print(
                f"\n[{current_run}/{total_runs}] Split {split} | Microbatch Size: {mb_size} ..."
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
                    print("Failed (OOM/Crash).")
                else:
                    try:
                        lat = q.get(timeout=60.0)
                        print(
                            f"Success: {lat:.2f} ms per step (Prefill + Decode Average)"
                        )
                        results.append(
                            {
                                "split": str(split),
                                "batch_size": GLOBAL_BATCH_SIZE,
                                "microbatch_size": mb_size,
                                "latency_ms": lat,
                            }
                        )
                    except queue.Empty:
                        print("Timeout.")
            finally:
                q.close()
                q.join_thread()

            for p in procs:
                if p.is_alive():
                    p.terminate()

            current_run += 1

    pd.DataFrame(results).to_csv("mig_microbatching_results.csv", index=False)
    print("\nDone. Results saved to mig_microbatching_results.csv")


if __name__ == "__main__":
    main()
