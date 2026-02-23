import os
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
from transformers import LlamaConfig, AutoTokenizer
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
SEQ_LEN = 512
HEADS = 32

# --- CORRECT UUID MAPPING (Based on your output) ---
MIG_UUIDS = [
    "MIG-39575ee3-5daf-5ea3-978d-921d7fe80356",  # Rank 0: 20GB (Big Slice)
    "MIG-131d9102-cc9b-577d-a56e-1922e6e94d21",  # Rank 1: 10GB (Medium Slice)
    "MIG-4c5fa4db-de54-596b-9031-1619d41d2ab8",  # Rank 2: 5GB  (Small Slice)
]

# Max layers per slice.
# Rank 2 is strictly capped at 8 to prevent OOM.
LAYER_LIMITS = [32, 20, 8]


def get_wiki_sample():
    """Fetches real data."""
    print("Loading WikiText dataset...")
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
        return inputs.input_ids
    except Exception:
        print("Warning: Could not load WikiText. Using Random Data.")
        return torch.randint(0, 32000, (1, SEQ_LEN))


def load_specific_weights(rank, my_layers, my_layer_indices, model_components):
    """
    Surgically loads ONLY the weights needed for this specific rank.
    """
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

    # Iterate over each shard file
    for shard_file in tqdm(
        shard_files, desc=f"Rank {rank} Loading Shards", leave=False
    ):
        file_path = os.path.join(folder_path, shard_file)

        # Load to CPU memory first
        state_dict = torch.load(file_path, map_location="cpu")

        keys_to_process = list(state_dict.keys())
        for key in keys_to_process:
            # -- Embeddings (Rank 0 only) --
            if rank == 0 and "embed_tokens" in key and "embed" in model_components:
                model_components["embed"].weight.data.copy_(state_dict[key])

            # -- Final Norm (Last Rank only) --
            if rank == 2 and "norm.weight" in key and "norm" in model_components:
                model_components["norm"].weight.data.copy_(state_dict[key])

            # -- Layers --
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

        # Aggressive Cleanup to protect the 5GB slice
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[Rank {rank}] Weights Loaded.", flush=True)


def run_pipeline(
    rank, world_size, split_config, result_queue, device_uuid, input_ids_seed
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

        # 1. Configure Architecture
        config = LlamaConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "eager"

        print(
            f"[Rank {rank}] head_dim={config.head_dim}, hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}",
            flush=True,
        )

        # 2. Determine Which Layers I Own
        start_layer = sum(split_config[:rank])
        end_layer = start_layer + split_config[rank]
        my_layer_indices = list(range(start_layer, end_layer))
        print(f"[Rank {rank}] Owning Global Layers: {my_layer_indices}")

        # 3. Create Components
        model_components = {}
        if rank == 0:
            model_components["embed"] = (
                nn.Embedding(config.vocab_size, config.hidden_size).to(device).half()
            )

        layers = nn.ModuleList()
        for idx in my_layer_indices:
            layer = LlamaDecoderLayer(config, layer_idx=idx).half()
            layers.append(layer.to(device))
            torch.cuda.empty_cache()

        if rank == world_size - 1:
            model_components["norm"] = (
                LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                .to(device)
                .half()
            )

        # Rotary embedding — config already has correct head_dim=128, no overrides needed
        # Keep in float32 (no .half())
        rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

        # 4. Load Real Weights
        load_specific_weights(rank, layers, my_layer_indices, model_components)

        gc.collect()
        torch.cuda.empty_cache()

        # 5. Pipeline Execution
        input_ids = input_ids_seed.to(device)
        batch_size, seq_length = input_ids.shape

        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=device
        ).unsqueeze(0)

        mask = torch.full(
            (1, 1, seq_length, seq_length),
            torch.finfo(torch.float16).min,
            device=device,
        )
        mask = torch.triu(mask, diagonal=1).half()

        buffer_shape = (batch_size, seq_length, config.hidden_size)
        recv_buffer = torch.zeros(buffer_shape, dtype=torch.float16, device=device)
        send_buffer = torch.zeros(buffer_shape, dtype=torch.float16, device=device)

        dist.barrier()

        steps = 5
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.no_grad():
            for _ in range(steps):
                # RECV
                if rank == 0:
                    current_hidden = model_components["embed"](input_ids).half()
                else:
                    dist.recv(recv_buffer, src=rank - 1)
                    current_hidden = recv_buffer.clone()

                # Compute RoPE using the real hidden states — correct shape (batch, seq_len, hidden_size)
                position_embeddings = rotary_emb(current_hidden, position_ids)
                cos, sin = position_embeddings
                print(
                    f"[DEBUG Rank {rank}] current_hidden shape={current_hidden.shape}",
                    flush=True,
                )
                print(f"[DEBUG Rank {rank}] cos shape={cos.shape}", flush=True)
                print(f"[DEBUG Rank {rank}] sin shape={sin.shape}", flush=True)
                # Only run 1 step then break to avoid cascade failures
                # break

                # COMPUTE
                for layer in layers:
                    # Manual forward — bypass all wrappers
                    residual = current_hidden
                    hidden_states = layer.input_layernorm(current_hidden)

                    # Call attention directly with correct args
                    hidden_states, _ = layer.self_attn(
                        hidden_states=hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=mask,
                    )
                    hidden_states = residual + hidden_states

                    # MLP
                    residual = hidden_states
                    hidden_states = layer.post_attention_layernorm(hidden_states)
                    hidden_states = layer.mlp(hidden_states)
                    current_hidden = residual + hidden_states

                # FINAL NORM
                if rank == world_size - 1:
                    current_hidden = model_components["norm"](current_hidden)

                # SEND
                if rank < world_size - 1:
                    send_buffer.copy_(current_hidden)
                    dist.send(send_buffer, dst=rank + 1)

        dist.barrier()
        end_event.record()
        torch.cuda.synchronize()

        avg_ms = start_event.elapsed_time(end_event) / steps
        if rank == 0:
            result_queue.put(avg_ms)

        dist.destroy_process_group()

    except Exception:
        import traceback

        traceback.print_exc()
        return


def generate_layer_splits():
    valid_splits = []
    # Rank 0 (20GB) can take many layers
    # Rank 2 (5GB) is strictly limited to 8
    for l0 in range(1, LAYER_LIMITS[0] + 1):
        for l1 in range(1, LAYER_LIMITS[1] + 1):
            l2 = TOTAL_LAYERS - (l0 + l1)
            if 1 <= l2 <= LAYER_LIMITS[2]:  # Ensure l2 <= 8
                valid_splits.append([l0, l1, l2])
    return valid_splits


def main():
    input_ids_seed = get_wiki_sample()
    selected_splits = generate_layer_splits()
    splits_amt = len(selected_splits)

    print(f"Benchmarking {splits_amt} configurations with REAL WEIGHTS...")
    print(f"Rank 0 (20GB) | Rank 1 (10GB) | Rank 2 (5GB - Limit 8 Layers)")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    results = []

    for i, split in enumerate(selected_splits):
        print(f"\n[{i+1}/{splits_amt}] Split {split} ...")
        q = mp.Queue()
        procs = []

        try:
            # Start processes
            for rank in range(3):
                p = mp.Process(
                    target=run_pipeline,
                    args=(rank, 3, split, q, MIG_UUIDS[rank], input_ids_seed),
                )
                p.start()
                procs.append(p)

            # Wait for processes to finish
            for p in procs:
                p.join()

            exit_codes = [p.exitcode for p in procs]
            if any(code != 0 for code in exit_codes):
                print("Failed (OOM/Crash).")
            else:
                try:
                    lat = q.get(timeout=60.0)
                    print(f"Success: {lat:.2f} ms")
                    results.append({"split": str(split), "latency_ms": lat})
                except queue.Empty:
                    print("Timeout.")
        finally:
            # Cleanup queues
            q.close()
            q.join_thread()

        # Terminate any alive processes
        for p in procs:
            if p.is_alive():
                p.terminate()

    pd.DataFrame(results).to_csv("mig_real_weights_final.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
