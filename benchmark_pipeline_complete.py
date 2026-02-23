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
SEQ_LEN = 512
HEADS = 32
MAX_NEW_TOKENS = 5  # Number of autoregressive steps to perform

# --- CORRECT UUID MAPPING ---
MIG_UUIDS = [
    "MIG-997bbe72-d0a5-53f4-ba78-f3b7b2a21687",  # Rank 0: 20GB (Big Slice)
    "MIG-93c2210e-e768-54a7-a5d7-bf8e2bb54d4d",  # Rank 1: 10GB (Medium Slice)
    "MIG-b3b486dd-a81a-5a46-910c-950d02628804",  # Rank 2: 5GB  (Small Slice)
]

LAYER_LIMITS = [32, 20, 8]


def get_wiki_sample():
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
    # Announce that this rank is starting its weight loading phase.
    # flush=True forces the print to appear immediately even in multiprocessing.
    print(f"[Rank {rank}] Surgical weight loading started...", flush=True)

    try:
        # Ask Hugging Face's hub utility to find the locally cached copy of the
        # model's index file. This JSON file is a map of every weight tensor name
        # to the shard file (.bin) that contains it. It does NOT contain the
        # weights themselves — just the directory.
        cached_index = hub.cached_file(MODEL_NAME, "pytorch_model.bin.index.json")

        # Extract the folder that contains the index file. All the weight shard
        # files live in this same folder, so we'll need this path to open them.
        folder_path = os.path.dirname(cached_index)

        # Open and parse the index JSON file into a Python dictionary.
        with open(cached_index, "r") as f:
            index_data = json.load(f)

        # Pull out the "weight_map" sub-dictionary. Its structure is:
        # { "model.layers.0.self_attn.q_proj.weight": "pytorch_model-00001-of-00002.bin", ... }
        weight_map = index_data["weight_map"]

        # Collect the unique set of shard filenames. Using a set automatically
        # deduplicates — many weight keys map to the same shard file, and we
        # only want to open each file once.
        shard_files = set(weight_map.values())

    except Exception:
        # If the index file isn't cached locally (e.g. no internet, wrong model name),
        # warn and bail out early rather than crashing the whole process.
        print(
            f"[Rank {rank}] Warning: Could not find weight map. Skipping weight load."
        )
        return

    # Iterate over each shard file with a progress bar.
    # leave=False means the progress bar disappears when done (keeps output tidy
    # when many ranks are printing simultaneously).
    for shard_file in tqdm(
        shard_files, desc=f"Rank {rank} Loading Shards", leave=False
    ):
        # Build the full path to this shard file on disk.
        file_path = os.path.join(folder_path, shard_file)

        # Load the shard into CPU RAM, not GPU. This is intentional — we'll copy
        # only the tensors we need to the GPU, avoiding an OOM from loading
        # everything onto the device at once.
        # The type hint Dict[str, torch.Tensor] documents the shape of state_dict.
        # CRITICAL
        # The code silently assumes CPU RAM is large enough to hold at least one full shard file at a time.
        # For Vicuna-7B, each shard is typically around 4–9GB,
        # so you need that much free CPU RAM just to load one shard before you start copying and deleting.
        state_dict: Dict[str, torch.Tensor] = torch.load(file_path, map_location="cpu")

        # Snapshot the keys before we start iterating. This is a safety measure —
        # if anything modified state_dict during iteration it could cause issues.
        keys_to_process = list(state_dict.keys())

        for key in keys_to_process:

            # --- EMBEDDINGS: only loaded by Rank 0 ---
            # The embedding table converts token IDs to dense vectors.
            # Only the first rank in the pipeline needs it (it's the entry point).
            # We also guard with "embed" in model_components to make sure this rank
            # actually instantiated an embedding module — defensive programming.
            if rank == 0 and "embed_tokens" in key and "embed" in model_components:
                # Copy the weight tensor from CPU into the already-allocated GPU
                # embedding module. .data.copy_() does an in-place copy without
                # creating a new tensor or tracking gradients.
                model_components["embed"].weight.data.copy_(state_dict[key])

            # --- FINAL NORM & LM HEAD: only loaded by Rank 2 (last rank) ---
            # The final RMSNorm and LM head (vocab projection) are only needed at
            # the end of the pipeline, so only the last rank loads them.
            if rank == 2:
                if "norm.weight" in key and "norm" in model_components:
                    # Copy the RMSNorm scale parameter.
                    model_components["norm"].weight.data.copy_(state_dict[key])
                if "lm_head.weight" in key and "lm_head" in model_components:
                    # Copy the LM head projection matrix (hidden_size -> vocab_size).
                    model_components["lm_head"].weight.data.copy_(state_dict[key])

            # --- DECODER LAYERS: each rank loads only its assigned layers ---
            if "layers." in key:
                # A typical key looks like:
                # "model.layers.7.self_attn.q_proj.weight"
                # Splitting by "." gives: ["model", "layers", "7", "self_attn", ...]
                parts = key.split(".")

                try:
                    # parts[2] is the global layer index as a string — parse it.
                    layer_idx = int(parts[2])
                except ValueError:
                    # If parts[2] isn't a number for some reason, skip this key.
                    continue

                # Check if this global layer index belongs to THIS rank.
                # my_layer_indices is a list like [12, 13, 14, ...] for rank 1.
                if layer_idx in my_layer_indices:
                    # Convert the global layer index to a local index within
                    # this rank's my_layers list (which starts at 0 locally).
                    local_idx = my_layer_indices.index(layer_idx)

                    # Get the actual nn.Module (LlamaDecoderLayer) for this layer.
                    module = my_layers[local_idx]

                    # Strip the "model.layers.N." prefix to get just the
                    # sub-parameter path, e.g. "self_attn.q_proj.weight".
                    local_key = ".".join(parts[3:])

                    try:
                        # Navigate down the module tree to find the exact parameter.
                        # Start at the top-level decoder layer module.
                        sub_mod = module

                        # Split the local key into path segments.
                        # e.g. ["self_attn", "q_proj", "weight"]
                        sub_parts = local_key.split(".")

                        # Walk all segments EXCEPT the last one (which is the
                        # parameter name itself), drilling into nested submodules.
                        # e.g. module -> module.self_attn -> module.self_attn.q_proj
                        for sp in sub_parts[:-1]:
                            sub_mod = getattr(sub_mod, sp)

                        # The final segment is the actual parameter name (e.g. "weight").
                        param_name = sub_parts[-1]

                        # Copy the weight from the CPU shard into the GPU parameter,
                        # in-place. This avoids allocating a new tensor.
                        getattr(sub_mod, param_name).data.copy_(state_dict[key])

                    except AttributeError:
                        # If the attribute path doesn't exist on this module
                        # (e.g. an unexpected key in the checkpoint), silently skip.
                        pass

        # Explicitly delete the shard dict and trigger garbage collection.
        # Without this, all loaded shards would accumulate in CPU RAM across
        # the loop, potentially causing an OOM before we finish loading.
        # We only need enough CPU RAM for one shard at a time, not the whole model at once.
        # So the peak CPU RAM usage is roughly the size of the largest shard
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # Confirm this rank finished loading all its weights.
    print(f"[Rank {rank}] Weights Loaded.", flush=True)


def run_pipeline(
    rank,  # Which process this is (0, 1, or 2) — determines its role in the pipeline
    world_size,  # Total number of processes (3, one per MIG slice)
    split_config,  # List like [12, 12, 8] — how many layers each rank owns
    result_queue,  # Multiprocessing queue used by rank 0 to report timing back to main()
    device_uuid,  # The MIG UUID string for this rank's GPU slice
    input_ids_seed,  # The tokenized input tensor (shared across all ranks as a starting point)
):
    try:
        # Tell PyTorch's CUDA allocator to use expandable memory segments.
        # This reduces fragmentation on small MIG slices where VRAM is tight.
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        # Restrict this process to only see its assigned MIG slice.
        # After this line, "cuda:0" inside this process refers to THIS slice only,
        # not the whole GPU. This is how MIG isolation is enforced at the process level.
        os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid

        # Address and port for the distributed process group rendezvous.
        # All 3 processes meet here to coordinate before communication starts.
        # 127.0.0.1 = localhost, since all ranks are on the same machine.
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        # Initialize the distributed process group using the "gloo" backend.
        # Gloo is a CPU-based communication library. We use it instead of NCCL
        # because MIG slices cannot do direct GPU-to-GPU peer transfers (no NVLink
        # between slices), so tensor passing goes through CPU/system memory.
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Register custom transport hooks from the mig_transport module.
        # These hooks intercept distributed send/recv operations and route them
        # correctly across MIG slice boundaries.
        mig_transport.register_hooks()

        # Since CUDA_VISIBLE_DEVICES is set to just this rank's slice,
        # cuda:0 is always the correct device regardless of rank number.
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # Load the model's architecture configuration (number of heads, hidden size,
        # number of layers, etc.) from the Hugging Face hub cache.
        config = LlamaConfig.from_pretrained(MODEL_NAME)

        # Force "eager" (standard PyTorch) attention instead of flash attention or
        # SDPA. Flash attention has memory layout requirements that may not play
        # well with the manual KV cache management done here.
        config._attn_implementation = "eager"

        # Compute which global layer indices belong to this rank.
        # split_config[:rank] sums up all layers assigned to ranks before this one,
        # giving the starting layer index for this rank.
        # e.g. split_config=[12,12,8], rank=1 → start=12, end=24
        start_layer = sum(split_config[:rank])
        end_layer = start_layer + split_config[rank]
        my_layer_indices = list(range(start_layer, end_layer))

        # Dictionary to hold non-layer components (embedding, norm, lm_head).
        # Not every rank has these — they're conditionally added below.
        model_components = {}

        # Only rank 0 needs the token embedding table, since it's the entry point
        # of the pipeline and converts input token IDs into hidden states.
        # .half() converts to float16 to match the rest of the model and save VRAM.
        if rank == 0:
            model_components["embed"] = (
                nn.Embedding(config.vocab_size, config.hidden_size).to(device).half()
            )

        # Build this rank's decoder layers and move them to the GPU one at a time.
        # We call empty_cache() after each layer to release any temporary allocations
        # made during layer construction, keeping peak VRAM as low as possible.
        layers = nn.ModuleList()
        for idx in my_layer_indices:
            layer = LlamaDecoderLayer(config, layer_idx=idx).half()
            layers.append(layer.to(device))
            torch.cuda.empty_cache()

        # Only the last rank needs the final RMSNorm and LM head, since they are
        # applied after all transformer layers to produce token logits.
        if rank == world_size - 1:
            model_components["norm"] = (
                LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                .to(device)
                .half()
            )
            # LM head projects from hidden_size (4096) to vocab_size (32000),
            # producing a score for each possible next token. bias=False matches
            # the original Llama architecture.
            model_components["lm_head"] = (
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                .to(device)
                .half()
            )

        # Every rank needs rotary embeddings because positional encoding is applied
        # inside each attention layer, not just at the input. Each rank computes
        # its own rotary embeddings locally rather than passing them through the pipeline.
        rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

        # Load the actual pretrained weights into the modules this rank owns.
        # This is the surgical loading described in load_specific_weights().
        load_specific_weights(rank, layers, my_layer_indices, model_components)

        # Clean up any lingering CPU and GPU memory after weight loading.
        gc.collect()
        torch.cuda.empty_cache()

        # Move the input token IDs to this rank's GPU slice.
        # All ranks receive the same input_ids_seed but only rank 0 actually
        # uses it — the others need it on-device just to have the shape info.
        input_ids = input_ids_seed.to(device)
        batch_size, seq_length = input_ids.shape  # e.g. (1, 512)

        # DynamicCache is Hugging Face's built-in KV cache object.
        # It accumulates past key and value tensors across decode steps so that
        # attention doesn't recompute over the full history on every new token.
        past_key_values = DynamicCache()

        # Pre-allocate a buffer to hold the next token ID sent back from rank 2.
        # Initialized to zeros; will be overwritten on each decode step.
        # Shape (batch_size, 1) = one token per batch item.
        next_token_tensor = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device
        )

        # Synchronize all ranks before starting the timed generation loop.
        # Without this, a fast rank might start sending tensors before a slow rank
        # has finished loading weights and is ready to receive them.
        dist.barrier()

        # CUDA events are the correct way to time GPU operations.
        # Unlike Python's time.time(), they account for GPU asynchrony —
        # CUDA kernels are launched asynchronously, so a CPU timer would measure
        # kernel launch time, not actual execution time.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # Place a timestamp marker in the CUDA stream

        # Disable gradient tracking for the entire generation loop.
        # We're doing inference only, so gradients are wasted memory and compute.
        with torch.no_grad():
            # Step 0 is the prefill: process all 512 input tokens in parallel.
            # Steps 1..MAX_NEW_TOKENS are decode steps: generate one new token each.
            for step in range(MAX_NEW_TOKENS + 1):
                is_prefill = step == 0

                # Prefill processes the full sequence; decode processes 1 token.
                current_seq_len = seq_length if is_prefill else 1

                # Allocate send/recv buffers sized for the current step.
                # Using separate buffers for send and recv avoids aliasing issues
                # where a tensor being received overwrites one being sent.
                # Shape: (batch, seq_len, hidden_size)
                buffer_shape = (batch_size, current_seq_len, config.hidden_size)
                recv_buffer = torch.zeros(
                    buffer_shape, dtype=torch.float16, device=device
                )
                send_buffer = torch.zeros(
                    buffer_shape, dtype=torch.float16, device=device
                )

                # ----------------------------------------------------------------
                # STAGE 1: INPUT — get the hidden states for this step
                # ----------------------------------------------------------------
                if rank == 0:
                    # On prefill, embed the full 512-token input sequence.
                    # On decode steps, embed only the single new token predicted
                    # in the previous step (stored in next_token_tensor).
                    current_input_ids = input_ids if is_prefill else next_token_tensor
                    current_hidden = model_components["embed"](current_input_ids).half()
                else:
                    # All other ranks wait to receive hidden states from the
                    # previous rank. dist.recv() is blocking — this rank will stall
                    # here until rank-1 sends its output.
                    dist.recv(recv_buffer, src=rank - 1)
                    # Clone to get a contiguous tensor we own; recv_buffer will be
                    # overwritten on the next step.
                    current_hidden = recv_buffer.clone()

                # ----------------------------------------------------------------
                # STAGE 2: POSITION IDS & ATTENTION MASK
                # ----------------------------------------------------------------
                if is_prefill:
                    # Position IDs are 0, 1, 2, ..., seq_len-1.
                    # unsqueeze(0) adds the batch dimension: shape (1, seq_len).
                    position_ids = torch.arange(
                        0, current_seq_len, dtype=torch.long, device=device
                    ).unsqueeze(0)

                    # Build a causal (upper-triangular) attention mask.
                    # Start with a matrix filled with -inf (the minimum float16 value).
                    # After softmax, -inf becomes 0, effectively masking future tokens.
                    mask = torch.full(
                        (1, 1, current_seq_len, current_seq_len),
                        torch.finfo(torch.float16).min,
                        device=device,
                    )
                    # torch.triu with diagonal=1 keeps only the upper triangle
                    # (above the main diagonal), which represents "future" positions.
                    # Everything on and below the diagonal is zeroed out (visible).
                    mask = torch.triu(mask, diagonal=1).half()
                else:
                    # During decode, the single new token's position is
                    # seq_length + step - 1. e.g. after prefilling 512 tokens,
                    # the first new token is at position 512 (step=1, so 512+1-1=512).
                    position_ids = torch.tensor(
                        [[seq_length + step - 1]], dtype=torch.long, device=device
                    )
                    # No causal mask needed when processing a single token —
                    # there are no future tokens to mask. The KV cache handles
                    # all past context automatically.
                    mask = None

                # Compute rotary positional embeddings (sin/cos values) for the
                # current positions. These are passed into each attention layer
                # to encode relative position information into Q and K projections.
                position_embeddings = rotary_emb(current_hidden, position_ids)

                # ----------------------------------------------------------------
                # STAGE 3: FORWARD PASS THROUGH THIS RANK'S LAYERS
                # ----------------------------------------------------------------
                for layer in layers:
                    # Save the input as the residual connection. This is the
                    # standard "pre-norm" transformer block pattern:
                    # output = input + sublayer(norm(input))
                    residual = current_hidden

                    # Apply input layer norm before attention (pre-norm architecture).
                    hidden_states = layer.input_layernorm(current_hidden)

                    # Run multi-head self-attention.
                    # Passing past_key_values (the DynamicCache object) enables
                    # the cache to automatically store and retrieve K/V tensors.
                    # use_cache=True tells the layer to update the cache in-place.
                    attn_outputs = layer.self_attn(
                        hidden_states=hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    # attn_outputs is a tuple; index 0 is the attention output tensor.
                    # The cache is updated internally by DynamicCache — no manual
                    # assignment needed.
                    hidden_states = attn_outputs[0]

                    # First residual connection: add attention output to the pre-attention input.
                    hidden_states = residual + hidden_states

                    # Save for the second residual connection (around the MLP block).
                    residual = hidden_states

                    # Apply post-attention layer norm before the MLP (pre-norm again).
                    hidden_states = layer.post_attention_layernorm(hidden_states)

                    # Feed-forward / MLP block: two linear projections with a
                    # nonlinearity (SiLU gating in Llama). This is where most of the
                    # model's "knowledge" is stored.
                    hidden_states = layer.mlp(hidden_states)

                    # Second residual connection: add MLP output to pre-MLP input.
                    current_hidden = residual + hidden_states

                # ----------------------------------------------------------------
                # STAGE 4: TOKEN GENERATION (last rank only)
                # ----------------------------------------------------------------
                if rank == world_size - 1:
                    # Apply the final RMSNorm to stabilize the output hidden states
                    # before projecting to vocabulary logits.
                    current_hidden = model_components["norm"](current_hidden)

                    # During prefill we only need the LAST token's hidden state to
                    # predict the first new token. [:, -1:, :] slices the last
                    # position while keeping the 3D shape (batch, 1, hidden).
                    # During decode there's only one token anyway, so no slicing needed.
                    last_token_hidden = (
                        current_hidden[:, -1:, :] if is_prefill else current_hidden
                    )

                    # Project from hidden_size (4096) to vocab_size (32000).
                    # Each value in logits is the unnormalized score for a token.
                    logits = model_components["lm_head"](last_token_hidden)

                    # Greedy decoding: pick the token with the highest logit score.
                    # dim=-1 takes the argmax over the vocabulary dimension.
                    # Shape of next_token: (batch_size, 1)
                    next_token = torch.argmax(logits, dim=-1)

                    # Only print during decode steps (not prefill) to avoid noise.
                    if not is_prefill:
                        print(
                            f"[Rank 2] Generated Token ID: {next_token.item()}",
                            flush=True,
                        )

                # ----------------------------------------------------------------
                # STAGE 5: COMMUNICATION — pass tensors to the next stage
                # ----------------------------------------------------------------

                # Forward pass: each rank (except the last) sends its output
                # hidden states to the next rank in the pipeline.
                if rank < world_size - 1:
                    # Copy into the pre-allocated send_buffer before sending.
                    # This ensures the tensor being sent is contiguous in memory,
                    # which is required by dist.send().
                    send_buffer.copy_(current_hidden)
                    dist.send(send_buffer, dst=rank + 1)

                # Backward token feedback: rank 2 sends the newly generated token
                # ID back to rank 0 so rank 0 can embed it in the next decode step.
                # This only happens between decode steps (not after the last one).
                if step < MAX_NEW_TOKENS:
                    if rank == world_size - 1:
                        # Last rank sends the predicted token to rank 0.
                        dist.send(next_token, dst=0)
                    elif rank == 0:
                        # Rank 0 receives the token and stores it in next_token_tensor,
                        # which will be embedded at the top of the next iteration.
                        dist.recv(next_token_tensor, src=world_size - 1)
                        # Note: middle ranks (rank 1) do nothing here —
                        # they don't need the token ID since they don't do embedding.

        # Final barrier to ensure all ranks finish before we record the end time.
        # Without this, rank 0 might record end_event before rank 2 finishes.
        dist.barrier()

        # Place the end timestamp marker in the CUDA stream.
        end_event.record()

        # Block the CPU until all pending CUDA operations complete,
        # so elapsed_time() returns an accurate measurement.
        torch.cuda.synchronize()

        # elapsed_time() returns milliseconds between the two CUDA events.
        # Dividing by (MAX_NEW_TOKENS + 1) gives average ms per step
        # (1 prefill + N decode steps).
        avg_ms = start_event.elapsed_time(end_event) / (MAX_NEW_TOKENS + 1)

        # Only rank 0 reports the result to avoid duplicates in the queue.
        # main() will read this value to record the benchmark result.
        if rank == 0:
            result_queue.put(avg_ms)

        # Cleanly shut down the distributed process group to free resources.
        # Skipping this can leave ports occupied and cause the next benchmark
        # configuration to fail when it tries to init a new process group.
        dist.destroy_process_group()

    except Exception:
        # Print the full traceback so failures are diagnosable.
        # We catch broadly here because any unhandled exception in a child
        # process would otherwise silently produce a non-zero exit code and
        # the parent would only know "it crashed", not why.
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
    splits_amt = len(selected_splits)

    print(
        f"Benchmarking {splits_amt} configurations with REAL WEIGHTS & AUTOREGRESSIVE DECODING..."
    )

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    results = []

    for i, split in enumerate(selected_splits):
        print(f"\n[{i+1}/{splits_amt}] Split {split} ...")
        q = mp.Queue()
        procs: list[mp.Process] = []

        try:
            for rank in range(3):
                p = mp.Process(
                    target=run_pipeline,
                    args=(rank, 3, split, q, MIG_UUIDS[rank], input_ids_seed),
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
                    print(f"Success: {lat:.2f} ms per step (Prefill + Decode Average)")
                    results.append({"split": str(split), "latency_ms": lat})
                except queue.Empty:
                    print("Timeout.")
        finally:
            q.close()
            q.join_thread()

        for p in procs:
            if p.is_alive():
                p.terminate()

    pd.DataFrame(results).to_csv("mig_real_weights_autoregressive.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
