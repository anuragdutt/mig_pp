import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import time
import sys

# Import our custom MIG-aware transport hooks
import mig_transport_tensor

# Hugging Face dataset + tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer


# --- CONFIGURATION ---

# Total attention heads in the model (e.g., Vicuna-7B uses 32)
TOTAL_HEADS = 32

# Dimension per attention head
HEAD_DIM = 128

# Total hidden size (32 heads × 128 dim = 4096)
HIDDEN_SIZE = 4096

# Sequence length used for benchmarking
SEQ_LEN = 512


# Hardware UUIDs for 3 MIG slices (Big -> Medium -> Small)
# Each rank will bind itself to exactly one MIG slice
MIG_UUIDS = [
    "MIG-eb0d0042-96cf-5e2c-8993-4d98f85b44fa",  # Rank 0 (4g.20gb)
    "MIG-652be015-0967-5714-a9df-a3c8f90e7a5f",  # Rank 1 (2g.10gb)
    "MIG-a4fdc65d-78bf-5af3-b80c-a73680f3bc19",  # Rank 2 (1g.5gb)
]

# Maximum number of heads each MIG slice can handle (memory constraint)
# Big -> 32 heads
# Medium -> 19 heads
# Small -> 9 heads
LIMITS = [32, 19, 9]


def generate_valid_splits():
    """
    Generate all valid ways to split 32 attention heads across 3 GPUs
    such that:
      - Each GPU gets at least 1 head
      - Each GPU does not exceed its memory limit
      - Total heads sum to 32
    """
    valid_splits = []

    # Try every possible allocation for GPU 0
    for h0 in range(1, LIMITS[0] + 1):

        # Try every possible allocation for GPU 1
        for h1 in range(1, LIMITS[1] + 1):

            # GPU 2 gets whatever remains
            h2 = TOTAL_HEADS - (h0 + h1)

            # Ensure GPU 2's allocation is valid
            if 1 <= h2 <= LIMITS[2]:
                valid_splits.append([h0, h1, h2])

    return valid_splits


def get_dataloader(batch_size=1):
    """
    Loads a small subset of WikiText-2 dataset and tokenizes it.
    Used only for realistic input during benchmarking.
    """

    # Load 20 training samples for stable benchmarking
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20]")

    # Load tokenizer from Vicuna model
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

    # Set pad token equal to EOS token (common workaround)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenization function applied to dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],  # Raw text
            padding="max_length",  # Pad to fixed length
            truncation=True,  # Truncate long text
            max_length=SEQ_LEN,  # Fixed sequence length
        )

    # Apply tokenization
    tokenized = dataset.map(tokenize_function, batched=True)

    # Convert output to PyTorch tensors
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Return DataLoader
    return torch.utils.data.DataLoader(tokenized, batch_size=batch_size)


def run_inference(rank, world_size, split_config, result_queue, device_uuid):
    """
    Each process runs this function.
    Simulates sharded attention layer across 3 MIG GPUs.
    """

    # --- 1. ISOLATE DEVICE ---

    # Restrict this process to ONLY its assigned MIG device
    os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid

    # --- 2. Distributed Setup ---

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # Initialize distributed process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # --- 3. Enable Custom MIG Transport Hooks ---
    mig_transport_tensor.register_hooks()

    # --- 4. Set CUDA Device ---
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # --- 5. Setup Sharded Weights ---

    # Number of heads this GPU is responsible for
    my_heads = split_config[rank]

    # Size of its projection shard
    my_shard_size = my_heads * HEAD_DIM

    # Local shard of QKV projection
    w_qkv = torch.randn(HIDDEN_SIZE, my_shard_size, device=device, dtype=torch.float16)

    # Local shard of output projection
    w_out = torch.randn(my_shard_size, HIDDEN_SIZE, device=device, dtype=torch.float16)

    # --- 6. Dataset Loading (only on rank 0) ---

    data_iter = None
    if rank == 0:
        dataloader = get_dataloader()
        data_iter = iter(dataloader)

    # --- 7. Benchmark Setup ---

    steps = 10  # number of measured iterations

    # --- WARMUP PHASE ---
    # Warmup ensures CUDA kernels are compiled and memory allocated

    dummy = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.float16)

    for _ in range(3):
        h = torch.matmul(dummy, w_qkv)
        out_warmup = torch.matmul(h, w_out)

        # Flatten tensor for safe custom transport communication
        out_flat = out_warmup.view(-1)

        # All GPUs sum their partial outputs
        dist.all_reduce(out_flat)

    # Synchronize all ranks before timing
    dist.barrier()

    # --- TIMING SETUP ---

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    # --- MAIN BENCHMARK LOOP ---

    for i in range(steps):

        # Allocate input tensor (ensure contiguous memory)
        x = torch.zeros(
            1, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.float16
        ).contiguous()

        # Only rank 0 loads real input data
        if rank == 0:
            try:
                batch = next(data_iter)

                input_ids = batch["input_ids"].to(device)

                # Expand token IDs into fake embedding-like tensor
                x = input_ids.unsqueeze(-1).expand(-1, -1, HIDDEN_SIZE).half() / 1000.0

                x = x.contiguous()

            except StopIteration:
                # Fallback if dataset exhausted
                x = torch.randn(
                    1, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.float16
                )

        # Broadcast input from rank 0 to all other GPUs
        dist.broadcast(x, src=0)

        # Forward pass (sharded attention projection)
        hidden = torch.matmul(x, w_qkv)
        out_partial = torch.matmul(hidden, w_out)

        # Flatten before all_reduce for transport safety
        out_flat = out_partial.view(-1)

        # Sum partial outputs across all GPUs
        dist.all_reduce(out_flat)

    end_event.record()

    # Wait for all kernels to complete
    torch.cuda.synchronize()

    # Compute average latency per step
    avg_ms = start_event.elapsed_time(end_event) / steps

    # Only rank 0 reports result
    if rank == 0:
        result_queue.put(avg_ms)

    # Clean up distributed group
    dist.destroy_process_group()


def main():
    """
    Main benchmark driver.
    Tests all valid head splits across 3 MIG slices.
    """

    splits = generate_valid_splits()
    print(f"Generated {len(splits)} valid split configurations.")

    results = []
    selected_splits = splits

    print(f"Benchmarking {len(selected_splits)} configurations...")

    # Remove global CUDA visibility
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    # Use spawn start method (required for CUDA multiprocessing)
    ctx = mp.get_context("spawn")

    # Test each valid split
    for split in selected_splits:
        print(f"Testing Split: {split} ...", end="", flush=True)

        q = ctx.Queue()
        procs = []

        # Launch 3 processes (one per MIG slice)
        for rank in range(3):
            uuid = MIG_UUIDS[rank]

            p = ctx.Process(target=run_inference, args=(rank, 3, split, q, uuid))

            p.start()
            procs.append(p)

        # Wait for all processes to finish
        for p in procs:
            p.join()

        # Collect result from rank 0
        if not q.empty():
            time_ms = q.get()
            print(f" {time_ms:.2f} ms")
            results.append({"split": str(split), "latency_ms": time_ms})
        else:
            print(" Failed.")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("mig_3way_benchmark.csv", index=False)

    print("\nSaved results to mig_3way_benchmark.csv")


if __name__ == "__main__":
    main()
