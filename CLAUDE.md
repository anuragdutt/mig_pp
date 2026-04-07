# MIG Pipeline Parallelism Benchmark

## Project Overview

This project benchmarks pipeline-parallel LLM inference across NVIDIA A100 MIG (Multi-Instance GPU) partitions. The goal is to find the optimal layer split and microbatch configuration for running **Vicuna-13B** across 4 MIG slices of different sizes, with the primary metric being **GPU memory utilization per MIG instance** (not latency).

The benchmark sweeps all valid (layer split × batch size × microbatch size) configurations, records peak and average memory via DCGM, and produces a CSV for offline analysis.

---

## Hardware Setup

NVIDIA A100 40GB split into 4 MIG instances (7 compute slices total):

| Rank | MIG Profile | VRAM | Role |
|------|------------|------|------|
| 0 | `3g.20gb` | 20GB | Embedding + first chunk of decoder layers |
| 1 | `2g.10gb` | 10GB | Middle decoder layers |
| 2 | `1g.5gb` | 5GB | Later decoder layers |
| 3 | `1g.5gb` | 5GB | Final decoder layers + RMSNorm + lm_head |

MIG instances are identified by UUID, hardcoded in `benchmark_pipeline_microbatching.py` under `MIG_UUIDS`. Update these after every MIG re-partition.

---

## Model: Vicuna-13B v1.5

| Property | Value |
|----------|-------|
| HuggingFace ID | `lmsys/vicuna-13b-v1.5` |
| Architecture | LLaMA-based |
| Total decoder layers | 40 |
| Hidden size | 5120 |
| Attention heads | 40 |
| Intermediate (MLP) size | 13824 |
| Vocab size | 32000 |

### Per-Layer Memory Budget (fp16)

Each `LlamaDecoderLayer` contains ~317M parameters:

| Component | Shape | Params |
|-----------|-------|--------|
| Q / K / V projections | 3 × (5120 × 5120) | 78.6M |
| Output projection | 5120 × 5120 | 26.2M |
| gate_proj | 5120 × 13824 | 70.8M |
| up_proj | 5120 × 13824 | 70.8M |
| down_proj | 13824 × 5120 | 70.8M |
| 2× RMSNorm | 2 × 5120 | ~10K |
| **Total per layer** | | **~317M params → ~634 MB in fp16** |

### Additional Components

| Component | Location | Size (fp16) |
|-----------|----------|-------------|
| Embedding (`nn.Embedding`) | Rank 0 only | 32000 × 5120 × 2 ≈ 327 MB |
| Final RMSNorm | Last rank only | ~10 KB (negligible) |
| lm_head (`nn.Linear`) | Last rank only | 5120 × 32000 × 2 ≈ 327 MB |

### Rough VRAM Capacity Per MIG Slice (weights only)

| Slice | VRAM | Layers (weights only) | Practical limit (with KV cache + activations) |
|-------|------|-----------------------|------------------------------------------------|
| 20GB | 20GB | ~31 layers | ~20–22 layers |
| 10GB | 10GB | ~15 layers | ~8–10 layers |
| 5GB | 5GB | ~7 layers | ~4–5 layers |

These practical limits account for KV cache growth during 512 decode steps and activation memory during prefill. The exact limit depends on batch size and microbatch size.

---

## Key Architecture Decisions

### Why Shared Memory Transport (not NCCL)?

MIG instances on the same physical GPU are hardware-isolated — they cannot use NVLink P2P, and NCCL does not work across MIG slices. Instead, activations are passed via **POSIX shared memory** (`/dev/shm/mig_pipe_shm_<rank>_slot<N>`), with tiny handshake messages sent over PyTorch's **gloo backend** for synchronization.

### Transport Layer: SHM + ACK Protocol

`mig_transport_pipeline_non_blocking.py` monkey-patches `dist.send/recv/isend/irecv` at the module level. The protocol for each tensor transfer:

1. **Sender** writes tensor data into a shared memory ring-buffer slot (GPU → pinned CPU → SHM for fp16; GPU → CPU → SHM for other dtypes)
2. **Sender** sends a 1-int handshake (slot index) via gloo
3. **Receiver** reads the slot index from the handshake, copies SHM data into its GPU tensor
4. **Receiver** sends an ACK back so the sender can mark the slot as free

`AsyncHandle` / `AsyncHandleRecv` wrap non-blocking variants (isend/irecv) so callers can call `.wait()` later.

### Ring Buffer Slots

`NUM_SLOTS = 32` ring buffer slots per rank. This must be **≥ max microbatches in flight at once**. The calculation:

```
max_microbatches = max_batch_size / min_microbatch_size = 64 / 2 = 32
```

With the current `BATCH_MB_PAIRS`, this is exactly saturated. If you add configurations with smaller microbatch sizes (e.g., mb=1), increase `NUM_SLOTS` accordingly.

### Dynamic Slot Sizing

Slot size is computed dynamically based on the largest possible tensor (prefill activation at max microbatch size):

```
max_bytes = max_mb_size × max_seq_len × hidden_size × 2 (fp16)
         = 32 × 64 × 5120 × 2
         ≈ 20 MB
```

The `_compute_slot_size_mb()` function calculates this automatically, avoiding the previous static 128MB allocation which wasted ~6× host memory per slot.

### Pipeline Microbatching

- **Prefill**: Rank >0 posts all `irecv`s up front to maximize compute-communication overlap, then processes microbatches sequentially as data arrives
- **Decode**: Same pattern repeated for each of the 512 decode steps
- **Token exchange**: `next_tokens` is sent from last rank → rank 0 after each decode step (unavoidable sync point due to autoregressive dependency)

### Buffer Management (Memory-Critical)

To ensure accurate memory measurements, all GPU receive buffers are **allocated exactly once** before the compute loop:

- `prefill_recv_bufs` — allocated before `dist.barrier()`, reused for all prefill microbatches
- `decode_recv_bufs` — allocated before `dist.barrier()`, zeroed in-place via `.zero_()` each decode step

This avoids spurious GPU memory spikes from repeated allocation/deallocation that would corrupt DCGM readings.

### Layer Distribution Constraint

Larger MIG slices must receive strictly more layers: `l0 > l1 > l2 >= l3` (the two 5GB slices can be equal). `generate_layer_splits()` enumerates all valid combinations within:

```python
LAYER_LIMITS = [22, 10, 5, 5]  # max layers per rank
```

**Important**: The sum of `LAYER_LIMITS` must be ≥ `TOTAL_LAYERS` (40). If not, `generate_layer_splits()` returns an empty list and no benchmarks run.

---

## Configuration Reference

### Benchmark Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `SEQ_LEN` | 64 | Input sequence length (prefill) |
| `MAX_NEW_TOKENS` | 512 | Decode steps per run |
| `TOTAL_LAYERS` | 40 | Vicuna-13B decoder layers |
| `HIDDEN_SIZE` | 5120 | Model hidden dimension |
| `LAYER_LIMITS` | [22, 10, 5, 5] | Max layers per rank; sum must be ≥ 40 |

### Batch / Microbatch Pairs

```python
BATCH_MB_PAIRS = [
    (32, 16), (32, 8), (32, 4), (32, 2),
    (64, 32), (64, 16), (64, 8), (64, 4), (64, 2),
]
```

Each pair produces `batch_size / mb_size` microbatches. The number of microbatches must not exceed `NUM_SLOTS` (32).

### Communication Tags

Tags prevent collisions across concurrent microbatch sends:

| Tag Range | Purpose |
|-----------|---------|
| `1000 + mb_idx` | Prefill activations |
| `2000 + step × 10_000 + mb_idx` | Decode activations |
| `9000 + step` | next_token exchange (last rank → rank 0) |
| `tag + 10_000_000` | ACK messages in SHM transport |

---

## File Reference

### Active Files

| File | Purpose |
|------|---------|
| `benchmark_pipeline_microbatching.py` | Main benchmark driver: spawns 4 workers, sweeps splits × batch/mb pairs |
| `mig_transport_pipeline_non_blocking.py` | SHM transport with ACK protocol; monkey-patches `dist.*` |
| `dcgm_mem_monitor.py` | DCGM-based GPU memory monitor; tracks whole GPU + per-MIG-instance memory |
| `setup_mig_flexible.sh` | Interactive shell script to partition A100 into MIG instances |
| `kill_mig.sh` | Tears down all MIG compute/GPU instances on GPU 0 |
| `CLAUDE.md` | This file |

### Superseded / Historical Files

| File | Purpose |
|------|---------|
| `mig_patch.py` | Earlier 2-rank prototype (blocking only) |
| `benchmark_tensor.py` | Earlier tensor-only benchmark (no pipeline) |
| `mig_transport_tensor.py` | Earlier transport for tensor-only benchmarks |
| `mig_transport_pipeline.py` | Earlier blocking pipeline transport |
| `run_vicuna_7b_optimized.py` | Standalone Vicuna-7B runner (non-MIG) |

---

## Running the Benchmark

### 1. Set up MIG partitions
```bash
sudo ./setup_mig_flexible.sh
# Enter: 4 instances → 3g.20gb, 2g.10gb, 1g.5gb, 1g.5gb
```

### 2. Update MIG UUIDs
```bash
nvidia-smi -L
```
Update `MIG_UUIDS` in `benchmark_pipeline_microbatching.py` with the new UUIDs.

### 3. Activate the Python environment
```bash
source mig_env/bin/activate
```

### 4. Run the benchmark
```bash
python benchmark_pipeline_microbatching.py
```

### 5. Tear down MIG when done
```bash
sudo ./kill_mig.sh
```

---

## Output Files

| File | Contents |
|------|----------|
| `mig_benchmark_results.csv` | Per-config results: split, batch, microbatch, latency, peak/avg 5GB memory, status |
| `mig_memory_trace.csv` | DCGM framebuffer memory samples over time (timestamped) |
| `benchmark.log` | Detailed run log with per-rank latency, OOM events, errors |

Historical results are kept with date suffixes (e.g., `mig_benchmark_results-April-2.csv`).

### Status Values in Results CSV

| Status | Meaning |
|--------|---------|
| `ok` | Run completed successfully |
| `OOM_rankN` | Out-of-memory on rank N |
| `crash` | Non-zero exit code from a worker |
| `hang` | Workers did not finish within timeout (1200s) |
| `timeout` | Workers finished but no results received from queue |

---

## Known Constraints and Pitfalls

1. **`/dev/shm` size limit** — Many systems cap shared memory at half of RAM. With 4 ranks × 32 slots, ensure `/dev/shm` has enough space. Check with `df -h /dev/shm`. Dynamic slot sizing helps significantly.

2. **Stale SHM after crashes** — If the benchmark crashes, `/dev/shm/mig_pipe_shm_*` files may persist. The transport cleans up its own rank's stale files on startup, but a full manual cleanup may be needed: `rm /dev/shm/mig_pipe_shm_*`

3. **`LAYER_LIMITS` must sum to ≥ `TOTAL_LAYERS`** — If not, `generate_layer_splits()` returns an empty list, no benchmarks run, and the results CSV will be empty (causing a `KeyError` on analysis).

4. **KV cache growth** — `DynamicCache` grows with each decode step. Over 512 steps with batch=64, this can consume significant VRAM on the 5GB slices. This is the primary source of OOM, not the model weights.

5. **`MASTER_PORT` collision** — All 4 workers use port 29500. If another distributed job is running on the same machine, change this in `run_pipeline()`.

6. **Weight loading requires HuggingFace cache** — The model must be pre-downloaded. If the cache is missing, weight loading silently skips and the model runs with random weights (inference results will be garbage, but memory measurements remain valid).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch (with CUDA) | Core compute + distributed communication (gloo) |
| `transformers` | LlamaConfig, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding |
| `datasets` | WikiText-2 input data |
| `pandas` | Results CSV handling |
| `tqdm` | Progress bars for weight loading |
| `numpy` | SHM buffer management |
| DCGM / `dcgmi` CLI | GPU memory monitoring (system-level, requires sudo) |