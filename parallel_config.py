"""
Which models run on which GPU. Read by parallel_plan.py (plain Python, torch-free).

One lane per GPU; lanes run at the same time, a lane's models one after
another. Steps on the box:
  1. set MODEL_ROOT below, python3 download_models.py, create the MIG slices
  2. ./run_parallel.sh discover   -> paste its GPUS block below, put the models back
  3. ./run_parallel.sh check
  4. ./run_parallel.sh smoke, then ./run_parallel.sh start --detach
Layer limits are in layer_limits.py. Unknown keys are errors, so typos fail loudly.
"""
import os

# Where download_models.py puts the weights (~129 GB for these four models).
MODEL_ROOT = os.path.expanduser("~/models")

# path: local model dir, or an HF repo id already in the HF cache (never downloads).
# Optional per model: seq_len, max_new_tokens, max_runs, batch_mb_pairs, splits,
# layer_limits, min_last_rank_layers (these override SWEEP / layer_limits.py).
# The 7B models have layer limits for 20_10_5_5 (A100-40GB) only; their MODELS
# lines are in commit 2236a5e.
# On 40/20/10/10 every model runs all 14 batch pairs: the 40GB-box caps (13B at
# B8/B16, Qwen2.5-14B at B<=32) were for 20/10/5/5.
MODELS = {
    "vicuna_13b": dict(path=f"{MODEL_ROOT}/vicuna-13b-v1.5"),
    "llama_13b": dict(path=f"{MODEL_ROOT}/llama-2-13b-hf"),
    "qwen_14b": dict(path=f"{MODEL_ROOT}/Qwen2.5-14B"),
    "mistral_24b": dict(path=f"{MODEL_ROOT}/Mistral-Small-24B-Base-2501"),
}

# One entry per GPU. slice_gb and mig_uuids in RANK ORDER (largest slice first);
# `./run_parallel.sh discover` prints this block with the real UUIDs.
# models: MODELS keys (or dict(model=key, <overrides>)) run in order; [] = GPU unused.
# A100-80GB: 3g.40gb + 2g.20gb + 1g.10gb + 1g.10gb per GPU.
GPUS = [
    dict(gpu=0, slice_gb=[40, 20, 10, 10], models=["vicuna_13b"],
         mig_uuids=["MIG-0cfc69bb-d780-5095-9656-be2b83fb379d", "MIG-f6677aed-01af-56f3-83f3-91e511d24e6c", "MIG-4035a060-869a-58e6-9794-21dcff020348", "MIG-b2e8e895-ae5f-5e49-863d-35b4ad3bc0d5"]),  # NVIDIA A100-SXM4-80GB: 3g.40gb 2g.20gb 1g.10gb 1g.10gb
    dict(gpu=1, slice_gb=[40, 20, 10, 10], models=["llama_13b"],
         mig_uuids=["MIG-3300dd9c-59a5-5978-866a-9a4b62713e56", "MIG-0c88400e-21c0-5b39-a698-dd1230d974b7", "MIG-18295ce0-0f18-520b-9ae0-1ff8980abd2c", "MIG-b42afb91-2f72-526f-b908-b4f956ecc0e8"]),  # NVIDIA A100-SXM4-80GB: 3g.40gb 2g.20gb 1g.10gb 1g.10gb
    dict(gpu=2, slice_gb=[40, 20, 10, 10], models=["qwen_14b"],
         mig_uuids=["MIG-a1190fb5-b8a0-57aa-9ad0-58dd183cae3b", "MIG-4b6fbb23-1c51-5293-b9b4-fd4964e7cded", "MIG-b9edd97b-d54e-56da-b43a-e252908cd96f", "MIG-8daa456f-0b2e-5d62-9608-c3d51797a857"]),  # NVIDIA A100-SXM4-80GB: 3g.40gb 2g.20gb 1g.10gb 1g.10gb
    dict(gpu=3, slice_gb=[40, 20, 10, 10], models=["mistral_24b"],
         mig_uuids=["MIG-13c176dc-1fb4-5e69-af95-14744f1a6dc1", "MIG-c325e781-a6cc-5d8e-bf85-074ad7ede982", "MIG-b22269c6-b770-5ecd-86a5-25c3901391ac", "MIG-7da53c6b-6132-5d50-a1e3-f2c991b4ffd0"]),  # NVIDIA A100-SXM4-80GB: 3g.40gb 2g.20gb 1g.10gb 1g.10gb
]

# The benchmark's own constants (benchmark_pipeline_microbatching.py).
SWEEP = dict(
    seq_len=64,
    max_new_tokens=512,
    max_runs=None,
    enforce_slice_ordering=True,
    # All 14 pairs. The 16/32-microbatch ones ((32, 2), (64, 4), (64, 2)) are
    # ~1/3 of a split's time on the current harness (46% on April's pre-ACK-fix
    # one, where (64, 2) overran the 1200 s join timeout and was logged "hang");
    # layer_limits.py's 40_20_10_10 split counts are sized for all 14.
    batch_mb_pairs=[
        # (8, 4), (8, 2),
        (16, 8), (16, 4), (16, 2),
        (32, 16), (32, 8), (32, 4), (32, 2),
        # (64, 32), (64, 16), (64, 8), (64, 4), (64, 2),
    ],
)

RUNNER = dict(
    base_port=29500,  # lane port = base_port + 10 * gpu
    mem_monitor="nvml",  # "nvml" (per MIG UUID), "dcgm" (one lane on GPU 0 only), "off"
    # Extra env for every job. MIG_LOG_LEVEL: "summary" = end-of-run transport
    # summaries only (T22-T29), "off" = no transport log; unset = per-microbatch
    # INFO (the transport's default, GBs per sweep).
    env={"MIG_LOG_LEVEL": "summary"},
)
