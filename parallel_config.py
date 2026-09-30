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

# Where download_models.py puts the weights (~70 GB for these four models).
MODEL_ROOT = "/data/models"

# path: local model dir, or an HF repo id already in the HF cache (never downloads).
# Optional per model: seq_len, max_new_tokens, max_runs, batch_mb_pairs, splits,
# layer_limits, min_last_rank_layers (these override SWEEP / layer_limits.py).
# vicuna_7b, llama_13b, qwen_14b and nemo_12b are still known to layer_limits.py
# and download_models.py; their MODELS lines are in commit 2236a5e.
MODELS = {
    # Same shapes and the same 938 configs as the 9/28 Vicuna-7B sweep (Vicuna
    # v1.5 is a Llama-2-7B fine-tune), so its latencies also show how much the
    # new box + four lanes at once move things.
    "llama_7b": dict(path=f"{MODEL_ROOT}/llama-2-7b-hf"),
    "mistral_7b": dict(path=f"{MODEL_ROOT}/Mistral-7B-Instruct-v0.3"),
    "qwen_7b": dict(path=f"{MODEL_ROOT}/Qwen2.5-7B"),
    # 13B weights + KV cannot fit on 20/10/5/5 at B32 or B64 (memory_limits.py;
    # the old 13B runs swept B8/B16 only).
    "vicuna_13b": dict(path=f"{MODEL_ROOT}/vicuna-13b-v1.5", batch_mb_pairs=[(8, 4), (8, 2), (16, 8), (16, 4), (16, 2)]),
}

# One entry per GPU. slice_gb and mig_uuids in RANK ORDER (largest slice first);
# `./run_parallel.sh discover` prints this block with the real UUIDs.
# models: MODELS keys (or dict(model=key, <overrides>)) run in order; [] = GPU unused.
GPUS = [
    dict(gpu=0, slice_gb=[20, 10, 5, 5], mig_uuids=["MIG-PASTE-0-0", "MIG-PASTE-0-1", "MIG-PASTE-0-2", "MIG-PASTE-0-3"], models=["llama_7b"]),
    dict(gpu=1, slice_gb=[20, 10, 5, 5], mig_uuids=["MIG-PASTE-1-0", "MIG-PASTE-1-1", "MIG-PASTE-1-2", "MIG-PASTE-1-3"], models=["mistral_7b"]),
    dict(gpu=2, slice_gb=[20, 10, 5, 5], mig_uuids=["MIG-PASTE-2-0", "MIG-PASTE-2-1", "MIG-PASTE-2-2", "MIG-PASTE-2-3"], models=["qwen_7b"]),
    dict(gpu=3, slice_gb=[20, 10, 5, 5], mig_uuids=["MIG-PASTE-3-0", "MIG-PASTE-3-1", "MIG-PASTE-3-2", "MIG-PASTE-3-3"], models=["vicuna_13b"]),
]

# The benchmark's own constants (benchmark_pipeline_microbatching.py).
SWEEP = dict(
    seq_len=64,
    max_new_tokens=512,
    max_runs=5,
    enforce_slice_ordering=True,
    batch_mb_pairs=[
        (8, 4), (8, 2),
        (16, 8), (16, 4), (16, 2),
        (32, 16), (32, 8), (32, 4), (32, 2),
        (64, 32), (64, 16), (64, 8), (64, 4), (64, 2),
    ],
)

RUNNER = dict(
    base_port=29500,  # lane port = base_port + 10 * gpu
    mem_monitor="nvml",  # "nvml" (per MIG UUID), "dcgm" (one lane on GPU 0 only), "off"
    env={},  # extra env for every job, e.g. {"MIG_LOG_LEVEL": "summary"} for smaller logs
)
