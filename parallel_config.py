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

# Where download_models.py puts the weights (~129 GB for the four heavy models,
# ~57 GB more for the four 7B ones).
MODEL_ROOT = os.path.expanduser("~/models")

# path: local model dir, or an HF repo id already in the HF cache (never downloads).
# Optional per model: seq_len, max_new_tokens, max_runs, batch_mb_pairs, splits,
# layer_limits, min_last_rank_layers (these override SWEEP / layer_limits.py).
# On 40/20/10/10 every model runs all 14 batch pairs: the 40GB-box caps (13B at
# B8/B16, Qwen2.5-14B at B<=32) were for 20/10/5/5.
MODELS = {
    "vicuna_13b": dict(path=f"{MODEL_ROOT}/vicuna-13b-v1.5"),
    "llama_13b": dict(path=f"{MODEL_ROOT}/llama-2-13b-hf"),
    "qwen_14b": dict(path=f"{MODEL_ROOT}/Qwen2.5-14B"),
    "mistral_24b": dict(path=f"{MODEL_ROOT}/Mistral-Small-24B-Base-2501"),
    "llama_7b": dict(path=f"{MODEL_ROOT}/llama-2-7b-hf"),
    "mistral_7b": dict(path=f"{MODEL_ROOT}/Mistral-7B-Instruct-v0.3"),
    "qwen_7b": dict(path=f"{MODEL_ROOT}/Qwen2.5-7B"),
    "vicuna_7b": dict(path=f"{MODEL_ROOT}/vicuna-7b-v1.5"),
}

# Two runs, one model per GPU: the heavy models first; once that run is done,
# switch RUN to SMALL and start again.
HEAVY = ["vicuna_13b", "llama_13b", "qwen_14b", "mistral_24b"]
SMALL = ["llama_7b", "mistral_7b", "vicuna_7b", "qwen_7b"]


# The 14 batch pairs every split is swept at (as in SWEEP below).
ALL_PAIRS = {(8, 4), (8, 2), (16, 8), (16, 4), (16, 2), (32, 16), (32, 8), (32, 4), (32, 2),
             (64, 32), (64, 16), (64, 8), (64, 4), (64, 2)}


def _qwen_chunks(sizes):
    """Qwen-7B's splits within layer_limits.py not yet measured at all 14 pairs
    (done_configs.py), dealt in turn to len(sizes) jobs until each holds its size (the
    last takes any rest). A split measured at only some pairs stays in, so its missing
    pairs run (with skip_done the job re-runs it at all its pairs)."""
    import experiment_config as ec
    import layer_limits as ll
    from done_configs import DONE
    measured = {sp for sp, pairs in DONE["40_20_10_10"]["qwen_7b"].items()
                if {tuple(p) for p in pairs} >= ALL_PAIRS}
    every = ec.generate_layer_splits(28, ll.LAYER_LIMITS["40_20_10_10"]["qwen_7b"], [40, 20, 10, 10])
    rest = [s for s in every if tuple(s) not in measured]
    chunks = [[] for _ in sizes]
    i = 0
    for sp in rest:
        while len(chunks[i % len(sizes)]) >= sizes[i % len(sizes)] and any(len(c) < n for c, n in zip(chunks, sizes)):
            i += 1
        chunks[i % len(sizes)].append(sp)
        i += 1
    return [dict(model="qwen_7b", splits=c) if c else None for c in chunks]  # None: nothing left


# 10/04 night: each GPU runs its jobs in order -- a 7B model's new splits (~4.9 h), then a
# share of Qwen-7B's 60 new splits (12/12/13/23 sized so every GPU ends ~10-11 h in).
QWEN = _qwen_chunks([12, 12, 13, 23])
NIGHT = [
    ["llama_7b", QWEN[0]],      # GPU 0
    ["mistral_7b", QWEN[1]],    # GPU 1
    ["vicuna_7b", QWEN[2]],     # GPU 2
    [QWEN[3]],                  # GPU 3
]

# PCIe vs SXM4 check (checking-diff.md): re-run already-measured configs on the PCIe box.
# Run this set twice, as two separate `start`s: the second pass measures the box's own noise.
CARD_CHECK_PAIRS = [(16, 4), (16, 8), (32, 16), (64, 2)]
CARD_CHECK = [
    dict(model="vicuna_13b", batch_mb_pairs=CARD_CHECK_PAIRS,  # GPU 0
         splits=[[23, 11, 4, 2], [20, 11, 5, 4], [23, 9, 4, 4], [23, 8, 6, 3], [21, 7, 6, 6],
                 [17, 11, 6, 6]]),
    dict(model="qwen_14b", batch_mb_pairs=CARD_CHECK_PAIRS,  # GPU 1
         splits=[[28, 10, 7, 3], [27, 10, 7, 4], [26, 9, 8, 5], [28, 8, 6, 6], [20, 10, 9, 9],
                 [26, 10, 6, 6], [25, 10, 7, 6]]),
    dict(model="mistral_7b", batch_mb_pairs=CARD_CHECK_PAIRS,  # GPU 2
         splits=[[16, 10, 4, 2], [15, 10, 4, 3], [17, 11, 2, 2], [20, 7, 4, 1], [19, 5, 4, 4],
                 [20, 8, 3, 1], [12, 11, 5, 4]]),
    dict(model="qwen_7b", batch_mb_pairs=CARD_CHECK_PAIRS,  # GPU 3
         splits=[[14, 9, 4, 1], [11, 10, 5, 2], [18, 8, 1, 1], [20, 5, 2, 1], [9, 7, 6, 6],
                 [20, 6, 1, 1], [9, 8, 7, 4]]),
]

RUN = CARD_CHECK

def _lane(i):  # RUN[i]: a model key, a job dict, or a list of them run in order; None is skipped
    return [j for j in (RUN[i] if isinstance(RUN[i], list) else [RUN[i]]) if j is not None]


# One entry per GPU. slice_gb and mig_uuids in RANK ORDER (largest slice first);
# `./run_parallel.sh discover` prints this block with the real UUIDs.
# models: MODELS keys (or dict(model=key, <overrides>)) run in order; [] = GPU unused.
# A100-80GB: 3g.40gb + 2g.20gb + 1g.10gb + 1g.10gb per GPU.
GPUS = [
    dict(gpu=0, slice_gb=[40, 20, 10, 10], models=_lane(0),
         mig_uuids=["MIG-3b24b0ee-34c8-53d9-8fef-a4b0ea58ae58", "MIG-78342107-fa11-5a12-8d86-02f35bd5c2c4", "MIG-00891593-1c26-564f-bb71-f039c79f5c68", "MIG-92b8af9d-c19e-5299-b97c-7b56fa6c2217"]),  # NVIDIA A100 80GB PCIe: 3g.40gb 2g.20gb 1g.10gb 1g.10gb
    dict(gpu=1, slice_gb=[40, 20, 10, 10], models=_lane(1),
         mig_uuids=["MIG-a33f9552-2e64-5376-938f-0d5059fd3aef", "MIG-1853e45a-112b-56c2-8b82-955e3559661a", "MIG-ef3edd2a-6260-544d-8ad2-f50173de33e7", "MIG-76f6b5a9-95a8-5647-80df-b6e08214a39f"]),  # NVIDIA A100 80GB PCIe: 3g.40gb 2g.20gb 1g.10gb 1g.10gb
    dict(gpu=2, slice_gb=[40, 20, 10, 10], models=_lane(2),
         mig_uuids=["MIG-cf073845-28c9-50f3-b6f9-1eaf5566c511", "MIG-88ab4898-e490-5abb-8317-eb019998f050", "MIG-c8b2731e-53ad-52c7-a445-b2aa103e9c4c", "MIG-87efb1d2-e0fb-5fa9-9b89-5875cb59a028"]),  # NVIDIA A100 80GB PCIe: 3g.40gb 2g.20gb 1g.10gb 1g.10gb
    dict(gpu=3, slice_gb=[40, 20, 10, 10], models=_lane(3),
         mig_uuids=["MIG-289c8561-ae9f-5c39-b734-d583736677b6", "MIG-d3ee3b66-106a-5c5e-ba3a-031e11c22b45", "MIG-8361ffbf-f96d-5ffd-a2dd-2daca06871fb", "MIG-1531a7db-ef6a-57d6-be1d-e135aea4dcf4"]),  # NVIDIA A100 80GB PCIe: 3g.40gb 2g.20gb 1g.10gb 1g.10gb
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
        (8, 4), (8, 2),
        (16, 8), (16, 4), (16, 2),
        (32, 16), (32, 8), (32, 4), (32, 2),
        (64, 32), (64, 16), (64, 8), (64, 4), (64, 2),
    ],
)

RUNNER = dict(
    base_port=29500,  # lane port = base_port + 10 * gpu
    mem_monitor="nvml",  # "nvml" (per MIG UUID), "dcgm" (one lane on GPU 0 only), "off"
    # Extra env for every job. MIG_LOG_LEVEL: "summary" = end-of-run transport
    # summaries only (T22-T29), "off" = no transport log; unset = per-microbatch
    # INFO (the transport's default, GBs per sweep).
    env={"MIG_LOG_LEVEL": "summary"},
    # Skip configs already measured: done_configs.py, built by
    # `python3 make_done.py <results dirs>` (rebuild it after every run).
    # False for CARD_CHECK, which re-runs measured configs on purpose.
    skip_done=False,
)
