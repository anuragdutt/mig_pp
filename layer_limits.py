"""
Layer limits: the most decoder layers each rank (MIG slice) may hold, per MIG
layout and model. Edit here; nothing else hardcodes them.

The sweep runs every split with at most LAYER_LIMITS[layout][model][r] layers on
rank r that sums to the model's layer count, giving a bigger slice strictly
more layers than a smaller one. A limit is a cap: a split that fits at batch 8
may OOM at 64, which the benchmark records as OOM_rank<N>.

Layout key = the GPU's slice_gb joined by "_" (rank 0 first). A model or layout
missing here fails `./run_parallel.sh check` -- no silent fallback.

`python3 memory_limits.py` shows the theory behind the values: [18, 12, 5, 5]
(Vicuna-7B, 9/28) is rank 0 = its capacity at B64 minus 1, ranks 1-3 = capacity
at B32 minus 1. The 13B vector applies the same rule; the other 7B models
reuse Vicuna's vector so they sweep the same splits.
"""

LAYER_LIMITS = {
    # 3g.20gb + 2g.10gb + 1g.5gb + 1g.5gb on A100-40GB
    "20_10_5_5": {
        "vicuna_7b": [18, 12, 5, 5],  # swept 9/28
        "llama_7b": [18, 12, 5, 5],  # same shapes as Vicuna-7B
        "mistral_7b": [18, 12, 5, 5],  # GQA: 4x smaller KV, all of these fit even at B64
        # 152k-vocab lm_head is built fp32 then halved: 3118 MiB of the last
        # 5GB slice for a moment, leaving room for 3 layers at most.
        "qwen_7b": [18, 12, 5, 3],
        "vicuna_13b": [23, 12, 5, 5],  # cannot fit at B32/B64 at all on this layout
        "llama_13b": [23, 12, 5, 5],
        # lm_head build needs 4455 MiB of the last slice: rank 3 holds only
        # norm + lm_head. [30, 15, 7] is exactly the capacity at B32.
        "qwen_14b": [30, 15, 7, 0],
        # Mistral-Nemo-12B, 40 layers: its 131k-vocab lm_head build needs 3840 MiB
        # of the last slice, so rank 3 is head-only as for Qwen2.5-14B. By the
        # rule: fits up to B32, B64 is the stress batch (~18% OOM predicted).
        "nemo_12b": [26, 15, 6, 0],
    },
}

# Models whose LAST rank may hold no decoder layers (only norm + lm_head), as
# the qwen-7b / qwen-14b branches swept (e.g. the oracle's [16, 8, 4, 0]).
HEAD_ONLY_LAST_RANK = {"qwen_7b", "qwen_14b", "nemo_12b"}


def layout_key(slice_gb) -> str:
    return "_".join(str(int(g)) for g in slice_gb)


def limits_for(model_key: str, slice_gb):
    """(layer_limits, min_last_rank_layers). KeyError names what is missing."""
    key = layout_key(slice_gb)
    if key not in LAYER_LIMITS:
        raise KeyError(f"layer_limits.py has no layout '{key}' (have: {', '.join(sorted(LAYER_LIMITS))})")
    if model_key not in LAYER_LIMITS[key]:
        raise KeyError(f"layer_limits.py has no entry for model '{model_key}' on layout '{key}'")
    limits = list(LAYER_LIMITS[key][model_key])
    if len(limits) != len(list(slice_gb)):
        raise KeyError(f"layer_limits.py: '{model_key}' has {len(limits)} limits for {len(list(slice_gb))} slices")
    return limits, (0 if model_key in HEAD_ONLY_LAST_RANK else 1)
