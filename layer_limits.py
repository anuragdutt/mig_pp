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
    # 3g.40gb + 2g.20gb + 1g.10gb + 1g.10gb on A100-80GB: the 20_10_5_5 compute
    # split with twice the memory. The rule above would admit hundreds of splits
    # per model here, so these are sized to a ~2.5-day sweep (29-33 splits x 14
    # batch pairs, ~43 h per lane estimated) instead: small-slice caps at their
    # B64 capacity (memory_limits.py), then as wide on ranks 0-1 as the budget
    # allows. The April 80GB runs used [24, 10, 5, 5] (13B) and [30, 10, 4, 2]
    # (24B) -- small-slice caps left over from the 5GB slices.
    "40_20_10_10": {
        # 10GB slices hold 6 13B layers at B64 (MHA: big KV). 33 splits:
        # rank 0 17-22, rank 1 7-11, rank 2 4-6, rank 3 1-6 layers.
        "vicuna_13b": [22, 11, 6, 6],
        "llama_13b": [22, 11, 6, 6],  # same shapes as Vicuna-13B
        # 31 splits: 20-27 / 8-10 / 6-9 / 2-9. No longer head-only: the
        # 152k-vocab lm_head build leaves room for 9 layers on a 10GB slice
        # (within ~300 MiB).
        "qwen_14b": [27, 10, 9, 9],
        # Mistral-Small-24B, 1060 MiB/layer: rank 2 holds 7 at B64, rank 3 5
        # beside the 131k-vocab lm_head build (within ~340 MiB). 29 splits:
        # 18-22 / 7-10 / 4-7 / 1-5.
        "mistral_24b": [22, 10, 7, 5],
        # 7B models: every split fits even at B64 (10GB slices hold 8-17 of their
        # layers), so the ordering rule, not memory, caps the small slices (at 9 / 7
        # for 32 layers, 8 / 6 for Qwen's 28). Sized for ~12 h of B32 only (4 pairs,
        # ~11 min per split at the 9/28 7B pace + 15%), small slices as high as the
        # ordering allows. The user widened rank 0 to 18: 112 splits, 10-18 / 6-11 /
        # 2-9 / 1-7, ~21 h at that pace -- a bet that the 80GB box runs ~2x faster.
        "llama_7b": [18, 12, 5, 5],
        "mistral_7b": [18, 12, 5, 5],
        # 53 splits: 9-13 / 6-12 / 2-8 / 1-6.
        "qwen_7b": [13, 12, 8, 6],
        "vicuna_7b": [18, 12, 5, 5],
    },
}

# Per layout, models whose LAST rank may hold no decoder layers (only norm +
# lm_head), as the qwen-7b / qwen-14b branches swept (e.g. the oracle's
# [16, 8, 4, 0]): on a 5GB slice their lm_head build leaves no room. A 10GB
# slice has room, so 40_20_10_10 has none.
HEAD_ONLY_LAST_RANK = {"20_10_5_5": {"qwen_7b", "qwen_14b", "nemo_12b"}}


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
    return limits, (0 if model_key in HEAD_ONLY_LAST_RANK.get(key, ()) else 1)
