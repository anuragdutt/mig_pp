"""
Mistral-7B on one A100-PCIE-40GB, 20/10/5/5, P3 Run B step 1 (agreed 2026-11-26): the split list,
the same 136 splits at every pair. Built here from rules, never from measured data:

1. CORE (120): every split of 32 layers within layer caps [28, 12, 5, 5] under the sweep's
   ordering rule r0 > r1 > r2 >= r3 (experiment_config.generate_layer_splits, the function the
   Vicuna-7B sweep used). The same 120 splits oracle_v2 measured for Mistral-7B (cross-node).
2. RANK0_HEAVY (16): every split with 26-29 layers on rank 0 (>= 1 layer on each other rank) that
   CORE leaves out (rank-0 cap 28, or the ordering rule).

All 136 fit in memory at all 9 pairs (memory_limits.py, checked by tests/test_mistral7b_split_order.py).
Deferred to after S1's deadline (not in ALL): rank 0 = 25, and ranks 2 / 3 swapped (rank 3 also
runs the output head, so the swap is a different configuration).

Run order: CORE in generate_layer_splits order (balanced to rank-0 heavy), then RANK0_HEAVY by
rank 0 descending. clock_study.sh reverses the list on reversed passes.
"""

import experiment_config as ec

LAYERS = 32
SLICE_GB = [20, 10, 5, 5]
CORE_CAPS = [28, 12, 5, 5]
RANK0_HEAVY_MIN = 26
PAIRS = [(8, 4), (16, 8), (32, 16), (8, 2), (16, 4), (32, 8), (16, 2), (32, 4), (32, 2)]


def core_splits():
    return ec.generate_layer_splits(LAYERS, CORE_CAPS, SLICE_GB)


def rank0_heavy_splits(core):
    """Splits with RANK0_HEAVY_MIN..LAYERS-3 layers on rank 0 and >= 1 elsewhere, not in core."""
    out = []
    for r0 in range(LAYERS - 3, RANK0_HEAVY_MIN - 1, -1):
        rest = LAYERS - r0
        for r1 in range(rest - 2, 0, -1):
            for r2 in range(rest - r1 - 1, 0, -1):
                s = [r0, r1, r2, rest - r1 - r2]
                if s not in core:
                    out.append(s)
    return out


def all_splits():
    core = core_splits()
    return core + rank0_heavy_splits(core)


ALL = {pair: all_splits() for pair in PAIRS}
