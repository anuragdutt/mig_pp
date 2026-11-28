"""
Vicuna-7B on mig5 (= mig3's card, so run 17's main-condition data stands), after the Mistral runs:
only the baselines and ILP picks never measured under the main condition (TO-DO.md, 2026-11-27).
No anchors, no repeats of measured splits. Even [8, 8, 8, 8] only where it fits in memory: it does
not fit at (16,8), (32,16), (16,4), (32,8), (32,4), (32,2) (rank 3, and at batch 32 rank 2 too).
(32,4) and (32,2) have nothing missing, so they are not here. Run with 2 passes (forward, reversed).
"""

ALL = {
    (8, 4): [[8, 8, 8, 8], [14, 9, 5, 4], [20, 10, 1, 1]],  # even, GPC 3:2:1:1, GPipe pick
    (16, 8): [[14, 9, 5, 4]],
    (32, 16): [[14, 9, 5, 4]],
    (8, 2): [[8, 8, 8, 8], [14, 9, 5, 4]],
    (16, 4): [[14, 9, 5, 4], [16, 9, 3, 4]],  # GPC, GPipe pick
    (32, 8): [[14, 9, 5, 4], [18, 10, 1, 3]],  # GPC, GPipe pick
    (16, 2): [[8, 8, 8, 8], [14, 9, 5, 4]],
}
