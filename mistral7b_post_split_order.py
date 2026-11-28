"""
Mistral-7B on mig5, the pass after Sat 11/28 11:45 pm EST (TO-DO.md): the splits Run B step 1 leaves
out by its caps / ordering rule, built from rules, never from measured data. Same 88 at every pair:

1. every split with 25 layers on rank 0 (>= 1 layer elsewhere) not in step 1: 12;
2. ranks 2 and 3 swapped for every step-1 core split with r2 > r3 (rank 3 also runs the output head,
   so the swap is a different configuration), not in step 1: 77 (one overlaps item 1).

All fit in memory at all 9 pairs (tests/test_runb_presets.py). Not part of the primary score
(SCORING.md, "What is scored"): a coverage check reported separately.
"""

import mistral7b_40gb_split_order as step1

PAIRS = step1.PAIRS


def post_splits():
    have = {tuple(s) for s in step1.all_splits()}
    rank0_25 = [[25, b, c, 7 - b - c] for b in range(5, 0, -1) for c in range(6 - b, 0, -1) if 7 - b - c >= 1]
    swaps = [[a, b, d, c] for a, b, c, d in step1.core_splits() if c > d]
    out = []
    for s in rank0_25 + swaps:
        if tuple(s) not in have and s not in out:
            out.append(s)
    return out


ALL = {pair: post_splits() for pair in PAIRS}
