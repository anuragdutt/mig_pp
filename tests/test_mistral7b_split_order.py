"""P3 Run B's split list (mistral7b_40gb_split_order.py): 136 splits per pair, built from rules.

    .venv/bin/python -m unittest tests/test_mistral7b_split_order.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import memory_limits as ml  # noqa: E402
import mistral7b_40gb_split_order as m  # noqa: E402


class SplitList(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(m.core_splits()), 120)
        self.assertEqual(set(m.ALL), set(m.PAIRS))
        for pair, splits in m.ALL.items():
            self.assertEqual(len(splits), 136, pair)
            self.assertEqual(len({tuple(s) for s in splits}), 136, f"duplicates at {pair}")

    def test_every_split_is_valid(self):
        for s in m.ALL[m.PAIRS[0]]:
            self.assertEqual(len(s), 4)
            self.assertEqual(sum(s), 32, s)
            self.assertTrue(all(x >= 1 for x in s), s)

    def test_core_rule_and_caps(self):
        for s in m.core_splits():
            self.assertTrue(s[0] > s[1] > s[2] >= s[3], s)
            self.assertTrue(all(x <= c for x, c in zip(s, [28, 12, 5, 5])), s)

    def test_rank0_26_to_29_complete(self):
        have = {tuple(s) for s in m.ALL[m.PAIRS[0]]}
        want = {(a, b, c, 32 - a - b - c) for a in range(26, 30) for b in range(1, 32)
                for c in range(1, 32) if 32 - a - b - c >= 1}
        self.assertEqual(len(want), 1 + 3 + 6 + 10)
        self.assertEqual(want - have, set())
        self.assertEqual(max(s[0] for s in have), 29)

    def test_same_list_every_pair(self):
        first = m.ALL[m.PAIRS[0]]
        for pair in m.PAIRS:
            self.assertEqual(m.ALL[pair], first, pair)

    def test_all_fit_in_memory(self):
        mm = ml.MemoryModel(ml.REFERENCE_SHAPES["mistral_7b"])
        for b, mb in m.PAIRS:
            bad = [s for s in m.ALL[(b, mb)] if mm.oom_ranks(s, b, mb, m.SLICE_GB)]
            self.assertEqual(bad, [], f"({b},{mb})")


if __name__ == "__main__":
    unittest.main()
