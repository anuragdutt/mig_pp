"""The frozen ILP versions: their pre-registered picks must not change silently (Run A, Mistral-7B).

P3 Run A (SCORING.md, "Pre-registered: (32,16) and (16,8) at max + 1 thread"): probe set mig4
(probe v7), live kernel-boundary gap, no pool term (1 thread), clock="fast", exact search ->
[23, 6, 2, 1] at 43.50 s / 36.10 s. Needs PuLP < 3 (as tests/test_pulp_layer_placement.py).

    PYTHONPATH=<scratch>/pulplib .venv/bin/python -m unittest tests/test_frozen_picks.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "mig_pp", "milp"), os.path.join(HERE, "..")]

try:
    import graph_placement as g
    import gpipe_placement as gp
    import hardware_constants as h
except ImportError as err:  # no PuLP here
    g = None
    SKIP = str(err)


def config(probe_set, gap_mode):
    constants = {k: v for k, v in h.CONSTANTS[probe_set].items() if not k.startswith(("recv_copy_pool", "pool_"))}
    constants["gap_mode"] = gap_mode
    return dict(model="vicuna_7b", slice_gb=[20, 10, 5, 5], card="A100-40GB", seq_len=64, max_new_tokens=512,
                samples=8, min_last_rank_layers=1, clock="fast", constants=constants)


@unittest.skipIf(g is None, "PuLP not importable")
class FrozenPicks(unittest.TestCase):
    def test_run_a_preregistered(self):
        cfg = config("mig4", "live")
        for (b, mb), seconds in (((32, 16), 43.50), ((16, 8), 36.10)):
            pick, _ = g.optimize_placement(cfg, b, mb)
            self.assertEqual(list(pick), [23, 6, 2, 1], (b, mb))
            self.assertAlmostEqual(g.predict_ms(cfg, list(pick), b, mb) / 1000, seconds, places=2)

    def test_mistral_preregistered(self):
        # SCORING.md "Pre-registered: Mistral-7B" (2026-11-26): mig5 set, live gap, clock="fast"
        cfg = dict(config("mig5", "live"), model="mistral_7b")
        graph = {(8, 4): [27, 3, 1, 1], (16, 8): [27, 3, 1, 1], (32, 16): [26, 4, 1, 1],
                 (8, 2): [19, 9, 3, 1], (16, 4): [19, 9, 3, 1], (32, 8): [20, 8, 3, 1],
                 (16, 2): [16, 9, 4, 3], (32, 4): [17, 9, 4, 2], (32, 2): [16, 9, 4, 3]}
        gpipe = {(8, 4): [20, 10, 1, 1], (16, 8): [20, 10, 1, 1], (32, 16): [20, 10, 1, 1],
                 (8, 2): [16, 9, 3, 4], (16, 4): [18, 10, 1, 3], (32, 8): [18, 10, 3, 1],
                 (16, 2): [16, 9, 3, 4], (32, 4): [16, 9, 3, 4], (32, 2): [16, 9, 3, 4]}
        graph_s = {(8, 4): 31.00, (16, 8): 31.81, (32, 16): 34.69, (8, 2): 49.89, (16, 4): 50.40,
                   (32, 8): 51.56, (16, 2): 77.34, (32, 4): 78.53, (32, 2): 130.27}
        for pair in graph:
            pick, _ = g.optimize_placement(cfg, *pair)
            self.assertEqual(list(pick), graph[pair], pair)
            self.assertAlmostEqual(g.predict_ms(cfg, list(pick), *pair) / 1000, graph_s[pair], places=2)
            gpick, _ = gp.optimize_placement(cfg, *pair)
            self.assertEqual(list(gpick), gpipe[pair], pair)

    def test_gap_adds_time(self):
        # the gap only adds GPU time: with it, every rank's composed time is at least as large
        with_gap, no_gap = config("mig5", "live"), config("mig5", None)
        for s in ([21, 8, 2, 1], [16, 8, 4, 4], [28, 2, 1, 1]):
            self.assertGreater(g.predict_ms(with_gap, s, 32, 16), g.predict_ms(no_gap, s, 32, 16), s)

    def test_mig5_set_has_no_clock_model(self):
        # probe v7.1 under the main condition: no governor / pool data, so clock="schedutil" is refused
        self.assertNotIn("clock_slow_factor", h.CONSTANTS["mig5"])
        self.assertNotIn("recv_copy_pool_us", h.CONSTANTS["mig5"])
        cfg = dict(config("mig5", "live"), clock="schedutil")
        with self.assertRaises(ValueError):
            g.predict_ms(cfg, [21, 8, 2, 1], 32, 16)


if __name__ == "__main__":
    unittest.main()
