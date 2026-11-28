"""Run B tools (runb_lists.py, runb_finish_check.py, runb_noise_check.py) on synthetic exports laid out
like the box's: investigation_runs.txt with '# run' blocks, runs/<dir>/gpu0/01_mistral_7b/
mig_benchmark_results.csv. Latencies are made up; nothing here reads Run B data.

    .venv/bin/python -m unittest tests/test_runb_tools.py -v
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mistral7b_40gb_split_order as m  # noqa: E402
import runb_finish_check as fc  # noqa: E402
import runb_lists as rl  # noqa: E402
import runb_noise_check as nc  # noqa: E402


def fake_ms(split, pair, offset=0.0):
    """A smooth made-up latency with its best near [20, 8, 3, 1]; distinct per split."""
    a, b, c, d = split
    return 30000 + pair[0] * 100 + 400 * abs(a - 20) + 300 * abs(b - 8) + 200 * abs(c - 3) + 150 * abs(d - 1) \
        + a * 0.01 + b * 0.001 + offset


def make_export(root, runs, stale=True, prefix="mistral7b_clock_", dup_last=False):
    """runs: [(pair, {split: (latency, status)})], one oracle each, in one '# run' block (after a stale one)."""
    os.makedirs(root, exist_ok=True)
    lines = ["# run 2026-11-25 old block\n", "old_step runs/old\n"] if stale else []
    lines.append("# run 2026-11-27 clock_study.sh\n")
    for i, (pair, rows) in enumerate(runs):
        d = f"runs/r{i}"
        os.makedirs(os.path.join(root, d, "gpu0", "01_mistral_7b"))
        with open(os.path.join(root, d, "gpu0", "01_mistral_7b", "mig_benchmark_results.csv"), "w") as f:
            f.write("split,batch_size,microbatch_size,total_latency_ms,status\n")
            for s, (lat, st) in rows.items():
                f.write(f'"{json.dumps(list(s))}",{pair[0]},{pair[1]},{lat},{st}\n')
        lines.append(f"{prefix}blk01_max-t1_b{pair[0]}_mb{pair[1]} {d}\n")
    if dup_last:
        lines.append(lines[-1])
    with open(os.path.join(root, "investigation_runs.txt"), "w") as f:
        f.writelines(lines)


def step1_export(root, drop=None, status=None):
    runs = []
    for pair in m.PAIRS:
        rows = {tuple(s): (fake_ms(s, pair), "ok") for s in m.ALL[pair]}
        if drop and pair == drop[0]:
            del rows[drop[1]]
        if status and pair == status[0]:
            rows[status[1]] = (0, status[2])
        runs.append((pair, rows))
    make_export(root, runs)


def run_quiet(fn, *a, **k):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            result = fn(*a, **k)
        except SystemExit as e:  # as on the command line: a message means exit 1, shown on the terminal
            result = e.code
            if isinstance(e.code, str):
                out.write(e.code)
                result = 1
    return result, out.getvalue()


class Tools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.mm = rl.memory_limits.MemoryModel(rl.memory_limits.REFERENCE_SHAPES["mistral_7b"])

    def tearDown(self):
        self.tmp.cleanup()

    # --- finish check
    def test_finish_all_ok(self):
        step1_export(self.d)
        code, out = run_quiet(fc.main, dict(export=self.d, module="mistral7b_40gb_split_order", prefix="mistral7b_clock_", passes=1))
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("ok 136 of 136 expected"), 9)

    def test_finish_missing_and_oom(self):
        step1_export(self.d, drop=((32, 2), (16, 8, 4, 4)))
        code, out = run_quiet(fc.main, dict(export=self.d, module="mistral7b_40gb_split_order", prefix="mistral7b_clock_", passes=1))
        self.assertEqual(code, 1)
        self.assertIn("missing 1", out)
        step1_export(os.path.join(self.d, "b"), status=((8, 4), (29, 1, 1, 1), "oom"))
        code, out = run_quiet(fc.main, dict(export=os.path.join(self.d, "b"), module="mistral7b_40gb_split_order", prefix="mistral7b_clock_", passes=1))
        self.assertEqual(code, 1)
        self.assertIn("not ok {'oom': 1}: [[29, 1, 1, 1]]", out)

    def test_last_block_only(self):
        step1_export(self.d)
        self.assertNotIn(("old_step", "runs/old"), rl.last_run_block(self.d))
        self.assertEqual(len(rl.last_run_block(self.d)), 9)

    # --- step 2
    def test_step2_rules(self):
        step1_export(self.d)
        data = rl.load(self.d)
        for pair in m.PAIRS:
            chosen, n_cands, baselines = rl.step2_list(self.mm, data[pair], pair, 3.0, 25)
            measured = set(data[pair])
            self.assertEqual(len(chosen), len(set(chosen)), pair)              # each split once
            self.assertFalse(set(chosen) & measured, pair)                    # only unmeasured
            self.assertTrue(all(rl.fits(self.mm, s, pair) for s in chosen))   # all fit
            self.assertLessEqual(len(chosen) - len(baselines), 25)            # cap on (a) + (b)
            self.assertIn((8, 8, 8, 8), chosen)                               # even fits Mistral: always added
            gp = rl.GPIPE_PICK[pair]
            self.assertEqual(gp in chosen, gp not in measured, pair)          # GPipe pick iff unmeasured

    def test_step2_deterministic_tie_break(self):
        # every candidate's parent has the same latency: the order must then be the split tuples, ascending
        pair = (32, 16)
        measured = {tuple(s): [1000.0] for s in m.ALL[pair]}
        chosen, n_cands, baselines = rl.step2_list(self.mm, measured, pair, 3.0, 25)
        refine = [s for s in chosen if s not in baselines]
        self.assertGreater(n_cands, 25)
        self.assertEqual(refine, sorted(refine))
        self.assertEqual(rl.step2_list(self.mm, measured, pair, 3.0, 25)[0], chosen)

    def test_step2_main_prints_no_latency(self):
        step1_export(self.d)
        out_dir = os.path.join(self.d, "mod")
        os.makedirs(out_dir)
        _, out = run_quiet(rl.main, dict(step=2, step1=self.d, step2=None, step1_prefix="mistral7b_clock_",
                                         step2_prefix="mistral7b_s2_", group_pct=3.0, cap=25, out_dir=out_dir))
        self.assertIsNone(re.search(r"\d+\.\d", out), out)  # no decimal number anywhere: no latency
        ns = {}
        with open(os.path.join(out_dir, "mistral7b_step2_split_order.py")) as f:
            exec(f.read(), ns)
        self.assertEqual(set(ns["ALL"]), set(m.PAIRS))

    # --- step 3 from the builder's entry point
    def step2_export(self, drop_even=False):
        """A step-1 export, then a step-2 export of the step-2 list it implies (made-up numbers)."""
        step1_export(self.d)
        data = rl.load(self.d)
        runs = []
        for pair in m.PAIRS:
            chosen = rl.step2_list(self.mm, data[pair], pair, 3.0, 25)[0]
            runs.append((pair, {x: (fake_ms(x, pair), "ok") for x in chosen if not (drop_even and x == (8, 8, 8, 8))}))
        make_export(os.path.join(self.d, "s2"), runs, prefix="mistral7b_s2_clock_")
        return os.path.join(self.d, "s2")

    def build3(self, step2):
        out_dir = os.path.join(self.d, "mod3")
        os.makedirs(out_dir, exist_ok=True)
        code, out = run_quiet(rl.main, dict(step=3, step1=self.d, step2=step2, step1_prefix="mistral7b_clock_",
                                            step2_prefix="mistral7b_s2_", group_pct=3.0, cap=25, out_dir=out_dir))
        return code, out, os.path.join(out_dir, "mistral7b_step3_split_order.py")

    def test_step3_refuses_same_export_twice(self):
        # review 2026-11-28: step 1 given as step 2 wrote a list without [8, 8, 8, 8]
        self.step2_export()
        code, out, path = self.build3(self.d)
        self.assertEqual(code, 1)
        self.assertIn("NO LIST WRITTEN", out)
        self.assertIn("same export folder", out)
        self.assertIn("does not start with mistral7b_s2_", out)
        self.assertFalse(os.path.exists(path))

    def test_step3_refuses_missing_baseline(self):
        code, out, path = self.build3(self.step2_export(drop_even=True))
        self.assertIn("NO LIST WRITTEN", out)
        self.assertIn("frozen picks / baselines never measured: [[8, 8, 8, 8]]", out)
        self.assertFalse(os.path.exists(path))

    def test_step3_list_has_every_pick_and_baseline(self):
        code, out, path = self.build3(self.step2_export())
        self.assertIn("wrote", out)
        ns = {}
        with open(path) as f:
            exec(f.read(), ns)
        for pair in m.PAIRS:
            for x in [rl.GRAPH_PICK[pair], rl.GPIPE_PICK[pair]] + rl.RULE_BASELINES:
                self.assertIn(list(x), ns["ALL"][pair], (pair, x))

    def test_step3_set(self):
        step1_export(self.d)
        data = rl.load(self.d)
        pair = (16, 4)
        out = rl.step3_list(self.mm, data[pair], pair, 3.0)
        med = {s: v[0] for s, v in data[pair].items()}
        best = min(med.values())
        for s, v in med.items():
            self.assertEqual(s in out, v <= best * 1.03 or s in
                             [rl.GRAPH_PICK[pair], rl.GPIPE_PICK[pair]] + rl.RULE_BASELINES, s)

    def test_finish_duplicate_oracle_and_wrong_prefix(self):
        # review 2026-11-28: a 10th, repeated oracle entry must not pass
        runs = [(pair, {tuple(x): (fake_ms(x, pair), "ok") for x in m.ALL[pair]}) for pair in m.PAIRS]
        make_export(self.d, runs, dup_last=True)
        code, out = run_quiet(fc.main, dict(export=self.d, module="mistral7b_40gb_split_order", prefix="mistral7b_clock_", passes=1))
        self.assertEqual(code, 1)
        self.assertIn("listed 2 times", out)
        self.assertIn("(32, 2): 2 oracles, want 1", out)
        step1_export(os.path.join(self.d, "b"))  # a step-1 export checked as step 2
        code, out = run_quiet(fc.main, dict(export=os.path.join(self.d, "b"), module="mistral7b_40gb_split_order",
                                            prefix="mistral7b_s2_", passes=1))
        self.assertEqual(code, 1)
        self.assertIn("does not start with mistral7b_s2_", out)

    # --- noise check
    def noise_inputs(self, partial=False):
        """A step-1 export, a step-3 module and a step-3 export built from them (made-up numbers)."""
        step1_export(self.d)
        data = rl.load(self.d)
        sets = {pair: [list(x) for x in rl.step3_list(self.mm, data[pair], pair, 3.0)] for pair in m.PAIRS}
        mod_dir = os.path.join(self.d, "mod")
        os.makedirs(mod_dir, exist_ok=True)
        rl.write_module(os.path.join(mod_dir, "fake_step3_split_order.py"), sets, "test")
        s3 = [(pair, {tuple(x): (data[pair][tuple(x)][0] * 1.001, "ok") for x in v[: -1 if partial and pair == (8, 4) else None]})
              for pair, v in sets.items()]
        make_export(os.path.join(self.d, "s3"), s3, prefix="mistral7b_s3_clock_")
        return mod_dir

    def run_noise(self, mod_dir, step3, earlier=None):
        sys.path.insert(0, mod_dir)
        try:
            return run_quiet(nc.main, dict(earlier=earlier or [(self.d, "mistral7b_clock_")], step3=step3, median_limit=0.5, any_limit=2.0,
                                           step3_module="fake_step3_split_order", step3_prefix="mistral7b_s3_",
                                           out_dir=mod_dir))
        finally:
            sys.path.remove(mod_dir)
            sys.modules.pop("fake_step3_split_order", None)

    def test_noise_refuses_step1_as_step3(self):
        # review 2026-11-28: the step-1 export given as the confirmation run compared each latency with itself
        mod_dir = self.noise_inputs()
        code, out = self.run_noise(mod_dir, self.d)
        self.assertEqual(code, 1)
        self.assertIn("NO VERDICT", out)
        self.assertIn("export folder given twice", out)
        self.assertIn("does not start with mistral7b_s3_", out)

    def test_noise_refuses_partial_step3(self):
        mod_dir = self.noise_inputs(partial=True)
        code, out = self.run_noise(mod_dir, os.path.join(self.d, "s3"))
        self.assertEqual(code, 1)
        self.assertIn("pair (8, 4): 0 splits without an earlier ok run, 1 without a step-3 ok run", out)

    def test_noise_refuses_step3_copy_as_earlier(self):
        # review 2026-11-28: a copy of the step-3 export given as the earlier measurement
        import shutil
        mod_dir = self.noise_inputs()
        copy = os.path.join(self.d, "s3_copy")
        shutil.copytree(os.path.join(self.d, "s3"), copy)
        code, out = self.run_noise(mod_dir, os.path.join(self.d, "s3"), earlier=[(copy, "mistral7b_clock_")])
        self.assertEqual(code, 1)
        self.assertIn("NO VERDICT", out)
        self.assertIn("does not start with mistral7b_clock_", out)

    def test_finish_vicuna_two_passes(self):
        # review 2026-11-28: the Vicuna top-up runs 2 passes per pair; that is correct, not a duplicate
        import vicuna7b_topup_split_order as vt
        runs = [(pair, {tuple(x): (fake_ms(x, pair, k), "ok") for x in v}) for k in (0, 1) for pair, v in vt.ALL.items()]
        make_export(self.d, runs, prefix="vicuna7b_vtop_clock_")
        cfg = dict(export=self.d, module="vicuna7b_topup_split_order", prefix="vicuna7b_vtop_")
        code, out = run_quiet(fc.main, dict(cfg, passes=2))
        self.assertEqual(code, 0, out)
        code, out = run_quiet(fc.main, dict(cfg, passes=1))
        self.assertEqual(code, 1)
        self.assertIn("2 oracles, want 1", out)

    def test_noise_complete_inputs_give_a_verdict(self):
        mod_dir = self.noise_inputs()
        third, out = self.run_noise(mod_dir, os.path.join(self.d, "s3"))
        self.assertEqual(third, [], out)

    def test_repeat_diff(self):
        self.assertAlmostEqual(nc.repeat_diff(100, 102), nc.repeat_diff(102, 100))
        self.assertAlmostEqual(nc.repeat_diff(99, 101), 2.0)

    def test_noise_trigger(self):
        self.assertFalse(nc.needs_third(dict(median=0.5, worst=2.0)))
        self.assertTrue(nc.needs_third(dict(median=0.51, worst=1.0)))
        self.assertTrue(nc.needs_third(dict(median=0.1, worst=2.01)))

    def test_noise_main_prints_aggregates_only(self):
        step1_export(self.d)
        data = rl.load(self.d)
        s3 = []
        for pair in m.PAIRS:
            conf = rl.step3_list(self.mm, data[pair], pair, 3.0)
            # (32,2) gets a 3% repeat difference on one split: must be flagged
            s3.append((pair, {s: (data[pair][s][0] * (1.03 if pair == (32, 2) and i == 0 else 1.001), "ok")
                              for i, s in enumerate(conf)}))
        make_export(os.path.join(self.d, "s3"), s3, prefix="mistral7b_s3_clock_")
        mod_dir = os.path.join(self.d, "mod")
        os.makedirs(mod_dir)
        rl.write_module(os.path.join(mod_dir, "fake_step3_split_order.py"),
                        {pair: list(rows) for pair, rows in s3}, "test")
        sys.path.insert(0, mod_dir)
        try:
            third, out = run_quiet(nc.main, dict(earlier=[(self.d, "mistral7b_clock_")], step3=os.path.join(self.d, "s3"),
                                                 median_limit=0.5, any_limit=2.0,
                                                 step3_module="fake_step3_split_order", step3_prefix="mistral7b_s3_",
                                                 out_dir=mod_dir))
        finally:
            sys.path.remove(mod_dir)
        self.assertEqual(third, [(32, 2)])
        ns = {}
        with open(os.path.join(mod_dir, "mistral7b_step3b_split_order.py")) as f:
            exec(f.read(), ns)
        self.assertEqual(list(ns["ALL"]), [(32, 2)])  # only the flagged pair, with its step-3 set
        self.assertEqual(ns["ALL"][(32, 2)], [list(x) for x in dict(s3)[(32, 2)]])
        self.assertNotIn("[", out.split("3rd run (same")[0].replace("pair", ""))  # no split names in the table


if __name__ == "__main__":
    unittest.main()
