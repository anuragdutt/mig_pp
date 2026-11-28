"""clock_study.sh presets through the real --dry-run entry point (no GPU, nothing touched), and the
rule-based list modules they use (mistral7b_post_split_order, vicuna7b_topup_split_order).

Each case copies clock_study.sh into a temp dir; generated list modules (step 2, 3, 3b) are written
there from made-up lists; the rest of the repo comes through PYTHONPATH.

    .venv/bin/python -m unittest tests/test_runb_presets.py -v
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import memory_limits as ml  # noqa: E402
import mistral7b_40gb_split_order as step1  # noqa: E402
import mistral7b_post_split_order as post  # noqa: E402
import vicuna7b_topup_split_order as vtop  # noqa: E402

PAIRS = step1.PAIRS


def dry_run(args, modules=None):
    """(exit code, output) of `clock_study.sh <args> --dry-run`; modules: {name: ALL dict} written beside it."""
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(os.path.join(REPO, "clock_study.sh"), d)
        for name, lists in (modules or {}).items():
            with open(os.path.join(d, name + ".py"), "w") as f:
                f.write("ALL = " + repr(lists) + "\n")
        env = dict(os.environ, INV_HOST="mig5", PYTHON=sys.executable, PYTHONPATH=REPO, CLOCK_NOTIFY_TOPIC="")
        r = subprocess.run(["bash", os.path.join(d, "clock_study.sh"), *args, "--dry-run"], cwd=d, env=env,
                           capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr


def blocks(out):
    """[(pair, n splits)] of the '== block' lines, in order."""
    return [(m.group(1), int(m.group(2))) for m in re.finditer(r"\((\d+,\d+)\): (\d+) splits", out)]


FAKE = {pair: [[20, 8, 3, 1], [21, 7, 3, 1], [19, 9, 3, 1]][: 1 + i % 3] for i, pair in enumerate(PAIRS)}


class Presets(unittest.TestCase):
    def test_default_equals_runB_step1(self):
        code0, out0 = dry_run([])
        code1, out1 = dry_run(["runB_step1"])
        self.assertEqual((code0, code1), (0, 0), out0 + out1)
        self.assertEqual(blocks(out0), [(f"{b},{mb}", 136) for b, mb in PAIRS])
        self.assertEqual(blocks(out0), blocks(out1))

    def test_step2_and_step3(self):
        for preset, module in (("runB_step2", "mistral7b_step2_split_order"),
                               ("runB_step3", "mistral7b_step3_split_order")):
            code, out = dry_run([preset], {module: FAKE})
            self.assertEqual(code, 0, out)
            self.assertIn(f"preset {preset}", out)
            self.assertEqual(blocks(out), [(f"{b},{mb}", len(FAKE[(b, mb)])) for b, mb in PAIRS])

    def test_step2_without_its_module_stops(self):
        code, out = dry_run(["runB_step2"])
        self.assertNotEqual(code, 0)
        self.assertIn("STOP: split lists", out)
        self.assertEqual(blocks(out), [])

    def test_rerun_only_flagged_pairs(self):
        flagged = {(32, 2): FAKE[(32, 2)], (8, 4): FAKE[(8, 4)]}
        code, out = dry_run(["runB_step3_rerun"], {"mistral7b_step3b_split_order": flagged})
        self.assertEqual(code, 0, out)
        self.assertEqual(blocks(out), [("32,2", len(flagged[(32, 2)])), ("8,4", len(flagged[(8, 4)]))])

    def test_rerun_without_module_stops_before_anything(self):
        code, out = dry_run(["runB_step3_rerun"])
        self.assertEqual(code, 2, out)
        self.assertIn("no pairs from mistral7b_step3b_split_order", out)

    def test_post(self):
        code, out = dry_run(["runB_post"])
        self.assertEqual(code, 0, out)
        self.assertEqual(blocks(out), [(f"{b},{mb}", 88) for b, mb in PAIRS])

    def test_vicuna_topup(self):
        code, out = dry_run(["vicuna_topup"])
        self.assertEqual(code, 0, out)
        want = [(f"{b},{mb}", len(v)) for (b, mb), v in vtop.ALL.items()]
        self.assertEqual(blocks(out), want + want)  # 2 passes
        self.assertIn("model vicuna_7b", out)

    def test_unknown_and_two_presets(self):
        self.assertEqual(dry_run(["runB_step4"])[0], 2)
        self.assertEqual(dry_run(["runB_step2", "runB_post"])[0], 2)


class Lists(unittest.TestCase):
    def test_post_rules(self):
        have = {tuple(s) for s in step1.all_splits()}
        lst = post.ALL[PAIRS[0]]
        self.assertEqual(len(lst), 88)
        self.assertEqual(len({tuple(s) for s in lst}), 88)
        self.assertFalse({tuple(s) for s in lst} & have)
        self.assertEqual(sum(s[0] == 25 for s in lst), 12)
        self.assertTrue(all(sum(s) == 32 and min(s) >= 1 for s in lst))
        mm = ml.MemoryModel(ml.REFERENCE_SHAPES["mistral_7b"])
        for b, mb in PAIRS:
            self.assertEqual([s for s in post.ALL[(b, mb)] if mm.oom_ranks(s, b, mb, [20, 10, 5, 5])], [])

    def test_vicuna_topup_fits_and_skips_oom(self):
        mm = ml.MemoryModel(ml.REFERENCE_SHAPES["vicuna_7b"])
        for (b, mb), splits in vtop.ALL.items():
            for s in splits:
                self.assertEqual(mm.oom_ranks(s, b, mb, [20, 10, 5, 5]), [], (b, mb, s))
        for pair in [(16, 8), (32, 16), (16, 4), (32, 8), (32, 4), (32, 2)]:
            self.assertNotIn([8, 8, 8, 8], vtop.ALL.get(pair, []))
            self.assertTrue(mm.oom_ranks([8, 8, 8, 8], *pair, [20, 10, 5, 5]))


if __name__ == "__main__":
    unittest.main()
