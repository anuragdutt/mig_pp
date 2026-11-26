"""clock_study.sh's split-list check, through the real --dry-run entry point (no GPU, nothing touched).

Each case copies clock_study.sh into a temp dir with a (possibly broken) mistral7b_40gb_split_order.py
beside it. The rest of the repo comes through PYTHONPATH: a second temp dir linking every top-level
repo module except the split module, so a missing module is really missing.

    .venv/bin/python -m unittest tests/test_clock_study_lists.py -v
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = "mistral7b_40gb_split_order.py"
PAIRS = ["8,4", "16,8", "32,16", "8,2", "16,4", "32,8", "16,2", "32,4", "32,2"]


def dry_run(extra=None, module=True):
    """(exit code, output) of `clock_study.sh --dry-run`; extra = Python appended to the list module."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as lib:
        for name in os.listdir(REPO):
            if name.endswith(".py") and name != MODULE:
                os.symlink(os.path.join(REPO, name), os.path.join(lib, name))
        shutil.copy(os.path.join(REPO, "clock_study.sh"), d)
        if module:
            src = open(os.path.join(REPO, MODULE)).read()
            open(os.path.join(d, MODULE), "w").write(src + ("\n" + extra + "\n" if extra else ""))
        env = dict(os.environ, INV_HOST="mig5", PYTHON=sys.executable, PYTHONPATH=lib,
                   CLOCK_NOTIFY_TOPIC="")
        r = subprocess.run(["bash", os.path.join(d, "clock_study.sh"), "--dry-run"], cwd=d, env=env,
                           capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr


class ClockStudyLists(unittest.TestCase):
    def assertStops(self, extra=None, module=True, why=""):
        code, out = dry_run(extra, module)
        self.assertNotEqual(code, 0, out)
        self.assertIn("STOP: split lists", out)
        self.assertRegex(out, why)
        self.assertNotIn("== block", out)  # stopped before any block

    def test_current_list_passes(self):
        code, out = dry_run()
        self.assertEqual(code, 0, out)
        line = [l for l in out.splitlines() if "split lists ok" in l]
        self.assertEqual(len(line), 1, out)
        self.assertEqual(re.findall(r"(\d+,\d+) (\d+)", line[0]), [(p, "136") for p in PAIRS])
        self.assertEqual(out.count("136 splits (all)"), 9)

    def test_one_split_short(self):
        self.assertStops("ALL = {p: s[:-1] for p, s in ALL.items()}", why="135 splits, want 136")

    def test_one_pair_short(self):
        self.assertStops("ALL[(32, 2)] = ALL[(32, 2)][:-1]", why=r"32,2 all: 135 splits, want 136")

    def test_one_split_too_many(self):
        # [11, 11, 5, 5] breaks r0 > r1 and has 11 on rank 0: in neither part of the list
        self.assertStops("assert [11, 11, 5, 5] not in ALL[(8, 4)]\nALL[(8, 4)] = ALL[(8, 4)] + [[11, 11, 5, 5]]",
                         why="8,4 all: 137 splits, want 136")

    def test_duplicate(self):
        self.assertStops("ALL[(8, 4)] = ALL[(8, 4)][:-1] + [ALL[(8, 4)][0]]", why="duplicate")

    def test_wrong_layer_total_everywhere(self):
        self.assertStops("ALL = {p: [[s[0] + 1] + s[1:] for s in v] for p, v in ALL.items()}",
                         why=r"want \(4, 32\)")

    def test_five_ranks(self):
        self.assertStops("ALL = {p: [s[:3] + [1, s[3] - 1] for s in v if s[3] > 1] for p, v in ALL.items()}",
                         why=r"want \(4, 32\)")

    def test_empty(self):
        self.assertStops("ALL[(16, 8)] = []", why="16,8 all: empty")

    def test_missing_pair(self):
        self.assertStops("del ALL[(16, 2)]", why="16,2 all: KeyError")

    def test_module_missing(self):
        self.assertStops(module=False, why="cannot import")


if __name__ == "__main__":
    unittest.main()
