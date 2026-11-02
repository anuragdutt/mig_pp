"""make_done.py: results CSVs -> done_configs.py. Stdlib only.

    python3 -m unittest discover -s tests -p 'test_make_done.py' -v
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import make_done  # noqa: E402


def job(root, run, rel, key, rows, smoke=False):
    d = Path(root, run, rel)
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"model_key": key, "slice_gb": [40, 20, 10, 10], "smoke": smoke}))
    with open(d / "mig_benchmark_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "batch_size", "microbatch_size", "status"])
        w.writerows(rows)


class MakeDone(unittest.TestCase):
    def test_collects_measured_rows_by_layout_model_split(self):
        with tempfile.TemporaryDirectory() as root:
            job(root, "r1", "gpu0/01_vicuna_13b", "vicuna_13b",
                [["[22, 11, 6, 1]", 64, 32, "ok"], ["[22, 11, 6, 1]", 64, 16, "OOM_rank2"],
                 ["[21, 11, 6, 2]", 64, 32, "hang"]])
            job(root, "r2", "gpu0/01_vicuna_13b", "vicuna_13b", [["[22, 11, 6, 1]", 8, 4, "ok"]])
            job(root, "r3", "gpu0/01_vicuna_13b", "vicuna_13b", [["[20, 12, 6, 2]", 8, 4, "ok"]], smoke=True)
            done = make_done.collect([root])
            self.assertEqual(done, {"40_20_10_10": {"vicuna_13b": {(22, 11, 6, 1): {(64, 32), (64, 16), (8, 4)}}}})
            ns = {}
            exec(make_done.render(done, [root]), ns)  # the written file is plain Python
            self.assertEqual(ns["DONE"]["40_20_10_10"]["vicuna_13b"][(22, 11, 6, 1)], [(8, 4), (64, 16), (64, 32)])


if __name__ == "__main__":
    unittest.main()
