"""Torch-free regression checks for partial benchmark results."""

import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import partial_results  # noqa: E402


class PartialResults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / "gpu0" / "01_vicuna_7b" / "benchmark.log"
        self.log.parent.mkdir(parents=True)

    def write_log(self, *lines):
        self.log.write_text("\n".join(f"12:00:00.000 INFO {line}" for line in lines) + "\n")

    def start(self, n=1, total=1, split="[20, 9, 2, 1]"):
        return f"[{n}/{total}] Split {split} | Batch: 32 | Microbatch: 16"

    def test_rank_completion_waits_for_parent_outcome(self):
        self.write_log(self.start(), "[Rank 0] Finished. Latency: 1234 ms")
        row = partial_results.parse(self.log)[0]
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["latency_ms"], 1234)

    def test_parent_success_supplies_latency_and_preserves_metrics(self):
        self.write_log(
            self.start(),
            "[B03][Rank 2] phase split: prefill=234.0ms (19.0%) decode=1000.0ms (81.0%)",
            "[B05][Rank 2] decode compute/mb: mean=4.50ms median=4.30ms",
            "[B06][Rank 2] barrier idle: post_prefill=12.3ms final=45.6ms",
            "Total latency:     1234 ms",
            "Peak 5GB memory (rank 2): 4321 MB",
            "Avg  5GB memory (rank 2): 4000 MB",
        )
        row = partial_results.parse(self.log)[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["latency_ms"], 1234)
        self.assertEqual(row["prefill_ms_rank2"], 234.0)
        self.assertEqual(row["decode_ms_rank2"], 1000.0)
        self.assertEqual(row["decode_compute_mean_ms_rank2"], 4.5)
        self.assertEqual(row["barrier_idle_ms_rank2"], 57.9)
        self.assertEqual(row["tight_rank"], 2)
        self.assertEqual(row["tight_peak_mb"], 4321)
        self.assertEqual(row["tight_avg_mb"], 4000)

    def test_parent_failures_with_and_without_rank_completion(self):
        outcomes = {
            "crash": "Crashed. Exit codes: [0, 1, 0, 0]",
            "hang": "Hang detected: ranks still alive after timeout: 2",
            "timeout": "Timeout — no results received.",
            "OOM_rank2": "OOM on Rank 2 — skipping.",
        }
        for status, line in outcomes.items():
            for finished in (False, True):
                with self.subTest(status=status, finished=finished):
                    rank_lines = ["[Rank 0] Finished. Latency: 100 ms"] if finished else []
                    self.write_log(self.start(1, 2), *rank_lines, line, self.start(2, 2))
                    rows = partial_results.parse(self.log)
                    self.assertEqual(rows[0]["status"], status)
                    self.assertEqual(rows[1]["status"], "running")

    def test_child_messages_do_not_override_parent_failure(self):
        self.write_log(
            self.start(),
            "[Rank 2] OOM",
            "Hang detected: ranks still alive after timeout: 0, 1, 3",
            "[Rank 0] Finished. Latency: 100 ms",
            "[Rank 3] OOM",
        )
        self.assertEqual(partial_results.parse(self.log)[0]["status"], "hang")

    def test_child_oom_waits_for_parent_outcome(self):
        self.write_log(self.start(), "[Rank 2] OOM")
        self.assertEqual(partial_results.parse(self.log)[0]["status"], "running")

    def test_cli_excludes_failed_and_unconfirmed_runs_from_fastest(self):
        self.write_log(
            self.start(1, 3),
            "[Rank 0] Finished. Latency: 100 ms",
            "Crashed. Exit codes: [0, 1, 0, 0]",
            self.start(2, 3, "[17, 9, 4, 2]"),
            "[Rank 0] Finished. Latency: 2000 ms",
            "Total latency:     2000 ms",
            self.start(3, 3, "[12, 10, 5, 5]"),
            "[Rank 0] Finished. Latency: 50 ms",
        )
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["partial_results.py", str(self.root)]), contextlib.redirect_stdout(output):
            partial_results.main()
        self.assertIn("2 of 3 configs done", output.getvalue())
        self.assertIn("fastest so far: [17, 9, 4, 2] 2.0 s", output.getvalue())
        with self.log.with_name("partial_results.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["status"] for r in rows], ["crash", "ok", "running"])


if __name__ == "__main__":
    unittest.main()
