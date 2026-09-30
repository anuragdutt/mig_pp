"""Tests for nvml_mem_monitor.py against a fake pynvml: no GPU, torch or
nvidia-ml-py needed.

    python3 -m unittest discover -s tests -p 'test_nvml_mem_monitor.py' -v
"""

import csv
import ctypes
import importlib.util
import io
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / "nvml_mem_monitor.py"
MIB = 2**20
UUIDS = [f"MIG-{c * 8}-0000-0000-0000-000000000000" for c in "cadb"]  # rank order
# Distinct per slice, so a value names its slice, and not in rank order, so a
# positional or sorted mapping would show: rank columns must be [300, 100, 400, 200].
USED = dict(zip(sorted(UUIDS), [100, 200, 300, 400]))
SLICE_GB = [20, 10, 5, 5]
PARENT_MIB = 30000  # deliberately not the slices' sum
# The header dcgm_mem_monitor writes for this topology.
HEADER = [
    "timestamp",
    "label",
    "gpu_mb",
    "rank0_20gb_mb",
    "rank1_10gb_mb",
    "rank2_5gb_mb",
    "rank3_5gb_mb",
]


class NVMLError(Exception):
    pass


class FakePynvml:
    """The five pynvml calls the monitor makes, over one GPU holding USED's slices."""

    def __init__(self):
        self.used = dict(USED)  # MIG UUID -> used MiB
        self.calls = []  # nvmlInit and every UUID lookup, in order
        self.init_error = self.str_error = self.parent_error = self.read_error = None
        self.failed_reads = 0

    def nvmlInit(self):
        self.calls.append("nvmlInit")
        if self.init_error:
            raise self.init_error

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self.calls.append(uuid)
        if isinstance(uuid, str) and self.str_error:  # an old, bytes-only pynvml
            raise self.str_error
        uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        if uuid not in self.used:
            raise NVMLError("Not Found")
        return uuid

    def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(self, handle):
        return "GPU-0"

    def nvmlDeviceGetMemoryInfo(self, handle):
        if handle == "GPU-0":
            if self.parent_error:
                raise self.parent_error
            return types.SimpleNamespace(used=PARENT_MIB * MIB)
        if self.read_error:
            self.failed_reads += 1
            raise self.read_error
        # Not a whole MiB: the monitor must floor it.
        return types.SimpleNamespace(used=self.used[handle] * MIB + 12345)


def wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the sampler")
        time.sleep(0.002)


class NvmlMemMonitorTest(unittest.TestCase):
    def setUp(self):
        out = mock.patch("sys.stdout", new_callable=io.StringIO)
        self.out = out.start()
        self.addCleanup(out.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.csv_path = os.path.join(tmp.name, "mig_memory_trace.csv")
        self.nvml = FakePynvml()
        self.mon = self.fresh(self.nvml)

    def fresh(self, pynvml):
        """A new copy of the module (its state is globals); `import pynvml` gets pynvml."""
        patch = mock.patch.dict(sys.modules, {"pynvml": pynvml})
        patch.start()
        self.addCleanup(patch.stop)
        spec = importlib.util.spec_from_file_location("nvml_mem_monitor", MODULE)
        mon = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mon)
        mon.SAMPLE_INTERVAL_MS = 5  # fast ticks
        self.addCleanup(mon.stop)  # runs before the patches are undone
        return mon

    def setup_mon(self, mon=None, uuids=UUIDS, slice_gb=SLICE_GB):
        (mon or self.mon).setup_dcgm_group(mig_uuids=list(uuids), slice_gb=slice_gb)

    def sample(self, n=2):
        """One benchmark config: start, wait for n rows, stop. Returns the rows."""
        self.mon.start()
        wait_for(lambda: len(self.mon._samples) >= n)
        self.mon.stop()
        return list(self.mon._samples)

    def csv_rows(self, mon):
        mon.save_csv(self.csv_path)
        with open(self.csv_path, newline="") as f:
            return list(csv.reader(f))

    def test_rank_r_column_is_mig_uuids_r(self):
        self.setup_mon()
        self.assertTrue(self.mon._enabled)
        self.assertEqual(self.mon.NUM_MIG_INSTANCES, 4)
        self.assertIs(self.mon.setup_group, self.mon.setup_dcgm_group)
        for ts, _label, _gpu_mb, *ranks in self.sample():
            self.assertRegex(ts, r"^\d\d:\d\d:\d\d\.\d{3}$")
            self.assertEqual(ranks, [USED[u] for u in UUIDS])

    def test_gpu_mb_is_the_parent_gpu(self):
        self.setup_mon()
        self.assertEqual({row[2] for row in self.sample()}, {PARENT_MIB})

    def test_gpu_mb_falls_back_to_the_slice_sum(self):
        # NVML refuses a MIG-mode GPU's aggregate memory to unprivileged callers.
        self.nvml.parent_error = NVMLError("Insufficient Permissions")
        self.setup_mon()
        self.assertTrue(self.mon._enabled)
        self.assertEqual({row[2] for row in self.sample(n=3)}, {sum(USED.values())})
        self.assertEqual(self.out.getvalue().count("Insufficient Permissions"), 1)

    def test_str_uuid_is_retried_as_bytes(self):
        for err in (TypeError("bytes expected"), ctypes.ArgumentError("wrong type")):
            with self.subTest(type(err).__name__):
                nvml = FakePynvml()
                nvml.str_error = err
                mon = self.fresh(nvml)
                self.setup_mon(mon)
                self.assertTrue(mon._enabled)
                retried = [x for u in UUIDS for x in (u, u.encode())]
                self.assertEqual(nvml.calls, ["nvmlInit"] + retried)

    def test_setup_failure_disables_without_raising(self):
        bad_init = FakePynvml()
        bad_init.init_error = NVMLError("Driver Not Loaded")
        cases = [
            ("no pynvml", None, UUIDS, "pynvml"),  # None in sys.modules: import fails
            ("nvmlInit", bad_init, UUIDS, "Driver Not Loaded"),
            ("unknown UUID", FakePynvml(), UUIDS[:3] + ["MIG-unknown"], "MIG-unknown"),
        ]
        for name, pynvml, uuids, reason in cases:
            with self.subTest(name):
                seen = len(self.out.getvalue())
                mon = self.fresh(pynvml)  # importing the module needs no pynvml
                self.setup_mon(mon, uuids)
                printed = self.out.getvalue()[seen:]
                self.assertTrue(printed.startswith("[MemMonitor] DISABLED — "), printed)
                self.assertIn(reason, printed)
                self.assertFalse(mon._enabled)
                self.assertEqual(mon.NUM_MIG_INSTANCES, 0)
                mon.start()
                time.sleep(0.03)
                mon.stop()
                self.assertEqual(self.csv_rows(mon), [HEADER])  # header, no rows

    def test_stop_joins_the_sampler_and_is_idempotent(self):
        self.setup_mon()
        self.mon.start()
        thread = self.mon._thread
        wait_for(lambda: len(self.mon._samples) >= 2)
        self.mon.stop()
        self.assertFalse(thread.is_alive())  # joined: the rows are final on return
        rows = list(self.mon._samples)
        time.sleep(0.03)  # six more ticks' worth
        self.mon.stop()  # the benchmark's finally: calls it a second time
        self.assertEqual(self.mon._samples, rows)

    def test_start_after_stop_samples_again(self):
        self.setup_mon()
        self.sample()
        self.mon.clear()
        self.nvml.used[UUIDS[0]] = 5000
        self.assertEqual({row[3] for row in self.sample()}, {5000})

    def test_rows_carry_the_label_and_clear_empties_them(self):
        self.setup_mon()
        self.mon.set_label("s13_10_9_b24_mb24")
        self.assertEqual({row[1] for row in self.sample()}, {"s13_10_9_b24_mb24"})
        self.mon.clear()
        self.assertEqual(self.mon._samples, [])
        self.mon.set_label("s12_11_9_b24_mb8")
        self.assertEqual({row[1] for row in self.sample()}, {"s12_11_9_b24_mb8"})

    def test_save_csv_writes_header_and_rows(self):
        self.setup_mon()
        rows = self.sample()
        expected = [HEADER] + [[str(v) for v in row] for row in rows]
        self.assertEqual(self.csv_rows(self.mon), expected)
        self.setup_mon(slice_gb=None)  # without sizes the rank columns are numbered
        self.assertEqual(
            self.csv_rows(self.mon)[0][3:], ["rank0_mb", "rank1_mb", "rank2_mb", "rank3_mb"]
        )

    def test_disable_makes_every_later_call_a_no_op(self):
        # The benchmark's order for mem_monitor="off": disable() at import, then
        # the usual setup, per-config calls and save_csv.
        self.mon.disable()
        self.setup_mon()
        self.mon.set_label("s13_10_9_b24_mb24")
        self.mon.clear()
        self.mon.start()
        time.sleep(0.03)
        self.mon.stop()
        self.mon.stop()
        self.assertEqual(self.nvml.calls, [])  # NVML never touched
        self.assertFalse(self.mon._enabled)
        self.assertEqual(self.csv_rows(self.mon), [HEADER])
        # Called mid-sampling, it stops the sampler and save_csv is still header-only.
        mon = self.fresh(FakePynvml())
        self.setup_mon(mon)
        mon.start()
        wait_for(lambda: mon._samples)
        mon.disable()
        self.assertEqual(self.csv_rows(mon), [HEADER])

    def test_failed_read_skips_the_tick_and_prints_once(self):
        self.setup_mon()
        self.nvml.read_error = NVMLError("GPU is lost")
        self.mon.start()
        wait_for(lambda: self.nvml.failed_reads >= 3)
        self.assertEqual(self.mon._samples, [])  # no zero rows for failed ticks
        self.nvml.read_error = None
        wait_for(lambda: self.mon._samples)
        self.mon.stop()
        self.assertEqual(self.out.getvalue().count("GPU is lost"), 1)
        expected = tuple(USED[u] for u in UUIDS)
        self.assertEqual({row[3:] for row in self.mon._samples}, {expected})


if __name__ == "__main__":
    unittest.main()
