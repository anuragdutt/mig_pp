"""The probe's torch-free helpers (the cases themselves need CUDA and run on the box).

    python3 -m unittest discover -s tests -p test_probe_two_node.py -v
"""

import queue
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe_two_node as probe  # noqa: E402
import two_node_control as tc  # noqa: E402


class FakeProc:
    """What collect_local uses of a multiprocessing.Process."""

    def __init__(self, exitcode=0, alive=False, ignores_sigterm=False):
        self.exitcode, self.alive, self.ignores_sigterm = exitcode, alive, ignores_sigterm
        self.killed = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        if not self.ignores_sigterm:
            self.alive, self.exitcode = False, -15

    def kill(self):
        self.alive, self.exitcode, self.killed = False, -9, True


PASSING = [("N2", True, "ok"), ("N3", True, "ok")]


def items(rank, rows=PASSING, done=True):
    """What rank_main puts: one item per case as it ends, then the finished marker."""
    out = [(rank, "B", [row], False) for row in rows]
    return out + ([(rank, "B", [], True)] if done else [])


class CollectLocal(unittest.TestCase):
    def collect(self, procs, reports, deadline_s=5.0):
        q = queue.Queue()
        for item in reports:
            q.put(item)
        return probe.collect_local(procs, [2, 3], q, "B", deadline_s, poll_s=0.05)

    def test_finished_then_nonzero_exit_fails_the_verdict(self):
        # The review's repro: rank 3 reports success, then exits with code 7.
        entries = self.collect([FakeProc(0), FakeProc(7)], items(2) + items(3))
        lines, fails = probe.render(entries, "title")
        self.assertEqual(fails, 1)
        self.assertEqual(lines[-1], "VERDICT: FAIL (1)")
        self.assertIn("r3[B]: finished, then exited with code 7", "\n".join(lines))

    def test_clean_exits_pass(self):
        entries = self.collect([FakeProc(0), FakeProc(0)], items(2) + items(3))
        lines, fails = probe.render(entries, "title")
        self.assertEqual((fails, lines[-1]), (0, "VERDICT: PASS"))
        self.assertIn("PASS  exit", lines)

    def test_silent_death_is_noticed_without_waiting_out_the_deadline(self):
        t0 = time.monotonic()
        entries = self.collect([FakeProc(0), FakeProc(-11)], items(2), deadline_s=60)
        self.assertLess(time.monotonic() - t0, 2)
        self.assertIn((3, "B", "exit", False, "exited with code -11 before finishing"), entries)

    def test_hung_rank_is_terminated_and_fails(self):
        procs = [FakeProc(0), FakeProc(None, alive=True)]
        entries = self.collect(procs, items(2), deadline_s=0.2)
        self.assertFalse(procs[1].alive)
        self.assertIn((3, "B", "exit", False, "hung; terminated before finishing"), entries)

    def test_rank_hung_at_exit_after_finishing_fails(self):
        procs = [FakeProc(0), FakeProc(None, alive=True)]
        entries = self.collect(procs, items(2) + items(3))
        self.assertIn((3, "B", "exit", False, "finished, then hung at exit; terminated"), entries)

    def test_rows_sent_before_a_hang_survive(self):
        # Rank 3 fails N3 and then hangs in teardown: its N3 row, the one that
        # explains the hang, must still reach the table.
        procs = [FakeProc(0), FakeProc(None, alive=True)]
        rows = [("N2", True, "ok"), ("N3", False, "RuntimeError: payload differs")]
        entries = self.collect(procs, items(2) + items(3, rows, done=False), deadline_s=0.2)
        self.assertIn((3, "B", "N3", False, "RuntimeError: payload differs"), entries)
        self.assertIn((3, "B", "exit", False, "hung; terminated before finishing"), entries)
        lines = "\n".join(probe.render(entries, "t")[0])
        self.assertIn("FAIL  N3\n      r2[B]: ok\n      r3[B]: RuntimeError: payload differs", lines)

    def test_rank_that_ignores_sigterm_is_killed(self):
        # A live child would otherwise block the parent at exit.
        procs = [FakeProc(0), FakeProc(None, alive=True, ignores_sigterm=True)]
        self.collect(procs, items(2), deadline_s=0.2)
        self.assertTrue(procs[1].killed)
        self.assertFalse(procs[1].alive)


def child(rank, q, mode):
    """A rank stand-in: report and exit 0, report and exit 7, or die silently."""
    if mode != "silent":
        for item in items(rank):
            q.put(item)
    sys.exit({"ok": 0, "report_then_7": 7, "silent": 3}[mode])


class RealProcesses(unittest.TestCase):
    def test_review_repro_with_real_children(self):
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        modes = {2: "ok", 3: "report_then_7", 4: "silent"}
        procs = [ctx.Process(target=child, args=(r, q, m)) for r, m in modes.items()]
        for proc in procs:
            proc.start()
        try:
            entries = probe.collect_local(procs, list(modes), q, "B", 60, poll_s=0.2)
        finally:
            for proc in procs:
                proc.join(5)
        exits = {e[0]: (e[3], e[4]) for e in entries if e[2] == "exit"}
        self.assertEqual(exits[2], (True, "exit code 0"))
        self.assertEqual(exits[3], (False, "finished, then exited with code 7"))
        self.assertEqual(exits[4], (False, "exited with code 3 before finishing"))
        self.assertEqual(probe.render(entries, "t")[1], 1)


def socket_pair():
    """(lead socket, other socket) over localhost, through the real control module."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    box = {}
    t = threading.Thread(target=lambda: box.update(lead=tc.listen_for_peer("127.0.0.1", port, 5)))
    t.start()
    other = tc.connect_to_lead("127.0.0.1", port, 5)
    t.join()
    return box["lead"], other


class TradeRows(unittest.TestCase):
    def test_both_get_the_others_rows_however_long_they_are(self):
        # Both sending before reading deadlocks once rows exceed the socket
        # buffers; the lead sends first and the other node replies.
        lead, other = socket_pair()
        self.addCleanup(lead.close)
        self.addCleanup(other.close)
        big = "x" * (4 << 20)  # far past any socket buffer
        lead_rows = [(0, "A", "N3", True, big), (-1, "A", "N1", True, "ok")]
        other_rows = [(2, "B", "N3", False, big)]
        box = {}

        def other_side():
            box["got"] = probe.trade_rows(other, other_rows, False, 20)

        t = threading.Thread(target=other_side)
        t.start()
        t0 = time.monotonic()
        got = probe.trade_rows(lead, lead_rows, True, 20)
        t.join(20)
        self.assertLess(time.monotonic() - t0, 15)
        self.assertEqual(got, other_rows)
        self.assertEqual(box["got"], lead_rows)


class Nic(unittest.TestCase):
    IP_OUT = ("2: eno1    inet 10.0.0.1/24 brd 10.0.0.255 scope global eno1\\       valid_lft forever\n"
              "2: eno1    inet 10.0.0.9/24 scope global secondary eno1\\       valid_lft forever\n")

    def test_ifname_addrs_parses_ip(self):
        out = types.SimpleNamespace(stdout=self.IP_OUT)
        with mock.patch.object(probe.subprocess, "run", return_value=out):
            self.assertEqual(probe.ifname_addrs("eno1"), ["10.0.0.1", "10.0.0.9"])
        with mock.patch.object(probe.subprocess, "run", return_value=types.SimpleNamespace(stdout="")):
            self.assertEqual(probe.ifname_addrs("nope0"), [])  # unknown NIC: `ip` prints to stderr
        with mock.patch.object(probe.subprocess, "run", side_effect=FileNotFoundError("ip")):
            self.assertIsNone(probe.ifname_addrs("eno1"))  # macOS

    def test_nic_row(self):
        self.assertEqual(probe.nic_row("A", "eno1", "10.0.0.1", ["10.0.0.1"])[3], True)
        row = probe.nic_row("A", "eno2", "10.0.0.1", ["192.168.1.5"])
        self.assertFalse(row[3])
        self.assertIn("eno2 carries 192.168.1.5, not NODES['A']['addr'] 10.0.0.1", row[4])
        self.assertIn("no IPv4 address", probe.nic_row("A", "eno2", "10.0.0.1", [])[4])
        self.assertIsNone(probe.nic_row("A", "lo0", "127.0.0.1", None)[3])
        self.assertEqual(probe.combine([row])["nic"][0], "FAIL")


class Identity(unittest.TestCase):
    def test_n1_compares_the_probe_file_and_so_its_config(self):
        # Two different probes (or two different CONFIG dicts, which live in the
        # probe file) must fail N1 instead of hanging later in N5/N6.
        self.assertIn("probe_two_node.py", probe.PROBE_CODE_FILES)
        job = {"rank_nodes": ["A", "B"], "slice_gb": [40, 40]}
        with tempfile.TemporaryDirectory() as d:
            for f in probe.PROBE_CODE_FILES:
                Path(d, f).write_text(f"# {f}\n")
            a = probe.probe_hello("A", job, repo=d)
            Path(d, "probe_two_node.py").write_text("CONFIG = dict(steps=30)\n")
            b = probe.probe_hello("B", job, repo=d)
        self.assertNotEqual(a["code"], b["code"])
        self.assertEqual(a["job"], b["job"])

    def test_default_hello_still_hashes_only_the_benchmark_files(self):
        with tempfile.TemporaryDirectory() as d:
            for f in tc.CODE_FILES:
                Path(d, f).write_text(f)
            before = tc.make_hello("A", "r", d, {})["code"]
            Path(d, "probe_two_node.py").write_text("anything")
            self.assertEqual(tc.make_hello("A", "r", d, {})["code"], before)


class Helpers(unittest.TestCase):
    def test_first_cross_pair(self):
        self.assertEqual(probe.first_cross_pair(["A", "A", "B", "B"]), [1, 2])
        self.assertEqual(probe.first_cross_pair(["A", "B", "A", "B"]), [0, 1])
        self.assertIsNone(probe.first_cross_pair(["A", "A"]))

    def test_host_problems(self):
        nodes = ["A", "A", "B", "B"]
        self.assertEqual(probe.host_problems(["mig1", "mig1", "mig2", "mig2"], nodes), [])
        # Both "nodes" on one machine: the network hops were loopback.
        bad = probe.host_problems(["mig1"] * 4, nodes)
        self.assertEqual(len(bad), 1)
        self.assertIn("all run on host mig1", bad[0])
        # A node's ranks on two hosts: the placement is not what the config says.
        self.assertIn("several hosts", probe.host_problems(["mig1", "mig2", "mig2", "mig2"], nodes)[0])

    def test_token_source(self):
        self.assertEqual(probe.token_source(["A", "A", "B", "B"]), 3)  # last rank, as in the benchmark
        self.assertEqual(probe.token_source(["A", "B", "B", "A"]), 1)  # last is local to rank 0
        self.assertIsNone(probe.token_source(["A", "A"]))

    def test_combine(self):
        entries = [
            (0, "A", "N3", True, "ok"), (2, "B", "N3", True, "ok"),
            (0, "A", "N5", True, "ok"), (2, "B", "N5", False, "boom"), (2, "B", "N5", None, "x"),
            (0, "A", "N6", None, "skipped"), (2, "B", "N6", None, "skipped"),
        ]
        out = probe.combine(entries)
        self.assertEqual([out[n][0] for n in ("N3", "N5", "N6")], ["PASS", "FAIL", "SKIP"])
        self.assertNotIn("N1", out)
        self.assertEqual(out["N3"][1], ["r0[A]: ok", "r2[B]: ok"])

    def test_payload_is_the_same_everywhere(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("no torch")
        a, b = probe.make_payload(7, (2, 3)), probe.make_payload(7, (2, 3))
        self.assertTrue((a == b).all())
        self.assertFalse((a == probe.make_payload(8, (2, 3))).all())
        self.assertEqual(str(a.dtype), "torch.float16")


if __name__ == "__main__":
    unittest.main()
