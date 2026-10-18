"""Tests for two_node_control.py: two parents on localhost, one per thread.

    python3 -m unittest discover -s tests -p test_two_node_control.py -v
"""

import json
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import two_node_control as tc  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pair(lead_timeout=5, peer_timeout=5):
    """(lead socket, other socket) connected over localhost."""
    port, box = free_port(), {}
    t = threading.Thread(
        target=lambda: box.update(lead=tc.listen_for_peer("127.0.0.1", port, lead_timeout))
    )
    t.start()
    other = tc.connect_to_lead("127.0.0.1", port, peer_timeout)
    t.join()
    return box["lead"], other


def in_thread(fn, *args):
    box = {}

    def run():
        try:
            box["value"] = fn(*args)
        except Exception as e:  # noqa: BLE001 - handed to the test
            box["error"] = e

    t = threading.Thread(target=run)
    t.start()
    return t, box


JOB = {
    "model_key": "vicuna_13b", "model_type": "llama", "num_layers": 40, "hidden_size": 5120,
    "num_heads": 40, "seq_len": 64, "max_new_tokens": 8, "max_runs": None,
    "batch_mb_pairs": [[8, 4]], "splits": None, "enforce_slice_ordering": True,
    "min_last_rank_layers": 1, "rank_nodes": ["A", "A", "B", "B"], "slice_gb": [40, 20, 40, 20],
    "mig_uuids": ["MIG-a", "MIG-b", "MIG-c", "MIG-d"], "node_addrs": {"A": "10.0.0.1", "B": "10.0.0.2"},
    "master_port": 29600, "control_port": 29900, "layer_limits": [22, 11, 22, 11],
}


class Hello(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.repo_a, self.repo_b = Path(d.name) / "a", Path(d.name) / "b"
        for r in (self.repo_a, self.repo_b):
            r.mkdir()
            for f in tc.CODE_FILES:
                (r / f).write_text(f"# {f}\n")

    def hello(self, node, repo, job=JOB):
        return tc.make_hello(node, "run1", repo, job)

    def exchange(self, a, b):
        lead, other = pair()
        self.addCleanup(lead.close)
        self.addCleanup(other.close)
        t, box = in_thread(tc.exchange_hello, other, b)
        try:
            ra = tc.exchange_hello(lead, a)
        except tc.Mismatch as e:
            ra = e
        t.join()
        return ra, box.get("value", box.get("error"))

    def test_matching_nodes_see_each_other(self):
        a, b = self.hello("A", self.repo_a), self.hello("B", self.repo_b)
        ra, rb = self.exchange(a, b)
        self.assertEqual((ra["node"], rb["node"]), ("B", "A"))
        self.assertRegex(tc.hello_line(a, ra), r"^\[B09\] peer B matched: code [0-9a-f]{8} ")

    def test_stale_code_on_one_node_aborts_both(self):
        (self.repo_b / "mig_transport_pipeline.py").write_text("# stale\n")
        ra, rb = self.exchange(self.hello("A", self.repo_a), self.hello("B", self.repo_b))
        for r in (ra, rb):
            self.assertIsInstance(r, tc.Mismatch)
            self.assertIn("code", str(r))

    def test_different_job_aborts_both(self):
        other = dict(JOB, splits=[[22, 11, 4, 3]])
        ra, rb = self.exchange(self.hello("A", self.repo_a), self.hello("B", self.repo_b, other))
        self.assertIsInstance(ra, tc.Mismatch)
        self.assertIsInstance(rb, tc.Mismatch)
        self.assertIn("job", str(ra))

    def test_node_specific_fields_do_not_count(self):
        # The fingerprint ignores what legitimately differs per node.
        other = dict(JOB, node="B", gloo_ifname="ens5", shm_prefix="x", model_path="/elsewhere")
        self.assertEqual(tc.job_fingerprint(JOB), tc.job_fingerprint(other))

    def test_version_difference_is_reported(self):
        a, b = self.hello("A", self.repo_a), self.hello("B", self.repo_b)
        b["torch"] = "9.9.9"
        ra, rb = self.exchange(a, b)
        self.assertIn("torch", str(ra))
        self.assertIn("9.9.9", str(rb))

    def test_code_hash_ignores_files_outside_the_list(self):
        h = tc.code_hash(self.repo_a)
        (self.repo_a / "parallel_config.py").write_text("changed")
        self.assertEqual(tc.code_hash(self.repo_a), h)
        (self.repo_a / "helpers.py").write_text("changed")
        self.assertNotEqual(tc.code_hash(self.repo_a), h)


class Protocol(unittest.TestCase):
    def setUp(self):
        self.lead, self.other = pair()
        self.addCleanup(self.lead.close)
        self.addCleanup(self.other.close)

    def test_lockstep_and_result_shipping(self):
        payload = {"phase_rank2": {"prefill_ms": 1.5}, "step_latencies_rank2": [0.1, 0.2], "oom": 3}

        def other_side():
            for i in range(3):
                tc.sync_ready(self.other, i, f"s_b8_mb4_{i}", 5)
                tc.send_result(self.other, dict(payload, i=i))
            tc.send_bye(self.other)

        t, box = in_thread(other_side)
        for i in range(3):
            tc.sync_ready(self.lead, i, f"s_b8_mb4_{i}", 5)
            got = tc.recv_result(self.lead, 5)
            self.assertEqual(got["i"], i)
            self.assertEqual(got["step_latencies_rank2"], [0.1, 0.2])
        tc.recv_bye(self.lead, 5)
        t.join()
        self.assertNotIn("error", box)

    def test_out_of_step_is_refused(self):
        t, box = in_thread(tc.sync_ready, self.other, 4, "s_other", 5)
        with self.assertRaises(tc.Mismatch):
            tc.sync_ready(self.lead, 3, "s_lead", 5)
        t.join()
        self.assertIsInstance(box.get("error"), tc.Mismatch)

    def test_closed_peer_is_noticed_immediately(self):
        self.other.close()
        t0 = time.monotonic()
        with self.assertRaisesRegex(tc.PeerLost, "closed"):
            tc.recv_result(self.lead, 30)
        self.assertLess(time.monotonic() - t0, 2)

    def test_silent_peer_times_out(self):
        with self.assertRaisesRegex(tc.PeerLost, "timed out"):
            tc.recv_result(self.lead, 0.3)

    def test_unexpected_message_type_is_refused(self):
        tc.send_msg(self.other, {"type": "bye"})
        with self.assertRaises(tc.Mismatch):
            tc.recv_result(self.lead, 5)

    def test_big_message_round_trips(self):
        rows = [{"rank": r, "step": s, "ms": s * 0.001} for r in range(4) for s in range(512)]
        tc.send_result(self.other, rows)
        self.assertEqual(tc.recv_result(self.lead, 5), rows)

    def test_wire_is_json_not_pickle(self):
        tc.send_msg(self.other, {"type": "x"})
        raw = self.lead.recv(4)
        n = int.from_bytes(raw, "big")
        self.assertEqual(json.loads(self.lead.recv(n)), {"type": "x"})

    def test_finish_is_ok_only_when_both_are(self):
        for a_ok, b_ok in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(a=a_ok, b=b_ok):
                t, box = in_thread(tc.finish, self.other, b_ok, 5)
                self.assertEqual(tc.finish(self.lead, a_ok, 5), a_ok and b_ok)
                t.join()
                self.assertEqual(box["value"], a_ok and b_ok)

    def test_resume_skips_only_when_both_finished(self):
        for a_ok, b_ok, skip in ((True, True, True), (True, False, False),
                                 (False, True, False), (False, False, False)):
            with self.subTest(a=a_ok, b=b_ok):
                t, box = in_thread(tc.agree_on_resume, self.other, b_ok, 5)
                self.assertEqual(tc.agree_on_resume(self.lead, a_ok, 5), skip)
                t.join()
                self.assertEqual(box["value"], skip)

    # Every failure is PeerLost or Mismatch, so a caller catching both catches all.
    def test_send_to_a_peer_that_stopped_reading_is_peer_lost(self):
        # sendall keeps the last recv_msg's timeout; a stuck peer must not
        # surface as a bare socket.timeout.
        self.lead.settimeout(0.3)
        with self.assertRaisesRegex(tc.PeerLost, "send to the other node failed"):
            tc.send_msg(self.lead, {"type": "x", "blob": "a" * (64 << 20)})

    def test_send_to_a_closed_peer_is_peer_lost(self):
        self.other.close()
        with self.assertRaises(tc.PeerLost):
            for _ in range(50):  # the first sends may still land in the kernel buffer
                tc.send_msg(self.lead, {"type": "x", "blob": "a" * 65536})
                time.sleep(0.01)
        tc.send_bye(self.lead)  # swallows it: bye is best effort

    def test_unreadable_frames_are_peer_lost(self):
        for raw in (b"\xff\xfe", b"{", b"[1, 2]", b"3", b'"text"', b"null", b"[" * 100_000):
            with self.subTest(raw=raw[:10]):
                self.other.sendall(struct.pack(">I", len(raw)) + raw)
                with self.assertRaisesRegex(tc.PeerLost, "unreadable"):
                    tc.recv_msg(self.lead, 5)

    def test_messages_missing_fields_are_mismatches(self):
        tc.send_msg(self.other, {"type": "ready"})
        with self.assertRaises(tc.Mismatch):
            tc.sync_ready(self.lead, 0, "s", 5)
        tc.send_msg(self.other, {"type": "result"})
        with self.assertRaisesRegex(tc.Mismatch, "without a payload"):
            tc.recv_result(self.lead, 5)
        tc.send_msg(self.other, {"type": "hello"})
        with self.assertRaisesRegex(tc.Mismatch, "code"):
            tc.exchange_hello(self.lead, tc.make_hello("A", "run1", ROOT, JOB))


class Connect(unittest.TestCase):
    def test_other_node_may_start_first(self):
        port, box = free_port(), {}
        t = threading.Thread(
            target=lambda: box.update(other=tc.connect_to_lead("127.0.0.1", port, 10))
        )
        t.start()
        time.sleep(1.2)  # the lead is late
        lead = tc.listen_for_peer("127.0.0.1", port, 10)
        t.join()
        lead.close()
        box["other"].close()

    def test_lead_gives_up_when_nobody_comes(self):
        with self.assertRaisesRegex(tc.PeerLost, "no peer connected"):
            tc.listen_for_peer("127.0.0.1", free_port(), 0.3)

    def test_other_gives_up_when_the_lead_never_listens(self):
        with self.assertRaisesRegex(tc.PeerLost, "could not reach"):
            tc.connect_to_lead("127.0.0.1", free_port(), 0.6)


class Merge(unittest.TestCase):
    def test_union_with_the_leads_value_winning(self):
        lead = {"latency": 5.0, "phase_rank0": 1, "oom": 0}
        other = {"phase_rank2": 2, "oom": 2}
        self.assertEqual(
            tc.merge_queue_items(lead, other),
            {"latency": 5.0, "phase_rank0": 1, "phase_rank2": 2, "oom": 0},
        )


if __name__ == "__main__":
    unittest.main()
