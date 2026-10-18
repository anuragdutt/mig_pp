"""CPU tests for the cross-node hop: engine routing + the net send/recv handles.

Needs torch and numpy (the engine imports numpy); skips without them. On the
Mac: PYTHONPATH=<scratch>/pylib .venv/bin/python -m unittest discover -s tests -p test_net_transport.py -v

The real MIGPipelineTransport methods run (isend / irecv / _irecv_net / slot
pools / peer attach); only the CUDA parts (side-stream copies, events) are
replaced by synchronous CPU copies. The two-process test moves real bytes over
real gloo, with slots reused every round.
"""

from contextlib import contextmanager
from datetime import timedelta
import importlib
import multiprocessing
from pathlib import Path
import queue
import sys
import tempfile
import traceback
import types
import unittest
from unittest import mock

try:
    import torch
    import torch.distributed as dist
except ImportError:  # a torch-free interpreter: every test below skips
    torch = dist = None

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

HAVE_DEPS = torch is not None and numpy is not None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHAPE = (2, 3, 4)
N = 24


def load_transport():
    """(mig_transport_pipeline, async_handle_net) with `helpers` (datasets,
    transformers) stubbed out; sys.modules is restored afterwards."""
    helpers = types.ModuleType("helpers")
    helpers._compute_slot_size_mb = lambda **kw: 1
    names = ("mig_transport_pipeline", "async_handle_send", "async_handle_recv", "async_handle_net")
    with mock.patch.dict(sys.modules, {"helpers": helpers}):
        for n in names:
            sys.modules.pop(n, None)
        mod = importlib.import_module("mig_transport_pipeline")
        net = importlib.import_module("async_handle_net")
    return mod, net


class Evt:
    def __init__(self, on_sync=None):
        self.on_sync = on_sync

    def synchronize(self):
        if self.on_sync:
            self.on_sync()

    def query(self):
        return True


def make_engine(mod, rank, rank_nodes, num_slots=3, events=None):
    """A real MIGPipelineTransport without its CUDA/SHM __init__; the side-stream
    copies are replaced by synchronous CPU ones."""
    e = mod.MIGPipelineTransport.__new__(mod.MIGPipelineTransport)
    e.rank, e.world_size, e.rank_nodes = rank, len(rank_nodes), list(rank_nodes)
    e.num_slots, e.slot_size = num_slots, N * 2
    e.slot_free = [True] * num_slots
    e.pinned_staging = [torch.zeros(N, dtype=torch.float16) for _ in range(num_slots)]
    e.recv_pinned_staging = [torch.zeros(N, dtype=torch.float16) for _ in range(num_slots)]
    e.slot_events = [Evt() for _ in range(num_slots)]
    e._recv_slot_pending = [None] * num_slots
    e._recv_slot_reserved = [False] * num_slots
    e.shm_prefix = "test"
    for stat in (
        "flush_block_s", "flush_count", "ack_wait_s", "ack_count", "handshake_wait_s",
        "handshake_count", "h2d_block_s", "h2d_count", "slot_spins", "net_send_wait_s",
        "net_send_count", "net_recv_wait_s", "net_recv_count",
    ):
        setattr(e, f"stat_{stat}", 0)
    log = events if events is not None else []

    def copy_to_staging(tensor, slot):
        log.append("d2h")
        e.pinned_staging[slot][: tensor.numel()].copy_(tensor.reshape(-1))
        return tensor.numel()

    def h2d(tensor, rslot):
        def land():
            log.append("h2d_landed")
            tensor.copy_(e.recv_pinned_staging[rslot][: tensor.numel()].view(tensor.shape))

        return Evt(land)

    e._queue_copy_to_staging = copy_to_staging
    e._queue_h2d_from_staging = h2d
    return e


def payload(round_index, mb):
    return (torch.arange(N) + 100 * round_index + 10 * mb).half().reshape(SHAPE)


def net_worker(rank, rendezvous, results):
    try:
        dist.init_process_group(
            "gloo", init_method=rendezvous, rank=rank, world_size=2,
            timeout=timedelta(seconds=20),
        )
        mod, net = load_transport()
        engine = make_engine(mod, rank, ["A", "B"], num_slots=3)
        rounds, mbs = 4, 3  # 3 slots, 3 microbatches in flight: every round reuses all
        for r in range(rounds):
            tags = [1100 + r * 10 + mb for mb in range(mbs)]
            if rank == 0:
                handles = []
                for mb in range(mbs):
                    h = engine.isend(payload(r, mb), dst=1, tag=tags[mb])
                    assert isinstance(h, net.AsyncHandleSendNet), type(h)
                    handles.append(h)
                    if len(handles) > 1:
                        handles[-2].flush()  # one microbatch behind, as the benchmark does
                handles[-1].flush()
                assert not any(engine.slot_free), "slot freed before the send completed"
                for h in handles:
                    h.wait()
                    h.wait()
                assert all(engine.slot_free)
            else:
                bufs = [torch.empty(SHAPE, dtype=torch.float16) for _ in range(mbs)]
                handles = [engine.irecv(bufs[mb], src=0, tag=tags[mb]) for mb in range(mbs)]
                assert all(isinstance(h, net.AsyncHandleRecvNet) for h in handles)
                assert all(engine._recv_slot_reserved), "slots not reserved at post time"
                for mb in reversed(range(mbs)):  # delivery order must not matter
                    handles[mb].wait()
                    handles[mb].wait()
                    assert torch.equal(bufs[mb], payload(r, mb)), f"round {r} mb {mb} differs"
                assert not any(engine._recv_slot_reserved)
            dist.barrier()
        if rank == 0:
            assert (engine.stat_net_send_count, engine.stat_flush_count) == (12, 12)
        else:
            assert (engine.stat_net_recv_count, engine.stat_h2d_count) == (12, 12)
        dist.destroy_process_group()
        results.put((rank, None))
    except Exception:
        results.put((rank, traceback.format_exc()))


@unittest.skipUnless(HAVE_DEPS, "torch and numpy needed (the engine imports both)")
@unittest.skipUnless(dist is not None and dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
class GlooHopTests(unittest.TestCase):
    def test_bit_exact_hop_with_slot_reuse(self):
        ctx = multiprocessing.get_context("spawn")
        results = ctx.Queue()
        procs = []
        with tempfile.TemporaryDirectory() as d:
            rendezvous = (Path(d) / "rendezvous").as_uri()
            try:
                for rank in range(2):
                    p = ctx.Process(target=net_worker, args=(rank, rendezvous, results))
                    p.start()
                    procs.append(p)
                reports = [results.get(timeout=60) for _ in procs]
                failures = [f"rank {r}:\n{e}" for r, e in reports if e]
                self.assertFalse(failures, "\n".join(failures))
                for p in procs:
                    p.join(5)
                    self.assertEqual(p.exitcode, 0)
            except queue.Empty:
                self.fail("two-process hop test did not finish in time")
            finally:
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                    p.join(5)
                results.close()
                results.join_thread()


@unittest.skipUnless(HAVE_DEPS, "torch and numpy needed (the engine imports both)")
class EngineAndHandleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod, cls.net = load_transport()

    def test_routing_follows_rank_nodes(self):
        mod = self.mod
        same = make_engine(mod, 0, ["A", "A"])
        self.assertFalse(same._is_remote(1))
        self.assertIsInstance(same.isend(payload(0, 0), dst=1, tag=1), mod.AsyncHandleSend)
        cross = make_engine(mod, 0, ["A", "B"])
        self.assertTrue(cross._is_remote(1))
        self.assertIsInstance(cross.isend(payload(0, 0), dst=1, tag=1), self.net.AsyncHandleSendNet)
        none = make_engine(mod, 0, ["A", "A"])
        none.rank_nodes = None  # a single-node run passes no rank_nodes at all
        self.assertFalse(none._is_remote(1))

    def test_same_node_irecv_keeps_the_shm_handle(self):
        engine = make_engine(self.mod, 1, ["A", "A"])
        with mock.patch.object(self.mod, "_ORIGINAL_IRECV", lambda *a, **k: object()):
            h = engine.irecv(torch.empty(SHAPE, dtype=torch.float16), src=0, tag=1)
        self.assertIsInstance(h, self.mod.AsyncHandleRecv)
        self.assertFalse(any(engine._recv_slot_reserved))

    def test_remote_hop_refuses_non_fp16_without_taking_a_slot(self):
        send = make_engine(self.mod, 0, ["A", "B"])
        with self.assertRaises(TypeError):
            send.isend(torch.zeros(SHAPE, dtype=torch.float32), dst=1, tag=1)
        self.assertTrue(all(send.slot_free))
        recv = make_engine(self.mod, 1, ["A", "B"])
        with self.assertRaises(TypeError):
            recv.irecv(torch.zeros(SHAPE, dtype=torch.float32), src=0, tag=1)
        self.assertFalse(any(recv._recv_slot_reserved))

    def test_failed_post_gives_the_reserved_slot_back(self):
        recv = make_engine(self.mod, 1, ["A", "B"])

        def boom(*a, **k):
            raise RuntimeError("post failed")

        with mock.patch.object(self.mod, "_ORIGINAL_IRECV", boom):
            with self.assertRaises(RuntimeError):
                recv.irecv(torch.empty(SHAPE, dtype=torch.float16), src=0, tag=1)
        self.assertFalse(any(recv._recv_slot_reserved))

    def test_reserved_recv_slots_are_not_handed_out(self):
        recv = make_engine(self.mod, 1, ["A", "B"], num_slots=2)
        recv._recv_slot_reserved[0] = True
        self.assertEqual(recv._get_free_recv_slot(), 1)
        recv._recv_slot_reserved[1] = True
        with mock.patch.object(self.mod.time, "sleep", lambda s: None):
            with self.assertRaises(RuntimeError):
                recv._get_free_recv_slot()
        recv._release_recv_slot(0)
        self.assertEqual(recv._get_free_recv_slot(), 0)

    def test_peer_attach_skips_remote_peers_only(self):
        mod = self.mod
        cross = make_engine(mod, 0, ["A", "B"])
        with mock.patch.object(mod.time, "sleep", side_effect=AssertionError("tried a remote peer")):
            cross._connect_peer_slots()
        self.assertEqual(cross.peer_slots, {})
        same = make_engine(mod, 0, ["A", "A"])
        same.shm_prefix = "nx_absent"  # short: macOS caps shm names at 31 chars
        with mock.patch.object(mod.time, "sleep", lambda s: None):
            with self.assertRaisesRegex(RuntimeError, "failed to connect"):
                same._connect_peer_slots()

    def test_world_size_and_rank_nodes_must_agree(self):
        with self.assertRaises(ValueError):
            self.mod.MIGPipelineTransport(0, 4, rank_nodes=["A", "B"])

    # -- handles, one process, fake gloo work ----------------------------------
    def send_handle(self, work, events):
        engine = make_engine(self.mod, 0, ["A", "B"], events=events)
        engine.slot_events[0] = Evt(lambda: events.append("d2h_complete"))
        engine.slot_free[0] = False
        engine.pinned_staging[0][:N] = 1

        def isend(buf, dst, group, tag):
            events.append("isend_posted")
            self.assertEqual((buf.numel(), dst, tag), (N, 1, 7))
            return work

        patcher = mock.patch.object(self.net, "_ORIGINAL_ISEND", isend)
        patcher.start()
        self.addCleanup(patcher.stop)
        return self.net.AsyncHandleSendNet(0, engine, 1, 7, None, N), engine

    def test_send_posts_after_d2h_and_frees_only_after_completion(self):
        events = []
        work = types.SimpleNamespace(wait=lambda: events.append("send_complete"))
        h, engine = self.send_handle(work, events)
        h.flush()
        h.flush()
        self.assertEqual(events, ["d2h_complete", "isend_posted"])
        self.assertFalse(engine.slot_free[0])  # flush alone never frees it
        h.wait()
        h.wait()
        self.assertEqual(events[-1], "send_complete")
        self.assertTrue(engine.slot_free[0])
        self.assertEqual((engine.stat_net_send_count, engine.stat_flush_count), (1, 1))

    def test_wait_alone_publishes_and_frees(self):
        events = []
        h, engine = self.send_handle(types.SimpleNamespace(wait=lambda: None), events)
        h.wait()
        self.assertEqual(events, ["d2h_complete", "isend_posted"])
        self.assertTrue(engine.slot_free[0])

    def test_failed_send_keeps_the_slot(self):
        def fail():
            raise RuntimeError("peer failed")

        h, engine = self.send_handle(types.SimpleNamespace(wait=fail), [])
        h.flush()
        with self.assertRaisesRegex(RuntimeError, "peer failed"):
            h.wait()
        self.assertFalse(engine.slot_free[0])
        self.assertFalse(h._waited)
        self.assertEqual(engine.stat_net_send_count, 0)

    def recv_handle(self, work, events):
        engine = make_engine(self.mod, 1, ["A", "B"], events=events)
        engine._recv_slot_reserved[2] = True
        engine.recv_pinned_staging[2][:N] = 5
        out = torch.zeros(SHAPE, dtype=torch.float16)
        return self.net.AsyncHandleRecvNet(work, None, 2, out, 0, engine, 7, None), engine, out

    def test_recv_holds_the_slot_until_h2d_landed(self):
        events = []
        work = types.SimpleNamespace(wait=lambda: events.append("data_arrived"))
        h, engine, out = self.recv_handle(work, events)
        orig_release = engine._release_recv_slot

        def release(rslot):
            events.append("released")
            orig_release(rslot)

        engine._release_recv_slot = release
        h.wait()
        h.wait()
        self.assertEqual(events, ["data_arrived", "h2d_landed", "released"])
        self.assertTrue(torch.equal(out, torch.full(SHAPE, 5.0).half()))
        self.assertFalse(engine._recv_slot_reserved[2])
        self.assertEqual((engine.stat_net_recv_count, engine.stat_h2d_count), (1, 1))

    def test_failed_recv_keeps_the_slot_reserved(self):
        def fail():
            raise RuntimeError("connection closed")

        h, engine, _ = self.recv_handle(types.SimpleNamespace(wait=fail), [])
        with self.assertRaises(RuntimeError):
            h.wait()
        self.assertTrue(engine._recv_slot_reserved[2])
        self.assertFalse(h._waited)

    def test_summary_reports_cross_node_lines_only_when_used(self):
        e = make_engine(self.mod, 0, ["A", "B"])
        with mock.patch.object(self.mod._tlog, "warning") as warn:
            e.log_summary("t")
            self.assertNotIn("T33", " ".join(str(c.args[0]) for c in warn.call_args_list))
            e.stat_net_send_wait_s, e.stat_net_send_count = 0.5, 5
            e.log_summary("t")
            self.assertIn("T33", " ".join(str(c.args[0]) for c in warn.call_args_list))


if __name__ == "__main__":
    unittest.main()
