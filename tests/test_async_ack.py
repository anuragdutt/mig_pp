"""CPU-only ACK ownership and real Gloo progress regressions.

Run with: python3 -m unittest discover -s tests -p test_async_ack.py -v
CUDA events and the SHM engine are simulated; the integration test uses the
production send/receive handles and real distributed control messages.
"""

from contextlib import contextmanager
from datetime import timedelta
import importlib.util
import logging
import multiprocessing
from pathlib import Path
import queue
import tempfile
import traceback
import types
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist


ACK_TAG_OFFSET = 10_000_000
ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def handle_modules(send=dist.send, recv=dist.recv, irecv=dist.irecv):
    # Import only the handles: the full engine imports model/dataset helpers
    # and allocates CUDA resources, neither of which this protocol test needs.
    transport = types.ModuleType("mig_transport_pipeline")
    transport._ORIGINAL_SEND = send
    transport._ORIGINAL_RECV = recv
    transport._ORIGINAL_IRECV = irecv
    transport.ACK_TAG_OFFSET = ACK_TAG_OFFSET
    transport._tlog = logging.getLogger("test_async_ack")
    transport._fmt_ms = lambda seconds: f"{seconds * 1000:.2f}ms"
    modules = []
    with patch.dict("sys.modules", {"mig_transport_pipeline": transport}):
        for filename in ("async_handle_send", "async_handle_recv"):
            spec = importlib.util.spec_from_file_location(
                f"_ack_test_{filename}", ROOT / f"{filename}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        yield modules


class Event:
    def __init__(self, synchronize):
        self.synchronize = synchronize


def sender_engine(flush, synchronize=lambda: None, num_slots=1):
    return types.SimpleNamespace(
        rank=0,
        slot_events=[Event(synchronize) for _ in range(num_slots)],
        slot_free=[False] * num_slots,
        _flush_staging_to_shm=flush,
        stat_flush_block_s=0.0,
        stat_flush_count=0,
        stat_ack_wait_s=0.0,
        stat_ack_count=0,
    )


class AckOwnershipTests(unittest.TestCase):
    def make_sender(self, ack_value=0, wait_error=None):
        events = []
        engine = sender_engine(
            lambda slot, numel: events.append("shm_written"),
            lambda: events.append("d2h_complete"),
        )

        def irecv(tensor, src, group, tag):
            self.assertEqual((src, group, tag), (1, None, 17 + ACK_TAG_OFFSET))
            events.append("ack_posted")

            def wait():
                self.assertFalse(engine.slot_free[0])
                events.append("ack_complete")
                if wait_error:
                    raise wait_error
                tensor.fill_(ack_value)
                return True

            return types.SimpleNamespace(wait=wait)

        def send(tensor, dst, group, tag):
            self.assertEqual((int(tensor.item()), dst, group, tag), (0, 1, None, 17))
            self.assertIn("ack_posted", events)
            self.assertLess(events.index("d2h_complete"), events.index("shm_written"))
            events.append("handshake_sent")

        with handle_modules(send=send, irecv=irecv) as (send_module, _):
            handle = send_module.AsyncHandleSend(0, engine, 1, 17, None, 1)
        return handle, engine, events

    def test_publish_and_reclaim_order_and_idempotence(self):
        handle, engine, events = self.make_sender()
        handle.flush()
        handle.flush()
        self.assertFalse(engine.slot_free[0])
        self.assertNotIn("ack_complete", events)
        self.assertEqual(events.count("handshake_sent"), 1)
        self.assertLess(events.index("shm_written"), events.index("handshake_sent"))
        self.assertLess(events.index("ack_posted"), events.index("handshake_sent"))
        handle.wait()
        handle.wait()
        handle.flush()
        self.assertTrue(engine.slot_free[0])
        self.assertEqual(events.count("ack_posted"), 1)
        self.assertEqual(events.count("ack_complete"), 1)
        self.assertEqual((engine.stat_flush_count, engine.stat_ack_count), (1, 1))

    def test_wait_alone_publishes_and_reclaims(self):
        handle, engine, events = self.make_sender()
        handle.wait()
        self.assertTrue(engine.slot_free[0])
        self.assertLess(events.index("handshake_sent"), events.index("ack_complete"))

    def test_failed_ack_does_not_release_slot(self):
        handle, engine, _ = self.make_sender(wait_error=RuntimeError("peer failed"))
        handle.flush()
        with self.assertRaisesRegex(RuntimeError, "peer failed"):
            handle.wait()
        self.assertFalse(engine.slot_free[0])
        self.assertFalse(handle._waited)
        self.assertEqual(engine.stat_ack_count, 0)

    def test_wrong_slot_ack_does_not_release_slot(self):
        handle, engine, _ = self.make_sender(ack_value=1)
        handle.flush()
        with self.assertRaises((RuntimeError, ValueError)):
            handle.wait()
        self.assertFalse(engine.slot_free[0])
        self.assertFalse(handle._waited)
        self.assertEqual(engine.stat_ack_count, 0)

    def test_receiver_ack_follows_h2d_completion(self):
        events = []
        tensor = torch.empty(1)
        engine = types.SimpleNamespace(
            rank=1,
            stat_handshake_wait_s=0.0,
            stat_handshake_count=0,
            stat_h2d_block_s=0.0,
            stat_h2d_count=0,
        )

        def read(output, src, slot):
            self.assertEqual((src, slot), (0, 2))
            self.assertEqual(events, ["handshake_complete"])
            events.append("h2d_queued")
            return Event(lambda: events.append("h2d_complete"))

        engine._queue_read_from_slot = read

        def send(ack, dst, group, tag):
            self.assertEqual((int(ack.item()), dst, tag), (2, 0, 17 + ACK_TAG_OFFSET))
            self.assertEqual(events[-1], "h2d_complete")
            events.append("ack_sent")

        work = Event(lambda: events.append("handshake_complete"))
        work.wait = work.synchronize
        with handle_modules(send=send) as (_, recv_module):
            handle = recv_module.AsyncHandleRecv(
                work, torch.tensor([2]), tensor, 0, engine, 17, None
            )
        handle.wait()
        handle.wait()
        self.assertEqual(events.count("ack_sent"), 1)


def gloo_worker(rank, rendezvous, shared, copy_started, allow_copy, received, results):
    try:
        dist.init_process_group(
            "gloo", init_method=rendezvous, rank=rank, world_size=2,
            timeout=timedelta(seconds=15),
        )
        with handle_modules() as (send_module, recv_module):
            for round_index, slots in enumerate(([0], [0, 1], [0, 1])):
                if rank == 0:
                    def publish(slot, numel):
                        shared[slot] = round_index * 10 + slot + 1

                    engine = sender_engine(publish, num_slots=2)
                    handles = [
                        send_module.AsyncHandleSend(slot, engine, 1, 1100 + slot, None, 1)
                        for slot in slots
                    ]
                    for handle in handles:
                        handle.flush()
                    if round_index == 0:
                        assert copy_started.wait(10), "receiver never started H2D"
                        assert not engine.slot_free[0], "slot released before H2D"
                        allow_copy.set()
                    # This is the regression: receiver.wait() MUST return while
                    # the sender has not yet called any handle.wait(). The old
                    # blocking ACK receive deadlocks at this event gate.
                    assert received[round_index].wait(10), (
                        "receiver blocked on ACK until sender drain"
                    )
                    assert all(not engine.slot_free[slot] for slot in slots)
                    for handle in handles:
                        handle.wait()
                        handle.wait()
                    assert all(engine.slot_free[slot] for slot in slots)
                    assert engine.stat_ack_count == len(slots)
                else:
                    engine = types.SimpleNamespace(
                        rank=1, stat_handshake_wait_s=0.0, stat_handshake_count=0,
                        stat_h2d_block_s=0.0, stat_h2d_count=0,
                    )

                    def read(output, src, slot):
                        def complete():
                            if round_index == 0:
                                copy_started.set()
                                assert allow_copy.wait(10), "sender never released H2D gate"
                            output.fill_(shared[slot])
                        return Event(complete)

                    engine._queue_read_from_slot = read
                    handles = []
                    for slot in slots:
                        handshake = torch.empty(1, dtype=torch.int32)
                        output = torch.empty(1)
                        work = dist.irecv(handshake, src=0, tag=1100 + slot)
                        handle = recv_module.AsyncHandleRecv(
                            work, handshake, output, 0, engine, 1100 + slot, None
                        )
                        handles.append((slot, output, handle))
                    # Reverse delivery proves independent tags match the right
                    # slots; the final round reuses both slots and message tags.
                    for slot, output, handle in reversed(handles):
                        handle.wait()
                        handle.wait()
                        assert output.item() == round_index * 10 + slot + 1
                    received[round_index].set()
                dist.barrier()
        dist.destroy_process_group()
        results.put((rank, None))
    except Exception:
        results.put((rank, traceback.format_exc()))


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
class GlooProgressTests(unittest.TestCase):
    def test_receiver_progress_before_sender_drain_and_tag_reuse(self):
        ctx = multiprocessing.get_context("spawn")
        shared = ctx.Array("d", 2, lock=False)
        copy_started, allow_copy = ctx.Event(), ctx.Event()
        received = [ctx.Event() for _ in range(3)]
        results = ctx.Queue()
        processes = []
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            try:
                for rank in range(2):
                    process = ctx.Process(
                        target=gloo_worker,
                        args=(rank, rendezvous, shared, copy_started, allow_copy, received, results),
                    )
                    process.start()
                    processes.append(process)
                reports = [results.get(timeout=30) for _ in processes]
                failures = [f"rank {rank}:\n{error}" for rank, error in reports if error]
                self.assertFalse(failures, "\n".join(failures))
                for process in processes:
                    process.join(5)
                    self.assertEqual(process.exitcode, 0)
            except queue.Empty:
                self.fail("Gloo regression exceeded its bounded completion deadline")
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)
                results.close()
                results.join_thread()


if __name__ == "__main__":
    unittest.main()
