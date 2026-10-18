import time
from mig_transport_pipeline import (
    _ORIGINAL_ISEND,
    _tlog,
    _fmt_ms,
)


# Handles for a hop between two NODES. Same flush()/wait() interface as the
# shared-memory handles, so the benchmark's loops do not change; the data goes
# over gloo TCP instead of /dev/shm, straight out of / into pinned host buffers.
#
# The three rules of the shared-memory protocol, in this form:
#   1. Send only after writing: the isend is posted after the slot's D2H landed.
#   2. Release receive staging only after reading: the slot is reserved from
#      the irecv post until its H2D copy has landed.
#   3. Reuse send staging only after completion: the slot is freed in wait(),
#      after gloo reports the send complete, never in flush(). A completed
#      send is the acknowledgement, so there is no handshake and no ACK.


class AsyncHandleSendNet:
    """Cross-node send: pinned staging slot -> gloo isend."""

    def __init__(self, slot_idx, engine, dst, tag, group, numel):
        self._slot_idx = slot_idx
        self._engine = engine
        self._dst = dst
        self._tag = tag
        self._group = group
        self._numel = numel
        self._flushed = False
        self._waited = False
        # Keep buffer and request alive until the send completes: gloo holds
        # the pointer, and a garbage-collected buffer would be read mid-send.
        self._buf = None
        self._work = None

    def flush(self):
        """Wait for this slot's D2H copy, then post the gloo send. Gloo
        transmits from its own thread while the next microbatch computes."""
        if self._flushed:
            return
        rank = self._engine.rank

        _t0 = time.perf_counter()
        self._engine.slot_events[self._slot_idx].synchronize()
        _blocked = time.perf_counter() - _t0
        _tlog.info(
            "[T07][rank%d] flush slot=%d tag=%d: copy event landed, blocked=%s",
            rank,
            self._slot_idx,
            self._tag,
            _fmt_ms(_blocked),
        )
        self._engine.stat_flush_block_s += _blocked
        self._engine.stat_flush_count += 1

        self._buf = self._engine.pinned_staging[self._slot_idx][: self._numel]
        self._work = _ORIGINAL_ISEND(
            self._buf, dst=self._dst, group=self._group, tag=self._tag
        )
        _tlog.info(
            "[T35][rank%d] net isend slot=%d tag=%d -> rank%d: %d elems posted",
            rank,
            self._slot_idx,
            self._tag,
            self._dst,
            self._numel,
        )
        self._flushed = True

    def wait(self):
        """Wait for gloo to finish the send, then free the slot. If the wait
        raises, the slot stays taken."""
        if self._waited:
            return
        rank = self._engine.rank
        if not self._flushed:
            _tlog.warning(
                "[T06b][rank%d] wait slot=%d tag=%d: flush() was not called "
                "first — send is correct but unoverlapped",
                rank,
                self._slot_idx,
                self._tag,
            )
        self.flush()

        _t0 = time.perf_counter()
        self._work.wait()
        _wait = time.perf_counter() - _t0

        self._engine.slot_free[self._slot_idx] = True
        self._waited = True
        self._buf = self._work = None
        self._engine.stat_net_send_wait_s += _wait
        self._engine.stat_net_send_count += 1
        _tlog.info(
            "[T11][rank%d] wait slot=%d tag=%d: net send complete in %s, slot freed",
            rank,
            self._slot_idx,
            self._tag,
            _fmt_ms(_wait),
        )


class AsyncHandleRecvNet:
    """Cross-node receive: gloo irecv into a reserved pinned slot -> H2D."""

    def __init__(self, work, buf, rslot, tensor, src, engine, tag, group):
        self._work = work
        self._buf = buf  # keeps the posted receive buffer alive
        self._rslot = rslot
        self._tensor = tensor
        self._src = src
        self._engine = engine
        self._tag = tag
        self._group = group
        self._waited = False

    def wait(self):
        if self._waited:
            return
        rank = self._engine.rank

        # Large here is upstream compute plus wire time, not a transport cost.
        _t0 = time.perf_counter()
        self._work.wait()
        _recv_wait = time.perf_counter() - _t0
        self._engine.stat_net_recv_wait_s += _recv_wait
        self._engine.stat_net_recv_count += 1

        evt = self._engine._queue_h2d_from_staging(self._tensor, self._rslot)
        _t0 = time.perf_counter()
        # REQUIRED: the staging slot is reused after this, so the copy out of it
        # must have landed (rule 2). One slot's event, not a device-wide sync.
        evt.synchronize()
        _h2d = time.perf_counter() - _t0
        self._engine.stat_h2d_block_s += _h2d
        self._engine.stat_h2d_count += 1

        self._engine._release_recv_slot(self._rslot)
        self._waited = True
        self._buf = self._work = None
        _tlog.info(
            "[T36][rank%d] net recv tag=%d <- rank%d landed: wait=%s h2d=%s",
            rank,
            self._tag,
            self._src,
            _fmt_ms(_recv_wait),
            _fmt_ms(_h2d),
        )
