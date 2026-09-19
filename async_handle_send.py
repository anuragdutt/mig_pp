import time
import torch
from mig_transport_pipeline import (
    _ORIGINAL_SEND,
    _ORIGINAL_RECV,
    ACK_TAG_OFFSET,
    _tlog,
    _fmt_ms,
)


class AsyncHandleSend:
    """
    Async send handle.
    IMPORTANT: slot becomes reusable only after receiver ACKs it has read SHM.
    """

    def __init__(self, slot_idx, engine, dst, tag, group, numel):
        self._slot_idx = slot_idx
        self._engine = engine
        self._dst = dst
        self._tag = tag
        self._group = group
        self._numel = numel
        self._flushed = False
        self._waited = False

    def flush(self):
        """
        Phase 1 — publish the data to the receiver.

        Waits for THIS slot's D2H copy (queued on copy_stream at isend()
        time) to land, memcpys pinned staging into SHM, then sends the
        handshake so the receiver can start reading.

        Split out from wait() so the caller can publish microbatch k
        immediately after computing it, while still deferring the ACK wait
        (phase 2) to a batched drain later. Publishing early is what keeps
        the DOWNSTREAM rank fed — if the handshake only went out at drain
        time, rank N+1 would sit idle until rank N finished every
        microbatch, which would serialize the ranks.
        """
        if self._flushed:
            return

        rank = self._engine.rank

        # Targets ONE slot's event — not a full-device torch.cuda.synchronize().
        # Compute already queued on the default stream keeps running.
        #
        # How long this blocks is THE overlap metric: if the fix is working,
        # the D2H copy ran underneath the next microbatch's compute and this
        # is ~0ms. A large value here means the copy had no compute to hide
        # behind (or the copy never got queued asynchronously at all).
        _tlog.debug(
            "[T06][rank%d] flush slot=%d tag=%d: waiting on copy event",
            rank,
            self._slot_idx,
            self._tag,
        )
        _t0 = time.perf_counter()

        self._engine.slot_events[self._slot_idx].synchronize()

        _blocked = time.perf_counter() - _t0
        _tlog.info(
            "[T07][rank%d] flush slot=%d tag=%d: copy event landed, "
            "blocked=%s (≈0 means D2H overlapped with compute)",
            rank,
            self._slot_idx,
            self._tag,
            _fmt_ms(_blocked),
        )

        self._engine.stat_flush_block_s += _blocked
        self._engine.stat_flush_count += 1

        # Pinned staging is valid on CPU only now; publish it to SHM.
        _t0 = time.perf_counter()
        self._engine._flush_staging_to_shm(self._slot_idx, self._numel)
        _tlog.debug(
            "[T08][rank%d] flush slot=%d: staged->SHM memcpy %d elems in %s",
            rank,
            self._slot_idx,
            self._numel,
            _fmt_ms(time.perf_counter() - _t0),
        )

        # Handshake AFTER the data is actually in SHM — sending it earlier
        # would let the receiver race ahead and read a stale/garbage slot.
        handshake = torch.tensor([self._slot_idx], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(handshake, dst=self._dst, group=self._group, tag=self._tag)

        _tlog.debug(
            "[T09][rank%d] flush slot=%d tag=%d: handshake sent -> rank%d",
            rank,
            self._slot_idx,
            self._tag,
            self._dst,
        )

        self._flushed = True

    # sender side wait
    def wait(self):
        """
        Phase 2 — reclaim the slot.

        Waits for the receiver's ACK ("I finished reading that SHM slot"),
        then marks the slot reusable. Safe to call in a batched drain after
        the compute loop, since flush() already unblocked the receiver.
        """
        if self._waited:
            return

        rank = self._engine.rank

        # Idempotent: if the caller never called flush() explicitly, do it
        # now so wait() alone is still correct (just less overlapped).
        if not self._flushed:
            _tlog.warning(
                "[T06b][rank%d] wait slot=%d tag=%d: flush() was not called "
                "first — send is correct but unoverlapped",
                rank,
                self._slot_idx,
                self._tag,
            )

        self.flush()

        _tlog.debug(
            "[T10][rank%d] wait slot=%d tag=%d: waiting for ACK from rank%d",
            rank,
            self._slot_idx,
            self._tag,
            self._dst,
        )
        _t0 = time.perf_counter()

        ack = torch.empty((1,), dtype=torch.int32, device="cpu")
        _ORIGINAL_RECV(
            ack,
            src=self._dst,
            group=self._group,
            tag=self._tag + ACK_TAG_OFFSET,
        )
        _ack_wait = time.perf_counter() - _t0

        # Now safe to reuse slot
        self._engine.slot_free[self._slot_idx] = True
        self._waited = True

        _tlog.info(
            "[T11][rank%d] wait slot=%d tag=%d: ACK(%d) received in %s, slot freed",
            rank,
            self._slot_idx,
            self._tag,
            int(ack.item()),
            _fmt_ms(_ack_wait),
        )

        self._engine.stat_ack_wait_s += _ack_wait
        self._engine.stat_ack_count += 1
