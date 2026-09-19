import time
import torch
from mig_transport_pipeline import (
    _ORIGINAL_SEND,
    ACK_TAG_OFFSET,
    _tlog,
    _fmt_ms,
)


class AsyncHandleRecv:
    """
    Async recv handle.
    wait() completes handshake recv, reads SHM into tensor, then sends ACK.
    """

    def __init__(self, dist_handle, handshake, tensor, src, engine, tag, group):
        self._dist_handle = dist_handle
        self._handshake = handshake
        self._tensor = tensor
        self._src = src
        self._engine = engine
        self._tag = tag
        self._group = group
        self._waited = False

    # This function is for the receiver
    def wait(self):
        if self._waited:
            return

        # Wait until handshake arrives (slot index). By this point the sender
        # has already flushed the data into SHM (see AsyncHandleSend.flush — the
        # handshake is sent only after the flush), so the slot is safe to read
        # as soon as this handshake shows up.
        rank = self._engine.rank
        _tlog.debug(
            "[T13a][rank%d] recv wait tag=%d: blocking for handshake from rank%d",
            rank,
            self._tag,
            self._src,
        )
        _t0 = time.perf_counter()
        self._dist_handle.wait()
        _hs_wait = time.perf_counter() - _t0
        slot = int(self._handshake.item())
        _tlog.info(
            "[T13][rank%d] recv wait tag=%d: handshake slot=%d from rank%d "
            "after %s (large = upstream rank still busy)",
            rank,
            self._tag,
            slot,
            self._src,
            _fmt_ms(_hs_wait),
        )
        self._engine.stat_handshake_wait_s += _hs_wait
        self._engine.stat_handshake_count += 1

        # Queue SHM->pinned->GPU on recv_stream, then wait only THIS slot's
        # event — not a device-wide sync.
        #
        # LIMITATION: because the slot index only arrives with the handshake,
        # the H2D copy cannot be queued any earlier than this, so it does not
        # overlap with this microbatch's own wait. It overlaps only with work
        # still outstanding on the default stream. Fully hiding the receive
        # would need a background thread draining handshakes as they land;
        # that is deliberately not done here (a thread touching the CUDA
        # context is the main hang risk in this design).
        _t0 = time.perf_counter()
        evt = self._engine._queue_read_from_slot(self._tensor, self._src, slot)
        _queue_t = time.perf_counter() - _t0
        _tlog.debug(
            "[T14][rank%d] recv wait tag=%d slot=%d: H2D queued in %s",
            rank,
            self._tag,
            slot,
            _fmt_ms(_queue_t),
        )

        _t0 = time.perf_counter()
        evt.synchronize()
        _h2d_wait = time.perf_counter() - _t0
        _tlog.info(
            "[T15][rank%d] recv wait tag=%d slot=%d: H2D landed, blocked=%s",
            rank,
            self._tag,
            slot,
            _fmt_ms(_h2d_wait),
        )
        self._engine.stat_h2d_block_s += _h2d_wait
        self._engine.stat_h2d_count += 1

        # Send ACK back so sender can reuse slot
        ack = torch.tensor([slot], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(
            ack,
            dst=self._src,
            group=self._group,
            tag=self._tag + ACK_TAG_OFFSET,
        )
        _tlog.debug(
            "[T16][rank%d] recv wait tag=%d slot=%d: ACK sent -> rank%d",
            rank,
            self._tag,
            slot,
            self._src,
        )

        # Marking the job as done
        self._waited = True
