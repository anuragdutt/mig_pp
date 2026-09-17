import os
import time
import logging
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# TRANSPORT LOGGING
# ---------------------------------------------------------------------------
# Every log statement carries a unique [Txx] tag so any line in the log can be
# traced back to exactly one source location, and so you can grep a single
# phase across all ranks (e.g. `grep T07 run_*.log`).
#
# Tag map (INFO unless marked debug — set MIG_LOG_DEBUG=1 for the rest):
#   T01   engine construction / config + host memory cost
#   T04   engine ready (peers connected)
#   T05   isend: D2H queued on copy_stream  <- queue time must be ~0
#   T06   flush: waiting on copy event                          (debug)
#   T06b  flush() was skipped by caller — send unoverlapped     (warn)
#   T07   flush: copy event landed          <- BLOCK TIME = overlap metric
#   T08   flush: staged -> SHM memcpy done                      (debug)
#   T09   flush: handshake sent                                 (debug)
#   T10   wait: waiting for ACK                                 (debug)
#   T11   wait: ACK received, slot freed
#   T12   irecv: handshake listener posted                      (debug)
#   T13a  recv wait: blocking for handshake                     (debug)
#   T13   recv wait: handshake received (+ how long it blocked)
#   T14   recv wait: H2D queued                                 (debug)
#   T15   recv wait: H2D event landed
#   T16   recv wait: ACK sent                                   (debug)
#   T17   blocking send() path used (next_tokens exchange)
#   T18   blocking recv() path used (next_tokens exchange)
#   T19   send slot pool had to spin                            (warn)
#   T19b  send slot pool deadlocked after 10s                   (error)
#   T20   non-fp16 BLOCKING fallback on send                    (debug)
#   T20b  non-fp16 BLOCKING fallback on recv                    (debug)
#   T21   hooks registered
#   T22   ===== summary header =====
#   T23   summary: D2H flush block mean/total
#   T24   summary: H2D recv block mean/total
#   T25   summary: handshake wait mean
#   T26   summary: ACK wait mean
#   T27   summary: slot pool spins
#   T28   VERDICT: no async sends on this rank
#   T29   VERDICT: overlap working
#   T30   VERDICT: poor overlap                                 (warn)
#
# Quick triage after a run:
#   grep -E 'T29|T30' logs/transport_*.log     # the verdict, one per rank
#   grep T05 logs/transport_*.log | tail -20   # isend queue times (want ~0)
#   grep T07 logs/transport_*.log | tail -20   # flush block times (want ~0)
#   grep -E 'T19|T30|T06b' logs/transport_*.log # problems only
_tlog = logging.getLogger("mig_transport")


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f}ms"


# --- GLOBAL STATE ---
_MIG_PIPE_ENGINE = None
# Shared memory handles must be kept in memory;
# if Python's garbage collector destroys the object, the shared memory can become inaccessible
_MIG_SHM_HANDLES: List[SharedMemory] = []

# Saving the _ORIGINAL_* functions is necessary
# so we can intercept (monkey-patch) PyTorch's native calls later
# while still retaining the ability to use the original PyTorch backend to send the tiny handshake messages
_ORIGINAL_SEND = dist.send
_ORIGINAL_RECV = dist.recv
_ORIGINAL_ISEND = dist.isend
_ORIGINAL_IRECV = dist.irecv

# Number of ring buffer slots per rank.
# Must be > max microbatches in flight at once, with headroom.
# Max microbatches comes from the smallest microbatch size: batch=64 / mb=2 = 32.
# Slots are only held until the receiver ACKs, but with the batched ACK drain
# all 32 can be outstanding simultaneously — sitting exactly AT the limit made
# _get_free_slot() spin and then raise. Slots are host SHM + pinned RAM, not
# GPU memory, so headroom is cheap.
NUM_SLOTS = 48

# ACK tag is tag + ACK_TAG_OFFSET so it never collides with normal handshakes
ACK_TAG_OFFSET = 10_000_000

# Map torch dtype -> numpy dtype without forcing tensor.cpu()
_TORCH_TO_NUMPY = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


def _compute_slot_size_mb(hidden_size=5120, max_mb_size=32, max_seq_len=64):
    """Largest tensor: prefill activation (mb_size × seq_len × hidden × 2 bytes)

    NOTE: pass the mb_size the run ACTUALLY uses. Every slot is allocated at
    this size in BOTH host SHM and pinned RAM, once per slot per rank, so
    oversizing multiplies fast: 4 ranks x NUM_SLOTS x 2 (shm+pinned).
    """
    max_bytes = max_mb_size * max_seq_len * hidden_size * 2  # fp16
    mb = (max_bytes // (1024 * 1024)) + 1  # round up
    return mb


# When you use non-blocking communication (isend/irecv), PyTorch returns a "handle" you can wait on later.
# Because this script overrides the backend, it must provide custom handles.
class AsyncHandle:
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
            rank, self._slot_idx, self._tag,
        )
        _t0 = time.perf_counter()
        self._engine.slot_events[self._slot_idx].synchronize()
        _blocked = time.perf_counter() - _t0
        _tlog.info(
            "[T07][rank%d] flush slot=%d tag=%d: copy event landed, "
            "blocked=%s (≈0 means D2H overlapped with compute)",
            rank, self._slot_idx, self._tag, _fmt_ms(_blocked),
        )
        self._engine.stat_flush_block_s += _blocked
        self._engine.stat_flush_count += 1

        # Pinned staging is valid on CPU only now; publish it to SHM.
        _t0 = time.perf_counter()
        self._engine._flush_staging_to_shm(self._slot_idx, self._numel)
        _tlog.debug(
            "[T08][rank%d] flush slot=%d: staged->SHM memcpy %d elems in %s",
            rank, self._slot_idx, self._numel, _fmt_ms(time.perf_counter() - _t0),
        )

        # Handshake AFTER the data is actually in SHM — sending it earlier
        # would let the receiver race ahead and read a stale/garbage slot.
        handshake = torch.tensor([self._slot_idx], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(handshake, dst=self._dst, group=self._group, tag=self._tag)
        _tlog.debug(
            "[T09][rank%d] flush slot=%d tag=%d: handshake sent -> rank%d",
            rank, self._slot_idx, self._tag, self._dst,
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
                rank, self._slot_idx, self._tag,
            )
        self.flush()

        _tlog.debug(
            "[T10][rank%d] wait slot=%d tag=%d: waiting for ACK from rank%d",
            rank, self._slot_idx, self._tag, self._dst,
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
            rank, self._slot_idx, self._tag, int(ack.item()), _fmt_ms(_ack_wait),
        )
        self._engine.stat_ack_wait_s += _ack_wait
        self._engine.stat_ack_count += 1


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
        # has already flushed the data into SHM (see AsyncHandle.flush — the
        # handshake is sent only after the flush), so the slot is safe to read
        # as soon as this handshake shows up.
        rank = self._engine.rank
        _tlog.debug(
            "[T13a][rank%d] recv wait tag=%d: blocking for handshake from rank%d",
            rank, self._tag, self._src,
        )
        _t0 = time.perf_counter()
        self._dist_handle.wait()
        _hs_wait = time.perf_counter() - _t0
        slot = int(self._handshake.item())
        _tlog.info(
            "[T13][rank%d] recv wait tag=%d: handshake slot=%d from rank%d "
            "after %s (large = upstream rank still busy)",
            rank, self._tag, slot, self._src, _fmt_ms(_hs_wait),
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
            rank, self._tag, slot, _fmt_ms(_queue_t),
        )

        _t0 = time.perf_counter()
        evt.synchronize()
        _h2d_wait = time.perf_counter() - _t0
        _tlog.info(
            "[T15][rank%d] recv wait tag=%d slot=%d: H2D landed, blocked=%s",
            rank, self._tag, slot, _fmt_ms(_h2d_wait),
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
            rank, self._tag, slot, self._src,
        )

        # Marking the job as done
        self._waited = True


class MIGPipelineTransport:
    def __init__(self, rank, world_size, buffer_size_mb=128, num_slots=NUM_SLOTS):
        # GPU id allocation
        self.rank = rank
        # Total no of GPU's working together
        self.world_size = world_size
        self.num_slots = num_slots

        # Each slot holds raw bytes for largest tensor.
        self.slot_size = buffer_size_mb * 1024 * 1024

        _tlog.info(
            "[T01][rank%d] init transport: world_size=%d num_slots=%d "
            "slot_size=%dMB (host cost: %dMB SHM + %dMB pinned send + "
            "%dMB pinned recv)",
            rank, world_size, num_slots, buffer_size_mb,
            num_slots * buffer_size_mb,
            num_slots * buffer_size_mb,
            num_slots * buffer_size_mb,
        )
        print(
            f"[MIG-Pipe] Rank {rank} initializing transport "
            f"({num_slots} slots x {buffer_size_mb}MB)...",
            flush=True,
        )

        # --- CREATE SHM SLOTS FOR THIS RANK ---
        self.my_shm_slots: List[SharedMemory] = []
        self.my_np_slots: List[np.ndarray] = []
        self.slot_free = [True] * num_slots

        for slot in range(num_slots):
            name = f"mig_pipe_shm_{rank}_slot{slot}"

            # Clean stale /dev/shm entries if present
            try:
                if os.path.exists(f"/dev/shm/{name}"):
                    os.unlink(f"/dev/shm/{name}")
            except Exception:
                pass

            # Creating the memory slots
            try:
                shm = SharedMemory(name=name, create=True, size=self.slot_size)
            except FileExistsError:
                shm = SharedMemory(name=name, create=False, size=self.slot_size)

            # For preventing garbage collection
            _MIG_SHM_HANDLES.append(shm)
            self.my_shm_slots.append(shm)

            # Wrap the raw shared memory buffer in a NumPy array.
            # This creates a "view" into the memory, allowing us to easily read/write
            # raw bytes without needing complex memory address math or extra data copies.
            self.my_np_slots.append(
                np.ndarray((self.slot_size,), dtype=np.uint8, buffer=shm.buf)
            )

        # --- PINNED CPU STAGING (float16 fast-path) ---
        # This tells the operating system: "Lock this specific chunk of RAM in place. Do not ever move it to the hard drive."
        # Because it is permanently locked in place, the GPU can use Direct Memory Access (DMA) to bypass the CPU entirely
        # and shove the massive tensor directly into this memory at maximum hardware speeds.
        # We use float16 as 16-bit math is most widely used
        max_elements = self.slot_size // 2  # float16 = 2 bytes
        self.pinned_staging = [
            torch.zeros(max_elements, dtype=torch.float16).pin_memory()
            for _ in range(num_slots)
        ]

        # --- ASYNC COPY INFRASTRUCTURE (sender side) ---
        # A dedicated stream for D2H copies, separate from the default
        # (compute) stream. Work queued here runs concurrently with whatever
        # the default stream is doing — that's the whole point: the next
        # microbatch's forward pass keeps going while this microbatch's
        # activation is copied out to pinned CPU memory in the background.
        self.copy_stream = torch.cuda.Stream()

        # One event per slot. An event is a marker you can drop into a
        # stream and later ask "has execution reached this point yet?".
        # Per-slot (not one shared event) because multiple copies can be
        # in flight at once and each needs to be waited on independently —
        # a single shared event would get overwritten by the next record()
        # and you'd end up waiting on the wrong copy.
        self.slot_events = [torch.cuda.Event() for _ in range(num_slots)]

        # --- ASYNC COPY INFRASTRUCTURE (receiver side) ---
        # Pinned staging for incoming data too, symmetric with the sender's
        # pinned_staging. Lets the H2D copy in _read_tensor_from_slot run
        # non_blocking on a dedicated recv stream instead of a slow unpinned
        # synchronous copy.
        self.recv_stream = torch.cuda.Stream()
        self.recv_pinned_staging = [
            torch.zeros(max_elements, dtype=torch.float16).pin_memory()
            for _ in range(num_slots)
        ]
        self.recv_events = [torch.cuda.Event() for _ in range(num_slots)]
        # Receiver-side slot bookkeeping, independent of the sender's slot
        # indices. Entry holds the event still gating that staging buffer,
        # or None if free.
        self._recv_slot_pending: List[Optional[torch.cuda.Event]] = [
            None
        ] * num_slots

        # --- INSTRUMENTATION COUNTERS ---
        # Aggregated by log_summary(); these are what tell you whether the
        # async transport is actually overlapping or silently serializing.
        self.stat_flush_block_s = 0.0      # time blocked waiting on D2H events
        self.stat_flush_count = 0
        self.stat_ack_wait_s = 0.0         # time blocked waiting for ACKs
        self.stat_ack_count = 0
        self.stat_handshake_wait_s = 0.0   # time blocked waiting for handshakes
        self.stat_handshake_count = 0
        self.stat_h2d_block_s = 0.0        # time blocked waiting on H2D events
        self.stat_h2d_count = 0
        self.stat_slot_spins = 0           # times _get_free_slot had to spin

        # Wait for all the peers to be done with Memory slot creation
        # This has to be done before moving on to next step where we check for peer connections
        # dist.barrier is a blocking synchronization mechanism
        dist.barrier()

        # --- CONNECT TO PEER SLOTS ---
        self.peer_slots: Dict[int, List[np.ndarray]] = {}
        for peer_rank in range(world_size):
            if peer_rank == rank:
                continue

            self.peer_slots[peer_rank] = []
            for slot in range(num_slots):
                peer_name = f"mig_pipe_shm_{peer_rank}_slot{slot}"
                connected = False
                attempts = 0

                while not connected and attempts < 1000:
                    try:
                        shm = SharedMemory(
                            name=peer_name, create=False, size=self.slot_size
                        )
                        _MIG_SHM_HANDLES.append(shm)
                        self.peer_slots[peer_rank].append(
                            np.ndarray(
                                (self.slot_size,), dtype=np.uint8, buffer=shm.buf
                            )
                        )
                        connected = True
                    except FileNotFoundError:
                        time.sleep(0.01)
                        attempts += 1

                if not connected:
                    raise RuntimeError(
                        f"Rank {rank} failed to connect to {peer_name} "
                        f"after {attempts} attempts"
                    )

        _tlog.info(
            "[T04][rank%d] transport ready: connected to %d peers, %d slots",
            rank, len(self.peer_slots), num_slots,
        )
        print(f"[MIG-Pipe] Rank {rank} connected to all peers. Ready.", flush=True)

        # Nobody is allowed to start the actual workday (sending boxes) until everyone has finished mapping the building!
        dist.barrier()

    def _get_free_slot(self) -> int:
        for attempt in range(10000):
            for slot in range(self.num_slots):
                if self.slot_free[slot]:
                    self.slot_free[slot] = False
                    if attempt > 0:
                        # Spinning means every slot was in flight — the pool
                        # is undersized for this pipeline depth.
                        self.stat_slot_spins += 1
                        _tlog.warning(
                            "[T19][rank%d] send slot pool exhausted, spun %d x1ms "
                            "before slot=%d freed (num_slots=%d — increase it)",
                            self.rank, attempt, slot, self.num_slots,
                        )
                    return slot
            time.sleep(0.001)

        _tlog.error(
            "[T19b][rank%d] send slot pool DEADLOCK: no free slot after 10s "
            "(num_slots=%d, all in flight awaiting ACK)",
            self.rank, self.num_slots,
        )
        raise RuntimeError(
            f"[Rank {self.rank}] No free SHM slots available — "
            f"increase NUM_SLOTS or reduce pipeline depth."
        )

    def _get_free_recv_slot(self) -> int:
        """
        Pick a receiver-side staging buffer. A slot is reusable once the
        H2D copy that was reading it has completed (event.query() is the
        non-blocking 'is it done yet?' check).
        """
        for _ in range(10000):
            for slot in range(self.num_slots):
                pending = self._recv_slot_pending[slot]
                if pending is None or pending.query():
                    self._recv_slot_pending[slot] = None
                    return slot
            time.sleep(0.001)

        raise RuntimeError(
            f"[Rank {self.rank}] No free recv staging slots available — "
            f"increase NUM_SLOTS or reduce pipeline depth."
        )

    # -----------------------------
    # Public API used by patch hooks
    # -----------------------------

    def send(self, tensor, dst, group=None, tag: int = 0):
        """
        Blocking send:
          - write SHM
          - send handshake(slot)
          - wait ACK
        """
        _t0 = time.perf_counter()
        slot = self._get_free_slot()
        self._write_tensor_to_slot(tensor, slot)

        # Handshake is shared using original pytorch backend ie gloo
        handshake = torch.tensor([slot], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(handshake, dst=dst, group=group, tag=tag)

        # ACK ensures receiver finished reading this SHM slot
        ack = torch.empty((1,), dtype=torch.int32, device="cpu")
        _ORIGINAL_RECV(ack, src=dst, group=group, tag=tag + ACK_TAG_OFFSET)

        self.slot_free[slot] = True
        _tlog.info(
            "[T17][rank%d] BLOCKING send slot=%d tag=%d -> rank%d: %s in %s "
            "(sync path — used for next_tokens exchange, not activations)",
            self.rank, slot, tag, dst, str(tuple(tensor.shape)),
            _fmt_ms(time.perf_counter() - _t0),
        )

    def recv(self, tensor, src, group=None, tag: int = 0):
        """
        Blocking recv:
          - recv handshake(slot)
          - read SHM into tensor
          - send ACK
        """
        _t0 = time.perf_counter()
        handshake = torch.tensor([0], dtype=torch.int32, device="cpu")
        _ORIGINAL_RECV(handshake, src=src, group=group, tag=tag)
        slot = int(handshake.item())

        self._read_tensor_from_slot(tensor, src, slot)

        ack = torch.tensor([slot], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(ack, dst=src, group=group, tag=tag + ACK_TAG_OFFSET)
        _tlog.info(
            "[T18][rank%d] BLOCKING recv slot=%d tag=%d <- rank%d: %s in %s "
            "(sync path)",
            self.rank, slot, tag, src, str(tuple(tensor.shape)),
            _fmt_ms(time.perf_counter() - _t0),
        )

    def isend(self, tensor, dst, group=None, tag: int = 0) -> AsyncHandle:
        """
        Non-blocking send:
          - QUEUE the D2H copy on a side copy-stream and return immediately.
            No torch.cuda.synchronize() here — the default (compute) stream
            is free to keep running the next microbatch's forward pass while
            this copy happens in the background.
          - handle.wait() (called later, after other work) drains the copy,
            flushes to SHM, sends the handshake, and waits the ACK.
        """
        slot = self._get_free_slot()
        _t0 = time.perf_counter()
        numel = self._queue_copy_to_staging(tensor, slot)
        _queue_t = time.perf_counter() - _t0

        # This is the headline overlap check: queueing must be ~0ms. If this
        # is as large as the copy itself, isend() is still blocking and the
        # async path is not in effect.
        _tlog.info(
            "[T05][rank%d] isend slot=%d tag=%d -> rank%d: %d elems (%s) "
            "D2H queued in %s (should be ≈0 — non-blocking)",
            self.rank, slot, tag, dst, numel,
            str(tuple(tensor.shape)), _fmt_ms(_queue_t),
        )

        return AsyncHandle(slot, self, dst=dst, tag=tag, group=group, numel=numel)

    def irecv(self, tensor, src, group=None, tag: int = 0) -> AsyncHandleRecv:
        """
        Non-blocking recv:
          - irecv handshake(slot)
          - handle.wait() reads SHM into tensor and sends ACK
        """
        handshake = torch.tensor([0], dtype=torch.int32, device="cpu")
        dist_handle = _ORIGINAL_IRECV(handshake, src=src, group=group, tag=tag)
        _tlog.debug(
            "[T12][rank%d] irecv posted tag=%d <- rank%d, dst_shape=%s",
            self.rank, tag, src, str(tuple(tensor.shape)),
        )
        return AsyncHandleRecv(
            dist_handle, handshake, tensor, src, self, tag=tag, group=group
        )

    # -----------------------------
    # SHM read/write helpers
    # -----------------------------

    def _write_tensor_to_slot(self, tensor: torch.Tensor, slot: int):
        """
        Blocking GPU -> CPU -> SHM path, used only by the synchronous
        send()/recv() (blocking API). Kept as the old, simple, correct
        (if slow) reference path — isend()/irecv() use the async path below
        instead and never call this.
        """
        nbytes = tensor.numel() * tensor.element_size()

        if nbytes > self.slot_size:
            raise ValueError(
                f"Tensor {nbytes} bytes exceeds slot size {self.slot_size} bytes. "
                f"Increase buffer_size_mb."
            )

        if tensor.dtype == torch.float16:
            numel = tensor.numel()
            staging = self.pinned_staging[slot]
            staging[:numel].copy_(tensor.view(-1), non_blocking=True)
            torch.cuda.synchronize()
            raw = staging[:numel].numpy().view(np.uint8)
            self.my_np_slots[slot][:nbytes] = raw[:nbytes]
        else:
            cpu_tensor = tensor.detach().cpu().contiguous()
            raw = cpu_tensor.numpy().reshape(-1).view(np.uint8)
            self.my_np_slots[slot][:nbytes] = raw[:nbytes]

    # -----------------------------
    # Async sender path (isend/AsyncHandle)
    # -----------------------------

    def _queue_copy_to_staging(self, tensor: torch.Tensor, slot: int) -> int:
        """
        Queue the D2H copy on self.copy_stream and return immediately —
        does NOT wait for it to finish. This is the fix for the old
        torch.cuda.synchronize() stall: that call froze the ENTIRE device;
        this only queues work on a side stream, so the default (compute)
        stream is free to move on to the next microbatch's forward pass
        right away. AsyncHandle.wait() is what actually waits for this copy,
        called later (after other work), which is where the overlap comes
        from.

        Only the float16 fast path is async; other dtypes (rare — control
        tensors, not activations) fall back to the old blocking path.
        """
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.slot_size:
            raise ValueError(
                f"Tensor {nbytes} bytes exceeds slot size {self.slot_size} bytes. "
                f"Increase buffer_size_mb."
            )

        if tensor.dtype != torch.float16:
            # Fall back to the synchronous path for non-fp16 control tensors.
            _tlog.debug(
                "[T20][rank%d] send slot=%d: non-fp16 dtype=%s — BLOCKING "
                "fallback path (expected only for small control tensors)",
                self.rank, slot, tensor.dtype,
            )
            self._write_tensor_to_slot(tensor, slot)
            self.slot_events[slot].record(torch.cuda.current_stream())
            return tensor.numel()

        numel = tensor.numel()
        staging = self.pinned_staging[slot]

        # The copy must not start reading `tensor` before the compute that
        # produced it has finished. wait_stream makes copy_stream wait for
        # everything already queued on the current (compute) stream up to
        # this point — an ordering dependency, not a full sync.
        self.copy_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.copy_stream):
            staging[:numel].copy_(tensor.view(-1), non_blocking=True)

        # Tell the allocator: `tensor`'s memory is still being read by
        # copy_stream. Without this, the caller's tensor.clone() can be
        # freed and its memory block reused by the NEXT microbatch's
        # compute (on the default stream) while copy_stream is still
        # mid-copy — silent data corruption, no error, wrong tokens later.
        tensor.record_stream(self.copy_stream)

        # Mark completion of THIS slot's copy on copy_stream. wait()
        # later calls slot_events[slot].synchronize() to wait on exactly
        # this copy — not any other slot's, not the whole device.
        self.slot_events[slot].record(self.copy_stream)

        return numel

    def _flush_staging_to_shm(self, slot: int, numel: int) -> None:
        """
        Called from AsyncHandle.wait(), AFTER slot_events[slot].synchronize()
        has confirmed the D2H copy landed. Does the CPU-side memcpy from
        pinned staging into the SHM slot so the receiving process can see it.
        """
        nbytes = numel * 2  # float16 = 2 bytes
        raw = self.pinned_staging[slot][:numel].numpy().view(np.uint8)
        self.my_np_slots[slot][:nbytes] = raw[:nbytes]

    # -----------------------------
    # Async receiver path (irecv/AsyncHandleRecv)
    # -----------------------------

    def _queue_read_from_slot(self, tensor: torch.Tensor, src: int, slot: int):
        """
        Queue the H2D copy on self.recv_stream and return a CUDA event the
        caller waits on. Mirrors _queue_copy_to_staging on the sender side:
        memcpy SHM into pinned recv staging (cheap CPU work, stays inline),
        then queue the PCIe H2D transfer on recv_stream so it can overlap
        with work still outstanding on the default stream.

        Returns the event to wait on, rather than indexing a shared array:
        `slot` is an index into the SENDER's slot pool, so using it to index
        a receiver-side array is only safe by coincidence (equal lengths,
        single upstream peer). Returning the event removes that coupling.
        """
        nbytes = tensor.numel() * tensor.element_size()

        if tensor.dtype != torch.float16:
            # Fall back to the synchronous path for non-fp16 control tensors.
            # Already synchronous, so hand back an event that is trivially
            # complete on the current stream.
            _tlog.debug(
                "[T20b][rank%d] recv slot=%d: non-fp16 dtype=%s — BLOCKING "
                "fallback path",
                self.rank, slot, tensor.dtype,
            )
            self._read_tensor_from_slot(tensor, src, slot)
            evt = torch.cuda.Event()
            evt.record(torch.cuda.current_stream())
            return evt

        numel = tensor.numel()

        # Grab a receiver-side staging buffer + event from our OWN pool,
        # keyed by our own free-slot bookkeeping, not the sender's index.
        rslot = self._get_free_recv_slot()
        staging = self.recv_pinned_staging[rslot]
        evt = self.recv_events[rslot]

        # SHM -> pinned staging, one host memcpy. np.frombuffer gives a
        # zero-copy view of SHM; copy_ from it writes straight into pinned
        # memory. (The previous .copy() here made a throwaway intermediate
        # host buffer on every microbatch.)
        peer_view = np.frombuffer(
            self.peer_slots[src][slot][:nbytes], dtype=np.float16
        )
        staging[:numel].copy_(torch.from_numpy(peer_view))

        with torch.cuda.stream(self.recv_stream):
            tensor.copy_(staging[:numel].view(tensor.shape), non_blocking=True)

        # tensor is the destination (caller-owned recv buffer reused across
        # steps) — record_stream so the allocator does not hand its memory
        # to other work while recv_stream is still writing into it.
        tensor.record_stream(self.recv_stream)
        evt.record(self.recv_stream)

        # The staging buffer is in use until the H2D lands; release it once
        # the caller has synchronized on evt.
        self._recv_slot_pending[rslot] = evt
        return evt

    def _read_tensor_from_slot(self, tensor: torch.Tensor, src: int, slot: int):
        """
        Blocking SHM -> CPU -> GPU path, used only by the synchronous
        send()/recv() (blocking API) and as the non-fp16 fallback.
        """
        nbytes = tensor.numel() * tensor.element_size()

        np_dtype = _TORCH_TO_NUMPY.get(tensor.dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype for SHM transport: {tensor.dtype}")

        raw_bytes = self.peer_slots[src][slot][:nbytes]
        peer_data = np.frombuffer(raw_bytes, dtype=np_dtype)

        # copy() so numpy buffer doesn’t alias SHM as tensor lives beyond scope
        src_tensor = torch.from_numpy(peer_data.copy()).reshape(tensor.shape)
        # PCIE crossing
        tensor.copy_(src_tensor.to(tensor.device))


    # -----------------------------
    # Instrumentation summary
    # -----------------------------

    def log_summary(self, label: str = "") -> dict:
        """
        Emit the per-rank verdict on whether the async transport actually
        overlapped. Call once at the end of a run, per rank.

        The numbers that matter:
          flush_block  — mean time blocked waiting for D2H copies. Near zero
                         means the copy hid under compute (the fix worked).
                         Large means it did not.
          h2d_block    — mean time blocked waiting for H2D copies on receive.
          handshake    — mean time waiting for upstream. Large here is normal
                         and just means this rank is faster than its upstream.
          ack_wait     — mean time waiting for downstream ACKs.
        """
        def _mean(total, count):
            return (total / count * 1000.0) if count else 0.0

        stats = {
            "rank": self.rank,
            "label": label,
            "flush_block_mean_ms": _mean(self.stat_flush_block_s, self.stat_flush_count),
            "flush_block_total_ms": self.stat_flush_block_s * 1000.0,
            "flush_count": self.stat_flush_count,
            "h2d_block_mean_ms": _mean(self.stat_h2d_block_s, self.stat_h2d_count),
            "h2d_block_total_ms": self.stat_h2d_block_s * 1000.0,
            "h2d_count": self.stat_h2d_count,
            "handshake_wait_mean_ms": _mean(
                self.stat_handshake_wait_s, self.stat_handshake_count
            ),
            "handshake_count": self.stat_handshake_count,
            "ack_wait_mean_ms": _mean(self.stat_ack_wait_s, self.stat_ack_count),
            "ack_count": self.stat_ack_count,
            "slot_spins": self.stat_slot_spins,
        }

        _tlog.info(
            "[T22][rank%d] ===== TRANSPORT SUMMARY %s =====", self.rank, label
        )
        _tlog.info(
            "[T23][rank%d] D2H flush block: mean=%.2fms total=%.1fms over %d sends",
            self.rank, stats["flush_block_mean_ms"],
            stats["flush_block_total_ms"], stats["flush_count"],
        )
        _tlog.info(
            "[T24][rank%d] H2D recv block:  mean=%.2fms total=%.1fms over %d recvs",
            self.rank, stats["h2d_block_mean_ms"],
            stats["h2d_block_total_ms"], stats["h2d_count"],
        )
        _tlog.info(
            "[T25][rank%d] handshake wait:  mean=%.2fms over %d recvs "
            "(large = upstream slower than us, not a transport problem)",
            self.rank, stats["handshake_wait_mean_ms"], stats["handshake_count"],
        )
        _tlog.info(
            "[T26][rank%d] ACK wait:        mean=%.2fms over %d sends",
            self.rank, stats["ack_wait_mean_ms"], stats["ack_count"],
        )
        _tlog.info(
            "[T27][rank%d] slot pool spins: %d (nonzero = pool undersized)",
            self.rank, stats["slot_spins"],
        )

        # Verdict line — the one thing to grep for.
        if self.stat_flush_count == 0:
            _tlog.info("[T28][rank%d] VERDICT: no async sends on this rank", self.rank)
        elif stats["flush_block_mean_ms"] < 1.0:
            _tlog.info(
                "[T29][rank%d] VERDICT: OVERLAP WORKING — D2H hidden under "
                "compute (mean block %.2fms)",
                self.rank, stats["flush_block_mean_ms"],
            )
        else:
            _tlog.warning(
                "[T30][rank%d] VERDICT: POOR OVERLAP — blocked %.2fms/send on "
                "D2H. Copy is not hiding under compute; check that isend() "
                "queue time (T05) is ~0 and that flush() is called one "
                "microbatch behind.",
                self.rank, stats["flush_block_mean_ms"],
            )

        return stats


# --- MODULE LEVEL FUNCTIONS ---


def setup_transport_logging(log_dir: str = "logs", run_label: str = "") -> str:
    """
    Configure the transport logger to write to a timestamped file.

    Returns the log file path. Call this ONCE PER PROCESS (i.e. inside each
    rank's worker), before register_hooks(). Each rank appends to the same
    file; the rank is in every line, so `grep 'rank2' <file>` separates them.

    The timestamp is generated by the PARENT and passed in via run_label so
    all four ranks land in one file. If run_label is empty, each process
    would otherwise create its own file, so a shared env var is used.
    """
    os.makedirs(log_dir, exist_ok=True)

    # All ranks must agree on the filename. The parent sets this env var
    # before spawning; if unset (standalone use), fall back to now().
    stamp = os.environ.get("MIG_LOG_STAMP")
    if not stamp:
        stamp = time.strftime("%Y%m%d_%H%M%S")

    suffix = f"_{run_label}" if run_label else ""
    log_path = os.path.join(log_dir, f"transport_{stamp}{suffix}.log")

    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    _tlog.handlers.clear()
    _tlog.addHandler(handler)
    _tlog.setLevel(
        logging.DEBUG if os.environ.get("MIG_LOG_DEBUG") else logging.INFO
    )
    # Don't double-log into the root logger's benchmark.log
    _tlog.propagate = False

    return log_path


def log_summary(label: str = "") -> Optional[dict]:
    """Module-level convenience wrapper around the engine's log_summary."""
    if _MIG_PIPE_ENGINE is None:
        return None
    return _MIG_PIPE_ENGINE.log_summary(label)


def register_hooks(mb_size=32, seq_len=64, hidden_size=5120, num_slots=None):
    global _MIG_PIPE_ENGINE

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if _MIG_PIPE_ENGINE is None:
        # Size slots to the mb_size this run actually uses, and allocate only
        # the slots this run can have in flight (num_microbatches) plus
        # headroom — not the global worst case. Host RAM cost is
        # num_slots x slot_mb x 2 (SHM + pinned) x 4 ranks.
        _MIG_PIPE_ENGINE = MIGPipelineTransport(
            rank,
            world_size,
            buffer_size_mb=_compute_slot_size_mb(
                hidden_size=hidden_size, max_mb_size=mb_size, max_seq_len=seq_len
            ),
            num_slots=num_slots if num_slots is not None else NUM_SLOTS,
        )

    # Patch both blocking and non-blocking variants
    dist.send = patched_send
    dist.recv = patched_recv
    dist.isend = patched_isend
    dist.irecv = patched_irecv

    torch.distributed.send = patched_send
    torch.distributed.recv = patched_recv
    torch.distributed.isend = patched_isend
    torch.distributed.irecv = patched_irecv

    _tlog.info(
        "[T21][rank%d] hooks registered (send/recv/isend/irecv patched), "
        "mb_size=%d seq_len=%d hidden=%d",
        rank, mb_size, seq_len, hidden_size,
    )
    print(f"[MIG-Pipe] Hooks registered for Rank {rank} (ACK + tags)", flush=True)


def patched_send(tensor, dst, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        return _MIG_PIPE_ENGINE.send(tensor, dst, group=group, tag=tag)
    return _ORIGINAL_SEND(tensor, dst, group=group, tag=tag)


def patched_recv(tensor, src=None, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        return _MIG_PIPE_ENGINE.recv(tensor, src, group=group, tag=tag)
    return _ORIGINAL_RECV(tensor, src, group=group, tag=tag)


def patched_isend(tensor, dst, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        return _MIG_PIPE_ENGINE.isend(tensor, dst, group=group, tag=tag)
    return _ORIGINAL_ISEND(tensor, dst, group=group, tag=tag)


def patched_irecv(tensor, src=None, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        return _MIG_PIPE_ENGINE.irecv(tensor, src, group=group, tag=tag)
    return _ORIGINAL_IRECV(tensor, src, group=group, tag=tag)
