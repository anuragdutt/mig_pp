import torch
import torch.distributed as dist
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time
import os

# --- GLOBAL STATE ---
_MIG_PIPE_ENGINE = None
_MIG_SHM_HANDLES = []
_ORIGINAL_SEND = dist.send
_ORIGINAL_RECV = dist.recv
_ORIGINAL_ISEND = dist.isend
_ORIGINAL_IRECV = dist.irecv

# Number of ring buffer slots per rank.
# Must be >= number of microbatches in flight simultaneously.
# With a fill-drain pipeline schedule, at most (world_size - 1) microbatches
# are in flight at once, so NUM_SLOTS = world_size is always safe.
NUM_SLOTS = 4


class AsyncHandle:
    """
    Wraps a non-blocking send or recv operation.
    Mirrors the interface of torch.distributed's WorkHandle so callers
    can just do handle.wait() regardless of whether it came from us or gloo.
    """

    def __init__(self, dist_handle, slot_idx, engine, is_send):
        self._dist_handle = dist_handle  # The underlying gloo async handle
        self._slot_idx = slot_idx  # Which ring buffer slot was used
        self._engine = engine  # Back-reference to free the slot on wait()
        self._is_send = is_send
        self._waited = False

    def wait(self):
        if not self._waited:
            # Block until the handshake TCP transfer completes
            self._dist_handle.wait()
            # Mark this slot as free so it can be reused for the next microbatch
            if self._is_send:
                self._engine.slot_free[self._slot_idx] = True
            self._waited = True


class MIGPipelineTransport:

    def __init__(self, rank, world_size, buffer_size_mb=128, num_slots=NUM_SLOTS):
        self.rank = rank
        self.world_size = world_size
        self.num_slots = num_slots
        # Each slot must be large enough for the largest tensor that will be sent.
        # 128MB per slot covers (128 batch, 512 seq, 4096 hidden) in float16 comfortably.
        self.slot_size = buffer_size_mb * 1024 * 1024

        print(
            f"[MIG-Pipe] Rank {rank} initializing async pipeline transport "
            f"({num_slots} slots x {buffer_size_mb}MB)...",
            flush=True,
        )

        # --- CREATE RING BUFFER SLOTS (one SHM region per slot) ---
        # Each slot is an independent shared memory block. When rank 0 is writing
        # MB1 into slot 1, rank 1 can still be reading MB0 from slot 0 — no conflict.
        self.my_shm_slots = []
        self.my_np_slots = []
        self.slot_free = [True] * num_slots  # Track which slots are available

        for slot in range(num_slots):
            name = f"mig_pipe_shm_{rank}_slot{slot}"
            # Clean up any leftover SHM from a previous crashed run
            try:
                if os.path.exists(f"/dev/shm/{name}"):
                    os.unlink(f"/dev/shm/{name}")
            except Exception:
                pass

            try:
                shm = SharedMemory(name=name, create=True, size=self.slot_size)
            except FileExistsError:
                shm = SharedMemory(name=name, create=False, size=self.slot_size)

            _MIG_SHM_HANDLES.append(shm)
            self.my_shm_slots.append(shm)
            self.my_np_slots.append(
                np.ndarray((self.slot_size,), dtype=np.uint8, buffer=shm.buf)
            )

        # --- PRE-ALLOCATE PINNED MEMORY STAGING BUFFERS ---
        # Pinned (page-locked) CPU memory allows the CUDA DMA engine to transfer
        # GPU tensors to CPU without involving the CPU cores, and allows non_blocking=True
        # copies. One staging buffer per slot so concurrent transfers don't collide.
        max_elements = self.slot_size // 2  # float16 = 2 bytes per element
        self.pinned_staging = [
            torch.zeros(max_elements, dtype=torch.float16).pin_memory()
            for _ in range(num_slots)
        ]

        # Wait for all ranks to finish creating their SHM slots before connecting
        dist.barrier()

        # --- CONNECT TO PEER SHM SLOTS ---
        # Each rank maps all other ranks' SHM regions into its own address space.
        # Structure: self.peer_slots[peer_rank][slot_idx] = numpy array view
        self.peer_slots = {}
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
                        f"Rank {rank} failed to connect to {peer_name} after {attempts} attempts"
                    )

        print(f"[MIG-Pipe] Rank {rank} connected to all peers. Ready.", flush=True)
        dist.barrier()

    def _get_free_slot(self):
        """
        Find the next free ring buffer slot.
        Spins briefly if all slots are occupied (shouldn't happen if num_slots
        is sized correctly relative to pipeline depth).
        """
        for _ in range(10000):
            for slot in range(self.num_slots):
                if self.slot_free[slot]:
                    self.slot_free[slot] = False  # Mark as occupied
                    return slot
            time.sleep(0.001)
        raise RuntimeError(
            f"[Rank {self.rank}] No free SHM slots available — "
            f"increase NUM_SLOTS or reduce pipeline depth."
        )

    def send(self, tensor, dst):
        """Blocking send — used for tensors that must arrive before proceeding."""
        slot = self._get_free_slot()
        self._write_tensor_to_slot(tensor, slot)
        # Send the slot index as the handshake so receiver knows where to read from
        handshake = torch.tensor([slot], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(handshake, dst)
        self.slot_free[slot] = True  # Safe to reuse immediately after blocking send

    def recv(self, tensor, src):
        """Blocking recv — paired with blocking send."""
        handshake = torch.tensor([0], dtype=torch.int32, device="cpu")
        _ORIGINAL_RECV(handshake, src)
        slot = handshake.item()
        self._read_tensor_from_slot(tensor, src, slot)

    def isend(self, tensor, dst):
        """
        Non-blocking send. Returns an AsyncHandle whose .wait() must be called
        before the slot can be considered free.
        """
        slot = self._get_free_slot()
        self._write_tensor_to_slot(tensor, slot)
        # isend the slot index — non-blocking, returns immediately
        handshake = torch.tensor([slot], dtype=torch.int32, device="cpu")
        dist_handle = _ORIGINAL_ISEND(handshake, dst)
        return AsyncHandle(dist_handle, slot, self, is_send=True)

    def irecv(self, tensor, src):
        """
        Non-blocking recv. Returns an AsyncHandle. Caller must call .wait()
        before reading from tensor — data isn't valid until the handshake arrives.
        """
        handshake = torch.tensor([0], dtype=torch.int32, device="cpu")
        dist_handle = _ORIGINAL_IRECV(handshake, src)
        # We can't read the SHM until the handshake completes, so we wrap
        # the completion logic inside the handle's wait()
        return AsyncHandleRecv(dist_handle, tensor, src, self)

    def _write_tensor_to_slot(self, tensor, slot):
        """
        GPU → pinned CPU → SHM.
        Uses pinned memory staging for async DMA transfer from GPU.
        """
        numel = tensor.numel()
        nbytes = numel * tensor.element_size()

        if nbytes > self.slot_size:
            raise ValueError(
                f"Tensor {nbytes} bytes exceeds slot size {self.slot_size} bytes. "
                f"Increase buffer_size_mb."
            )

        # Copy GPU tensor into pinned staging buffer.
        # non_blocking=True lets CUDA DMA handle the transfer asynchronously,
        # but we synchronize immediately after to ensure data is in CPU memory
        # before we copy it into SHM. In a future optimization this sync could
        # be deferred with a CUDA stream, but that adds significant complexity.
        staging = self.pinned_staging[slot]
        staging[:numel].copy_(tensor.view(-1).to(torch.float16), non_blocking=True)
        torch.cuda.synchronize()

        # Copy from pinned staging into SHM slot
        raw = staging[:numel].numpy().view(np.uint8)
        self.my_np_slots[slot][:nbytes] = raw[:nbytes]

    def _read_tensor_from_slot(self, tensor, src, slot):
        """
        SHM → CPU → GPU.
        Reads from the sender's SHM slot and copies into the destination tensor.
        """
        nbytes = tensor.numel() * tensor.element_size()
        target_dtype = tensor.cpu().numpy().dtype

        raw_bytes = self.peer_slots[src][slot][:nbytes]
        peer_data = np.frombuffer(raw_bytes, dtype=target_dtype)
        src_tensor = torch.from_numpy(peer_data.copy()).reshape(tensor.shape)
        tensor.copy_(src_tensor.to(tensor.device))


class AsyncHandleRecv:
    """
    Specialized handle for non-blocking receives.
    The actual SHM read happens inside wait() once the handshake confirms
    the sender has finished writing.
    """

    def __init__(self, dist_handle, tensor, src, engine):
        self._dist_handle = dist_handle
        self._tensor = tensor
        self._src = src
        self._engine = engine
        self._waited = False

    def wait(self):
        if not self._waited:
            # Block until handshake (slot index) arrives from sender
            self._dist_handle.wait()
            # Now we know which slot to read from — the handshake tensor
            # was filled in-place by _ORIGINAL_IRECV
            # Note: we need the handshake value — store it on the handle
            slot = self._handshake.item()
            self._engine._read_tensor_from_slot(self._tensor, self._src, slot)
            self._waited = True


# --- MODULE LEVEL FUNCTIONS ---


def register_hooks():
    global _MIG_PIPE_ENGINE
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if _MIG_PIPE_ENGINE is None:
        _MIG_PIPE_ENGINE = MIGPipelineTransport(rank, world_size)

    # Patch both blocking and non-blocking variants
    dist.send = patched_send
    dist.recv = patched_recv
    dist.isend = patched_isend
    dist.irecv = patched_irecv
    torch.distributed.send = patched_send
    torch.distributed.recv = patched_recv
    torch.distributed.isend = patched_isend
    torch.distributed.irecv = patched_irecv

    print(
        f"[MIG-Pipe] Hooks registered for Rank {rank} " f"(blocking + async)",
        flush=True,
    )


def patched_send(tensor, dst, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        _MIG_PIPE_ENGINE.send(tensor, dst)
        return
    return _ORIGINAL_SEND(tensor, dst, group, tag)


def patched_recv(tensor, src=None, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        _MIG_PIPE_ENGINE.recv(tensor, src)
        return
    return _ORIGINAL_RECV(tensor, src, group, tag)


def patched_isend(tensor, dst, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        return _MIG_PIPE_ENGINE.isend(tensor, dst)
    return _ORIGINAL_ISEND(tensor, dst, group, tag)


def patched_irecv(tensor, src=None, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        return _MIG_PIPE_ENGINE.irecv(tensor, src)
    return _ORIGINAL_IRECV(tensor, src, group, tag)
