import os
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

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
# Must be >= max microbatches in flight at once.
NUM_SLOTS = 10

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


# When you use non-blocking communication (isend/irecv), PyTorch returns a "handle" you can wait on later.
# Because this script overrides the backend, it must provide custom handles.
class AsyncHandle:
    """
    Async send handle.
    IMPORTANT: slot becomes reusable only after receiver ACKs it has read SHM.
    """

    def __init__(self, dist_handle, slot_idx, engine, dst, tag, group):
        self._dist_handle = dist_handle
        self._slot_idx = slot_idx
        self._engine = engine
        self._dst = dst
        self._tag = tag
        self._group = group
        self._waited = False

    # sender side wait
    def wait(self):
        if self._waited:
            return

        # 1) Wait until handshake send finishes (receiver got slot index)
        self._dist_handle.wait()

        # 2) Wait for ACK from receiver: "I have finished reading SHM slot"
        ack = torch.empty((1,), dtype=torch.int32, device="cpu")
        _ORIGINAL_RECV(
            ack,
            src=self._dst,
            group=self._group,
            tag=self._tag + ACK_TAG_OFFSET,
        )

        # Now safe to reuse slot
        self._engine.slot_free[self._slot_idx] = True
        self._waited = True


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

        # Wait until handshake arrives (slot index)
        self._dist_handle.wait()
        slot = int(self._handshake.item())

        # Read data from SHM slot into destination tensor
        self._engine._read_tensor_from_slot(self._tensor, self._src, slot)

        # Send ACK back so sender can reuse slot
        ack = torch.tensor([slot], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(
            ack,
            dst=self._src,
            group=self._group,
            tag=self._tag + ACK_TAG_OFFSET,
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

        print(f"[MIG-Pipe] Rank {rank} connected to all peers. Ready.", flush=True)

        # Nobody is allowed to start the actual workday (sending boxes) until everyone has finished mapping the building!
        dist.barrier()

    def _get_free_slot(self) -> int:
        for _ in range(10000):
            for slot in range(self.num_slots):
                if self.slot_free[slot]:
                    self.slot_free[slot] = False
                    return slot
            time.sleep(0.001)

        raise RuntimeError(
            f"[Rank {self.rank}] No free SHM slots available — "
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
        slot = self._get_free_slot()
        self._write_tensor_to_slot(tensor, slot)

        # Handshake is shared using original pytorch backend ie gloo
        handshake = torch.tensor([slot], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(handshake, dst=dst, group=group, tag=tag)

        # ACK ensures receiver finished reading this SHM slot
        ack = torch.empty((1,), dtype=torch.int32, device="cpu")
        _ORIGINAL_RECV(ack, src=dst, group=group, tag=tag + ACK_TAG_OFFSET)

        self.slot_free[slot] = True

    def recv(self, tensor, src, group=None, tag: int = 0):
        """
        Blocking recv:
          - recv handshake(slot)
          - read SHM into tensor
          - send ACK
        """
        handshake = torch.tensor([0], dtype=torch.int32, device="cpu")
        _ORIGINAL_RECV(handshake, src=src, group=group, tag=tag)
        slot = int(handshake.item())

        self._read_tensor_from_slot(tensor, src, slot)

        ack = torch.tensor([slot], dtype=torch.int32, device="cpu")
        _ORIGINAL_SEND(ack, dst=src, group=group, tag=tag + ACK_TAG_OFFSET)

    def isend(self, tensor, dst, group=None, tag: int = 0) -> AsyncHandle:
        """
        Non-blocking send:
          - write SHM now
          - isend handshake(slot)
          - handle.wait() will wait ACK and free the slot
        """
        slot = self._get_free_slot()
        self._write_tensor_to_slot(tensor, slot)

        handshake = torch.tensor([slot], dtype=torch.int32, device="cpu")
        dist_handle = _ORIGINAL_ISEND(handshake, dst=dst, group=group, tag=tag)

        return AsyncHandle(dist_handle, slot, self, dst=dst, tag=tag, group=group)

    def irecv(self, tensor, src, group=None, tag: int = 0) -> AsyncHandleRecv:
        """
        Non-blocking recv:
          - irecv handshake(slot)
          - handle.wait() reads SHM into tensor and sends ACK
        """
        handshake = torch.tensor([0], dtype=torch.int32, device="cpu")
        dist_handle = _ORIGINAL_IRECV(handshake, src=src, group=group, tag=tag)
        return AsyncHandleRecv(
            dist_handle, handshake, tensor, src, self, tag=tag, group=group
        )

    # -----------------------------
    # SHM read/write helpers
    # -----------------------------

    def _write_tensor_to_slot(self, tensor: torch.Tensor, slot: int):
        """
        GPU → CPU → SHM for float16 (via pinned staging).
        Other dtypes copied safely CPU-contiguous → bytes.
        """
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.slot_size:
            raise ValueError(
                f"Tensor {nbytes} bytes exceeds slot size {self.slot_size} bytes. "
                f"Increase buffer_size_mb."
            )

        if tensor.dtype == torch.float16:
            # For float16 we have pinned_staging (fast cart)
            # Loading into fast cart
            numel = tensor.numel()
            staging = self.pinned_staging[slot]
            staging[:numel].copy_(tensor.view(-1), non_blocking=True)

            # Stop right there. Freeze.
            # Do not execute another line of Python code
            # until the GPU confirms it has 100% finished all of its pending tasks
            torch.cuda.synchronize()

            # Writing into memory from the fast cart
            raw = staging[:numel].numpy().view(np.uint8)
            self.my_np_slots[slot][:nbytes] = raw[:nbytes]
        else:
            # Slow path move from GPU to CPU
            cpu_tensor = tensor.detach().cpu().contiguous()
            raw = cpu_tensor.numpy().reshape(-1).view(np.uint8)
            self.my_np_slots[slot][:nbytes] = raw[:nbytes]

    def _read_tensor_from_slot(self, tensor: torch.Tensor, src: int, slot: int):
        """
        SHM → CPU → GPU. Uses dtype map without forcing tensor.cpu().
        """
        nbytes = tensor.numel() * tensor.element_size()

        np_dtype = _TORCH_TO_NUMPY.get(tensor.dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype for SHM transport: {tensor.dtype}")

        raw_bytes = self.peer_slots[src][slot][:nbytes]
        peer_data = np.frombuffer(raw_bytes, dtype=np_dtype)

        # copy() so numpy buffer doesn’t alias SHM as tensor lives beyond scope
        src_tensor = torch.from_numpy(peer_data.copy()).reshape(tensor.shape)
        tensor.copy_(src_tensor.to(tensor.device))


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
