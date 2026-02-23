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


class MIGPipelineTransport:

    def __init__(self, rank, world_size, buffer_size_mb=64):
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size_mb * 1024 * 1024
        self.peer_buffers = {}

        print(f"[MIG-Pipe] Rank {rank} initializing Pipeline SHM...")

        # 1. Cleanup old shared memory
        my_name = f"mig_pipe_shm_{rank}"
        try:
            if os.path.exists(f"/dev/shm/{my_name}"):
                os.unlink(f"/dev/shm/{my_name}")
        except Exception:
            pass

        # 2. Create My Write Buffer
        try:
            self.my_shm = SharedMemory(name=my_name, create=True, size=self.buffer_size)
        except FileExistsError:
            self.my_shm = SharedMemory(
                name=my_name, create=False, size=self.buffer_size
            )

        _MIG_SHM_HANDLES.append(self.my_shm)

        self.my_buffer_np = np.ndarray(
            (self.buffer_size,), dtype=np.uint8, buffer=self.my_shm.buf
        )

        dist.barrier()

        # 3. Connect to Peers
        for peer_rank in range(world_size):
            if peer_rank == rank:
                continue

            peer_name = f"mig_pipe_shm_{peer_rank}"
            connected = False
            attempts = 0

            while not connected and attempts < 1000:
                try:
                    shm = SharedMemory(
                        name=peer_name, create=False, size=self.buffer_size
                    )
                    _MIG_SHM_HANDLES.append(shm)

                    self.peer_buffers[peer_rank] = np.ndarray(
                        (self.buffer_size,), dtype=np.uint8, buffer=shm.buf
                    )
                    connected = True
                except FileNotFoundError:
                    time.sleep(0.01)
                    attempts += 1

            if not connected:
                raise RuntimeError(f"Rank {rank} failed to connect to {peer_name}")

        print(f"[MIG-Pipe] Rank {rank} Connected. Pipeline Ready.")
        dist.barrier()

    def _tensor_to_bytes(self, tensor):
        cpu_view = tensor.detach().cpu().view(-1)
        dtype = cpu_view.numpy().dtype
        tensor_bytes = cpu_view.numpy().view(np.uint8)
        return tensor_bytes, dtype

    def send(self, tensor, dst):
        """
        P2P Send:
        1. Write to Shared Memory
        2. Send tiny 'token' via TCP to signal completion.
        """
        tensor_bytes, _ = self._tensor_to_bytes(tensor)
        nbytes = len(tensor_bytes)

        if nbytes > self.buffer_size:
            raise ValueError(f"Tensor size {nbytes} exceeds buffer {self.buffer_size}")

        # 1. Write Data to SHM
        self.my_buffer_np[:nbytes] = tensor_bytes[:]

        # 2. Send "Handshake" (Signal that write is done)
        # We send a dummy byte. The receiver blocks until they get this.
        # This acts as a localized barrier for just these two ranks.
        handshake = torch.tensor([1], dtype=torch.uint8, device="cpu")
        _ORIGINAL_SEND(handshake, dst)

    def recv(self, tensor, src):
        """
        P2P Recv:
        1. Wait for 'token' from Sender (TCP blocking).
        2. Read from Shared Memory.
        """
        # 1. Wait for "Handshake" (Signal that data is ready)
        handshake = torch.tensor([0], dtype=torch.uint8, device="cpu")
        _ORIGINAL_RECV(handshake, src)

        # 2. Read Data from SHM
        target_dtype = tensor.cpu().numpy().dtype
        nbytes = tensor.numel() * tensor.element_size()

        raw_bytes = self.peer_buffers[src][:nbytes]
        peer_data_flat = np.frombuffer(raw_bytes, dtype=target_dtype)

        # 3. Reshape and Copy
        src_tensor = torch.from_numpy(peer_data_flat).reshape(tensor.shape)
        tensor.copy_(src_tensor.to(tensor.device))


# --- MODULE LEVEL FUNCTIONS ---


def register_hooks():
    global _MIG_PIPE_ENGINE
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if _MIG_PIPE_ENGINE is None:
        _MIG_PIPE_ENGINE = MIGPipelineTransport(rank, world_size)

    dist.send = patched_send
    dist.recv = patched_recv
    torch.distributed.send = patched_send
    torch.distributed.recv = patched_recv
    print(f"[MIG-Pipe] Hooks registered for Rank {rank}")


def patched_send(tensor, dst, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        _MIG_PIPE_ENGINE.send(tensor, dst)
        return
    return _ORIGINAL_SEND(tensor, dst, group, tag)


def patched_recv(tensor, src, group=None, tag=0):
    if _MIG_PIPE_ENGINE is not None:
        _MIG_PIPE_ENGINE.recv(tensor, src)
        return
    return _ORIGINAL_RECV(tensor, src, group, tag)
