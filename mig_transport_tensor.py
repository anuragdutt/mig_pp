import torch
import torch.distributed as dist
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time
import os

# --- GLOBAL STATE ---
_MIG_ENGINE = None
_MIG_SHM_HANDLES = []  # Prevent Garbage Collection
_ORIGINAL_ALL_REDUCE = dist.all_reduce


class MIGTransport:

    def __init__(self, rank, world_size, buffer_size_mb=64):
        self.rank = rank
        # Integer ID of this process
        # Example: rank = 0, 1, 2, ...

        self.world_size = world_size
        # Total number of processes participating
        # Example: world_size = 4

        self.buffer_size = buffer_size_mb * 1024 * 1024
        # Convert MB → bytes
        # Example:
        #   buffer_size_mb = 64
        #   buffer_size = 64 * 1024 * 1024 = 67,108,864 bytes (64MB)

        self.peer_buffers = {}
        # Dictionary mapping:
        #   peer_rank -> numpy uint8 shared memory view
        # Example:
        #   {1: <np array>, 2: <np array>, 3: <np array>}

        print(f"[MIG-Mesh] Rank {rank} initializing SHM mesh...")

        # -----------------------------------------
        # 1. Cleanup old shared memory file
        # -----------------------------------------

        my_name = f"mig_shm_{rank}"
        # Unique shared memory name per rank
        # Example:
        #   rank 0 -> "mig_shm_0"
        #   rank 1 -> "mig_shm_1"

        try:
            # Check if leftover shared memory file exists in /dev/shm
            if os.path.exists(f"/dev/shm/{my_name}"):

                os.unlink(f"/dev/shm/{my_name}")
                # Delete old shared memory file
                # Prevents conflicts from previous runs

        except Exception:
            pass
            # Best-effort cleanup — ignore failures

        # -----------------------------------------
        # 2. Create My Write Buffer
        # -----------------------------------------

        try:
            self.my_shm = SharedMemory(name=my_name, create=True, size=self.buffer_size)
            # Creates a new shared memory region
            # Allocates buffer_size bytes in /dev/shm
            # Example:
            #   64MB continuous memory region

        except FileExistsError:
            # If cleanup failed and memory already exists
            self.my_shm = SharedMemory(
                name=my_name, create=False, size=self.buffer_size
            )
            # Attach to existing shared memory instead of creating

        _MIG_SHM_HANDLES.append(self.my_shm)
        # Store handle globally to prevent garbage collection
        # Ensures memory stays alive

        # -----------------------------------------
        # Create Numpy View (Zero Copy)
        # -----------------------------------------

        self.my_buffer_np = np.ndarray(
            (self.buffer_size,), dtype=np.uint8, buffer=self.my_shm.buf
        )
        # Interpret shared memory as:
        #   1D array of uint8 (bytes)
        # Length = buffer_size
        #
        # Example:
        #   If buffer_size = 64MB
        #   self.my_buffer_np.shape = (67108864,)
        #   dtype = uint8
        #
        # Zero-copy:
        #   This NumPy array directly maps to shared memory.
        #   Writing to this array writes to /dev/shm.

        # -----------------------------------------
        # 3. Global Sync
        # -----------------------------------------

        dist.barrier()
        # All processes wait here.
        # Ensures every rank has created its shared memory file
        # before anyone tries to connect to peers.

        # -----------------------------------------
        # 4. Connect to Peers (Read Buffers)
        # -----------------------------------------

        for peer_rank in range(world_size):

            if peer_rank == rank:
                continue
            # Skip myself — already have my own buffer

            peer_name = f"mig_shm_{peer_rank}"
            # Example:
            #   Trying to connect to "mig_shm_1", "mig_shm_2", etc.

            connected = False
            attempts = 0

            while not connected and attempts < 1000:

                try:
                    shm = SharedMemory(
                        name=peer_name, create=False, size=self.buffer_size
                    )
                    # Attach to peer's shared memory region
                    # Does NOT create new memory
                    # Just maps existing /dev/shm file

                    _MIG_SHM_HANDLES.append(shm)
                    # Keep reference alive

                    self.peer_buffers[peer_rank] = np.ndarray(
                        (self.buffer_size,), dtype=np.uint8, buffer=shm.buf
                    )
                    # Create zero-copy NumPy byte view of peer memory
                    #
                    # Example entry:
                    #   self.peer_buffers[1] -> array of shape (64MB,) uint8
                    #
                    # Now we can directly read peer_rank's memory

                    connected = True

                except FileNotFoundError:
                    # Peer hasn't created its shared memory yet
                    time.sleep(0.01)
                    # Wait 10ms
                    attempts += 1

            if not connected:
                raise RuntimeError(f"Rank {rank} failed to connect to {peer_name}")
                # Safety: fail if peer never became available

        print(f"[MIG-Mesh] Rank {rank} Connected. Mesh Ready.")

        dist.barrier()

        # Final synchronization:
        # Ensure all ranks are fully connected
        # before starting communication

    def all_reduce_sum(self, tensor):
        """
        1. Copy GPU Tensor -> CPU SHM
        2. Barrier
        3. Read Peers + Sum
        4. Copy Result -> GPU
        """

        # ----------------------------
        # A. Convert Tensor to Raw Bytes
        # ----------------------------

        cpu_view = tensor.detach().cpu().view(-1)
        # tensor.detach()  -> removes gradient tracking (no autograd graph)
        # .cpu()           -> moves tensor from GPU to CPU memory
        # .view(-1)        -> flattens tensor to 1D
        # Example:
        #   tensor = [[1.0, 2.0], [3.0, 4.0]] (float32, CUDA)
        #   cpu_view = [1.0, 2.0, 3.0, 4.0] (float32, CPU)

        dtype = cpu_view.numpy().dtype
        # Convert to NumPy view and read dtype
        # Example:
        #   dtype = float32
        # This is important so we reconstruct peer data correctly later.

        tensor_bytes = cpu_view.numpy().view(np.uint8)
        # Reinterpret the SAME memory as raw bytes (uint8 = 1 byte)
        # No data conversion, just reinterpretation.
        # Example:
        #   float32 has 4 bytes per value
        #   If cpu_view has 4 float32 values → 16 bytes total
        #   tensor_bytes = array([byte1, byte2, ..., byte16], dtype=uint8)

        nbytes = len(tensor_bytes)
        # Total memory occupied in bytes
        # Example:
        #   4 float32 values × 4 bytes = 16
        #   nbytes = 16

        if nbytes > self.buffer_size:
            raise ValueError(f"Tensor size {nbytes} exceeds buffer {self.buffer_size}")
        # Safety check:
        # Ensure shared memory buffer is large enough to hold this tensor

        # ----------------------------
        # B. Write My Data into Shared Memory
        # ----------------------------

        self.my_buffer_np[:nbytes] = tensor_bytes
        # Copy raw byte representation into my shared memory buffer.
        # Example:
        #   shared_memory[0:16] = [byte1, byte2, ..., byte16]
        # Now other processes can read my tensor.

        # ----------------------------
        # C. Synchronize
        # ----------------------------

        dist.barrier()
        # All processes wait here until everyone has written their data.
        # Ensures shared memory is fully populated.

        # ----------------------------
        # D. Accumulate Results
        # ----------------------------

        accumulated = cpu_view.clone()
        # Start accumulation with my own tensor values.
        # Example:
        #   accumulated = [1.0, 2.0, 3.0, 4.0]
        # Clone ensures we don't modify original CPU view directly.

        for peer_rank, peer_buffer in self.peer_buffers.items():

            raw_bytes = peer_buffer[:nbytes]
            # Read first nbytes from peer's shared memory buffer.
            # Example:
            #   raw_bytes = [byte1, byte2, ..., byte16]

            peer_data = np.frombuffer(raw_bytes, dtype=dtype)
            # Interpret those bytes as the original dtype (e.g., float32).
            # Example:
            #   peer_data = [5.0, 6.0, 7.0, 8.0]
            # This does NOT copy memory; just reinterprets.

            peer_tensor = torch.from_numpy(peer_data)
            # Convert NumPy array to PyTorch tensor (CPU).
            # Shares memory with peer_data.

            accumulated.add_(peer_tensor)
            # In-place addition.
            # Example:
            #   accumulated = [1,2,3,4]
            #   peer_tensor = [5,6,7,8]
            #   accumulated becomes [6,8,10,12]

        # After loop:
        # accumulated = sum of my tensor + all peer tensors

        # ----------------------------
        # E. Copy Result Back to GPU
        # ----------------------------

        tensor.copy_(accumulated.to(tensor.device))
        # Move accumulated CPU tensor back to original device (likely CUDA)
        # Then overwrite original tensor values in-place.
        # Now tensor contains the reduced (summed) result.

        # ----------------------------
        # F. Final Synchronization
        # ----------------------------

        dist.barrier()
        # Ensure all processes have finished reading before next round.


# --- MODULE LEVEL FUNCTIONS ---


def register_hooks():
    """
    Called by the user script to manually enable the custom transport.
    """

    global _MIG_ENGINE
    # Allows this function to modify the global _MIG_ENGINE variable
    # Instead of creating a local variable

    rank = dist.get_rank()
    # Integer ID of this process in the distributed group
    # Example:
    #   rank = 0, 1, 2, ...

    world_size = dist.get_world_size()
    # Total number of participating processes
    # Example:
    #   world_size = 4

    # Initialize the engine (only once per process)
    if _MIG_ENGINE is None:
        _MIG_ENGINE = MIGTransport(rank, world_size)
        # Creates shared memory mesh
        # After this:
        #   _MIG_ENGINE.my_buffer_np exists
        #   _MIG_ENGINE.peer_buffers populated
        #   SHM fully connected

    # -----------------------------------------
    # Monkey Patch PyTorch
    # -----------------------------------------

    dist.all_reduce = patched_all_reduce
    # Replace PyTorch's default all_reduce with our custom function
    #
    # Before:
    #   dist.all_reduce → NCCL or Gloo implementation
    #
    # After:
    #   dist.all_reduce → patched_all_reduce
    #
    # This affects ALL future calls in this process.

    print(f"[MIG] Hooks registered for Rank {rank}")


def patched_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    """
    The replacement function that PyTorch will call.
    """

    # Only intercept SUM operations if Engine is active
    if _MIG_ENGINE is not None and op == dist.ReduceOp.SUM:

        _MIG_ENGINE.all_reduce_sum(tensor)
        # Calls your shared-memory implementation
        #
        # tensor:
        #   - Initially contains local GPU values
        #   - After this call contains SUM across all ranks
        #
        # Operation is BLOCKING

        # If async_op=True was requested:
        # PyTorch normally returns a Work handle
        # We fake one here (but operation is already complete)
        return dist.distributed_c10d.Work()

    # -----------------------------------------
    # Fallback to original implementation
    # -----------------------------------------

    return _ORIGINAL_ALL_REDUCE(tensor, op, group, async_op)
    # If:
    #   - op != SUM
    #   - or _MIG_ENGINE not initialized
    #
    # Then call original backend (NCCL/Gloo)
    #
    # Example:
    #   broadcast
    #   max reduction
    #   product reduction
