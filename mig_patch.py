# Import PyTorch's main library for tensor operations
import torch

# Import PyTorch's distributed training module for multi-process communication
import torch.distributed as dist

# Import shared memory class for inter-process memory sharing
from multiprocessing.shared_memory import SharedMemory

# Import NumPy for efficient array operations and memory views
import numpy as np

# Import time module for sleep operations during polling
import time

# Import os module for file system operations (unlinking shared memory files)
import os

# --- GLOBAL STORAGE (Prevents Garbage Collection) ---
# Global list to keep references to SharedMemory objects alive
# Without this, Python's garbage collector might destroy the shared memory
# before other processes finish using it
_MIG_SHM_HANDLES = []


class MIGPatch:
    # This code was meant for pipeline parallelism for strictly 2mig instances
    # Constructor initializes the custom transport layer
    # rank: The process ID (0 or 1 in this case)
    # world_size: Total number of processes (2 in this case)
    # buffer_size_mb: Size of shared memory buffer in megabytes (default 128MB)
    def __init__(self, rank, world_size, buffer_size_mb=128):
        # Store the current process's rank (0 or 1)
        self.rank = rank

        # Calculate buffer size in bytes (MB * 1024 * 1024)
        self.buffer_size = buffer_size_mb * 1024 * 1024

        # 1. SYNC
        # Print status message showing this rank is waiting for other processes
        print(f"[MIG-Patch] Rank {rank} waiting for peers...")

        # Wait for all processes to reach this point before continuing
        # This ensures all processes are ready before creating/accessing shared memory
        dist.barrier()

        # 2. SETUP SENDER
        # Only rank 0 creates the shared memory segment (it will be the sender)
        if rank == 0:
            # Name for the shared memory segment (visible in /dev/shm/)
            name = "mig_link_0_1"

            # Try to remove any existing shared memory file with this name
            # This cleanup prevents errors if previous runs didn't clean up properly
            try:
                os.unlink(f"/dev/shm/{name}")
            except:
                # If file doesn't exist, that's fine - just ignore the error
                pass

            # Create a new shared memory segment
            # create=True: This process creates the segment
            # size=self.buffer_size: Allocate the specified number of bytes
            shm = SharedMemory(name=name, create=True, size=self.buffer_size)

            # Add the SharedMemory object to global list to prevent garbage collection
            _MIG_SHM_HANDLES.append(shm)

            # Create a NumPy array view of the shared memory buffer
            # This allows us to write bytes directly into shared memory
            # dtype=np.uint8: Treat memory as unsigned 8-bit integers (raw bytes)
            # buffer=shm.buf: Use the shared memory buffer as backing storage
            self.send_buff_np = np.ndarray(
                (self.buffer_size,), dtype=np.uint8, buffer=shm.buf
            )

            # Print confirmation that shared memory was created successfully
            print(f"[MIG-Patch] Rank {rank} created shared memory: {name}")

        # 3. SAFETY BARRIER
        # Wait again to ensure rank 0 has finished creating the shared memory
        # before rank 1 tries to connect to it
        dist.barrier()

        # 4. SETUP RECEIVER
        # Only rank 1 connects to existing shared memory (it will be the receiver)
        if rank == 1:
            # Use the same name that rank 0 created
            name = "mig_link_0_1"

            # Keep trying to connect until the shared memory is available
            # This handles any timing issues where rank 1 arrives before rank 0 finishes
            while True:
                try:
                    # Try to open the existing shared memory segment
                    # create=False: Don't create, just attach to existing segment
                    shm = SharedMemory(name=name, create=False, size=self.buffer_size)

                    # Add to global list to prevent garbage collection
                    _MIG_SHM_HANDLES.append(shm)

                    # Successfully connected, exit the retry loop
                    break
                except FileNotFoundError:
                    # Shared memory doesn't exist yet, wait 10ms and try again
                    time.sleep(0.01)

            # Create a NumPy array view of the shared memory buffer for reading
            # This points to the same physical memory that rank 0 writes to
            self.recv_buff_np = np.ndarray(
                (self.buffer_size,), dtype=np.uint8, buffer=shm.buf
            )

            # Print confirmation that rank 1 successfully connected
            print(f"[MIG-Patch] Rank {rank} connected to shared memory: {name}")

    # Method to send a tensor through shared memory
    def send(self, tensor):
        # 1. Move to CPU (Synchronizes implicitly)
        # Copy the tensor from GPU to CPU memory
        # .view(-1) flattens the tensor into a 1D array
        # This is a blocking operation that waits for GPU to finish any pending work
        cpu_tensor = tensor.cpu().view(-1)

        # 2. Direct Memory Copy
        # Get a byte-level view of the tensor data
        # .numpy() converts PyTorch tensor to NumPy array (shares memory, no copy)
        # .view(np.uint8) reinterprets the data as raw bytes
        tensor_bytes = cpu_tensor.numpy().view(np.uint8)

        # Get the total number of bytes to copy
        nbytes = len(tensor_bytes)

        # Copy the tensor bytes into shared memory
        # This writes directly to the memory that rank 1 can read
        self.send_buff_np[:nbytes] = tensor_bytes

    # Method to receive a tensor from shared memory
    def recv(self, tensor):
        # 1. Prepare target on CPU (Flat)
        # Create an empty CPU tensor with the same total number of elements
        # This will hold the flattened data read from shared memory
        flat_cpu_tensor = torch.empty(tensor.numel(), dtype=tensor.dtype)

        # Calculate how many bytes to read based on tensor size
        # element_size() returns bytes per element (e.g., 4 for float32)
        nbytes = flat_cpu_tensor.numel() * flat_cpu_tensor.element_size()

        # 2. Read bytes from Shared Memory
        # Copy the raw bytes from shared memory into a Python bytes object
        raw_data = self.recv_buff_np[:nbytes]

        # 3. Convert back to Tensor
        # Interpret the raw bytes as a NumPy array with the correct data type
        # This converts bytes back into the original numeric format (float32, int64, etc.)
        src_array = np.frombuffer(raw_data, dtype=tensor.cpu().numpy().dtype)

        # Convert the NumPy array to a PyTorch tensor (shares memory, no copy)
        t_cpu = torch.from_numpy(src_array)

        # 4. Copy to the flat CPU buffer
        # Copy the data into our prepared CPU tensor
        flat_cpu_tensor.copy_(t_cpu)

        # 5. Move back to GPU (The Fix!)
        # Flatten the destination GPU tensor to match the source shape
        # .view(-1) makes it 1D
        # .copy_() copies data from flat_cpu_tensor
        # .to(tensor.device) moves flat_cpu_tensor to the same GPU as tensor
        # This overwrites the original tensor's data with received data
        tensor.view(-1).copy_(flat_cpu_tensor.to(tensor.device))


# --- PATCH LOGIC ---
# Save references to the original PyTorch distributed functions
# We'll call these from our patched versions
_original_init = dist.init_process_group
_original_send = dist.send
_original_recv = dist.recv

# Global variable to hold the single MIGTransport instance
_mig_engine_instance = None


# Patched version of init_process_group
# *args: Accept any positional arguments
# **kwargs: Accept any keyword arguments
def patched_init(*args, **kwargs):
    # Access the global MIGTransport instance variable
    global _mig_engine_instance

    # Check if user requested "mig" backend
    if kwargs.get("backend") == "mig":
        # Replace "mig" with "gloo" (a CPU-based PyTorch backend)
        # We need an underlying backend for synchronization
        kwargs["backend"] = "gloo"

        # Call the original init_process_group with modified backend
        _original_init(*args, **kwargs)

        # Create our custom MIGTransport instance
        # dist.get_rank() returns this process's rank (0 or 1)
        # dist.get_world_size() returns total number of processes
        _mig_engine_instance = MIGTransport(dist.get_rank(), dist.get_world_size())

        # Return early since we're done
        return

    # If backend is not "mig", just call original function normally
    return _original_init(*args, **kwargs)


# Patched version of dist.send
# tensor: The tensor to send
# dst: Destination rank
# group: Process group (optional)
# tag: Message tag for matching sends/receives (optional)
def patched_send(tensor, dst, group=None, tag=0):
    # Check if MIG engine is active AND destination is rank 1
    if _mig_engine_instance and dst == 1:
        # Send the actual tensor through our custom shared memory transport
        _mig_engine_instance.send(tensor)

        # Create a tiny dummy tensor (1 byte) on CPU
        # This is used for synchronization through the gloo backend
        dummy = torch.tensor([1], dtype=torch.uint8).cpu()

        # Send the dummy tensor through original gloo backend
        # This ensures rank 1 knows data is ready in shared memory
        return _original_send(dummy, dst, group, tag)

    # For all other cases, use the original send function
    return _original_send(tensor, dst, group, tag)


# Patched version of dist.recv
# tensor: The tensor to receive data into
# src: Source rank (optional)
# group: Process group (optional)
# tag: Message tag for matching sends/receives (optional)
def patched_recv(tensor, src=None, group=None, tag=0):
    # Check if MIG engine is active AND source is rank 0
    if _mig_engine_instance and src == 0:
        # Create a dummy tensor to receive the synchronization signal
        dummy = torch.tensor([0], dtype=torch.uint8).cpu()

        # Receive the dummy tensor through gloo backend (blocks until ready)
        # This tells us that rank 0 has finished writing to shared memory
        ret = _original_recv(dummy, src, group, tag)

        # Now read the actual tensor data from shared memory
        _mig_engine_instance.recv(tensor)

        # Return the result from the dummy receive
        return ret

    # For all other cases, use the original recv function
    return _original_recv(tensor, src, group, tag)


# Replace PyTorch's distributed functions with our patched versions
# This is called "monkey patching" - modifying library behavior at runtime
dist.init_process_group = patched_init
dist.send = patched_send
dist.recv = patched_recv
