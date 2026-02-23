# Import os module for environment variable management and system operations
import os

# Import PyTorch's main library for tensor operations
import torch

# Import PyTorch's distributed communication library for multi-process training
import torch.distributed as dist

# Import PyTorch's multiprocessing utilities for spawning parallel processes
import torch.multiprocessing as mp

# Import Hugging Face transformers components for loading pre-trained models
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# Import the MIG patch module which monkey-patches distributed functions
# This enables shared memory communication between MIG GPU instances
import mig_patch  # <--- Activates your secret tunnel

# --- CONFIG ---
# Specify the Hugging Face model identifier for Vicuna 7B v1.5
MODEL_ID = "lmsys/vicuna-7b-v1.5"

# Master node address for distributed communication (localhost since both processes on same machine)
MASTER_ADDR = "localhost"

# Master node port for distributed communication coordination
MASTER_PORT = "29500"

# A slightly longer prompt to make the output interesting
# Using Vicuna's chat format with USER/ASSISTANT tags
TEST_PROMPT = "USER: Explain quantum entanglement to a 5-year old.\nASSISTANT:"


# Main function that runs on each process (each MIG GPU instance)
# rank: The process ID (0 or 1)
# world_size: Total number of processes (2 in this case)
# mig_uuids: List of MIG GPU UUIDs to assign to each process
def run_model_shard(rank, world_size, mig_uuids):
    # 1. SETUP MIG ISOLATION
    # Set CUDA_VISIBLE_DEVICES to only see this process's assigned MIG GPU
    # This isolates each process to its own GPU partition
    os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuids[rank]

    # Create a torch device pointing to cuda:0 (which is actually our assigned MIG GPU
    # due to CUDA_VISIBLE_DEVICES remapping)
    device = torch.device("cuda:0")

    # 2. INIT ENGINE
    # Initialize the distributed process group using the "mig" backend
    # The mig_patch module intercepts this and converts it to "gloo" + shared memory
    # rank: This process's ID (0 or 1)
    # world_size: Total number of processes (2)
    dist.init_process_group(backend="mig", rank=rank, world_size=world_size)

    # 3. LOAD MODEL
    # Print status message showing which rank is loading the model
    print(f"[Rank {rank}] Loading Vicuna 7B...")

    # Load the pre-trained Vicuna model from Hugging Face
    # torch_dtype=torch.float16: Use half-precision (16-bit) floats to save memory
    # low_cpu_mem_usage=True: Load model more efficiently to avoid OOM errors
    # device_map="cpu": Initially load all weights to CPU before moving to GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu"
    )

    # 4. SLICE LAYERS
    # Get the total number of transformer layers in the model (Vicuna 7B has 32 layers)
    total_layers = len(model.model.layers)

    # Calculate the midpoint to split the model in half
    mid_point = total_layers // 2

    # Rank 0 handles the FIRST HALF of the model
    if rank == 0:
        # Keep only the first half of transformer layers (layers 0-15)
        model.model.layers = model.model.layers[:mid_point]

        # Replace the layer normalization with Identity (do nothing)
        # Rank 1 will handle the actual normalization
        model.model.norm = torch.nn.Identity()

        # Replace the language model head with Identity (do nothing)
        # Rank 1 will handle generating the final logits
        model.lm_head = torch.nn.Identity()

    # Rank 1 handles the SECOND HALF of the model
    elif rank == 1:
        # Keep only the second half of transformer layers (layers 16-31)
        model.model.layers = model.model.layers[mid_point:]

        # Replace the embedding layer with Identity (do nothing)
        # Rank 0 already computed embeddings, we'll receive them via distributed comm
        model.model.embed_tokens = torch.nn.Identity()

    # Move the model (with only its assigned layers) to the GPU
    model.to(device)

    # Clear any cached memory from model loading to free up GPU memory
    torch.cuda.empty_cache()

    # 5. INFERENCE PIPELINE
    # Load the tokenizer to convert text to token IDs and vice versa
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # RANK 0: First stage of the pipeline (embedding + first half of layers)
    if rank == 0:
        # Tokenize the input prompt
        # return_tensors="pt": Return PyTorch tensors
        # .to(device): Move the token IDs to GPU
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(device)

        # Print status message
        print(f"[Rank 0] Processing prompt...")

        # Disable gradient computation (we're doing inference, not training)
        with torch.no_grad():
            # Forward pass through rank 0's portion of the model
            # This computes embeddings and processes through layers 0-15
            output = model.model(inputs.input_ids)

            # Extract the hidden states (the output activations from the last layer)
            # Shape: [batch_size, sequence_length, hidden_size]
            # For this prompt: [1, 23, 4096]
            hidden_states = output.last_hidden_state

            # --- CRITICAL FIX: SYNC BEFORE SEND ---
            # Wait for all GPU operations to complete before accessing the data
            # Without this, we might read incomplete/garbage data from GPU memory
            torch.cuda.synchronize()
            # --------------------------------------

            # Compute a checksum by summing all values in the hidden states tensor
            # This helps verify that data was computed correctly and isn't all zeros
            # .item() converts the single-element tensor to a Python scalar
            checksum = hidden_states.sum().item()

            # Print the checksum for debugging (should be a large number, not zero)
            print(f"[Rank 0] Sending. Data Checksum: {checksum:.2f}")

            # Send the hidden states to rank 1 via distributed communication
            # This uses the MIG shared memory transport under the hood
            # dst=1: Send to process with rank 1
            dist.send(hidden_states, dst=1)

    # RANK 1: Second stage of the pipeline (second half of layers + output head)
    elif rank == 1:
        # Print status message
        print(f"[Rank 1] Listening...")

        # Create a buffer to receive the hidden states from rank 0
        # Must match the exact shape that rank 0 will send
        # Shape: [batch_size=1, seq_len=23, hidden_size=4096]
        # dtype=torch.float16: Match the data type used by rank 0
        # device=device: Allocate on GPU for faster processing
        rec_buffer = torch.zeros(1, 23, 4096, dtype=torch.float16, device=device)

        # Receive the hidden states from rank 0
        # This blocks until data arrives
        # The MIG shared memory transport handles the actual data transfer
        # src=0: Receive from process with rank 0
        dist.recv(rec_buffer, src=0)

        # Compute checksum of received data to verify it matches what rank 0 sent
        checksum = rec_buffer.sum().item()

        # Print the received checksum (should match rank 0's checksum exactly)
        print(f"[Rank 1] Received. Data Checksum: {checksum:.2f}")

        # Check if we received all zeros (indicates pipeline failure)
        if checksum == 0:
            print("[Rank 1] ERROR: Received all zeros! Pipeline failed.")
            return

        # Disable gradient computation for inference
        with torch.no_grad():
            # Create an attention mask (all ones) indicating all tokens should be attended to
            # Shape must match the first two dimensions of rec_buffer: [batch_size, seq_len]
            # Shape: [1, 23]
            att_mask = torch.ones(rec_buffer.shape[:2], device=device)

            # Forward pass through rank 1's portion of the model
            # inputs_embeds=rec_buffer: Use the hidden states from rank 0 as input
            #   (instead of token IDs, since we already have embeddings)
            # attention_mask=att_mask: Tell the model which positions to attend to
            # This processes through layers 16-31, applies layer norm, and generates logits
            output = model(inputs_embeds=rec_buffer, attention_mask=att_mask)

            # Extract the logits (raw scores for each vocabulary token)
            # Shape: [batch_size, seq_len, vocab_size]
            # For Vicuna: [1, 23, 32000]
            logits = output.logits

            # Get the predicted next token by taking the argmax of the last position's logits
            # logits[:, -1, :] gets the logits for the last token position
            # argmax finds the token with the highest score
            # Shape: [batch_size] = [1]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)

            # Decode the token ID back to text
            # This converts the numeric token ID to its corresponding word/subword
            next_word = tokenizer.decode(next_token_id)

            # Print the final results with nice formatting
            print(f"================================================")
            print(f"[Rank 1] PIPELINE SUCCESS!")
            print(f"[Rank 1] Generated Token ID: {next_token_id.item()}")
            print(f"[Rank 1] GENERATED WORD: '{next_word}'")
            print(f"================================================")


# Entry point of the script (only runs when script is executed directly)
if __name__ == "__main__":
    # Import subprocess module for running shell commands
    import subprocess

    # Try to detect MIG GPU instances using nvidia-smi
    try:
        # Run nvidia-smi command to list all GPUs
        # -L flag: List all GPU devices
        # shell=True: Execute the command through the shell
        # .decode(): Convert bytes output to string
        # .strip(): Remove leading/trailing whitespace
        # .split("\n"): Split into individual lines (one per GPU)
        lines = (
            subprocess.check_output("nvidia-smi -L", shell=True)
            .decode()
            .strip()
            .split("\n")
        )

        # Extract UUIDs from lines that contain "MIG" (MIG GPU instances only)
        # For each line: split by "UUID: ", take second part, split by ")", take first part
        # This extracts the GPU UUID string like "MIG-12345678-..."
        # List comprehension filters to only lines containing "MIG"
        uuids = [l.split("UUID: ")[1].split(")")[0] for l in lines if "MIG" in l]
    except:
        # If nvidia-smi fails or no MIG GPUs found, use empty list
        # This will cause the spawn to create 0 processes (safe fallback)
        uuids = []

    # Set environment variables for PyTorch distributed to find the master node
    # These are read by dist.init_process_group
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    # Spawn multiple processes using PyTorch's multiprocessing
    # mp.spawn creates len(uuids) child processes (2 in this case)
    # Each process runs the run_model_shard function with a unique rank (0, 1, ...)
    # args=(len(uuids), uuids): Additional arguments passed to run_model_shard
    #   - world_size = len(uuids) = 2
    #   - mig_uuids = uuids (list of MIG GPU UUIDs)
    # nprocs=len(uuids): Number of processes to spawn (2)
    # join=True: Wait for all processes to complete before exiting
    mp.spawn(run_model_shard, args=(len(uuids), uuids), nprocs=len(uuids), join=True)
