import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# --- CONFIG ---
MODEL_ID = "lmsys/vicuna-7b-v1.5"
MASTER_ADDR = "localhost"
MASTER_PORT = "29500"
PROMPT = "USER: Explain quantum entanglement to a 5-year old.\nASSISTANT:"
MAX_NEW_TOKENS = 50  # Generate 50 words
BLOCK_SIZE = 128  # Increased to prevent early cutoff

# Import the transport patch
import mig_patch


def run_model_shard(rank, world_size, mig_uuids):
    try:
        # 1. SETUP
        os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuids[rank]
        device = torch.device("cuda:0")
        dist.init_process_group(backend="mig", rank=rank, world_size=world_size)

        # 2. LOAD
        if rank == 0:
            print(f"[Rank {rank}] Loading Vicuna...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

        # 3. SPLIT LAYERS
        mid = len(model.model.layers) // 2
        if rank == 0:
            model.model.layers = model.model.layers[:mid]
            model.model.norm = torch.nn.Identity()
            model.lm_head = torch.nn.Identity()
        else:
            model.model.layers = model.model.layers[mid:]
            model.model.embed_tokens = torch.nn.Identity()

        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # 4. GENERATION LOOP
        if rank == 0:
            input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
            print(f"\n[Rank 0] Prompt: '{PROMPT}'\n")
            print("Response:", end="", flush=True)

            for i in range(MAX_NEW_TOKENS):
                current_len = input_ids.shape[1]
                if current_len >= BLOCK_SIZE:
                    print("\n[Rank 0] Context limit reached.", flush=True)
                    break

                # A. Run First Half
                with torch.no_grad():
                    out = model.model(input_ids)
                    hidden = out.last_hidden_state

                # B. Pad to Fixed Size
                pad_len = BLOCK_SIZE - current_len
                padded_hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad_len))

                # C. Send to Rank 1 (Wait for GPU to finish first)
                torch.cuda.synchronize()
                dist.send(padded_hidden, dst=1)

                # D. Receive Token ID (on CPU to prevent deadlock)
                next_token_cpu = torch.tensor([0], dtype=torch.int64, device="cpu")
                dist.recv(next_token_cpu, src=1)

                # E. Decode
                next_token = next_token_cpu.to(device)
                word = tokenizer.decode(next_token, clean_up_tokenization_spaces=False)
                print(word, end="", flush=True)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        elif rank == 1:
            # Pre-allocate buffers
            recv_buffer = torch.zeros(
                1, BLOCK_SIZE, 4096, dtype=torch.float16, device=device
            )
            token_sender_cpu = torch.zeros(1, dtype=torch.int64, device="cpu")

            for i in range(MAX_NEW_TOKENS):
                # A. Receive Hidden States
                dist.recv(recv_buffer, src=0)

                # B. Check for valid data
                valid_rows = (recv_buffer.abs().sum(dim=-1)[0] > 0).sum().item()
                if valid_rows == 0:
                    break  # Stop if Rank 0 stopped sending

                with torch.no_grad():
                    # Run Second Half
                    out = model(inputs_embeds=recv_buffer)
                    logits = out.logits

                    # C. Predict Token
                    last_valid_idx = valid_rows - 1
                    next_token_id = torch.argmax(logits[:, last_valid_idx, :], dim=-1)

                # D. Send Back
                token_sender_cpu[0] = next_token_id.item()
                dist.send(token_sender_cpu, dst=0)

        # 5. CLEAN SHUTDOWN (The Fix)
        # Wait for both ranks to finish the loop before killing the process
        print(f"\n[Rank {rank}] Finished. Waiting for peer...", flush=True)
        dist.barrier()
        print(f"[Rank {rank}] Exiting safely.", flush=True)

    except Exception as e:
        print(f"\n[Rank {rank}] ERROR: {e}")
        import traceback

        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    import subprocess

    try:
        lines = (
            subprocess.check_output("nvidia-smi -L", shell=True)
            .decode()
            .strip()
            .split("\n")
        )
        uuids = [l.split("UUID: ")[1].split(")")[0] for l in lines if "MIG" in l]
    except:
        uuids = []

    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.system("rm /dev/shm/mig_link_* 2>/dev/null")

    mp.spawn(run_model_shard, args=(len(uuids), uuids), nprocs=len(uuids), join=True)
