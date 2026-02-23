import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import queue
import time
import sys
import uuid

# --- CONFIG ---
MODEL_ID = "lmsys/vicuna-7b-v1.5"
MASTER_ADDR = "localhost"
MASTER_PORT = "29500"
BLOCK_SIZE = 128

# Import your custom transport
import mig_patch

# --- API SETUP ---
app = FastAPI(title="MIG Distributed Inference API")


# Request Schema
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


# Global Queues for Inter-Process Communication
# We use a Manager to share queues between the API process and GPU workers
manager = mp.Manager()
input_queue = manager.Queue()
output_queue = manager.Queue()


# --- GPU WORKER LOGIC ---
def run_model_server(rank, world_size, mig_uuids, input_q, output_q):
    try:
        # 1. SETUP
        os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuids[rank]
        device = torch.device("cuda:0")
        dist.init_process_group(backend="mig", rank=rank, world_size=world_size)

        # 2. LOAD MODEL
        if rank == 0:
            print(f"[Rank {rank}] Loading Model Server...", flush=True)
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

        if rank == 0:
            print(f"[Rank 0] Server Ready. Waiting for requests...", flush=True)

            # --- RANK 0: THE ORCHESTRATOR ---
            while True:
                # Wait for a request from the API
                req = input_q.get()
                if req is None:
                    break  # Shutdown signal

                request_id, prompt, max_new_tokens = req

                print(f"[Rank 0] Processing Request {request_id[:4]}...", flush=True)

                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                generated_text = ""

                for _ in range(max_new_tokens):
                    current_len = input_ids.shape[1]
                    if current_len >= BLOCK_SIZE:
                        break

                    # Run First Half
                    with torch.no_grad():
                        out = model.model(input_ids)
                        hidden = out.last_hidden_state

                    # Pad
                    pad_len = BLOCK_SIZE - current_len
                    padded_hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad_len))

                    # Sync & Send
                    torch.cuda.synchronize()
                    dist.send(padded_hidden, dst=1)

                    # Receive Token
                    next_token_cpu = torch.tensor([0], dtype=torch.int64, device="cpu")
                    dist.recv(next_token_cpu, src=1)

                    # Decode
                    next_token = next_token_cpu.to(device)
                    word = tokenizer.decode(
                        next_token, clean_up_tokenization_spaces=False
                    )
                    generated_text += word

                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                # Send result back to API
                output_q.put((request_id, generated_text))
                print(f"[Rank 0] Finished Request {request_id[:4]}.", flush=True)

        elif rank == 1:
            # --- RANK 1: THE WORKER ---
            # Rank 1 loops forever. It doesn't know about requests.
            # It just processes whatever hidden states arrive.
            print(f"[Rank 1] Worker Ready.", flush=True)

            recv_buffer = torch.zeros(
                1, BLOCK_SIZE, 4096, dtype=torch.float16, device=device
            )
            token_sender_cpu = torch.zeros(1, dtype=torch.int64, device="cpu")

            while True:
                # Block until Rank 0 sends data
                # If Rank 0 is waiting for user input, Rank 1 simply sleeps here.
                try:
                    dist.recv(recv_buffer, src=0)
                except RuntimeError:
                    break  # Exit if connection dies

                # Process
                valid_rows = (recv_buffer.abs().sum(dim=-1)[0] > 0).sum().item()
                if valid_rows == 0:
                    continue

                with torch.no_grad():
                    out = model(inputs_embeds=recv_buffer)
                    logits = out.logits
                    last_valid_idx = valid_rows - 1
                    next_token_id = torch.argmax(logits[:, last_valid_idx, :], dim=-1)

                # Send Back
                token_sender_cpu[0] = next_token_id.item()
                dist.send(token_sender_cpu, dst=0)

    except Exception as e:
        print(f"[Rank {rank}] CRITICAL ERROR: {e}", flush=True)
        os._exit(1)


# --- API ROUTES ---


@app.post("/generate")
def generate_text(req: PromptRequest):
    request_id = uuid.uuid4().hex

    # Send to GPU Workers
    input_queue.put((request_id, req.prompt, req.max_tokens))

    # Wait for response (Blocking this thread)
    # In a production app, use asyncio or a proper result store (Redis)
    # Since we have 1 pipeline, we serialize requests anyway.
    while True:
        res_id, text = output_queue.get()
        if res_id == request_id:
            return {"status": "success", "generated_text": text}
        else:
            # If we got someone else's result (rare in single-thread), put it back
            output_queue.put((res_id, text))
            time.sleep(0.01)


@app.get("/health")
def health():
    return {"status": "MIG Pipeline Active"}


# --- MAIN ENTRYPOINT ---
if __name__ == "__main__":
    # 1. Detect MIG UUIDs
    try:
        lines = (
            subprocess.check_output("nvidia-smi -L", shell=True)
            .decode()
            .strip()
            .split("\n")
        )
        uuids = [l.split("UUID: ")[1].split(")")[0] for l in lines if "MIG" in l]
    except:
        import subprocess  # Re-import inside block if needed

        lines = (
            subprocess.check_output("nvidia-smi -L", shell=True)
            .decode()
            .strip()
            .split("\n")
        )
        uuids = [l.split("UUID: ")[1].split(")")[0] for l in lines if "MIG" in l]

    if len(uuids) < 2:
        print("Error: Need at least 2 MIG instances!")
        sys.exit(1)

    # 2. Cleanup Old IPC
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.system("rm /dev/shm/mig_link_* 2>/dev/null")

    # 3. Start GPU Workers in Background Process
    print("Starting Model Workers...")
    p = mp.Process(
        target=mp.spawn,
        args=(
            run_model_server,
            (len(uuids), uuids, input_queue, output_queue),
            len(uuids),
            True,
        ),
    )
    p.start()

    # 4. Start API Server
    print("Starting API Server on port 8000...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        print("Shutting down...")
        p.terminate()
        os.system("rm /dev/shm/mig_link_* 2>/dev/null")
