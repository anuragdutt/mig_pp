import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import mig_transport
import gc
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
SPLIT_CONFIG = [20, 7, 5]
HEAD_DIM = 128
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 11008
NUM_LAYERS = 32
VOCAB_SIZE = 32000
MODEL_NAME = "lmsys/vicuna-7b-v1.5"

MIG_UUIDS = [
    "MIG-ccbc8dda-f9b2-5ded-bd3f-52aca48ccb8f",  # Rank 0
    "MIG-919bf415-7bf2-50ac-87d9-69a403e99276",  # Rank 1
    "MIG-0582ee4b-48f6-52cc-8354-13d3e79f9d8c",  # Rank 2
]


def safe_all_reduce(tensor):
    orig_shape = tensor.shape
    flat = tensor.view(-1)
    dist.all_reduce(flat)
    return flat.view(orig_shape)


class ShardedVicuna(nn.Module):
    def __init__(self, rank, split_config):
        super().__init__()
        self.rank = rank
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

        # Build Layers
        self.layers = nn.ModuleList()
        for _ in range(NUM_LAYERS):
            # Attention
            attn = nn.Module()
            attn.n_heads = split_config[rank]
            attn.dim = attn.n_heads * HEAD_DIM
            attn.wq = nn.Linear(HIDDEN_SIZE, attn.dim, bias=False)
            attn.wk = nn.Linear(HIDDEN_SIZE, attn.dim, bias=False)
            attn.wv = nn.Linear(HIDDEN_SIZE, attn.dim, bias=False)
            attn.wo = nn.Linear(attn.dim, HIDDEN_SIZE, bias=False)

            # MLP
            mlp = nn.Module()
            total_heads = 32
            mlp_shard = int((split_config[rank] / 32) * INTERMEDIATE_SIZE)
            # Fix rounding for Rank 2
            if rank == 2:
                used = sum(
                    [int((c / 32) * INTERMEDIATE_SIZE) for c in split_config[:2]]
                )
                mlp_shard = INTERMEDIATE_SIZE - used

            mlp.gate_proj = nn.Linear(HIDDEN_SIZE, mlp_shard, bias=False)
            mlp.up_proj = nn.Linear(HIDDEN_SIZE, mlp_shard, bias=False)
            mlp.down_proj = nn.Linear(mlp_shard, HIDDEN_SIZE, bias=False)

            # Block
            block = nn.Module()
            block.attention = attn
            block.mlp = mlp
            block.input_layernorm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-5)
            block.post_attention_layernorm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-5)
            self.layers.append(block)

        self.norm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-5)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        seq_len = x.shape[1]
        # FIX: Ensure mask matches model dtype
        mask = (
            torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
            .to(x.device)
            .to(x.dtype)
        )

        for layer in self.layers:
            # Attention Block
            h = layer.input_layernorm(x)

            # Attn Calc
            xq = layer.attention.wq(h)
            xk = layer.attention.wk(h)
            xv = layer.attention.wv(h)

            b, s, _ = xq.shape
            xq = xq.view(b, s, -1, HEAD_DIM).transpose(1, 2)
            xk = xk.view(b, s, -1, HEAD_DIM).transpose(1, 2)
            xv = xv.view(b, s, -1, HEAD_DIM).transpose(1, 2)

            scores = torch.matmul(xq, xk.transpose(-2, -1)) / (HEAD_DIM**0.5)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(scores, xv)
            attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
            attn_out = layer.attention.wo(attn_out)
            attn_out = safe_all_reduce(attn_out)

            h = x + attn_out

            # MLP Block
            res = h
            h = layer.post_attention_layernorm(h)
            gate = F.silu(layer.mlp.gate_proj(h))
            up = layer.mlp.up_proj(h)
            inter = gate * up
            mlp_out = layer.mlp.down_proj(inter)
            mlp_out = safe_all_reduce(mlp_out)

            x = res + mlp_out

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def load_weights(model, hf_state_dict, rank, split_config):
    # Simplified loading logic for brevity - this matches your working script
    print(f"[Rank {rank}] Loading weights...")
    head_start = sum(split_config[:rank])
    dim_start = head_start * HEAD_DIM
    dim_end = dim_start + (split_config[rank] * HEAD_DIM)

    mlp_total = INTERMEDIATE_SIZE
    mlp_start = sum([int((c / 32) * mlp_total) for c in split_config[:rank]])
    if rank == 2:
        mlp_len = mlp_total - mlp_start
    else:
        mlp_len = int((split_config[rank] / 32) * mlp_total)
    mlp_end = mlp_start + mlp_len

    with torch.no_grad():
        model.embed.weight.copy_(hf_state_dict["model.embed_tokens.weight"])
        model.norm.weight.copy_(hf_state_dict["model.norm.weight"])
        model.lm_head.weight.copy_(hf_state_dict["lm_head.weight"])

        for i, layer in enumerate(model.layers):
            prefix = f"model.layers.{i}"
            layer.attention.wq.weight.copy_(
                hf_state_dict[f"{prefix}.self_attn.q_proj.weight"][dim_start:dim_end]
            )
            layer.attention.wk.weight.copy_(
                hf_state_dict[f"{prefix}.self_attn.k_proj.weight"][dim_start:dim_end]
            )
            layer.attention.wv.weight.copy_(
                hf_state_dict[f"{prefix}.self_attn.v_proj.weight"][dim_start:dim_end]
            )
            layer.attention.wo.weight.copy_(
                hf_state_dict[f"{prefix}.self_attn.o_proj.weight"][:, dim_start:dim_end]
            )

            layer.mlp.gate_proj.weight.copy_(
                hf_state_dict[f"{prefix}.mlp.gate_proj.weight"][mlp_start:mlp_end]
            )
            layer.mlp.up_proj.weight.copy_(
                hf_state_dict[f"{prefix}.mlp.up_proj.weight"][mlp_start:mlp_end]
            )
            layer.mlp.down_proj.weight.copy_(
                hf_state_dict[f"{prefix}.mlp.down_proj.weight"][:, mlp_start:mlp_end]
            )

            layer.input_layernorm.weight.copy_(
                hf_state_dict[f"{prefix}.input_layernorm.weight"]
            )
            layer.post_attention_layernorm.weight.copy_(
                hf_state_dict[f"{prefix}.post_attention_layernorm.weight"]
            )
    print(f"[Rank {rank}] Loaded.")


# --- GPU WORKER PROCESS ---
def gpu_worker(rank, world_size, input_queue, output_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = MIG_UUIDS[rank]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    mig_transport.register_hooks()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = ShardedVicuna(rank, SPLIT_CONFIG).half().to(device)

    # Load weights
    print(f"[Rank {rank}] Initializing...")
    hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    load_weights(model, hf.state_dict(), rank, SPLIT_CONFIG)
    del hf
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dist.barrier()

    if rank == 0:
        print(f"*** Rank {rank} READY TO SERVE ***")
        output_queue.put("READY")  # Signal to API that we are up

    while True:
        # Rank 0 waits for prompt
        if rank == 0:
            item = input_queue.get()
            if item == "SHUTDOWN":
                req_id, prompt = "SHUTDOWN", None
            else:
                req_id, prompt = item

            if req_id == "SHUTDOWN":
                # Signal others to stop
                sig = torch.tensor([-1], dtype=torch.long).to(device)
                dist.broadcast(sig, src=0)
                break

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            seq_len = torch.tensor([input_ids.shape[1]], dtype=torch.long).to(device)

            # Broadcast metadata
            dist.broadcast(seq_len, src=0)
            dist.broadcast(input_ids, src=0)

        else:
            # Workers wait for length
            seq_len = torch.tensor([0], dtype=torch.long).to(device)
            dist.broadcast(seq_len, src=0)

            if seq_len.item() == -1:  # Shutdown signal
                break

            input_ids = torch.zeros((1, seq_len.item()), dtype=torch.long).to(device)
            dist.broadcast(input_ids, src=0)

        # Inference
        for _ in range(100):  # Max new tokens
            with torch.no_grad():
                logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        # Return result
        if rank == 0:
            full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            # response = full_text[len(prompt):]
            output_queue.put((req_id, full_text))

    dist.destroy_process_group()


# --- FASTAPI SERVER ---

# Global Queues
q_in = mp.Queue()
q_out = mp.Queue()
gpu_procs = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Launch GPUs
    print("Starting GPU Workers...")
    mp.set_start_method("spawn", force=True)
    for i in range(3):
        p = mp.Process(target=gpu_worker, args=(i, 3, q_in, q_out))
        p.start()
        gpu_procs.append(p)

    # Wait for Rank 0 to signal ready
    print("Waiting for Model to Load...")
    msg = await asyncio.to_thread(q_out.get)
    if msg == "READY":
        print("Model Loaded successfully!")

    yield  # App runs here

    # Shutdown: Kill GPUs
    print("Shutting down GPU Workers...")
    q_in.put("SHUTDOWN")
    for p in gpu_procs:
        p.join()


app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate_text(req: PromptRequest):
    request_id = str(uuid.uuid4())

    # Send to GPU
    # We use asyncio.to_thread so we don't block the event loop while waiting for the Queue lock
    await asyncio.to_thread(q_in.put, (request_id, req.prompt))

    # Wait for Result
    # In a production app, we would use a dictionary of futures to handle multiple concurrent requests.
    # Since this GPU setup handles 1 request at a time, we just wait for the next output.

    result_id, result_text = await asyncio.to_thread(q_out.get)

    # Simple sanity check (though with 1 worker, order is guaranteed)
    if result_id != request_id:
        return {"error": "Synchronization error", "details": "Received wrong ID"}

    return {"id": result_id, "response": result_text}


if __name__ == "__main__":
    import uvicorn

    # Run on 0.0.0.0 so you can access it from outside
    uvicorn.run(app, host="0.0.0.0", port=8000)
