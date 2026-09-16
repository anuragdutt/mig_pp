"""
Counts the actual CUDA kernel launches per layer inside the real pipeline.
Uses your harness's forward_through_layers and dist transport so the count
includes all surrounding ops (clone, send, RoPE, etc.) -- not just the layer.

Run from the mig_pp root directory (same place you run the benchmark):
    python3 mig_pp/milp/count_kernels.py

EDIT THESE:
"""
import os
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import LlamaConfig, DynamicCache, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)
from torch.autograd import DeviceType

# ---- EDIT THESE ----
MIG_UUIDS = [
    "MIG-98f93df6-d522-5c00-9923-4326839cef2e",   # rank 0 — 20gb
    "MIG-153fcb3c-9412-5240-937b-67bc18179f24",   # rank 1 — 10gb
    "MIG-222909dc-5318-5493-8680-34be7bab2cc6",   # rank 2 — 10gb
]
SPLIT       = [16, 8, 8]    # any valid split, doesn't matter for counting
MODEL_NAME  = "lmsys/vicuna-7b-v1.5"
MASTER_PORT = 29700
HIDDEN_SIZE = 4096
SEQ_LEN     = 64
MB_SIZE     = 4             # microbatch size -- count at your typical value
# --------------------

# import forward_through_layers from your harness
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark_pipeline_microbatching import forward_through_layers


def worker(rank: int, world_size: int, result_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = MIG_UUIDS[rank]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(minutes=5),
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    config = LlamaConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "sdpa"

    start_layer = sum(SPLIT[:rank])
    my_indices  = list(range(start_layer, start_layer + SPLIT[rank]))

    # build layers (random weights -- we only care about kernel count, not values)
    layers = nn.ModuleList([
        LlamaDecoderLayer(config, layer_idx=i).half().to(device)
        for i in my_indices
    ])
    layers.eval()

    rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

    # build a realistic decode input: 1 new token, KV cache pre-filled
    hidden = torch.randn(MB_SIZE, 1, HIDDEN_SIZE, dtype=torch.float16, device=device)
    position_ids = torch.tensor([[SEQ_LEN]], dtype=torch.long, device=device).expand(MB_SIZE, -1)
    position_embeddings = rotary_emb(hidden, position_ids)
    cache = DynamicCache()

    # pre-fill KV cache so decode sees a real context
    prefill = torch.randn(MB_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16, device=device)
    pref_pos = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    pref_emb = rotary_emb(prefill, pref_pos)
    with torch.no_grad():
        forward_through_layers(layers, prefill, pref_emb, None, cache)

    dist.barrier()

    # ---- PROFILE ONE DECODE MICROBATCH (the thing we want to count) ----
    with torch.no_grad():
        # warmup (not profiled)
        out = forward_through_layers(layers, hidden, position_embeddings, None, cache)

        # profiled pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            out = forward_through_layers(
                layers, hidden, position_embeddings, None, cache
            )
            # include the cross-slice send (this is what your pipeline actually does)
            if rank < world_size - 1:
                h = dist.isend(out.clone(), dst=rank + 1, tag=9999)
                h.wait()

    # count unique CUDA kernel events
    cuda_events = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
    total_kernels = len(cuda_events)
    per_layer     = total_kernels / SPLIT[rank]

    result_queue.put({
        "rank":          rank,
        "layers":        SPLIT[rank],
        "total_kernels": total_kernels,
        "per_layer":     round(per_layer, 1),
        "top_kernels":   [(e.key, e.cuda_time_total) for e in
                          sorted(cuda_events, key=lambda e: e.cuda_time_total, reverse=True)[:5]],
    })

    dist.destroy_process_group()


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    q = mp.Queue()
    world_size = len(MIG_UUIDS)
    procs = []

    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, q))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=300)

    results = []
    while not q.empty():
        results.append(q.get())
    results.sort(key=lambda r: r["rank"])

    print()
    print("=" * 60)
    print(f"CUDA kernel count per layer  (MB_SIZE={MB_SIZE})")
    print("=" * 60)
    for r in results:
        print(f"\nRank {r['rank']} ({r['layers']} layers):")
        print(f"  total kernels in profiled pass : {r['total_kernels']}")
        print(f"  kernels per layer              : {r['per_layer']}")
        print(f"  top 5 kernels by CUDA time:")
        for name, t_us in r["top_kernels"]:
            print(f"    {t_us/1000:8.2f} ms   {name}")

    if results:
        avg_per_layer = sum(r["per_layer"] for r in results) / len(results)
        print()
        print(f"Suggested KERNELS_PER_LAYER = {round(avg_per_layer)}")
        print("(paste into pulp_layer_placement.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()