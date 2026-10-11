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
from transformers import AutoConfig, DynamicCache
from torch.autograd import DeviceType


def _resolve_classes(model_type: str):
    """Decoder-layer + rotary classes for the model family under test.

    Prefers the repo's model_family so this agrees with the benchmark.
    """
    try:
        import model_family
        fam = model_family.resolve(model_type)
        # Field names are decoder_layer / rotary_embedding (see
        # model_family.Family). The previous *_cls names did not exist, so the
        # broad except below swallowed an AttributeError and this ALWAYS fell
        # through to the hardcoded fallback -- silently, which is the worst case.
        return fam.decoder_layer, fam.rotary_embedding
    except (ImportError, KeyError, AttributeError) as e:
        print(f"[warn] model_family.resolve({model_type!r}) unavailable "
              f"({type(e).__name__}: {e}); using direct transformers imports")
    if model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import (
            MistralDecoderLayer as DL, MistralRotaryEmbedding as RE)
    elif model_type in ("qwen2", "qwen2_moe"):
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2DecoderLayer as DL, Qwen2RotaryEmbedding as RE)
    else:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer as DL, LlamaRotaryEmbedding as RE)
    return DL, RE

# ---- EDIT THESE (or override from the environment, see below) ----
# Env overrides so this runs on whatever cluster is up without editing the file:
#   MIG_UUIDS="uuid0,uuid1,uuid2,uuid3"   (world size follows the list length)
#   KM_SPLIT="12,10,5,5"                  (must have one entry per UUID)
#   KM_MODEL=... KM_MB_SIZE=8 KM_SEQ_LEN=64 KM_PORT=29700
MIG_UUIDS = [
    "MIG-98f93df6-d522-5c00-9923-4326839cef2e",  # rank 0 — 20gb
    "MIG-153fcb3c-9412-5240-937b-67bc18179f24",  # rank 1 — 10gb
    "MIG-222909dc-5318-5493-8680-34be7bab2cc6",  # rank 2 — 10gb
]
SPLIT = [16, 8, 8]  # any valid split, doesn't matter for counting
MODEL_NAME = "lmsys/vicuna-7b-v1.5"
MASTER_PORT = 29700
HIDDEN_SIZE = 4096
SEQ_LEN = 64
MB_SIZE = 4  # microbatch size -- count at your typical value
# --------------------

if os.environ.get("MIG_UUIDS"):
    MIG_UUIDS = [u.strip() for u in os.environ["MIG_UUIDS"].split(",") if u.strip()]
if os.environ.get("KM_SPLIT"):
    SPLIT = [int(x) for x in os.environ["KM_SPLIT"].split(",")]
MODEL_NAME = os.environ.get("KM_MODEL", MODEL_NAME)
MB_SIZE = int(os.environ.get("KM_MB_SIZE", MB_SIZE))
SEQ_LEN = int(os.environ.get("KM_SEQ_LEN", SEQ_LEN))
MASTER_PORT = int(os.environ.get("KM_PORT", MASTER_PORT))

if len(SPLIT) != len(MIG_UUIDS):
    raise SystemExit(
        f"SPLIT has {len(SPLIT)} entries but MIG_UUIDS has {len(MIG_UUIDS)}; "
        "they must match (one layer count per rank)."
    )

# import forward_through_layers from your harness
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark_pipeline_microbatching import forward_through_layers


def worker(rank: int, world_size: int, result_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = MIG_UUIDS[rank]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=5),
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Read the architecture rather than assuming Llama: the cluster now sweeps
    # mistral / qwen2 / 13B / 14B / 24B, and HIDDEN_SIZE=4096 is wrong for most
    # of them (and Mistral-Small-24B has head_dim != hidden/heads).
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "sdpa"
    hidden_size = config.hidden_size          # an int, stays an int
    model_type = getattr(config, "model_type", "llama")

    DL, RE = _resolve_classes(model_type)

    start_layer = sum(SPLIT[:rank])
    my_indices = list(range(start_layer, start_layer + SPLIT[rank]))

    # build layers (random weights -- we only care about kernel count, not values)
    layers = nn.ModuleList(
        [DL(config, layer_idx=i).half().to(device) for i in my_indices]
    )
    layers.eval()

    rotary_emb = RE(config=config, device=device)

    # build a realistic decode input: 1 new token, KV cache pre-filled
    # NOTE: keep the SIZE (hidden_size) and the TENSOR (hidden_states) in
    # separate names. Reusing one name made the size a tensor and then passed
    # that tensor to torch.randn as a dimension -> TypeError.
    hidden_states = torch.randn(
        MB_SIZE, 1, hidden_size, dtype=torch.float16, device=device
    )
    position_ids = torch.tensor([[SEQ_LEN]], dtype=torch.long, device=device).expand(
        MB_SIZE, -1
    )
    position_embeddings = rotary_emb(hidden_states, position_ids)
    cache = DynamicCache()

    # pre-fill KV cache so decode sees a real context
    prefill = torch.randn(
        MB_SIZE, SEQ_LEN, hidden_size, dtype=torch.float16, device=device
    )
    pref_pos = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    pref_emb = rotary_emb(prefill, pref_pos)
    with torch.no_grad():
        forward_through_layers(layers, prefill, pref_emb, None, cache)

    dist.barrier()

    # ---- PROFILE ONE DECODE MICROBATCH (the thing we want to count) ----
    with torch.no_grad():
        # warmup (not profiled)
        out = forward_through_layers(
            layers, hidden_states, position_embeddings, None, cache
        )

        # profiled pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            out = forward_through_layers(
                layers, hidden_states, position_embeddings, None, cache
            )
            # The cross-slice send is deliberately NOT included.
            #
            # It cannot work here: nothing posts a matching irecv, and
            # mig_transport.register_hooks() is never called, so dist.isend is
            # the stock gloo one with no peer -- h.wait() blocks until the
            # 5-minute join timeout and the profile is never collected. The
            # clone() would also be counted as layer work, inflating the
            # per-layer kernel count.
            #
            # KERNELS_PER_LAYER is a per-layer compute property; transport cost
            # is modelled separately from the transport constants. Measure the
            # real send with measure_constants.py (RECV_H2D_BW,
            # SHM_TO_PINNED_BW, HANDSHAKE_ACK_US), which exercises the actual
            # SHM path with a real peer.
            _ = out

    cuda_events = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]

    # TWO different counts, reported separately -- they are not the same number
    # and the model needs the LAUNCH count.
    #   distinct_names : how many different kernels appear (what this script
    #                    used to report, via len(cuda_events))
    #   total_launches : how many times a kernel was launched (what
    #                    measure_constants.calculate_KERNELS_PER_LAYER reports,
    #                    via sum(e.count))
    # KERNELS_PER_LAYER is multiplied by LAUNCH_US, so it must be LAUNCHES.
    # Reporting only distinct names understated it and silently disagreed with
    # measure_constants.py for the same constant.
    distinct_names = len(cuda_events)
    total_launches = int(sum(e.count for e in cuda_events))

    # Total CUDA time, so launch overhead can be compared against real work
    # rather than assumed to dominate.
    total_cuda_us = float(sum(e.self_device_time_total for e in cuda_events))

    result_queue.put(
        {
            "rank": rank,
            "layers": SPLIT[rank],
            "distinct_names": distinct_names,
            "total_launches": total_launches,
            "launches_per_layer": round(total_launches / SPLIT[rank], 1),
            "names_per_layer": round(distinct_names / SPLIT[rank], 1),
            "cuda_us_total": round(total_cuda_us, 1),
            "cuda_us_per_layer": round(total_cuda_us / SPLIT[rank], 1),
            "top_kernels": [
                (e.key, e.self_device_time_total, e.count)
                for e in sorted(
                    cuda_events, key=lambda e: e.self_device_time_total, reverse=True
                )[:8]
            ],
        }
    )

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
        print(f"  kernel LAUNCHES in pass  : {r['total_launches']}")
        print(f"  distinct kernel names    : {r['distinct_names']}")
        print(f"  LAUNCHES per layer       : {r['launches_per_layer']}  <-- "
              f"KERNELS_PER_LAYER")
        print(f"  names per layer          : {r['names_per_layer']}")
        print(f"  CUDA time per layer      : {r['cuda_us_per_layer'] / 1000:.3f} ms")
        print(f"  top kernels by self CUDA time:")
        for name, t_us, cnt in r["top_kernels"]:
            print(f"    {t_us / 1000:8.3f} ms  x{cnt:<4d} {name}")

    if results:
        avg_launch = sum(r["launches_per_layer"] for r in results) / len(results)
        avg_names = sum(r["names_per_layer"] for r in results) / len(results)
        print()
        print("=" * 60)
        print(f"KERNELS_PER_LAYER = {round(avg_launch)}   (launches -- use this; "
              f"it is multiplied by LAUNCH_US)")
        print(f"  distinct names/layer = {round(avg_names)}   (NOT the constant; "
              f"this is what the old version printed)")
        print()
        # Per-rank spread matters: ranks on smaller slices run the same kernels
        # but slower, so a single KERNELS_PER_LAYER is only valid if the LAUNCH
        # count really is slice-independent. Show whether it is.
        spread = [r["launches_per_layer"] for r in results]
        if spread and min(spread) > 0:
            print(f"  per-rank launches/layer: {spread}  "
                  f"(spread {max(spread) / min(spread):.2f}x)")
            if max(spread) / min(spread) > 1.1:
                print("  NOTE: launches/layer differs across ranks -- a single")
                print("        KERNELS_PER_LAYER is not valid; key it per rank.")
        print()
        print("Per-rank CUDA time/layer (compare the model's per-slice term):")
        for r in results:
            print(f"  rank{r['rank']} ({r['layers']} layers): "
                  f"{r['cuda_us_per_layer'] / 1000:.3f} ms/layer")
    print("=" * 60)


if __name__ == "__main__":
    main()
