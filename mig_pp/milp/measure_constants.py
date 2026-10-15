"""
Measures DEFAULT_CONSTANTS of pulp_layer_placement.py and prints them as a dict to
paste over it. Edit CONFIG in main(), then on the box:

    CUDA_VISIBLE_DEVICES=<gpu index> taskset -c <that GPU's NUMA CPUs> python3 measure_constants.py

WHERE TO RUN IT: one whole GPU with MIG disabled and nothing else on it. No sudo: the
probes are ordinary CUDA, gloo on localhost, host memcpy and torch.profiler kernel
tracing (tracing needs no admin rights; only hardware counters, e.g. ncu metrics, do).
A GPU with MIG enabled only shows CUDA its slices, so the script refuses to run there.

Whole-GPU numbers only for now. Not measured: how fast each MIG slice is -- the ILP
assumes card peak x GPCs/7 (3g -> 3/7). If the predictions are off, that is the first
thing to measure on real slices.

HOW EACH CONSTANT IS MEASURED, matching pulp_layer_placement.terms():
  eff_weight         one layer's weight bytes / its decode GEMMs' time / card peak
  eff_act            activation + KV bytes / (layer kernel time - weight time - launch
                     term); the ILP adds kernels_per_layer x launch_us to a layer's GPU time
  launch_us          CPU time to dispatch one decode layer call / its kernel count
  kernels_per_layer, rotary_kernels   kernels counted by torch.profiler
  pcie_bw, recv_h2d_bw                one activation's D2H / H2D as the transport does it
  memcpy_bw, shm_to_pinned_bw         the transport's two host copies
  handshake_ack_us, tok_roundtrip_us, barrier_us   gloo, CONFIG["world"] processes
  token_staging_us, step_start_us     TokenExchange's two copies; the position_ids sync
Layer kernel times are the profiler's summed kernel durations, so launch gaps are out.
"""

import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import DeviceType

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CARD_BW = {"A100-40GB": 1555e9, "A100-80GB": 2039e9,  # peak HBM bytes/s, as in the ILP
           "A100-80GB-PCIe": 1935e9}  # efficiencies are fractions of the card actually measured

# Architecture of the model being measured; load_model() sets them from config.json.
HIDDEN, INTER, VOCAB, N_HEADS, N_KV_HEADS, HEAD_DIM = 4096, 11008, 32000, 32, 32, 128


# ---------------------------------------------------------------------------
# bytes -- the same counts as pulp_layer_placement.weight_bytes / act_bytes (its test
# lifts these three functions out of this file and compares)
# ---------------------------------------------------------------------------
def kv_dim():
    return N_KV_HEADS * HEAD_DIM


def act_bytes_per_token(ctx):
    B = 2  # FP16
    kvd = kv_dim()
    q_out = N_HEADS * HEAD_DIM
    qkv = (HIDDEN + q_out + 2 * kvd) * B  # read x, write Q,K,V
    flash = (q_out + 2 * ctx * kvd + q_out) * B  # read Q + full-ctx K,V, write out
    oproj = (q_out + HIDDEN) * B
    gate = (HIDDEN + INTER) * B
    up = (HIDDEN + INTER) * B
    swiglu = (2 * INTER + INTER) * B
    down = (INTER + HIDDEN) * B
    misc = (6 * HIDDEN) * B
    return qkv + flash + oproj + gate + up + swiglu + down + misc


def weight_bytes_per_layer():
    kvd = kv_dim()
    q_out = N_HEADS * HEAD_DIM
    return (HIDDEN * q_out + 2 * HIDDEN * kvd + q_out * HIDDEN + 3 * HIDDEN * INTER) * 2


# ---------------------------------------------------------------------------
# model and timing helpers
# ---------------------------------------------------------------------------
def load_model(path):
    """One decoder layer + rotary of the model at path, random weights (timing only)."""
    global HIDDEN, INTER, VOCAB, N_HEADS, N_KV_HEADS, HEAD_DIM
    from transformers import AutoConfig
    import model_family

    cfg = AutoConfig.from_pretrained(os.path.expanduser(path))
    cfg._attn_implementation = "sdpa"  # as the benchmark
    HIDDEN, INTER, VOCAB = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    N_HEADS = cfg.num_attention_heads
    N_KV_HEADS = getattr(cfg, "num_key_value_heads", None) or N_HEADS
    HEAD_DIM = getattr(cfg, "head_dim", None) or HIDDEN // N_HEADS
    fam = model_family.resolve(cfg.model_type)
    layer = fam.decoder_layer(cfg, layer_idx=0).half().cuda().eval()
    rotary = fam.rotary_embedding(config=cfg, device=torch.device("cuda"))
    return layer, rotary


def layer_forward(layer, h, emb, cache):
    """forward_through_layers in benchmark_pipeline_microbatching.py, one layer."""
    r = h
    x = layer.self_attn(hidden_states=layer.input_layernorm(h), position_embeddings=emb,
                        attention_mask=None, past_key_values=cache, use_cache=True)[0]
    h = r + x
    return h + layer.mlp(layer.post_attention_layernorm(h))


def cache_at(layer, rotary, mb, ctx):
    """A DynamicCache holding ctx tokens for mb sequences (what decode step ctx-63 sees)."""
    from transformers import DynamicCache

    cache = DynamicCache()
    x = torch.randn(mb, ctx, HIDDEN, dtype=torch.float16, device="cuda")
    pos = torch.arange(ctx, device="cuda").unsqueeze(0).expand(mb, -1)
    with torch.no_grad():
        layer_forward(layer, x, rotary(x, pos), cache)
    return cache


def decode_input(rotary, mb, ctx):
    h = torch.randn(mb, 1, HIDDEN, dtype=torch.float16, device="cuda")
    pos = torch.full((mb, 1), ctx, dtype=torch.long, device="cuda")
    return h, rotary(h, pos), pos


def kernels(fn):
    """(kernel count, summed kernel time in us) of one call of fn, from torch.profiler."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ks = [e for e in prof.events() if e.device_type == DeviceType.CUDA
          and not e.name.startswith(("Memcpy", "Memset"))]
    if not ks:
        raise RuntimeError("torch.profiler recorded no CUDA kernels")
    return len(ks), sum(e.time_range.elapsed_us() for e in ks)


def wall_s(fn, reps=200, warmup=20):
    """Mean wall time of fn (fn must wait for its own work, as the transport does)."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def gemm_bytes_per_s(M, reps=50):
    """Weight streaming rate of one layer's decode GEMMs for M tokens: q,k,v,gate,up
    stacked in one matrix, then o and down. Big kernels, so launches do not show."""
    q, kv = N_HEADS * HEAD_DIM, kv_dim()
    wa = torch.randn(q + 2 * kv + 2 * INTER, HIDDEN, dtype=torch.float16, device="cuda")
    wo = torch.randn(HIDDEN, q, dtype=torch.float16, device="cuda")
    wd = torch.randn(HIDDEN, INTER, dtype=torch.float16, device="cuda")
    x = torch.randn(M, HIDDEN, dtype=torch.float16, device="cuda")
    y = torch.randn(M, q, dtype=torch.float16, device="cuda")
    z = torch.randn(M, INTER, dtype=torch.float16, device="cuda")

    def fn():
        F.linear(x, wa), F.linear(y, wo), F.linear(z, wd)

    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return weight_bytes_per_layer() / (start.elapsed_time(end) / reps * 1e-3)


def layer_kernel_us(layer, rotary, mb, ctx, reps=5):
    """(kernels, median kernel time in us) of one decode layer call at ctx cached tokens."""
    h, emb, _ = decode_input(rotary, mb, ctx)
    with torch.no_grad():
        layer_forward(layer, h, emb, cache_at(layer, rotary, mb, ctx))  # warm this shape
        runs = []
        for _ in range(reps):
            cache = cache_at(layer, rotary, mb, ctx)
            runs.append(kernels(lambda: layer_forward(layer, h, emb, cache)))
    return runs[0][0], statistics.median(us for _, us in runs)


def layer_dispatch_us(layer, rotary, mb, ctx, reps=30):
    """CPU time to queue one decode layer call (the GPU is idle at each start)."""
    h, emb, _ = decode_input(rotary, mb, ctx)
    cache = cache_at(layer, rotary, mb, ctx)
    out = []
    with torch.no_grad():
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            layer_forward(layer, h, emb, cache)
            out.append(time.perf_counter() - t0)
    return statistics.median(out) * 1e6


def device_copies(nbytes):
    """(D2H, H2D) bytes/s of one activation, as isend+FLUSH and RECV do them: a side
    stream, then a wait on that copy's event."""
    n = nbytes // 2
    src = torch.randn(n, dtype=torch.float16, device="cuda")
    dst = torch.empty(n, dtype=torch.float16, device="cuda")
    pinned = torch.empty(n, dtype=torch.float16).pin_memory()
    copy_stream, recv_stream = torch.cuda.Stream(), torch.cuda.Stream()
    ev = torch.cuda.Event()

    def d2h():
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            pinned.copy_(src, non_blocking=True)
        ev.record(copy_stream)
        ev.synchronize()

    def h2d():
        with torch.cuda.stream(recv_stream):
            dst.copy_(pinned, non_blocking=True)
        ev.record(recv_stream)
        ev.synchronize()

    return nbytes / wall_s(d2h), nbytes / wall_s(h2d)


def host_copies(nbytes):
    """(pinned -> SHM, SHM -> pinned) bytes/s: _flush_staging_to_shm (numpy slice
    assignment) and _queue_read_from_slot (torch copy_ from a numpy view of SHM)."""
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(create=True, size=nbytes)
    try:
        return _host_copy_rates(shm.buf, nbytes)  # its views of SHM are gone on return
    finally:
        shm.close()
        shm.unlink()


def _host_copy_rates(buf, nbytes):
    slot = np.ndarray((nbytes,), dtype=np.uint8, buffer=buf)
    pinned = torch.empty(nbytes // 2, dtype=torch.float16).pin_memory()
    raw = pinned.numpy().view(np.uint8)
    peer = torch.from_numpy(slot.view(np.float16))

    def send():
        slot[:nbytes] = raw[:nbytes]

    def recv():
        pinned.copy_(peer)

    return nbytes / wall_s(send), nbytes / wall_s(recv)


def token_staging_us(batch):
    """TokenExchange: D2H of next_tokens + stream sync, then H2D + stream sync."""
    dev_tok = torch.zeros((batch, 1), dtype=torch.long, device="cuda")
    host = torch.empty((batch, 1), dtype=torch.long).pin_memory()

    def exchange():
        host.copy_(dev_tok)
        torch.cuda.current_stream().synchronize()
        dev_tok.copy_(host)
        torch.cuda.current_stream().synchronize()

    return wall_s(exchange) * 1e6


def step_start_us(mb, seq):
    """Rank 0's work at a decode step start: position_ids built on the device (a pageable
    H2D that waits for itself), expand().contiguous(), the step event's first record."""
    def step():
        torch.tensor([[seq]], dtype=torch.long, device="cuda").expand(mb, -1).contiguous()
        torch.cuda.Event(enable_timing=True).record()

    return wall_s(step) * 1e6


def _gloo_worker(rank, world, port, batch, reps, q):
    """handshake + ACK (ranks 0 and 1, as FLUSH/RECV), next_tokens (last rank -> 0),
    barrier (all). Times are rank 0's. Second pass counts; the first is a warmup."""
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    hs, ack = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    tok = torch.zeros((batch, 1), dtype=torch.long)
    out = {}
    for _ in range(2):
        dist.barrier()
        t0 = time.perf_counter()
        for i in range(reps):
            if rank == 0:  # FLUSH: post the ACK irecv, send the handshake ... wait() later
                r = dist.irecv(ack, src=1, tag=2 * i + 1)
                dist.send(hs, dst=1, tag=2 * i)
                r.wait()
            elif rank == 1:  # RECV: handshake in, ACK out
                dist.recv(hs, src=0, tag=2 * i)
                dist.send(ack, dst=0, tag=2 * i + 1)
        out["handshake_ack_us"] = (time.perf_counter() - t0) / reps * 1e6
        dist.barrier()
        t0 = time.perf_counter()
        for i in range(reps):
            if rank == world - 1:
                dist.send(tok, dst=0, tag=100_000 + i)
            elif rank == 0:
                dist.recv(tok, src=world - 1, tag=100_000 + i)
        out["tok_roundtrip_us"] = (time.perf_counter() - t0) / reps * 1e6
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(reps):
            dist.barrier()
        out["barrier_us"] = (time.perf_counter() - t0) / reps * 1e6
    if rank == 0:
        q.put(out)
    dist.destroy_process_group()


def gloo_probes(world, port, batch, reps=200):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_gloo_worker, args=(r, world, port, batch, reps, q))
             for r in range(world)]
    for p in procs:
        p.start()
    out = q.get(timeout=300)
    for p in procs:
        p.join(timeout=60)
    return out


def detect_card(name):
    if "A100" in name:
        if torch.cuda.get_device_properties(0).total_memory < 45e9:
            return "A100-40GB"
        return "A100-80GB-PCIe" if "PCIe" in name else "A100-80GB"
    raise SystemExit(f"{name}: unknown card; set CONFIG['card'] and add its peak to CARD_BW")


# ---------------------------------------------------------------------------
def main():
    CONFIG = dict(
        model_path="lmsys/vicuna-7b-v1.5",  # HF id (only config.json is fetched) or a local model dir
        card=None,  # None: from the device; or "A100-40GB" / "A100-80GB"
        mbs=[1, 8, 32],  # microbatch sizes for the layer probes
        ctxs=[64, 192, 320, 448, 575],  # cached tokens: decode steps 1 .. 512 with seq 64
        transport_mb=8,  # activation of this microbatch for the copy probes (decode shape)
        seq_len=64,
        batch=32,  # next_tokens rows
        world=4,  # ranks for the gloo probes
        port=29900,
    )
    c = CONFIG

    name = torch.cuda.get_device_name(0)
    if "MIG" in name:
        raise SystemExit(f"{name} is a MIG slice: run on a whole GPU (MIG disabled), "
                         "e.g. CUDA_VISIBLE_DEVICES=<index>")
    card = c["card"] or detect_card(name)
    peak = CARD_BW[card]
    layer, rotary = load_model(c["model_path"])
    import transformers

    print(f"{name} as {card} (peak {peak / 1e9:.0f} GB/s); {c['model_path']}: hidden {HIDDEN}, "
          f"inter {INTER}, heads {N_HEADS}/{N_KV_HEADS} kv, head_dim {HEAD_DIM}; "
          f"torch {torch.__version__}, transformers {transformers.__version__}")

    # weight streaming
    rates = {M: gemm_bytes_per_s(M) for M in c["mbs"]}
    eff_weight = statistics.mean(rates.values()) / peak
    print("\neff_weight (decode GEMMs, M tokens): " + ", ".join(
        f"M={M}: {r / 1e9:.0f} GB/s = {r / peak:.3f}" for M, r in rates.items())
        + f"  -> mean {eff_weight:.3f}")

    # one layer: kernels, kernel time, CPU dispatch
    kpl, _ = layer_kernel_us(layer, rotary, 1, c["seq_len"], reps=1)
    dispatch = layer_dispatch_us(layer, rotary, c["transport_mb"], c["seq_len"])
    launch_us = dispatch / kpl
    with torch.no_grad():
        h, _, pos = decode_input(rotary, c["transport_mb"], c["seq_len"])
        rotary_kernels, _ = kernels(lambda: rotary(h, pos))
    print(f"kernels_per_layer {kpl}, rotary_kernels {rotary_kernels}, launch_us {launch_us:.1f} "
          f"(CPU {dispatch:.0f} us to queue one layer)")

    # activation + KV traffic, both ways of charging the launch term
    w_us = weight_bytes_per_layer() / (peak * eff_weight) * 1e6
    launch_term = kpl * launch_us
    print(f"\neff_act: one decode layer call; weight time {w_us:.0f} us, launch term "
          f"(kernels x launch_us) {launch_term:.0f} us")
    print(f"  {'mb':>3} {'ctx':>4} {'kernel us':>10} {'act ideal us':>13} {'ILP form':>9} {'no launch':>9}")
    ideal, ilp, nolaunch = [], [], []
    for mb in c["mbs"]:
        for ctx in c["ctxs"]:
            _, t = layer_kernel_us(layer, rotary, mb, ctx)
            a = mb * act_bytes_per_token(ctx) / peak * 1e6
            e1 = a / (t - w_us - launch_term) if t - w_us - launch_term > 0 else float("nan")
            e2 = a / (t - w_us) if t - w_us > 0 else float("nan")
            ideal.append(a), ilp.append(e1), nolaunch.append(e2)
            print(f"  {mb:3d} {ctx:4d} {t:10.0f} {a:13.1f} {e1:9.3f} {e2:9.3f}")

    def scalar(effs):  # time-equivalent: total ideal time / total measured time
        ok = [(a, e) for a, e in zip(ideal, effs) if e == e]
        return (sum(a for a, _ in ok) / sum(a / e for a, e in ok), len(ok)) if ok else (float("nan"), 0)

    eff_act, n_ok = scalar(ilp)
    eff_act_nl, n_nl = scalar(nolaunch)
    print(f"  ILP form ({n_ok}/{len(ilp)} samples usable): eff_act {eff_act:.3f}")
    print(f"  without the launch term in GPU time ({n_nl}/{len(ilp)}): eff_act {eff_act_nl:.3f}")

    # transport
    dec = c["transport_mb"] * HIDDEN * 2
    pre = dec * c["seq_len"]
    d2h, h2d = device_copies(dec)
    send, recv = host_copies(dec)
    d2h_p, h2d_p = device_copies(pre)
    send_p, recv_p = host_copies(pre)
    print(f"\ntransport, one activation of mb {c['transport_mb']} (GB/s)      decode {dec // 1024} KiB"
          f"   prefill {pre // 1024} KiB")
    for label, a, b in (("D2H (pcie_bw)", d2h, d2h_p), ("H2D (recv_h2d_bw)", h2d, h2d_p),
                        ("pinned -> SHM (memcpy_bw)", send, send_p),
                        ("SHM -> pinned (shm_to_pinned_bw)", recv, recv_p)):
        print(f"  {label:34s} {a / 1e9:8.2f} {b / 1e9:12.2f}")

    staging = token_staging_us(c["batch"])
    step = step_start_us(c["transport_mb"], c["seq_len"])
    g = gloo_probes(c["world"], c["port"], c["batch"])
    print(f"\ntoken_staging_us {staging:.1f}, step_start_us {step:.1f}, handshake_ack_us "
          f"{g['handshake_ack_us']:.1f}, tok_roundtrip_us {g['tok_roundtrip_us']:.1f}, "
          f"barrier_us {g['barrier_us']:.1f}")

    print("\n# measured on " + f"{name}, {os.path.basename(os.path.expanduser(c['model_path']))}"
          + f", mb {c['transport_mb']} transport")
    def num(x):  # a pasteable literal, also for a probe that came back NaN
        return f"{x:.3f}" if x == x else 'float("nan")'

    print(f"""DEFAULT_CONSTANTS = dict(
    eff_weight={eff_weight:.3f},
    eff_act={num(eff_act)},  # without the launch term in GPU time: {num(eff_act_nl)}
    launch_us={launch_us:.1f},
    kernels_per_layer={kpl},
    pcie_bw={d2h:.3g},
    recv_h2d_bw={h2d:.3g},
    shm_to_pinned_bw={recv:.3g},
    memcpy_bw={send:.3g},
    handshake_ack_us={g['handshake_ack_us']:.1f},
    tok_roundtrip_us={g['tok_roundtrip_us']:.1f},
    token_staging_us={staging:.1f},
    step_start_us={step:.1f},
    barrier_us={g['barrier_us']:.1f},
    rotary_kernels={rotary_kernels},
    timing_event_calls=2,  # from the benchmark code, not measured
    send_queue_calls=2,  # from the transport code, not measured
)""")


if __name__ == "__main__":
    main()
