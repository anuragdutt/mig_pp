"""
Shared probes for measure_constants_gpu.py, measure_constants_mig.py, measure_constants_cpu.py and
probe_cpu_interactions.py. Not run on its own.

primitives(): generic GPU primitives, no model, each fitted over its shape sweep as
    t = fixed + units / rate     (units: bytes moved, or FLOPs for prefill GEMMs)
for decode GEMMs (weight streaming), prefill GEMMs, decode attention, the KV-cache append,
elementwise kernels, device copies and D2H / H2D. Fitting a probe's own sweep is a hardware
measurement; nothing here is fitted to benchmark results.
validation_layers(): one decoder layer per model at a few points, a check only, never a constant.
The layer helpers (load_model, layer_kernel_us, ...) come from measure_constants.py on branch
test/intra-gpu-setup-2-A100-40gb. Kernel times are torch.profiler's summed kernel durations,
so CPU launch gaps are out. Needs torch, numpy and transformers (the box's venv). No sudo.
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

CARD_BW = {"A100-40GB": 1555e9, "A100-80GB": 2039e9}  # peak HBM bytes/s, as hardware_constants.CARD_BW

# Architecture of the model being measured; load_model() sets them from config.json.
HIDDEN, INTER, VOCAB, N_HEADS, N_KV_HEADS, HEAD_DIM = 4096, 11008, 32000, 32, 32, 128


# ---------------------------------------------------------------------------
# bytes: the same counts as graph_placement.weight_bytes / act_bytes
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


def layer_forward(layer, h, emb, cache, mask=None):
    """forward_through_layers in benchmark_pipeline_microbatching.py, one layer."""
    r = h
    x = layer.self_attn(hidden_states=layer.input_layernorm(h), position_embeddings=emb,
                        attention_mask=mask, past_key_values=cache, use_cache=True)[0]
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


def kernels(fn, copies=False):
    """(kernel count, summed kernel time in us) of one call of fn, from torch.profiler.
    copies=True also counts Memcpy/Memset (a device clone() or copy_() is one memcpy)."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ks = [e for e in prof.events() if e.device_type == DeviceType.CUDA
          and (copies or not e.name.startswith(("Memcpy", "Memset")))]
    if not ks:
        seen = [f"{e.name} ({e.device_type})" for e in prof.events()][:20]
        raise RuntimeError(f"torch.profiler recorded no CUDA kernels; events seen: {seen}")
    return len(ks), sum(e.time_range.elapsed_us() for e in ks)


def wall_s(fn, reps=200, warmup=20):
    """Mean wall time of fn (fn must wait for its own work, as the transport does)."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def fit_line(xs, ys):
    """Least squares y = a + b x: (a, b)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


# ---------------------------------------------------------------------------
# GPU probes
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


def prefill_layer_us(layer, rotary, mb, seq, reps=5):
    """Kernel time of one prefill layer call: mb sequences of seq tokens, empty cache,
    the benchmark's causal mask (1, 1, seq, seq)."""
    from transformers import DynamicCache

    x = torch.randn(mb, seq, HIDDEN, dtype=torch.float16, device="cuda")
    pos = torch.arange(seq, device="cuda").unsqueeze(0)
    mask = torch.triu(torch.full((1, 1, seq, seq), torch.finfo(torch.float16).min,
                                 device="cuda"), diagonal=1).half()
    with torch.no_grad():
        emb = rotary(x, pos)
        layer_forward(layer, x, emb, DynamicCache(), mask)  # warm this shape
        runs = [kernels(lambda: layer_forward(layer, x, emb, DynamicCache(), mask))[1]
                for _ in range(reps)]
    return statistics.median(runs)


def endpoint_us(layer, mb):
    """Kernel time (us) of the non-layer pieces of one decode microbatch: embed gather
    (rank 0), clone (sending ranks), and on the last rank norm, lm_head, argmax."""
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.float16, device="cuda")
    ids = torch.randint(0, VOCAB, (mb, 1), device="cuda")
    hid = torch.randn(mb, 1, HIDDEN, dtype=torch.float16, device="cuda")
    head = torch.randn(VOCAB, HIDDEN, dtype=torch.float16, device="cuda")
    tok = torch.zeros(mb, 1, dtype=torch.long, device="cuda")
    norm = layer.input_layernorm  # same RMSNorm class as the final norm
    out = {}
    with torch.no_grad():
        for name, fn in (
            ("embed", lambda: F.embedding(ids, table)),
            ("clone", lambda: hid.clone()),
            ("norm", lambda: norm(hid)),
            ("lm_head", lambda: F.linear(hid, head)),
            ("argmax", lambda: tok.copy_(torch.argmax(F.linear(hid, head), dim=-1))),
        ):
            fn()
            try:
                out[name] = statistics.median(kernels(fn, name in ("clone", "argmax"))[1] for _ in range(5))
            except RuntimeError as err:  # report it and keep going: nan in that column
                print(f"  [endpoint {name}, mb {mb}] {err}", flush=True)
                out[name] = float("nan")
    out["argmax"] -= out["lm_head"]  # the argmax line also ran lm_head to get logits
    return out


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


def kernel_us(fn, copies=False, reps=5):
    """Median summed kernel time (us) of one call of fn; one warm-up call first."""
    fn()
    return statistics.median(kernels(fn, copies)[1] for _ in range(reps))


def fit_rate(points):
    """[(units, us)] -> (fixed us, units per second) of t = fixed + units / rate, least squares on
    the relative error (weights 1 / t^2), so the small sizes (mostly fixed time) and the large ones
    (mostly rate) count alike."""
    w = [1 / t ** 2 for _, t in points]
    sw = sum(w)
    sx = sum(wi * x for wi, (x, _) in zip(w, points))
    sy = sum(wi * t for wi, (_, t) in zip(w, points))
    sxx = sum(wi * x * x for wi, (x, _) in zip(w, points))
    sxy = sum(wi * x * t for wi, (x, t) in zip(w, points))
    b = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
    a = (sy - b * sx) / sw
    return a, (1e6 / b if b > 0 else float("nan"))


# ---------------------------------------------------------------------------
# steady-state GPU timing: what a primitive takes while the GPU is busy, as in a running decode
# ---------------------------------------------------------------------------
L2_BYTES = 40 << 20  # A100's L2; a slice has at most this


def warm_gpu(seconds=1.0):
    """Keep the GPU busy so its clock is where it sits during a decode (it lowers it when idle)."""
    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    b = torch.randn_like(a)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        for _ in range(10):
            torch.mm(a, b)
        torch.cuda.synchronize()


def sm_clock_mhz():
    """The GPU's current SM clock (nvidia-smi; with MIG, the parent GPU's), or None."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0])
    except Exception:
        return None


def copies_for(nbytes):
    """Input copies to rotate through so two calls in a row never find their inputs in L2: in a
    decode every layer reads its own weights and KV cache."""
    return max(1, min(64, -(-2 * L2_BYTES // max(int(nbytes), 1))))


def graph_us(calls, target_ms=20.0, replays=5):
    """(GPU time in us per call, method, every replay's time), steady state: the calls (closures, rotated) captured n
    times in a CUDA graph and replayed back to back, so neither CPU launch gaps nor idle gaps are in
    it; median of `replays` replays. method "graph", or "eager" if capture failed (a back-to-back
    eager loop, launch gaps included: kept apart from graph points, never fitted with them)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm-up off the default stream, as CUDA graph capture needs
        for c in calls[:3]:
            c()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    calls[0]()
    e1.record()
    torch.cuda.synchronize()
    n = int(min(400, max(5, target_ms * 1000 / max(e0.elapsed_time(e1) * 1000, 1.0))))
    method = "graph"
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for k in range(n):
                calls[k % len(calls)]()
        g.replay()
        torch.cuda.synchronize()
        run = g.replay
    except Exception as err:  # report and keep going
        print(f"    (no CUDA graph: {type(err).__name__}; timing an eager loop)", flush=True)
        method = "eager"
        def run():
            for k in range(n):
                calls[k % len(calls)]()
    times = []
    for _ in range(replays):
        e0.record()
        run()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1000 / n)
    return statistics.median(times), method, times


def primitives(c, peak, gpcs):
    """Generic GPU primitives on the visible device (a whole GPU or one MIG slice), as functions
    of shape, no model, timed in steady state (graph_us). Each sweep is fitted as
    t = fixed + units / rate; attention and the KV append per head shape. peak = the device's peak
    HBM bytes/s; gpcs = its GPC count. Prints the tables; returns a dict."""
    f16 = dict(dtype=torch.float16, device="cuda")
    out, clocks = {"timing": {}}, {}

    def report(name, pts, unit, scale, per, timing="device time: CUDA graph replays, GPU warmed, back to back"):
        """pts: [(units, us, method, shape dict)]. Fits the graph-timed points only; every point is
        kept raw with its shape, method and residual against the fit."""
        fit_pts = [(x, t) for x, t, m, _ in pts if m in ("graph", "wall")]
        eager = len(pts) - len(fit_pts)
        if len({x for x, _ in fit_pts}) < 2:  # a slope needs two distinct sizes
            print(f"  {name:<22} NOT FITTED: {len(fit_pts)} graph-timed points at "
                  f"{len({x for x, _ in fit_pts})} distinct sizes ({eager} eager)", flush=True)
            fixed, rate = float("nan"), float("nan")
        else:
            fixed, rate = fit_rate(fit_pts)
            worst = max(abs((fixed + x / rate * 1e6) / t - 1) for x, t in fit_pts)
            print(f"  {name:<22} fixed {fixed:7.1f} us + {unit} / {rate / scale:8.1f} {per}"
                  f"   ({len(fit_pts)} points, worst {100 * worst:.1f}% off"
                  + (f"; {eager} eager points not fitted" if eager else "") + ")", flush=True)
        out[f"{name}_fixed_us"], out[f"{name}_rate"] = fixed, rate
        out[f"{name}_points"] = [dict(units=x, us=t, method=m, residual=((fixed + x / rate * 1e6) / t - 1) if rate == rate else None, **shape)
                                 for x, t, m, shape in pts]
        out["timing"][name] = timing

    def group(name):  # warm the GPU before each sweep, note its clock
        warm_gpu(c["warm_s"])
        clocks[name] = sm_clock_mhz()

    print(f"\nprimitives (steady state: CUDA graphs, GPU warmed; SM clock at start {sm_clock_mhz()} MHz):")
    # decode GEMMs (M <= 32): weight streaming, bytes = N x K x 2
    group("gemm_decode")
    pts = []
    for N, K in c["gemm_shapes"]:
        ws = [torch.randn(N, K, **f16) for _ in range(copies_for(N * K * 2))]
        for M in c["gemm_decode_m"]:
            x, y = torch.randn(M, K, **f16), torch.empty(M, N, **f16)
            calls = [(lambda w=w: torch.matmul(x, w.t(), out=y)) for w in ws]
            t, m, reps = graph_us(calls)
            pts.append((N * K * 2, t, m, dict(N=N, K=K, M=M, replays_us=reps)))
        del ws
    report("gemm_decode", pts, "weight bytes", 1e9, "GB/s")
    # prefill GEMMs (large M): FLOPs = 2 x M x N x K
    group("gemm_prefill")
    pts = []
    for N, K in c["gemm_prefill_shapes"]:
        ws = [torch.randn(N, K, **f16) for _ in range(copies_for(N * K * 2))]
        for M in c["gemm_prefill_m"]:
            x, y = torch.randn(M, K, **f16), torch.empty(M, N, **f16)
            calls = [(lambda w=w: torch.matmul(x, w.t(), out=y)) for w in ws]
            t, m, reps = graph_us(calls)
            pts.append((2 * M * N * K, t, m, dict(N=N, K=K, M=M, replays_us=reps)))
        del ws
    report("gemm_prefill", pts, "FLOPs", 1e12, "TFLOP/s")
    # decode attention, as transformers 5.14.1 / 5.18.0 call SDPA in decode (no mask): full attention
    # as is; GQA with enable_gqa=True, K/V not expanded. Fitted per head shape.
    # bytes = 2 (K, V) x mb x kv x L x head_dim x 2
    group("attention")
    for kind, configs, gqa in (("attention_mha", [(h, h, d) for h, d in c["attn_mha"]], False),
                               ("attention_gqa", c["attn_gqa"], True)):
        for heads, kv, d in configs:
            pts = []
            for L in c["attn_ctx"]:
                for mb in c["attn_mb"]:
                    nb = 2 * mb * kv * L * d * 2
                    q = torch.randn(mb, heads, 1, d, **f16)
                    kvs = [(torch.randn(mb, kv, L, d, **f16), torch.randn(mb, kv, L, d, **f16)) for _ in range(copies_for(nb))]
                    kw = dict(enable_gqa=True) if gqa else {}
                    calls = [(lambda k=k, v=v: F.scaled_dot_product_attention(q, k, v, **kw)) for k, v in kvs]
                    t, m, reps = graph_us(calls)
                    pts.append((nb, t, m, dict(heads=heads, kv=kv, d=d, L=L, mb=mb, replays_us=reps)))
                    del kvs
            report(f"{kind}_h{heads}_kv{kv}_d{d}", pts, "KV bytes", 1e9, "GB/s")
    # KV cache append (DynamicLayer.update: torch.cat of the cache and one new token), per K/V head
    # count; bytes read + written
    group("kv_append")
    for kv in c["cat_kv_heads"]:
        pts = []
        for L in c["attn_ctx"]:
            for mb in c["attn_mb"]:
                nb = 2 * mb * kv * (L + 1) * 128 * 2
                caches = [torch.randn(mb, kv, L, 128, **f16) for _ in range(copies_for(nb))]
                new, dst = torch.randn(mb, kv, 1, 128, **f16), torch.empty(mb, kv, L + 1, 128, **f16)
                calls = [(lambda cc=cc: torch.cat([cc, new], dim=-2, out=dst)) for cc in caches]
                t, m, reps = graph_us(calls)
                pts.append((nb, t, m, dict(kv=kv, d=128, L=L, mb=mb, replays_us=reps)))
                del caches
        report(f"kv_append_kv{kv}", pts, "bytes", 1e9, "GB/s")
    # elementwise (binary, in place): bytes = 3 x numel x 2. Small sizes give the fixed time per
    # kernel, sizes past L2 the rate.
    group("elementwise")
    pts = []
    for n in c["elem_numel"]:
        bs = [torch.randn(n, **f16) for _ in range(copies_for(3 * n * 2))]
        a = torch.randn(n, **f16)
        calls = [(lambda b=b: a.mul_(b)) for b in bs]
        t, m, reps = graph_us(calls)
        pts.append((3 * n * 2, t, m, dict(numel=n, replays_us=reps)))
        del bs
    report("elementwise", pts, "bytes", 1e9, "GB/s")
    # device copy (DtoD): bytes read + written
    group("device_copy")
    pts = []
    for n in c["elem_numel"]:
        srcs = [torch.randn(n, **f16) for _ in range(copies_for(2 * n * 2))]
        dst = torch.empty(n, **f16)
        calls = [(lambda s_=s_: dst.copy_(s_)) for s_ in srcs]
        t, m, reps = graph_us(calls)
        pts.append((2 * n * 2, t, m, dict(numel=n, replays_us=reps)))
        del srcs
    report("device_copy", pts, "bytes", 1e9, "GB/s")
    # host copies as the transport does them (wall time, a side stream + event wait; CPU in it)
    d2h, h2d = [], []
    for nb in c["copy_bytes"]:
        a, b = device_copies(nb)
        d2h.append((nb, nb / a * 1e6, "wall", dict(bytes=nb)))
        h2d.append((nb, nb / b * 1e6, "wall", dict(bytes=nb)))
    wall = "wall time: side-stream copy + event wait, as the transport does it (CPU and GPU, mean of 200)"
    report("d2h", d2h, "bytes", 1e9, "GB/s", timing=wall)
    report("h2d", h2d, "bytes", 1e9, "GB/s", timing=wall)

    out.update(peak=peak, gpcs=gpcs, sm_clock_mhz=clocks)
    print(f"  SM clock after each warm-up (MHz): {clocks}")
    rates = [k for k in out if k.endswith("_rate") and k not in ("gemm_prefill_rate", "d2h_rate", "h2d_rate") and out[k] == out[k]]
    print("  as fractions of peak: " + ", ".join(f"{k[:-5]} {out[k] / peak:.3f}" for k in rates)
          + f"; prefill {out['gemm_prefill_rate'] / gpcs / 1e12:.1f} TFLOP/s per GPC")
    return out


def layer_kernel_us_steady(layer, rotary, mb, ctx, calls=10):
    """Kernel time (us) of one decode layer call in steady state: the GPU warmed, `calls` calls
    back to back (the cache grows by one token each, negligible), profiler kernel time / calls."""
    h, emb, _ = decode_input(rotary, mb, ctx)
    with torch.no_grad():
        cache = cache_at(layer, rotary, mb, ctx)
        layer_forward(layer, h, emb, cache)
        warm_gpu(0.3)
        _, us = kernels(lambda: [layer_forward(layer, h, emb, cache) for _ in range(calls)])
    return us / calls


def prefill_layer_us_steady(layer, rotary, mb, seq, calls=5):
    """Kernel time (us) of one prefill layer call in steady state (fresh cache each call)."""
    from transformers import DynamicCache

    x = torch.randn(mb, seq, HIDDEN, dtype=torch.float16, device="cuda")
    pos = torch.arange(seq, device="cuda").unsqueeze(0)
    mask = torch.triu(torch.full((1, 1, seq, seq), torch.finfo(torch.float16).min, device="cuda"), diagonal=1).half()
    with torch.no_grad():
        emb = rotary(x, pos)
        layer_forward(layer, x, emb, DynamicCache(), mask)
        warm_gpu(0.3)
        _, us = kernels(lambda: [layer_forward(layer, x, emb, DynamicCache(), mask) for _ in range(calls)])
    return us / calls


def validation_layers(c):
    """VALIDATION ONLY, never a constant: one decoder layer's kernel time per model at a few
    (mb, ctx), in steady state like the primitives, to check that layer time composed from the
    primitives matches."""
    out = {}
    for path in c["validate_models"]:
        try:
            layer, rotary = load_model(path)
        except Exception as err:  # no config / no network: report and keep going
            print(f"  [validation {path}] skipped: {err}")
            continue
        rows = {}
        for mb in c["validate_mb"]:
            for ctx in c["validate_ctx"]:
                w = [layer_kernel_us_steady(layer, rotary, mb, ctx) for _ in range(c["windows"])]
                rows[f"decode mb{mb} ctx{ctx}"] = dict(median=statistics.median(w), spread=(max(w) - min(w)) / min(w), windows=w)
            w = [prefill_layer_us_steady(layer, rotary, mb, c["seq_len"]) for _ in range(c["windows"])]
            rows[f"prefill mb{mb} seq{c['seq_len']}"] = dict(median=statistics.median(w), spread=(max(w) - min(w)) / min(w), windows=w)
        out[path] = rows
        print(f"\nvalidation layer, {path} (kernel us, steady state, median of {c['windows']} windows, spread): "
              + ", ".join(f"{k} {v['median']:.0f} ({100 * v['spread']:.1f}%)" for k, v in rows.items()))
        del layer, rotary
        torch.cuda.empty_cache()
    return out


PRIMITIVE_CONFIG = dict(
    gemm_shapes=[(4096, 4096), (11008, 4096), (4096, 11008), (1024, 4096), (5120, 5120), (13824, 5120),
                 (3584, 3584), (18944, 3584), (32000, 4096)],  # (N out, K in), the 7B-14B family
    gemm_decode_m=[1, 8, 32],
    gemm_prefill_shapes=[(4096, 4096), (11008, 4096)],
    gemm_prefill_m=[256, 1024, 2048],
    attn_mha=[(32, 128), (40, 128)],  # full attention: (heads, head dim), e.g. Llama-2-7B / 13B
    attn_gqa=[(28, 4, 128), (32, 8, 128)],  # GQA: (query heads, K/V heads, head dim), e.g. Qwen2.5-7B, Mistral-7B
    attn_ctx=[64, 320, 576, 1088],
    attn_mb=[1, 8, 32],
    cat_kv_heads=[4, 8, 32, 40],
    elem_numel=[4096, 65536, 1 << 25, 1 << 26],  # small: fixed time per kernel; 64-128 MB: past L2, the rate
    copy_bytes=[16 << 10, 64 << 10, 512 << 10, 4 << 20],
    warm_s=1.0,  # seconds of GPU load before each sweep
    validate_models=["lmsys/vicuna-7b-v1.5", "Qwen/Qwen2.5-7B"],  # HF ids (config.json only) or local dirs
    validate_mb=[1, 8, 32],
    validate_ctx=[64, 320, 575],
    windows=5,  # validation layers: repeated windows per point (median and spread kept)
    seq_len=64,
)


def host_info():
    """Governor, CPU model, allowed CPUs, torch threads: what the CPU numbers depend on."""
    def read(path):
        try:
            return open(path).read().strip()
        except OSError:
            return "?"
    model = "?"
    for line in read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count()))
    try:
        import transformers
        tf = transformers.__version__
    except ImportError:
        tf = "not installed"
    return dict(cpu=model, governor=read(f"/sys/devices/system/cpu/cpu{cpus[0]}/cpufreq/scaling_governor"),
                allowed_cpus=len(cpus), torch_threads=torch.get_num_threads(),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", ""), torch=torch.__version__, transformers=tf)


def save_json(path, payload):
    import json

    with open(os.path.expanduser(path), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {path}")
