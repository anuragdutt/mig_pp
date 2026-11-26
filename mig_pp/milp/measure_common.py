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

DEVICE = "cuda"  # load_layers / stack_cache / harness_launch_us only (tests set "cpu")

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


def kernels(fn, copies=False, detail=False):
    """(kernel count, summed kernel time in us) of one call of fn, from torch.profiler.
    copies=True also counts Memcpy/Memset (a device clone() or copy_() is one memcpy).
    detail=True adds a third item: [(kernel name, us)] in the order the kernels started."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ks = [e for e in prof.events() if e.device_type == DeviceType.CUDA
          and (copies or not e.name.startswith(("Memcpy", "Memset")))]
    if not ks:
        seen = [f"{e.name} ({e.device_type})" for e in prof.events()][:20]
        raise RuntimeError(f"torch.profiler recorded no CUDA kernels; events seen: {seen}")
    total = (len(ks), sum(e.time_range.elapsed_us() for e in ks))
    if not detail:
        return total
    return total + ([(e.name, e.time_range.elapsed_us()) for e in sorted(ks, key=lambda e: e.time_range.start)],)


def kernel_detail(kernel_list, calls):
    """[(name, us)] of `calls` calls in a row -> what a validation point stores about its kernels:
    {"kernels": [[name, mean us]] per kernel position of ONE call} only when every call launched the
    same kernel sequence (same count, same names in the same order); otherwise {"kernels": None,
    "kernels_failed": why, "kernels_raw": [[name, us]] of all the calls}, never passed off as one call."""
    names = [n for n, _ in kernel_list]
    k, rest = divmod(len(names), calls)
    if rest:
        why = f"{len(names)} kernels for {calls} calls"
    elif any(names[j * k:(j + 1) * k] != names[:k] for j in range(1, calls)):
        why = "the calls launched different kernel sequences"
    else:
        return dict(kernels=[[names[i][:100], statistics.mean(kernel_list[i + j * k][1] for j in range(calls))]
                             for i in range(k)])
    return dict(kernels=None, kernels_failed=why, kernels_raw=[[n[:100], us] for n, us in kernel_list])


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


def layer_stack_dispatch_us(layer, rotary, mb, ctx, layers=16, stacks=10):
    """CPU time (us) to queue one decode layer call the way forward_through_layers does: a stack of
    layer calls back to back, no sync between them (the GPU works through the queue meanwhile);
    each call timed, the first of each stack left out (it starts cold after the sync), median.
    Probe v6 on; layer_dispatch_us before (one call after a sync: ~15% above the harness)."""
    h, emb, _ = decode_input(rotary, mb, ctx)
    cache = cache_at(layer, rotary, mb, ctx)
    out = []
    with torch.no_grad():
        for _ in range(stacks):
            torch.cuda.synchronize()
            for i in range(layers):
                t0 = time.perf_counter()
                layer_forward(layer, h, emb, cache)
                if i > 0:
                    out.append(time.perf_counter() - t0)
        torch.cuda.synchronize()
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
    if b <= 0:  # no measurable growth with size (tiny ops): fixed time only
        return sy / sw, float("inf")
    a = (sy - b * sx) / sw
    return a, 1e6 / b


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


def eager_us(calls, target_ms=20.0):
    """(GPU kernel time in us per call, "kernel", [that time]): the calls (closures, rotated) launched
    eagerly one by one, the way the harness launches them, GPU warmed by the caller; kernel time
    from the profiler (launch gaps not in it), summed over each call's kernels (e.g. a GEMM and
    its split-K reduce, flash attention and its split-KV combine). Probe v5 on; graph_us before."""
    for cl in calls[:3]:  # warm this shape
        cl()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    calls[0]()
    e1.record()
    torch.cuda.synchronize()
    n = int(min(400, max(5, target_ms * 1000 / max(e0.elapsed_time(e1) * 1000, 1.0))))

    def run():  # each result is dropped at once, as in normal eager execution (memory reused)
        for k in range(n):
            calls[k % len(calls)]()

    _, us = kernels(run, copies=True)
    return us / n, "kernel", [us / n]


def chain_us(fn, n_kernels, target_ms=10.0):
    """[us] per kernel position of fn, a chain of n_kernels small kernels (e.g. RMSNorm's steps),
    launched eagerly and timed by the profiler, mean over the calls; None if the calls did not all
    launch the same n_kernels kernels (kernel_detail)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    n = int(min(200, max(10, target_ms * 1000 / (5.0 * n_kernels))))  # ~5 us per small kernel

    def run():  # each result is dropped at once, as in normal eager execution (memory reused)
        for _ in range(n):
            fn()

    res = kernels(run, detail=True)
    det = kernel_detail(res[2], n)
    if det["kernels"] is None or len(det["kernels"]) != n_kernels:
        return None
    return [us for _, us in det["kernels"]]


def gpu_load(seconds):
    """A decode-like load on the visible device for `seconds`: M=8 GEMMs streaming rotated
    11008 x 4096 fp16 weights (past L2), as a pipeline rank does most of the time."""
    w_bytes = 11008 * 4096 * 2
    ws = [torch.randn(11008, 4096, dtype=torch.float16, device="cuda") for _ in range(max(2, copies_for(w_bytes)))]
    x = torch.randn(8, 4096, dtype=torch.float16, device="cuda")
    y = torch.empty(8, 11008, dtype=torch.float16, device="cuda")
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        for w in ws:
            torch.matmul(x, w.t(), out=y)
        torch.cuda.synchronize()


def concurrency(c, loader_cmd, others):
    """Slowdown of this slice's work while the other slices of the card are busy, as in the
    pipeline (the probes otherwise run one slice at a time and the card shares its power and
    clocks): a decode GEMM, decode attention and an RMSNorm chain, each timed as eager_us /
    chain_us alone, with the other slices running gpu_load (loader_cmd per slice in `others`, a
    list of MIG UUIDs), and alone again. Returns {kind: loaded / alone} and the raw medians."""
    import subprocess

    f16 = dict(dtype=torch.float16, device="cuda")
    ws = [torch.randn(11008, 4096, **f16) for _ in range(copies_for(11008 * 4096 * 2))]
    x, y = torch.randn(8, 4096, **f16), torch.empty(8, 11008, **f16)
    q = torch.randn(8, 32, 1, 128, **f16)
    kvs = [(torch.randn(8, 32, 321, 128, **f16), torch.randn(8, 32, 321, 128, **f16))
           for _ in range(copies_for(2 * 8 * 32 * 321 * 128 * 2))]
    h16, w16 = torch.randn(8, 4096, **f16), torch.randn(4096, **f16)

    def norm():
        v = h16.to(torch.float32)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return w16 * v.to(torch.float16)

    work = {
        "gemm": lambda: eager_us([(lambda w=w: torch.matmul(x, w.t(), out=y)) for w in ws])[0],
        "attention": lambda: eager_us([(lambda k=k, v=v: F.scaled_dot_product_attention(q, k, v)) for k, v in kvs])[0],
        "small": lambda: sum(chain_us(norm, 8) or [float("nan")]),
    }

    def measure(windows):
        return {kind: statistics.median(fn() for _ in range(windows)) for kind, fn in work.items()}

    warm_gpu(c["warm_s"])
    alone_before = measure(c["conc_windows"])
    env = dict(os.environ)
    loaders = [subprocess.Popen(loader_cmd, env=dict(env, CUDA_VISIBLE_DEVICES=uuid,
                                                     MEASURE_MIG_LOADER=str(c["load_seconds"])),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for uuid in others]
    try:
        time.sleep(c["load_warm_s"])  # loaders start, warm up and fill their slices
        clock_loaded = sm_clock_mhz()
        loaded = measure(c["conc_windows"])
    finally:
        for p in loaders:
            p.kill()
        for p in loaders:
            p.wait()
    warm_gpu(c["warm_s"])
    alone_after = measure(c["conc_windows"])
    alone = {k: (alone_before[k] + alone_after[k]) / 2 for k in work}
    factors = {k: loaded[k] / alone[k] for k in work}
    print(f"\nconcurrency ({len(others)} other slices busy; SM clock loaded {clock_loaded} MHz): " + ", ".join(
        f"{k} {alone_before[k]:.1f} / {loaded[k]:.1f} / {alone_after[k]:.1f} us (x{factors[k]:.3f})" for k in work),
        flush=True)
    return dict(factors=factors, alone_before_us=alone_before, loaded_us=loaded, alone_after_us=alone_after,
                others=len(others), sm_clock_loaded_mhz=clock_loaded)


# ---------------------------------------------------------------------------
# probe v7: the GPU's idle gap between kernels, and launch as the harness issues it
# ---------------------------------------------------------------------------
def gap_layer_set(mb, ctx, device="cuda", hidden=4096, inter=11008, heads=32, d=128):
    """Tensors for one generic decode layer (Llama-7B shapes, random): weights, the K/V cache of ctx
    tokens, and the microbatch's hidden state. A model-free stand-in for a decoder layer's ops."""
    f16 = dict(dtype=torch.float16, device=device)
    return dict(
        wq=torch.randn(hidden, hidden, **f16), wk=torch.randn(hidden, hidden, **f16),
        wv=torch.randn(hidden, hidden, **f16), wo=torch.randn(hidden, hidden, **f16),
        wg=torch.randn(inter, hidden, **f16), wu=torch.randn(inter, hidden, **f16),
        wd=torch.randn(hidden, inter, **f16), n1=torch.randn(hidden, **f16), n2=torch.randn(hidden, **f16),
        kc=torch.randn(mb, heads, ctx, d, **f16), vc=torch.randn(mb, heads, ctx, d, **f16),
        cos=torch.randn(mb, 1, 1, d, **f16), sin=torch.randn(mb, 1, 1, d, **f16),
        x=torch.randn(mb, 1, hidden, **f16), heads=heads, d=d)


def gap_layer(s):
    """One generic decode layer launched eagerly in the harness's op order (~40 kernels for
    Llama shapes): RMSNorm, q / k / v GEMMs, rotary on q and k, KV append (cat), SDPA, o GEMM,
    residual add, RMSNorm, gate / up GEMMs, SiLU x up, down GEMM, residual add. Not a model: the
    validation layers stay a check only."""
    def norm(h, w):
        v = h.to(torch.float32)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return w * v.to(torch.float16)

    def rope(t):
        half = t.shape[-1] // 2
        return t * s["cos"] + torch.cat((-t[..., half:], t[..., :half]), dim=-1) * s["sin"]

    x = s["x"]
    mb, H, d = x.shape[0], s["heads"], s["d"]
    h = norm(x, s["n1"])
    q = torch.matmul(h, s["wq"].t()).view(mb, 1, H, d).transpose(1, 2)
    k = torch.matmul(h, s["wk"].t()).view(mb, 1, H, d).transpose(1, 2)
    v = torch.matmul(h, s["wv"].t()).view(mb, 1, H, d).transpose(1, 2)
    q, k = rope(q), rope(k)
    kk = torch.cat([s["kc"], k], dim=-2)
    vv = torch.cat([s["vc"], v], dim=-2)
    a = F.scaled_dot_product_attention(q, kk, vv).transpose(1, 2).reshape(mb, 1, H * d)
    h = x + torch.matmul(a, s["wo"].t())
    m = norm(h, s["n2"])
    m = torch.matmul(F.silu(torch.matmul(m, s["wg"].t())) * torch.matmul(m, s["wu"].t()), s["wd"].t())
    return h + m


def boundary_gap(c):
    """Probe v7, item 1: the GPU's idle time between consecutive kernels of an eager decode sequence,
    per slice. A stack of c["gap_layers"] generic decode layers (gap_layer, two weight sets in turn so
    weights stream from HBM), per microbatch size in c["gap_mb"], context c["gap_ctx"]:
      kernel_us   summed profiler kernel durations of the stack (what the ILP composes)
      span_queued_us  CUDA-event span of the same stack with every launch queued before the GPU
                  reaches it (a sleep kernel holds the GPU while the CPU launches): GPU-side gaps only
      span_live_us    the span launched live, the GPU idle at the start (CPU launch can starve the GPU,
                  as in a pipeline rank that waited on a receive)
    gap per kernel = (span - kernel_us) / kernels. Median of c["gap_windows"] windows."""
    out = {}
    warm_gpu(c["warm_s"])
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for mb in c["gap_mb"]:
        sets = [gap_layer_set(mb, c["gap_ctx"]) for _ in range(2)]
        n = c["gap_layers"]

        def stack():
            for i in range(n):
                gap_layer(sets[i % 2])

        with torch.no_grad():
            for _ in range(3):  # warm these shapes
                stack()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            stack()
            launch_s = time.perf_counter() - t0
            torch.cuda.synchronize()
            sleep_cycles = int(max(3 * launch_s, 0.005) * 1410e6)  # SM cycles at ~1410 MHz
            rows = []
            for _ in range(c["gap_windows"]):
                warm_gpu(0.2)
                count, kus = kernels(stack, copies=True)
                torch.cuda.synchronize()
                torch.cuda._sleep(sleep_cycles)
                e0.record()
                stack()
                e1.record()
                queued_ok = not e0.query()  # the CPU finished launching before the GPU reached the stack
                torch.cuda.synchronize()
                span_q = e0.elapsed_time(e1) * 1000
                warm_gpu(0.2)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                e0.record()
                stack()
                e1.record()
                launch_live = (time.perf_counter() - t0) * 1e6
                torch.cuda.synchronize()
                span_l = e0.elapsed_time(e1) * 1000
                rows.append(dict(kernels=count, kernel_us=kus, span_queued_us=span_q, span_live_us=span_l,
                                 launch_live_us=launch_live, queued_ok=queued_ok))
        med = {k: statistics.median(r[k] for r in rows) for k in ("kernels", "kernel_us", "span_queued_us",
                                                                  "span_live_us", "launch_live_us")}
        med["gap_queued_us_per_kernel"] = (med["span_queued_us"] - med["kernel_us"]) / med["kernels"]
        med["gap_live_us_per_kernel"] = (med["span_live_us"] - med["kernel_us"]) / med["kernels"]
        med["queued_ok"] = all(r["queued_ok"] for r in rows)
        med["windows"] = rows
        out[str(mb)] = med
        print(f"  gap mb {mb:2d}: {med['kernels']:.0f} kernels, kernel sum {med['kernel_us']:.0f} us, span queued "
              f"{med['span_queued_us']:.0f} us ({med['gap_queued_us_per_kernel']:.2f} us/kernel"
              f"{'' if med['queued_ok'] else ', NOT fully queued'}), live {med['span_live_us']:.0f} us "
              f"({med['gap_live_us_per_kernel']:.2f} us/kernel; CPU launch {med['launch_live_us']:.0f} us)", flush=True)
        del sets
    return dict(by_mb=out, layers=c["gap_layers"], ctx=c["gap_ctx"], sm_clock_mhz=sm_clock_mhz(),
                timing="span = CUDA events around the stack; kernel_us = profiler kernel durations (incl. memcpy)")


class CpuFreqSampler:
    """scaling_cur_freq (kHz) of the allowed CPUs every period_s in a child process, so the
    measuring process keeps its CPU to itself. samples() -> [(time.time(), {cpu: kHz})]."""

    SCRIPT = ("import os, sys, time, json\n"
              "cpus = sorted(os.sched_getaffinity(0)); p = float(sys.argv[1])\n"
              "while True:\n"
              "    row = {}\n"
              "    for c in cpus:\n"
              "        try: row[c] = int(open(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq').read())\n"
              "        except OSError: pass\n"
              "    print(json.dumps([time.time(), row]), flush=True)\n"
              "    time.sleep(p)\n")

    def __init__(self, period_s=0.02):
        import subprocess
        import tempfile

        self.file = tempfile.TemporaryFile(mode="w+")
        self.proc = subprocess.Popen([sys.executable, "-c", self.SCRIPT, str(period_s)], stdout=self.file)

    def samples(self):
        import json

        self.proc.kill()
        self.proc.wait()
        self.file.seek(0)
        rows = []
        for line in self.file:
            try:
                t, row = json.loads(line)
                rows.append((t, {int(k): v for k, v in row.items()}))
            except ValueError:
                pass
        return rows


def cpu_khz(cpu):
    """scaling_cur_freq (kHz) of one CPU, or None."""
    try:
        return int(open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read())
    except (OSError, ValueError, TypeError):
        return None


def current_cpu():
    """The CPU this process runs on now (field 39 of /proc/self/stat), or None."""
    try:
        return int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[36])
    except (OSError, IndexError, ValueError):
        return None


def load_layers(path, n):
    """n decoder layers (layer_idx 0..n-1, distinct weights) + rotary of the model at path, random
    weights, and a DynamicCache factory filled for all n layers (as a rank holds its stack)."""
    from transformers import AutoConfig
    import model_family

    cfg = AutoConfig.from_pretrained(os.path.expanduser(path))
    cfg._attn_implementation = "sdpa"
    fam = model_family.resolve(cfg.model_type)
    layers = torch.nn.ModuleList([fam.decoder_layer(cfg, layer_idx=i).half().to(DEVICE).eval() for i in range(n)])
    rotary = fam.rotary_embedding(config=cfg, device=torch.device(DEVICE))
    return layers, rotary, cfg.hidden_size


def stack_cache(layers, rotary, hidden, mb, ctx):
    """A DynamicCache holding ctx tokens for mb sequences in every layer of the stack."""
    from transformers import DynamicCache

    cache = DynamicCache()
    x = torch.randn(mb, ctx, hidden, dtype=torch.float16, device=DEVICE)
    pos = torch.arange(ctx, device=DEVICE).unsqueeze(0).expand(mb, -1)
    with torch.no_grad():
        emb = rotary(x, pos)
        for layer in layers:
            layer_forward(layer, x, emb, cache)
    return cache


def harness_forward():
    """The harness's own forward_through_layers (benchmark_pipeline_microbatching.py), or None if
    the benchmark module does not import here; then a loop of layer_forward is timed instead."""
    try:
        import benchmark_pipeline_microbatching as bench
        return bench.forward_through_layers
    except Exception as err:  # report and keep going
        print(f"  (benchmark module not importable: {type(err).__name__}: {err}; timing layer_forward)", flush=True)
        return None


def harness_launch_us(c):
    """Probe v7, item 2: CPU time per decode layer as the harness launches a rank's stack, to find
    the ~10% between the probe's launch (v6) and the events runs' 0.67 ms per layer. c["launch_layers"]
    distinct layers (own weights, own cache slots), one decode microbatch of c["launch_mb"] at
    context c["launch_ctx"], called the way the harness does: the stack as one forward_through_layers
    call (time / layers), and the same layers one by one with layer_forward (each call timed, first
    left out, as v6). For comparison, v6's measurement (one layer object called again and again).
    The CPU's clock is sampled during every window (the CPU the process ran on, start and end).
    Windows alternate the three methods so drift hits them alike. Returns medians and raw windows."""
    layers, rotary, hidden = load_layers(c["model_path"], c["launch_layers"])
    n, mb, ctx = len(layers), c["launch_mb"], c["launch_ctx"]
    cache = stack_cache(layers, rotary, hidden, mb, ctx)
    ftl = harness_forward()
    h = torch.randn(mb, 1, hidden, dtype=torch.float16, device=DEVICE)
    pos = torch.full((mb, 1), ctx, dtype=torch.long, device=DEVICE)
    emb = rotary(h, pos)
    with torch.no_grad():  # as every timed call below (v7 counted outside no_grad: 35, not 40)
        kpl, _ = kernels(lambda: layer_forward(layers[0], h, emb, cache))

    def one_call():  # the harness's call, per layer
        t0 = time.perf_counter()
        if ftl is not None:
            ftl(layers, h, emb, None, cache)
        else:
            x = h
            for layer in layers:
                x = layer_forward(layer, x, emb, cache)
        return [(time.perf_counter() - t0) / n]

    def per_layer():  # each layer timed, the first left out (v6's method on distinct layers)
        out, x = [], h
        for i, layer in enumerate(layers):
            t0 = time.perf_counter()
            x = layer_forward(layer, x, emb, cache)
            if i:
                out.append(time.perf_counter() - t0)
        return out

    def same_layer():  # v6: one layer object, n calls
        out = []
        for i in range(n):
            t0 = time.perf_counter()
            layer_forward(layers[0], h, emb, cache)
            if i:
                out.append(time.perf_counter() - t0)
        return out

    methods = {"forward_through_layers" if ftl else "layer_forward_loop": one_call,
               "per_layer_distinct": per_layer, "per_layer_same_v6": same_layer}
    # v7.1: the CPU's clock read in this process right before and after every stack, on the CPU it
    # ran on (v7's sampler child recorded nothing on the box)
    windows = []
    nfreq = 0
    with torch.no_grad():
        for fn in methods.values():  # warm every path
            fn()
        torch.cuda.synchronize()
        for w in range(c["launch_windows"]):
            for name, fn in methods.items():
                vals, ks = [], []
                for _ in range(c["launch_stacks"]):
                    torch.cuda.synchronize()
                    cpu0 = current_cpu()
                    k0 = cpu_khz(cpu0)
                    vals += fn()
                    cpu1 = current_cpu()
                    k1 = cpu_khz(cpu1)
                    ks += [k for k in (k0, k1) if k is not None]
                torch.cuda.synchronize()
                nfreq += len(ks)
                windows.append(dict(method=name, window=w, us_per_layer=statistics.median(vals) * 1e6,
                                    cpu_khz_median=statistics.median(ks) if ks else None,
                                    cpu_khz_min=min(ks) if ks else None, cpu_last=cpu1))
    freq = range(nfreq)  # count only, for the summary below
    res = {}
    for name in methods:
        ws = [wd for wd in windows if wd["method"] == name]
        us = statistics.median(wd["us_per_layer"] for wd in ws)
        khz = [wd["cpu_khz_median"] for wd in ws if wd["cpu_khz_median"]]
        res[name] = dict(us_per_layer=us, us_per_kernel=us / kpl, cpu_khz_median=statistics.median(khz) if khz else None)
        print(f"  launch {name:24s}: {us:7.1f} us per layer = {us / kpl:5.2f} us per kernel"
              f" ({kpl} kernels); CPU clock {res[name]['cpu_khz_median'] or '?'} kHz", flush=True)
    del layers, cache
    torch.cuda.empty_cache()
    return dict(methods=res, kernels_per_layer=kpl, layers=n, mb=mb, ctx=ctx, windows=windows,
                harness_function=ftl is not None,
                freq_samples=len(freq), timing="wall time, CPU only (the GPU works through the queue meanwhile)")


def primitives(c, peak, gpcs):
    """Generic GPU primitives on the visible device (a whole GPU or one MIG slice), as functions
    of shape, no model, launched eagerly one by one as the harness does, kernel time from the
    profiler (eager_us; probe v4 and before: CUDA graph replays, graph_us). Each sweep is fitted as
    t = fixed + units / rate; attention and the KV append per head shape. peak = the device's peak
    HBM bytes/s; gpcs = its GPC count. Prints the tables; returns a dict."""
    f16 = dict(dtype=torch.float16, device="cuda")
    out, clocks = {"timing": {}}, {}

    def report(name, pts, unit, scale, per,
               timing="device time: profiler kernel time, launched eagerly one by one like the harness, GPU warmed"):
        """pts: [(units, us, method, shape dict)]. Fits the graph-timed points only; every point is
        kept raw with its shape, method and residual against the fit."""
        fit_pts = [(x, t) for x, t, m, _ in pts if m in ("kernel", "graph", "wall")]
        eager = len(pts) - len(fit_pts)
        if len({x for x, _ in fit_pts}) < 2:  # a slope needs two distinct sizes
            print(f"  {name:<22} NOT FITTED: {len(fit_pts)} timed points at "
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

    print(f"\nprimitives (eager launches, profiler kernel time, GPU warmed; SM clock at start {sm_clock_mhz()} MHz):")
    # decode GEMMs (M <= 32): weight streaming, bytes = N x K x 2
    group("gemm_decode")
    pts = []
    for N, K in c["gemm_shapes"]:
        ws = [torch.randn(N, K, **f16) for _ in range(copies_for(N * K * 2))]
        for M in c["gemm_decode_m"]:
            x, y = torch.randn(M, K, **f16), torch.empty(M, N, **f16)
            calls = [(lambda w=w: torch.matmul(x, w.t(), out=y)) for w in ws]
            t, m, reps = eager_us(calls)
            pts.append((N * K * 2, t, m, dict(N=N, K=K, M=M, replays_us=reps)))
        del ws
    report("gemm_decode", pts, "weight bytes", 1e9, "GB/s")
    # decode GEMMs with a bias (probe v7): F.linear(x, w, b), as nn.Linear(bias=True) runs Qwen2's
    # q / k / v projections; cuBLAS picks a different kernel (bias epilogue) than without bias
    if c.get("gemm_bias_shapes"):
        group("gemm_bias_decode")
        pts = []
        for N, K in c["gemm_bias_shapes"]:
            ws = [torch.randn(N, K, **f16) for _ in range(copies_for(N * K * 2))]
            b = torch.randn(N, **f16)
            for M in c["gemm_decode_m"]:
                x = torch.randn(M, K, **f16)
                calls = [(lambda w=w: F.linear(x, w, b)) for w in ws]
                t, m, reps = eager_us(calls)
                pts.append((N * K * 2, t, m, dict(N=N, K=K, M=M, replays_us=reps)))
            del ws
        report("gemm_bias_decode", pts, "weight bytes", 1e9, "GB/s")
        # the same shapes with a 3-D input (mb, 1, K), as a decoder layer's nn.Linear gets it (probe v7.1),
        # with and without the bias: is the layer's slower q projection the bias or the input shape?
        for name, with_bias in (("gemm3d_decode", False), ("gemm_bias3d_decode", True)):
            group(name)
            pts = []
            for N, K in c["gemm_bias_shapes"]:
                ws = [torch.randn(N, K, **f16) for _ in range(copies_for(N * K * 2))]
                b = torch.randn(N, **f16) if with_bias else None
                for M in c["gemm_decode_m"]:
                    x = torch.randn(M, 1, K, **f16)
                    calls = [(lambda w=w: F.linear(x, w, b)) for w in ws]
                    t, m, reps = eager_us(calls)
                    pts.append((N * K * 2, t, m, dict(N=N, K=K, M=M, replays_us=reps)))
                del ws
            report(name, pts, "weight bytes", 1e9, "GB/s")
    # prefill GEMMs (large M): FLOPs = 2 x M x N x K
    group("gemm_prefill")
    pts = []
    for N, K in c["gemm_prefill_shapes"]:
        ws = [torch.randn(N, K, **f16) for _ in range(copies_for(N * K * 2))]
        for M in c["gemm_prefill_m"]:
            x, y = torch.randn(M, K, **f16), torch.empty(M, N, **f16)
            calls = [(lambda w=w: torch.matmul(x, w.t(), out=y)) for w in ws]
            t, m, reps = eager_us(calls)
            pts.append((2 * M * N * K, t, m, dict(N=N, K=K, M=M, replays_us=reps)))
        del ws
    report("gemm_prefill", pts, "FLOPs", 1e12, "TFLOP/s")
    if c.get("gemm_bias_shapes"):  # prefill GEMMs with a bias (probe v7), as above
        group("gemm_bias_prefill")
        pts = []
        for N, K in c["gemm_bias_shapes"]:
            ws = [torch.randn(N, K, **f16) for _ in range(copies_for(N * K * 2))]
            b = torch.randn(N, **f16)
            for M in c["gemm_prefill_m"]:
                x = torch.randn(M, K, **f16)
                calls = [(lambda w=w: F.linear(x, w, b)) for w in ws]
                t, m, reps = eager_us(calls)
                pts.append((2 * M * N * K, t, m, dict(N=N, K=K, M=M, replays_us=reps)))
            del ws
        report("gemm_bias_prefill", pts, "FLOPs", 1e12, "TFLOP/s")
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
                    t, m, reps = eager_us(calls)
                    pts.append((nb, t, m, dict(heads=heads, kv=kv, d=d, L=L, mb=mb, replays_us=reps)))
                    del kvs
            report(f"{kind}_h{heads}_kv{kv}_d{d}", pts, "KV bytes", 1e9, "GB/s")
    # prefill attention, as transformers calls SDPA when a mask is passed (the harness's prefill):
    # K/V already repeated to every query head (repeat_kv, its own copy kernels), so full attention
    # with heads = query heads and an explicit causal mask. Per head count;
    # FLOPs = 4 x mb x heads x seq x seq x head_dim (QK^T and PV, mask not subtracted)
    group("attention_prefill")
    d = 128
    for heads in c["attn_prefill_heads"]:
        pts = []
        for seq in c["attn_prefill_seq"]:
            mask = torch.triu(torch.full((1, 1, seq, seq), torch.finfo(torch.float16).min, **f16), diagonal=1)
            for mb in c["attn_mb"]:
                nb = 3 * mb * heads * seq * d * 2
                qkvs = [tuple(torch.randn(mb, heads, seq, d, **f16) for _ in range(3)) for _ in range(copies_for(nb))]
                calls = [(lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)) for q, k, v in qkvs]
                t, m, reps = eager_us(calls)
                pts.append((4 * mb * heads * seq * seq * d, t, m, dict(heads=heads, d=d, seq=seq, mb=mb, replays_us=reps)))
                del qkvs
        report(f"attention_prefill_h{heads}_d{d}", pts, "FLOPs", 1e12, "TFLOP/s")
    # KV cache append (DynamicLayer.update: torch.cat of the cache and one new token), per K/V head
    # count; bytes read + written
    group("kv_append")
    for kv in c["cat_kv_heads"]:
        pts = []
        for L in c["kv_ctx"]:
            for mb in c["attn_mb"]:
                nb = 2 * mb * kv * (L + 1) * 128 * 2
                caches = [torch.randn(mb, kv, L, 128, **f16) for _ in range(copies_for(nb))]
                new, dst = torch.randn(mb, kv, 1, 128, **f16), torch.empty(mb, kv, L + 1, 128, **f16)
                calls = [(lambda cc=cc: torch.cat([cc, new], dim=-2, out=dst)) for cc in caches]
                t, m, reps = eager_us(calls)
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
        t, m, reps = eager_us(calls)
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
        t, m, reps = eager_us(calls)
        pts.append((2 * n * 2, t, m, dict(numel=n, replays_us=reps)))
        del srcs
    report("device_copy", pts, "bytes", 1e9, "GB/s")
    # a decoder layer's small ops by type, measured inside generic chains in their layer order and
    # launched eagerly (chain_us: profiler time per kernel), on T rows x W columns (rope: T tokens x
    # W heads of 128, one decode token each): RMSNorm's 8 steps; rotary's multiply by cos, half
    # negate, cat of the halves, multiply by sin, add; SwiGLU's SiLU and gate x up, and a residual
    # add. bytes = inputs + output, as layer_ops counts them. Fitted per type; points kept for lookups.
    group("small_ops")
    f32 = dict(dtype=torch.float32, device="cuda")
    small = {}  # op type -> points
    failed = []

    def keep(names_bytes, times, T, W):
        for (name, nb), us in zip(names_bytes, times):
            small.setdefault(name, []).append((nb, us, "kernel", dict(T=T, W=W, replays_us=[us])))

    for W in c["small_widths"]:
        for T in c["small_rows"]:
            x16, y16, w16 = torch.randn(T, W, **f16), torch.randn(T, W, **f16), torch.randn(W, **f16)

            def norm():  # RMSNorm, as the transformers code runs it
                x = x16.to(torch.float32)
                v = x.pow(2).mean(-1, keepdim=True)
                x = x * torch.rsqrt(v + 1e-6)
                return w16 * x.to(torch.float16)

            def mlp():  # SiLU(gate) x up, then the residual add
                return F.silu(x16) * y16 + x16

            for fn, names_bytes in (
                    (norm, [("to_f32", 6 * T * W), ("pow", 8 * T * W), ("mean", 4 * T * W + 4 * T), ("add_eps", 8 * T),
                            ("rsqrt", 8 * T), ("mul_rows", 8 * T * W + 4 * T), ("to_f16", 6 * T * W),
                            ("mul_weight", 4 * T * W + 2 * W)]),
                    (mlp, [("silu", 4 * T * W), ("mul", 6 * T * W), ("add", 6 * T * W)])):
                times = chain_us(fn, len(names_bytes))
                if times is None:
                    failed.append(f"{fn.__name__} T{T} W{W}")
                else:
                    keep(names_bytes, times, T, W)
    for H in c["small_rope_heads"]:
        for T in c["small_rows"]:
            q, cos, sin = torch.randn(T, H, 1, 128, **f16), torch.randn(T, 1, 1, 128, **f16), torch.randn(T, 1, 1, 128, **f16)

            def rope():  # q x cos + rotate_half(q) x sin, as apply_rotary_pos_emb runs it
                return q * cos + torch.cat((-q[..., 64:], q[..., :64]), dim=-1) * sin

            nb = T * H * 128
            names_bytes = [("rope_mul", 4 * nb + 2 * T * 128), ("rope_neg", 2 * nb), ("rope_cat", 4 * nb),
                           ("rope_mul", 4 * nb + 2 * T * 128), ("add", 6 * nb)]
            times = chain_us(rope, len(names_bytes))
            if times is None:
                failed.append(f"rope T{T} H{H}")
            else:
                keep(names_bytes, times, T, H)
    if failed:
        print(f"  small-op chains NOT GROUPED (calls launched different kernels): {failed}", flush=True)
    for name in c["small_ops"]:
        report(f"op_{name}", small.get(name, []), "bytes", 1e9, "GB/s")
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
    rates = [k for k in out if k.endswith("_rate") and k not in ("gemm_prefill_rate", "gemm_bias_prefill_rate", "d2h_rate", "h2d_rate")
             and not k.startswith("attention_prefill") and out[k] == out[k]]  # bandwidths only
    print("  as fractions of peak: " + ", ".join(f"{k[:-5]} {out[k] / peak:.3f}" for k in rates)
          + f"; prefill {out['gemm_prefill_rate'] / gpcs / 1e12:.1f} TFLOP/s per GPC")
    return out


def layer_kernel_us_steady(layer, rotary, mb, ctx, calls=10, detail=False):
    """Kernel time (us) of one decode layer call in steady state: the GPU warmed, `calls` calls
    back to back (the cache grows by one token each, negligible), profiler kernel time / calls.
    detail=True: (that time, kernel_detail() of the calls)."""
    h, emb, _ = decode_input(rotary, mb, ctx)
    with torch.no_grad():
        cache = cache_at(layer, rotary, mb, ctx)
        layer_forward(layer, h, emb, cache)
        warm_gpu(0.3)
        def run():  # each output dropped at once, as in normal eager execution
            for _ in range(calls):
                layer_forward(layer, h, emb, cache)

        res = kernels(run, detail=detail)
    return (res[1] / calls, kernel_detail(res[2], calls)) if detail else res[1] / calls


def prefill_layer_us_steady(layer, rotary, mb, seq, calls=5, detail=False):
    """Kernel time (us) of one prefill layer call in steady state (fresh cache each call).
    detail=True: (that time, kernel_detail() of the calls)."""
    from transformers import DynamicCache

    x = torch.randn(mb, seq, HIDDEN, dtype=torch.float16, device="cuda")
    pos = torch.arange(seq, device="cuda").unsqueeze(0)
    mask = torch.triu(torch.full((1, 1, seq, seq), torch.finfo(torch.float16).min, device="cuda"), diagonal=1).half()
    with torch.no_grad():
        emb = rotary(x, pos)
        layer_forward(layer, x, emb, DynamicCache(), mask)
        warm_gpu(0.3)
        def run():  # each output and its fresh cache dropped at once
            for _ in range(calls):
                layer_forward(layer, x, emb, DynamicCache(), mask)

        res = kernels(run, detail=detail)
    return (res[1] / calls, kernel_detail(res[2], calls)) if detail else res[1] / calls


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
                _, det = layer_kernel_us_steady(layer, rotary, mb, ctx, detail=True)  # which kernel takes what
                rows[f"decode mb{mb} ctx{ctx}"] = dict(median=statistics.median(w), spread=(max(w) - min(w)) / min(w),
                                                       windows=w, **det)
            w = [prefill_layer_us_steady(layer, rotary, mb, c["seq_len"]) for _ in range(c["windows"])]
            _, det = prefill_layer_us_steady(layer, rotary, mb, c["seq_len"], detail=True)
            rows[f"prefill mb{mb} seq{c['seq_len']}"] = dict(median=statistics.median(w), spread=(max(w) - min(w)) / min(w),
                                                            windows=w, **det)
        out[path] = rows
        print(f"\nvalidation layer, {path} (kernel us, steady state, median of {c['windows']} windows, spread; "
              f"kernels per call): " + ", ".join(
                  f"{k} {v['median']:.0f} ({100 * v['spread']:.1f}%; "
                  + (f"{len(v['kernels'])})" if v["kernels"] is not None else f"NOT GROUPED: {v['kernels_failed']})")
                  for k, v in rows.items()))
        del layer, rotary
        torch.cuda.empty_cache()
    return out


PRIMITIVE_CONFIG = dict(
    gemm_shapes=[(4096, 4096), (11008, 4096), (4096, 11008), (1024, 4096), (14336, 4096), (4096, 14336),
                 (5120, 5120), (13824, 5120), (5120, 13824), (3584, 3584), (512, 3584), (18944, 3584),
                 (3584, 18944), (32000, 4096), (32768, 4096), (152064, 3584)],  # (N out, K in): 7B-14B widths, lm_heads
    gemm_decode_m=[1, 2, 4, 8, 16, 32],  # the harness's microbatch sizes are 2-16
    gemm_prefill_shapes=[(4096, 4096), (11008, 4096), (4096, 11008), (1024, 4096), (14336, 4096), (4096, 14336),
                         (5120, 5120), (13824, 5120), (5120, 13824), (3584, 3584), (512, 3584), (18944, 3584),
                         (3584, 18944)],  # every decode shape but the lm_heads (prefill runs them on the last token only)
    gemm_prefill_m=[64, 128, 256, 512, 1024, 2048],  # microbatch x 64 prompt tokens
    attn_mha=[(32, 128), (40, 128)],  # full attention: (heads, head dim), e.g. Llama-2-7B / 13B
    attn_gqa=[(28, 4, 128), (32, 8, 128)],  # GQA: (query heads, K/V heads, head dim), e.g. Qwen2.5-7B, Mistral-7B
    # decode attention reads ctx + 1 keys and works in blocks of 64: both sides of each block edge
    attn_ctx=[L for k in range(1, 10) for L in (64 * k, 64 * k + 1)] + [1088, 1089],
    kv_ctx=[64, 320, 576, 1088],  # the KV append (a copy) has no block steps
    attn_mb=[1, 2, 4, 8, 16, 32],
    attn_prefill_heads=[28, 32, 40],  # prefill attention: query heads (K/V already repeated to them)
    attn_prefill_seq=[64, 256],
    cat_kv_heads=[4, 8, 32, 40],
    small_ops=["to_f32", "pow", "mean", "add_eps", "rsqrt", "mul_rows", "to_f16", "mul_weight", "add", "silu",
               "mul", "rope_mul", "rope_neg", "rope_cat"],  # a decoder layer's small ops (primitives(): chains)
    small_rows=[1, 2, 4, 8, 16, 32, 128, 512, 2048],  # decode: microbatch rows; prefill: microbatch x 64
    small_widths=[4096, 14336],  # hidden and MLP widths
    small_rope_heads=[4, 32],  # K/V and query heads, 128 each
    elem_numel=[4096, 65536, 1 << 25, 1 << 26],  # small: fixed time per kernel; 64-128 MB: past L2, the rate
    copy_bytes=[16 << 10, 64 << 10, 512 << 10, 4 << 20],
    warm_s=1.0,  # seconds of GPU load before each sweep
    validate_models=["lmsys/vicuna-7b-v1.5", "Qwen/Qwen2.5-7B"],  # HF ids (config.json only) or local dirs
    validate_mb=[1, 8, 32],
    validate_ctx=[64, 320, 575],
    windows=5,  # validation layers: repeated windows per point (median and spread kept)
    seq_len=64,
    conc_windows=3,  # concurrency(): windows per workload, alone / loaded / alone again
    load_seconds=600,  # a loader's lifetime cap (they are killed as soon as the loaded windows are done)
    load_warm_s=5.0,  # seconds the loaders run before the loaded windows
    # probe v7
    gemm_bias_shapes=[(3584, 3584), (512, 3584)],  # GEMMs with a bias: Qwen2.5-7B's q and k / v projections
    gap_mb=[1, 8, 32],  # boundary_gap(): microbatch sizes
    gap_ctx=320,  # boundary_gap(): cached tokens
    gap_layers=8,  # boundary_gap(): generic decode layers per stack
    gap_windows=5,  # boundary_gap(): windows per microbatch size (median kept)
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
