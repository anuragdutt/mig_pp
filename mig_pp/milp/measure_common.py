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


def primitives(c, peak, gpcs):
    """Generic GPU primitives on the visible device (a whole GPU or one MIG slice), as functions
    of shape, no model: each fitted as t = fixed + units / rate over its sweep. peak = the
    device's peak HBM bytes/s; gpcs = its GPC count. Prints the tables; returns a dict."""
    f16 = dict(dtype=torch.float16, device="cuda")
    out = {}

    def report(name, pts, unit, scale, per):
        fixed, rate = fit_rate(pts)
        worst = max(abs((fixed + x / rate * 1e6) / t - 1) for x, t in pts)
        out[f"{name}_fixed_us"], out[f"{name}_rate"] = fixed, rate
        print(f"  {name:<12} fixed {fixed:7.1f} us + {unit} / {rate / scale:8.1f} {per}"
              f"   ({len(pts)} points, worst {100 * worst:.1f}% off)")

    print("\nprimitives (kernel time, torch.profiler):")
    # decode GEMMs (M <= 32): weight streaming, bytes = N x K x 2
    pts = []
    for N, K in c["gemm_shapes"]:
        w = torch.randn(N, K, **f16)
        for M in c["gemm_decode_m"]:
            x = torch.randn(M, K, **f16)
            pts.append((N * K * 2, kernel_us(lambda: F.linear(x, w))))
        del w
    report("gemm_decode", pts, "weight bytes", 1e9, "GB/s")
    # prefill GEMMs (large M): FLOPs = 2 x M x N x K
    pts = []
    for N, K in c["gemm_prefill_shapes"]:
        w = torch.randn(N, K, **f16)
        for M in c["gemm_prefill_m"]:
            x = torch.randn(M, K, **f16)
            pts.append((2 * M * N * K, kernel_us(lambda: F.linear(x, w))))
        del w
    report("gemm_prefill", pts, "FLOPs", 1e12, "TFLOP/s")
    # decode attention, one query token per sequence against L cached, as transformers 5.14.1
    # (integrations/sdpa_attention.py) calls SDPA in decode (no mask): full attention (K/V heads =
    # query heads) as is; GQA with enable_gqa=True, K/V not expanded (torch >= 2.5). Prefill passes
    # a mask, so there GQA K/V are expanded (repeat_kv) first. bytes = 2 (K, V) x mb x kv x L x d x 2
    for name, configs, gqa in (("attention_mha", [(h, h, d) for h, d in c["attn_mha"]], False),
                               ("attention_gqa", c["attn_gqa"], True)):
        pts = []
        for heads, kv, d in configs:
            for L in c["attn_ctx"]:
                for mb in c["attn_mb"]:
                    q = torch.randn(mb, heads, 1, d, **f16)
                    k, v = torch.randn(mb, kv, L, d, **f16), torch.randn(mb, kv, L, d, **f16)
                    kw = dict(enable_gqa=True) if gqa else {}
                    pts.append((2 * mb * kv * L * d * 2, kernel_us(lambda: F.scaled_dot_product_attention(q, k, v, **kw))))
        report(name, pts, "KV bytes", 1e9, "GB/s")
    # KV cache append (DynamicCache: torch.cat of the cache and one new token); bytes read + written
    pts = []
    for kv in c["cat_kv_heads"]:
        for L in c["attn_ctx"]:
            for mb in c["attn_mb"]:
                cache, new = torch.randn(mb, kv, L, 128, **f16), torch.randn(mb, kv, 1, 128, **f16)
                pts.append((2 * mb * kv * (L + 1) * 128 * 2, kernel_us(lambda: torch.cat([cache, new], dim=-2))))
    report("kv_append", pts, "bytes", 1e9, "GB/s")
    # elementwise (binary, in place): bytes = 3 x numel x 2
    pts = []
    for n in c["elem_numel"]:
        a, b = torch.randn(n, **f16), torch.randn(n, **f16)
        pts.append((3 * n * 2, kernel_us(lambda: a.mul_(b))))
    report("elementwise", pts, "bytes", 1e9, "GB/s")
    # device copy (clone, a DtoD memcpy): bytes read + written
    pts = []
    for n in c["elem_numel"]:
        a = torch.randn(n, **f16)
        pts.append((2 * n * 2, kernel_us(lambda: a.clone(), copies=True)))
    report("device_copy", pts, "bytes", 1e9, "GB/s")
    # host copies as the transport does them (wall time, a side stream + event wait)
    d2h, h2d = [], []
    for nb in c["copy_bytes"]:
        a, b = device_copies(nb)
        d2h.append((nb, nb / a * 1e6))
        h2d.append((nb, nb / b * 1e6))
    report("d2h", d2h, "bytes", 1e9, "GB/s")
    report("h2d", h2d, "bytes", 1e9, "GB/s")

    out.update(peak=peak, gpcs=gpcs)
    print("  as fractions: " + ", ".join(f"{k[:-5]} {out[k] / peak:.3f} of peak"
                                       for k in ("gemm_decode_rate", "attention_mha_rate", "attention_gqa_rate", "kv_append_rate",
                                                 "elementwise_rate", "device_copy_rate"))
          + f"; prefill {out['gemm_prefill_rate'] / gpcs / 1e12:.1f} TFLOP/s per GPC")
    return out


def validation_layers(c):
    """VALIDATION ONLY, never a constant: one decoder layer's kernel time per model at a few
    (mb, ctx), to check later that layer time composed from the primitives matches."""
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
                rows[f"decode mb{mb} ctx{ctx}"] = layer_kernel_us(layer, rotary, mb, ctx, reps=3)[1]
            rows[f"prefill mb{mb} seq{c['seq_len']}"] = prefill_layer_us(layer, rotary, mb, c["seq_len"], reps=3)
        out[path] = rows
        print(f"\nvalidation layer, {path} (kernel us): " + ", ".join(f"{k} {v:.0f}" for k, v in rows.items()))
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
    elem_numel=[4096, 65536, 1 << 20, 1 << 24],
    copy_bytes=[16 << 10, 64 << 10, 512 << 10, 4 << 20],
    validate_models=["lmsys/vicuna-7b-v1.5", "Qwen/Qwen2.5-7B"],  # HF ids (config.json only) or local dirs
    validate_mb=[1, 8, 32],
    validate_ctx=[64, 320, 575],
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
