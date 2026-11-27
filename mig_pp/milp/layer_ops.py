"""
One decoder layer's GPU time, composed from generic primitives (measure_common.primitives) and the
layer's op list, which comes from the model code itself, never from benchmark results.

trace_ops(cfg, mb, ctx, prefill_seq): records every torch function call of one decoder layer's
forward on the "meta" device (shapes only, no compute), called as the harness's
forward_through_layers calls it. It records at the torch-function level, where
F.scaled_dot_product_attention, F.linear and torch.cat are single calls, as on the GPU (one
fused attention kernel, one GEMM, one cat); the aten level below would show SDPA's math fallback.
Needs torch + transformers.

trace_extras(cfg, mb, ctx, prefill_seq): the same for what the harness runs per microbatch outside
the layers: rotary (every rank; prefill also builds its position ids), embed (rank 0), the clone
before the send (every rank but the last), head = norm, lm_head (prefill: last position only),
argmax and the write into next_tokens (last rank).

classify(records): each call -> one of
    gemm        F.linear: M rows, N x K weight
    attention   F.scaled_dot_product_attention: heads, kv heads, head dim, query and key lengths
    kv_append   torch.cat growing the KV cache (dim -2, one input already holding the context)
    elementwise any other op producing a new tensor (casts, norms, rotary, residual adds, SiLU, cat
                of the rotary halves): bytes = inputs + output; the embedding gather reads only
                the rows it outputs
    copy        reshape / contiguous of a non-contiguous tensor, clone, the write into next_tokens
                (a device copy)
    free        views, slices, transposes, shape queries, same-dtype .to() / type_as() / .half() /
                .float()

layer_us(classified, prims): the sum of each op's primitive time on one slice (prims = one slice's
entry of constants_mig.json). Torch-free.

layer_ops_from_shape(shape, mb, ctx, prefill_seq) / extras_from_shape(...): the same op lists
(kernels only) written out from the model shape (memory_limits.REFERENCE_SHAPES), torch-free, for
the ILP. tests/test_layer_ops_torch.py checks them against trace_ops / trace_extras + classify.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

FREE = {"view", "reshape", "transpose", "permute", "unsqueeze", "squeeze", "expand", "__getitem__", "size", "dim",
        "contiguous", "t", "flatten", "unflatten", "split", "chunk", "narrow", "select", "dtype", "shape",
        "device", "is_contiguous", "__len__", "__iter__", "numel", "get_device", "__eq__", "__bool__", "data_ptr",
        "__format__", "__repr__", "__hash__", "requires_grad_", "detach", "item"}


def _tracer(cfg_dict):
    """(torch, Log, cfg, family): Log() records every torch function call made inside it as
    (name, args shapes/dtypes, kwargs, output shape/dtype, input contiguity); cfg is the model
    config with SDPA attention, following transformers' CUDA branch whatever this machine has."""
    import torch
    from torch.overrides import TorchFunctionMode
    from transformers import AutoConfig
    import model_family

    def desc(x):
        if isinstance(x, torch.Tensor):
            return ("T", tuple(x.shape), str(x.dtype).replace("torch.", ""), x.is_contiguous())
        if isinstance(x, (list, tuple)):
            return tuple(desc(y) for y in x)
        if isinstance(x, torch.dtype):
            return ("dtype", str(x).replace("torch.", ""))
        return x if isinstance(x, (int, float, bool, str, type(None))) else type(x).__name__

    class Log(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            name = getattr(func, "__name__", str(func))
            self.calls.append((name, desc(args), {k: desc(v) for k, v in kwargs.items()}, desc(out)))
            return out

    # follow transformers' CUDA branch whatever this machine has: its SDPA path has MPS / XPU / NPU
    # branches (e.g. masked GQA allowed on MPS) that the boxes never take
    import transformers.integrations.sdpa_attention as sdpa_attention
    for flag in ("_is_torch_mps_available", "_is_torch_xpu_available", "_is_torch_npu_available"):
        if hasattr(sdpa_attention, flag):
            setattr(sdpa_attention, flag, False)

    cfg = AutoConfig.for_model(**cfg_dict)
    cfg._attn_implementation = "sdpa"
    return torch, Log, cfg, model_family.resolve(cfg.model_type)


def trace_ops(cfg_dict, mb, ctx, prefill_seq=None):
    """[(name, args shapes/dtypes, kwargs, output shape/dtype, input contiguity)] for one decoder
    layer forward: decode (one token, ctx cached) or prefill (prefill_seq tokens, causal mask)."""
    from transformers import DynamicCache
    torch, Log, cfg, fam = _tracer(cfg_dict)
    with torch.device("meta"):
        layer = fam.decoder_layer(cfg, layer_idx=0).half()
        rotary = fam.rotary_embedding(config=cfg, device=torch.device("meta"))
    H = cfg.hidden_size

    def forward(h, emb, cache, mask=None):  # = measure_common.layer_forward = the harness's per-layer call
        r = h
        x = layer.self_attn(hidden_states=layer.input_layernorm(h), position_embeddings=emb, attention_mask=mask,
                            past_key_values=cache, use_cache=True)[0]
        h = r + x
        return h + layer.mlp(layer.post_attention_layernorm(h))

    with torch.no_grad():
        if prefill_seq:
            x = torch.empty(mb, prefill_seq, H, dtype=torch.float16, device="meta")
            emb = rotary(x, torch.arange(prefill_seq, device="meta").unsqueeze(0))
            mask = torch.empty(1, 1, prefill_seq, prefill_seq, dtype=torch.float16, device="meta")
            with Log() as log:
                forward(x, emb, DynamicCache(), mask)
        else:
            cache = DynamicCache()
            x = torch.empty(mb, ctx, H, dtype=torch.float16, device="meta")
            forward(x, rotary(x, torch.arange(ctx, device="meta").unsqueeze(0).expand(mb, -1)), cache)
            h = torch.empty(mb, 1, H, dtype=torch.float16, device="meta")
            emb = rotary(h, torch.full((mb, 1), ctx, dtype=torch.long, device="meta"))
            with Log() as log:
                forward(h, emb, cache)
    return log.calls


def trace_extras(cfg_dict, mb, ctx, prefill_seq=None):
    """{role: calls} for what the harness's decode / prefill loop runs per microbatch outside the
    layers, built as the harness builds them (benchmark_pipeline_microbatching.py):
        rotary  every rank; in prefill also the position ids (torch.arange); decode's position ids
                are built once per step (STEP), not per microbatch
        embed   rank 0: nn.Embedding(...).half() called on the token ids, then .half()
        clone   every rank but the last: the clone handed to isend
        head    last rank: norm, lm_head (prefill: last position only), argmax, written into
                next_tokens"""
    torch, Log, cfg, fam = _tracer(cfg_dict)
    meta = torch.device("meta")
    H, V = cfg.hidden_size, cfg.vocab_size
    with torch.device("meta"):
        embed = torch.nn.Embedding(V, H).half()
        norm = fam.rms_norm(H, eps=cfg.rms_norm_eps).half()
        lm_head = torch.nn.Linear(H, V, bias=False).half()
        rotary = fam.rotary_embedding(config=cfg, device=meta)
    q = prefill_seq or 1
    h = torch.empty(mb, q, H, dtype=torch.float16, device=meta)
    ids = torch.empty(mb, q, dtype=torch.long, device=meta)
    next_tokens = torch.empty(2 * mb, 1, dtype=torch.long, device=meta)  # the batch; one microbatch's rows
    calls = {}
    with torch.no_grad():
        if prefill_seq:
            with Log() as log:
                rotary(h, torch.arange(0, prefill_seq, dtype=torch.long, device=meta).unsqueeze(0))
        else:
            position_ids = torch.full((mb, 1), ctx, dtype=torch.long, device=meta)
            with Log() as log:
                rotary(h, position_ids)
        calls["rotary"] = log.calls
        with Log() as log:
            embed(ids).half()
        calls["embed"] = log.calls
        with Log() as log:
            h.clone()
        calls["clone"] = log.calls
        with Log() as log:
            normed = norm(h)
            logits = lm_head(normed[:, -1:, :] if prefill_seq else normed)
            next_tokens[0:mb] = torch.argmax(logits, dim=-1)
        calls["head"] = log.calls
    return calls


def _tensors(x):
    if isinstance(x, tuple) and x and x[0] == "T":
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for y in x for t in _tensors(y)]
    if isinstance(x, dict):
        return [t for y in x.values() for t in _tensors(y)]
    return []


def _nbytes(t):
    n = 1
    for s in t[1]:
        n *= s
    return n * {"float16": 2, "bfloat16": 2, "float32": 4, "int64": 8, "bool": 1}.get(t[2], 2)


def _op_tag(name, ins, outs):
    """The small-op type of an elementwise call (the probe's op_* primitives), or None: RMSNorm's
    fp32 cast / pow / mean / + eps / rsqrt / x rsqrt (rows) / fp16 cast / x weight, rotary's
    broadcast multiply / half negate / cat, residual and rotary add, SiLU, gate x up."""
    if name == "to" and ins and outs:
        a, b = ins[0][2], outs[0][2]
        if a in ("float16", "bfloat16") and b == "float32":
            return "to_f32"
        if a == "float32" and b in ("float16", "bfloat16"):
            return "to_f16"
        return None
    if name in ("pow", "mean", "rsqrt", "silu"):
        return name
    if name == "neg":
        return "rope_neg"
    if name == "cat":
        return "cat"
    if name == "add":
        return "add_eps" if len(ins) == 1 else "add"
    if name == "mul" and len(ins) == 2:
        sa, sb = ins[0][1], ins[1][1]
        if len(sa) == 1 or len(sb) == 1:
            return "mul_weight"
        if sa == sb:
            return "mul"
        return "rope_mul" if len(sa) == 4 and len(sb) == 4 else "mul_rows"
    return None


def classify(calls):
    """[(kind, info dict)] per call; see the module docstring."""
    out = []
    for name, args, kwargs, res in calls:
        base = name.strip("_").replace("Tensor.", "")
        ins, outs = _tensors(args) + _tensors(kwargs), _tensors(res)
        if name == "contiguous" and next((k for k, _ in reversed(out) if k != "free"), None) == "attention":
            # CUDA's SDPA kernels write the output already in (mb, seq, heads, d) order, so the
            # transpose back is contiguous there (the probe's per-kernel lists: no copy); only the
            # meta device's math path would copy
            out.append(("free", dict(name=name)))
            continue
        if name == "linear":
            x, w = ins[0], ins[1]
            M = 1
            for s in x[1][:-1]:
                M *= s
            out.append(("gemm", dict(M=M, N=w[1][0], K=w[1][1])))
        elif name == "scaled_dot_product_attention":
            q, k = ins[0], ins[1]
            out.append(("attention", dict(mb=q[1][0], heads=q[1][1], q_len=q[1][2], kv=k[1][1], L=k[1][2], d=q[1][3],
                                          gqa=bool(kwargs.get("enable_gqa", False)), masked=len(ins) > 3)))
        elif (name == "cat" and len(ins) == 2 and len(ins[0][1]) == 4 and len(ins[1][1]) == 4
              and ins[0][1][-2] > 1 and ins[1][1][-2] == 1):
            c = ins[0]
            out.append(("kv_append", dict(mb=c[1][0], kv=c[1][1], L=c[1][2], d=c[1][3], bytes=_nbytes(c) + _nbytes(ins[1]) + _nbytes(outs[0]))))
        elif name == "embedding":  # a gather: reads the ids and only the rows it outputs
            out.append(("elementwise", dict(name=name, bytes=_nbytes(ins[0]) + 2 * _nbytes(outs[0]), op=None)))
        elif name == "clone":
            out.append(("copy", dict(name=name, bytes=2 * _nbytes(ins[0]))))
        elif name == "__setitem__":  # x[rows] = value: one device copy of value
            out.append(("copy", dict(name=name, bytes=2 * _nbytes(ins[-1]))))
        elif base in FREE or name in FREE or not outs:
            if name in ("reshape", "contiguous") and ins and not ins[0][3]:  # non-contiguous: a device copy
                out.append(("copy", dict(name=name, bytes=2 * _nbytes(ins[0]))))
            else:
                out.append(("free", dict(name=name)))
        elif name in ("to", "type_as", "half", "float") and ins and outs and ins[0][2] == outs[0][2]:
            out.append(("free", dict(name=name)))  # same dtype: no kernel
        else:
            out.append(("elementwise", dict(name=name, bytes=sum(_nbytes(t) for t in ins + outs),
                                            op=_op_tag(name, ins, outs))))
    return out


def _attention_key(prims, op):
    """(key of the primitives' fit for this head shape, exact?). Per-shape keys (probe v2) or one key
    per kind (probe v1); a shape that was not measured takes the nearest measured one of its kind
    (by heads, then kv heads), else of the other kind, and is reported as not exact."""
    kind = "attention_gqa" if op["kv"] < op["heads"] else "attention_mha"
    exact = f"{kind}_h{op['heads']}_kv{op['kv']}_d{op['d']}"
    if f"{exact}_rate" in prims:
        return exact, True
    if f"{kind}_rate" in prims:  # probe v1: one fit per kind
        return kind, True
    measured = []
    for k in prims:
        if (k.startswith(("attention_mha_", "attention_gqa_")) and k.endswith("_rate") and "_h" in k
                and "_kv" in k):  # decode shapes only (attention_prefill_* has no K/V count)
            name = k[:-5]
            h, kv, d = (int(x) for x in name.split("_h")[1].replace("_kv", " ").replace("_d", " ").split())
            measured.append((name.startswith(kind) is False, abs(h - op["heads"]), abs(kv - op["kv"]), abs(d - op["d"]), name))
    if not measured:
        raise KeyError(f"no attention fit at all for {exact}")
    return min(measured)[-1], False


def _interp(xs, ys, x):
    """Piecewise linear through the points; below the first: the first; above the last: on along
    the last segment."""
    if x <= xs[0] or len(xs) == 1:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1])


def _curve(pairs):
    """[(x, y)] -> (xs, ys) sorted, equal x averaged."""
    by = {}
    for x, y in pairs:
        by.setdefault(x, []).append(y)
    xs = sorted(by)
    return xs, [sum(by[x]) / len(by[x]) for x in xs]


def _grid_us(points, outer, inner, a, b):
    """Time on a measured grid (e.g. outer mb, inner context): linear in the inner coordinate within
    each measured outer value, then linear in the outer one."""
    rows = {}
    for p in points:
        rows.setdefault(p[outer], []).append((p[inner], p["us"]))
    outs = sorted(rows)
    return _interp(outs, [_interp(*_curve(rows[o]), b) for o in outs], a)


def _gemm_points_us(prims, op):
    """(us, note) of one GEMM from the probe's points: rows <= 32 from the decode sweep of that exact
    shape, more rows from the prefill sweep; a shape not measured is scaled from the measured one
    nearest in weight size (same rows), and noted."""
    M, N, K = op["M"], op["N"], op["K"]
    sweep = prims["gemm_decode_points"] if M <= 32 else prims.get("gemm_prefill_points", [])
    shapes = {}
    for p in sweep:
        shapes.setdefault((p["N"], p["K"]), []).append((p["M"], p["us"]))
    if (N, K) in shapes:
        return _interp(*_curve(shapes[(N, K)]), M), None
    if not shapes:
        return None, f"gemm {N}x{K} M{M}: no points"
    import math
    n0, k0 = min(shapes, key=lambda s: abs(math.log(s[0] * s[1] / (N * K))))
    return _interp(*_curve(shapes[(n0, k0)]), M) * N * K / (n0 * k0), f"gemm {N}x{K} M{M}: from {n0}x{k0}"


CONCURRENCY_KIND = {"gemm": "gemm", "kv_append": "gemm", "attention": "attention", "elementwise": "small",
                    "copy": "small"}  # op kind -> the probe's concurrency workload (bandwidth-bound / attention / small)


def layer_us(classified, prims, loaded=False):
    """GPU time (us) of one layer on one slice: each op at its primitive's time. With the probe's raw
    points in prims (probe v3 on), from the points: GEMMs by shape and rows, attention and the KV
    append on their microbatch x context grid, prefill attention on microbatch x length, small ops
    by type and size. Without points, from the fitted lines (fixed + size / rate).
    loaded=True: each op x the slice's slowdown with the card's other slices busy (prims
    "concurrency", probe v6 on; 1 without it), as in the pipeline; step 3 composes alone, as the
    validation layers were measured.
    Returns (total, {kind: us}, kernels, [approximations made])."""
    def t(key, units):
        return prims[f"{key}_fixed_us"] + units / prims[f"{key}_rate"] * 1e6

    points = "gemm_decode_points" in prims
    parts, kernels, approx = {}, 0, []
    for kind, op in classified:
        if kind == "free":
            continue
        kernels += 1
        if kind == "gemm":
            us = None
            if points:
                us, note = _gemm_points_us(prims, op)
                if note:
                    approx.append(note)
            if us is None:
                W = op["N"] * op["K"] * 2
                us = max(t("gemm_decode", W), t("gemm_prefill", 2 * op["M"] * op["N"] * op["K"]))
        elif kind == "attention":
            pre = f"attention_prefill_h{op['heads']}_d{op['d']}"
            if points and op["q_len"] > 1 and f"{pre}_points" in prims:
                us = _grid_us(prims[f"{pre}_points"], "mb", "seq", op["mb"], op["q_len"])
            else:
                key, exact = _attention_key(prims, op)
                if not exact or op["q_len"] > 1:  # prefill attention without its primitive: charged by KV bytes
                    approx.append(f"attention {op['heads']}/{op['kv']} heads q_len {op['q_len']} -> {key}")
                if points and op["q_len"] == 1 and f"{key}_points" in prims:
                    us = _grid_us(prims[f"{key}_points"], "mb", "L", op["mb"], op["L"])
                else:
                    us = t(key, 2 * op["mb"] * op["kv"] * op["L"] * op["d"] * 2)
        elif kind == "kv_append":
            key = f"kv_append_kv{op['kv']}" if f"kv_append_kv{op['kv']}_rate" in prims else "kv_append"
            if points and f"{key}_points" in prims:
                us = _grid_us(prims[f"{key}_points"], "mb", "L", op["mb"], op["L"])
            else:
                us = t(key, op["bytes"])
        else:
            prim = "device_copy" if kind == "copy" else "elementwise"
            tag = op.get("op")
            if kind == "elementwise" and tag:
                if f"op_{'rope_cat' if tag == 'cat' else tag}_points" in prims:
                    prim = f"op_{'rope_cat' if tag == 'cat' else tag}"
                elif points:
                    approx.append(f"{tag}: no op_ points, generic elementwise")
            if points and f"{prim}_points" in prims:
                us = _interp(*_curve((p["units"], p["us"]) for p in prims[f"{prim}_points"]), op["bytes"])
            else:
                us = t(prim, op["bytes"])
        if loaded and "concurrency" in prims:
            us *= prims["concurrency"]["factors"][CONCURRENCY_KIND[kind]]
        parts[kind] = parts.get(kind, 0.0) + us
    return sum(parts.values()), parts, kernels, approx


# ---------------------------------------------------------------------------
# The same kernel lists written out from the model shape (torch-free, for the ILP). Bytes are each
# kernel's inputs + output, as classify counts them; tests/test_layer_ops_torch.py checks every
# list against the traced model code.
# ---------------------------------------------------------------------------
def _ew(name, nbytes, op=None):
    """An elementwise kernel; op = its small-op type (classify's _op_tag), None if it has none."""
    return ("elementwise", dict(name=name, bytes=nbytes, op=op))


def _rms_norm(T, H):
    """RMSNorm over T tokens of width H: fp32 upcast, pow, mean, + eps, rsqrt, scale, downcast, x weight."""
    return [_ew("to", 6 * T * H, "to_f32"), _ew("pow", 8 * T * H, "pow"), _ew("mean", 4 * T * H + 4 * T, "mean"),
            _ew("add", 8 * T, "add_eps"), _ew("rsqrt", 8 * T, "rsqrt"), _ew("mul", 8 * T * H + 4 * T, "mul_rows"),
            _ew("to", 6 * T * H, "to_f16"), _ew("mul", 4 * T * H + 2 * H, "mul_weight")]


def _rope(T, W, cos_bytes):
    """Rotary on T tokens of q or k (width W = heads x head dim): x cos, rotate_half (neg of one
    half, cat of the halves), x sin, add."""
    return [_ew("mul", 4 * T * W + cos_bytes, "rope_mul"), _ew("neg", 2 * T * W, "rope_neg"),
            _ew("cat", 4 * T * W, "cat"), _ew("mul", 4 * T * W + cos_bytes, "rope_mul"), _ew("add", 6 * T * W, "add")]


def layer_ops_from_shape(shape, mb, ctx, prefill_seq=None):
    """One decoder layer's kernels in classify's form, from the model shape (REFERENCE_SHAPES: H, I,
    heads, kv, hd): decode (one token, ctx cached) or prefill (prefill_seq tokens, causal mask, empty
    cache). The same list as classify(trace_ops(...)) without the free calls."""
    H, I, nh, kv, d = shape["H"], shape["I"], shape["heads"], shape["kv"], shape["hd"]
    Q, KV = nh * d, kv * d
    q_len = prefill_seq or 1
    T = mb * q_len
    cos_bytes = 2 * d * (prefill_seq or mb)  # cos / sin: prefill positions (1, seq), decode (mb, 1)
    ops = _rms_norm(T, H)
    ops += [("gemm", dict(M=T, N=Q, K=H)), ("gemm", dict(M=T, N=KV, K=H)), ("gemm", dict(M=T, N=KV, K=H))]
    ops += _rope(T, Q, cos_bytes) + _rope(T, KV, cos_bytes)
    if prefill_seq:  # empty cache: an empty tensor each for K and V, then a cat onto it
        ops += [_ew("tensor", 0), _ew("tensor", 0), _ew("cat", 4 * T * KV, "cat"), _ew("cat", 4 * T * KV, "cat")]
        if kv < nh:  # masked SDPA gets K and V repeated to every query head (repeat_kv: one copy each)
            ops += [("copy", dict(name="reshape", bytes=4 * T * Q)), ("copy", dict(name="reshape", bytes=4 * T * Q))]
        ops.append(("attention", dict(mb=mb, heads=nh, q_len=q_len, kv=nh, L=q_len, d=d, gqa=False, masked=True)))
        # the output's transpose back costs nothing on the GPU (classify)
    else:  # K and V appended to the cache; unmasked SDPA takes the kv heads as they are (enable_gqa)
        append = ("kv_append", dict(mb=mb, kv=kv, L=ctx, d=d, bytes=4 * mb * KV * (ctx + 1)))
        ops += [append, append]
        ops.append(("attention", dict(mb=mb, heads=nh, q_len=1, kv=kv, L=ctx + 1, d=d, gqa=kv < nh, masked=False)))
    ops += [("gemm", dict(M=T, N=H, K=Q)), _ew("add", 6 * T * H, "add")]
    ops += _rms_norm(T, H)
    ops += [("gemm", dict(M=T, N=I, K=H)), _ew("silu", 4 * T * I, "silu"), ("gemm", dict(M=T, N=I, K=H)),
            _ew("mul", 6 * T * I, "mul"), ("gemm", dict(M=T, N=H, K=I)), _ew("add", 6 * T * H, "add")]
    return ops


def extras_from_shape(shape, mb, ctx, prefill_seq=None):
    """{role: kernels} of the per-microbatch work outside the layers (see trace_extras), from the
    model shape: the same lists as classify(trace_extras(...)[role]) without the free calls."""
    H, V, d = shape["H"], shape["V"], shape["hd"]
    T = mb * (prefill_seq or 1)
    B, P = (1, prefill_seq) if prefill_seq else (mb, mb)  # position ids: prefill (1, seq), decode (mb, 1)
    rotary = [_ew("arange", 8 * P)] if prefill_seq else []
    rotary += [_ew("float", 12 * P), _ew("matmul", 2 * d * B + 4 * P + 2 * d * P), _ew("cat", 8 * d * P, "cat"),
               _ew("cos", 8 * d * P), _ew("mul", 8 * d * P), _ew("sin", 8 * d * P), _ew("mul", 8 * d * P),
               _ew("to", 6 * d * P, "to_f16"), _ew("to", 6 * d * P, "to_f16")]
    return dict(
        rotary=rotary,
        embed=[_ew("embedding", 8 * T + 4 * T * H)],
        clone=[("copy", dict(name="clone", bytes=4 * T * H))],
        head=_rms_norm(T, H) + [("gemm", dict(M=mb, N=V, K=H)), _ew("argmax", 2 * mb * V + 8 * mb),
                                ("copy", dict(name="__setitem__", bytes=16 * mb))],
    )


# model configs the validation layers use (config.json values; shapes only)
MODELS = {
    "lmsys/vicuna-7b-v1.5": dict(model_type="llama", hidden_size=4096, intermediate_size=11008, num_attention_heads=32,
                                 num_key_value_heads=32, vocab_size=32000, rms_norm_eps=1e-5, max_position_embeddings=4096),
    "Qwen/Qwen2.5-7B": dict(model_type="qwen2", hidden_size=3584, intermediate_size=18944, num_attention_heads=28,
                            num_key_value_heads=4, vocab_size=152064, rms_norm_eps=1e-6, max_position_embeddings=32768),
}
