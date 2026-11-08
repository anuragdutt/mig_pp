"""
Measures every empirical constant needed by pulp_layer_placement.py.

TWO MODES. The split matters -- it is what makes the model predictive.

  FULL MODE  (--mode full, run on the WHOLE GPU, no MIG)
      Measures hardware properties that do not depend on how the GPU is
      partitioned: bandwidth efficiency in each access regime, kernel launch
      overhead, kernel count, and the host-side transport costs. Every
      efficiency is normalized by the FULL GPU's peak bandwidth (A100_BW), so
      the numbers are pure properties of the hardware and the workload.

  SLICE MODE (--mode slice, run on ONE MIG slice)
      Measures only one thing: what fraction of full-GPU bandwidth this slice
      actually gets. That fraction is the scaling law the model needs, and it
      is DERIVED here rather than assumed.

Why the split: if you measure efficiency on a slice and divide by an ASSUMED
slice bandwidth, the two errors multiply and you cannot separate them --
        EFF_measured = (true_efficiency * true_slice_BW) / (A100_BW * assumed_frac)
A wrong assumed_frac silently corrupts EFF_WEIGHT, and you would have to
re-measure everything for every new MIG geometry. Measuring efficiency once on
the full GPU and calibrating the per-slice fraction separately means you can
predict a MIG configuration you have never run.

NOTE ON THE SCALING LAW: the model currently assumes bandwidth scales with GPC
count (3/7 for a 3g.20gb slice). That is not obviously right -- MIG also
partitions memory controllers, and on a 40GB A100 a 3g.20gb slice gets 2 of the
4 HBM stacks, which would be 1/2. Slice mode settles this by measurement. Run it
once per slice geometry and paste the measured fractions into SLICE_FRAC.

Usage:
    # once, on the unpartitioned GPU
    python3 measure_constants.py --mode full

    # once per slice geometry (MIG enabled)
    CUDA_VISIBLE_DEVICES=<mig_uuid> python3 measure_constants.py --mode slice

What FULL mode measures, and which constant it feeds:

    EFF_WEIGHT          large contiguous weight reads      -> weight streaming
    EFF_ACT(ctx)        single-token decode, SWEPT over    -> activation/KV traffic
                        context length (see note below)
    LAUNCH_US           per-kernel launch + dispatch       -> launch overhead
    KERNELS_PER_LAYER   profiler kernel count per layer    -> launch overhead
    PCIE_BW_PINNED      pinned host<->device DMA           -> transport send leg
    PCIE_BW_UNPINNED    pageable host->device (bounced)    -> transport recv leg
    MEMCPY_BW           host-to-host memcpy (SHM path)     -> transport both legs
    HANDOFF_FIXED_US    cuda.sync drain + handshake        -> transport fixed cost
    TOK_ROUNDTRIP_US    tiny blocking send/recv round-trip -> per-step feedback

NOTE ON EFF_ACT:
    Decode activation traffic is dominated by the KV-cache read, which grows
    linearly with context. At ctx=64 it is a small scattered read (latency-bound,
    low efficiency); by ctx=576 it is a ~9MB contiguous stream (much closer to
    the efficient regime). A single scalar EFF_ACT is therefore wrong at one end
    no matter what you measure. This script sweeps ctx and reports the whole
    curve plus a traffic-weighted average, so you can either (a) paste the
    weighted scalar, or (b) make EFF_ACT a function of ctx in the model.

EDIT THIS:
"""

# Peak HBM bandwidth per card, bytes/s (spec sheet). The 40GB and 80GB A100s
# differ by 1.31x, so using the wrong one scales EVERY efficiency constant by
# that factor. --card picks one; the default is detected from the device name.
CARD_BW = {
    "A100-40GB": 1555e9,   # HBM2
    "A100-80GB": 2039e9,   # HBM2e
    "H100-80GB": 3350e9,   # HBM3 SXM
}


def detect_card() -> str:
    """Best-effort card identification from the CUDA device name."""
    try:
        import torch as _t
        name = _t.cuda.get_device_name(0)
    except Exception:
        return "A100-40GB"
    if "H100" in name:
        return "H100-80GB"
    if "A100" in name:
        # MIG slice names do not carry the card's total memory, so ask the device.
        try:
            total_gb = _t.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            total_gb = 40
        # A MIG slice reports only its own slice memory; fall back on the name.
        if "80GB" in name or total_gb > 45:
            return "A100-80GB"
        return "A100-40GB"
    return "A100-40GB"


A100_BW = 1555e9  # overwritten in main() from --card / detect_card()

# Slice mode only: which slice you are calibrating, and what the model currently
# ASSUMES its bandwidth fraction is. Slice mode measures the real fraction and
# tells you whether this assumption holds.
MIG_UUID = "MIG-98f93df6-d522-5c00-9923-4326839cef2e"  # 3g.20gb slice
SLICE_LABEL = "3g.20gb"
SLICE_FRAC_ASSUMED = 3 / 7  # what pulp_layer_placement.py assumes today

# Model under test. --model overrides. Architecture (hidden/inter/vocab/layers,
# and GQA's kv-head count) is read from config.json rather than hardcoded, so
# mistral / qwen2 / 13B / 24B all work. GQA matters a lot: a model with 8 kv
# heads out of 32 moves 1/4 the KV bytes, and the old hardcoded MHA formula
# overstated KV traffic by that factor.
MODEL_NAME = "lmsys/vicuna-7b-v1.5"

# workload shape -- must match benchmark_pipeline_microbatching.py
SEQ_LEN = 64
MAX_NEW_TOKENS = 512
MAX_SEQ = SEQ_LEN + MAX_NEW_TOKENS

# context lengths at which to sample EFF_ACT
EFF_ACT_CTXS = [64, 128, 192, 320, 448, 576]

# microbatch size to use for the transport measurements (payload scale)
TRANSPORT_MB = 16

# ---- imports ----
import os
import time
import statistics

# NOTE: CUDA_VISIBLE_DEVICES is NOT set here. Full mode must see the whole
# GPU; slice mode expects you to export the MIG UUID before running.

import numpy as np
import torch
from transformers import DynamicCache

# Filled by load_arch() from the model's config.json. Defaults are Vicuna-7B so
# the module still imports without a model present.
HIDDEN, INTER, VOCAB, LAYERS = 4096, 11008, 32000, 32
N_HEADS, N_KV_HEADS, HEAD_DIM = 32, 32, 128
MODEL_TYPE = "llama"
WARMUP = 20
REPEATS = 200


def load_arch(model_name: str = None):
    """Read architecture from config.json. Sets the module-level globals.

    Uses model_family.resolve when available so this stays in step with the
    benchmark's own notion of each family; falls back to AutoConfig.
    """
    global HIDDEN, INTER, VOCAB, LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
    global MODEL_TYPE, MODEL_NAME
    if model_name:
        MODEL_NAME = model_name

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    HIDDEN = cfg.hidden_size
    INTER = cfg.intermediate_size
    VOCAB = cfg.vocab_size
    LAYERS = cfg.num_hidden_layers
    N_HEADS = cfg.num_attention_heads
    # GQA / MQA: absent means MHA (kv heads == attention heads).
    N_KV_HEADS = getattr(cfg, "num_key_value_heads", None) or N_HEADS
    # Mistral-Small-24B has an explicit head_dim != hidden/heads (128 vs 160).
    HEAD_DIM = getattr(cfg, "head_dim", None) or (HIDDEN // N_HEADS)
    MODEL_TYPE = getattr(cfg, "model_type", "llama")
    return cfg


def resolve_classes():
    """Decoder layer + rotary classes for this model family.

    Prefers the repo's model_family so measurement and benchmark agree.
    """
    try:
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "..", ".."))
        import model_family
        fam = model_family.resolve(MODEL_TYPE)
        # Field names are decoder_layer / rotary_embedding (model_family.Family).
        # The previous *_cls names did not exist and the bare `except Exception`
        # swallowed the AttributeError, so this ALWAYS fell through to the
        # hardcoded imports below -- silently agreeing with the benchmark only by
        # luck. Narrow the except so a real mismatch is visible.
        return fam.decoder_layer, fam.rotary_embedding
    except (ImportError, KeyError, AttributeError) as e:
        print(f"[warn] model_family.resolve({MODEL_TYPE!r}) unavailable "
              f"({type(e).__name__}: {e}); using direct transformers imports")
    if MODEL_TYPE == "mistral":
        from transformers.models.mistral.modeling_mistral import (
            MistralDecoderLayer as DL, MistralRotaryEmbedding as RE)
    elif MODEL_TYPE in ("qwen2", "qwen2_moe"):
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2DecoderLayer as DL, Qwen2RotaryEmbedding as RE)
    else:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer as DL, LlamaRotaryEmbedding as RE)
    return DL, RE

# Normalization basis for every efficiency constant. Full mode runs on the whole
# GPU, so efficiencies are fractions of the FULL device's peak bandwidth. That is
# what makes them hardware properties rather than slice-specific numbers, and it
# is what lets the model predict a MIG geometry you have not run.
PEAK_BW = A100_BW


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

_CARD = "A100-40GB"      # set in main()
_JSON_PATH = None        # set by --json; constants are also printed either way


def _emit_json(payload: dict):
    """Write measured constants to JSON so the MILP can import rather than
    have numbers pasted by hand -- paste-by-hand is how a stale A100_BW or a
    stale SLICE_FRAC silently corrupts every downstream cost."""
    if not _JSON_PATH:
        return
    import json
    payload = dict(payload)
    payload["_measured_at_unix"] = time.time()
    with open(_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[json] wrote {_JSON_PATH}")


def kv_dim() -> int:
    """Per-token K (or V) width in elements. GQA/MQA aware."""
    return N_KV_HEADS * HEAD_DIM


def act_bytes_per_token(ctx: int) -> int:
    """Activation + KV traffic per token per layer (bytes), flash-attn aware.

    GQA-aware: the KV read is N_KV_HEADS*HEAD_DIM wide, not HIDDEN. For
    Llama-2-7B (MHA) these coincide; for Mistral-7B / Qwen2 (8 kv heads of 32)
    the KV term is 4x smaller, and the old HIDDEN-based formula overstated decode
    traffic by nearly that factor on exactly the models now being swept.

    Keep in step with act_bytes_per_token in pulp_layer_placement.py.
    """
    B = 2  # FP16
    kvd = kv_dim()
    q_out = N_HEADS * HEAD_DIM
    qkv = (HIDDEN + q_out + 2 * kvd) * B        # read x, write Q,K,V
    flash = (q_out + 2 * ctx * kvd + q_out) * B  # read Q + full-ctx K,V, write out
    oproj = (q_out + HIDDEN) * B
    gate = (HIDDEN + INTER) * B
    up = (HIDDEN + INTER) * B
    swiglu = (2 * INTER + INTER) * B
    down = (INTER + HIDDEN) * B
    misc = (6 * HIDDEN) * B
    return qkv + flash + oproj + gate + up + swiglu + down + misc


def weight_bytes_per_layer() -> int:
    """FP16 weight bytes in one decoder layer. GQA-aware."""
    kvd = kv_dim()
    q_out = N_HEADS * HEAD_DIM
    attn = (HIDDEN * q_out + 2 * HIDDEN * kvd + q_out * HIDDEN)
    mlp = 3 * HIDDEN * INTER
    return (attn + mlp) * 2


def kv_bytes_per_token_per_layer() -> int:
    """KV cache growth per token per layer (bytes) -- K and V, FP16."""
    return 2 * kv_dim() * 2


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _layer_out(ret):
    """Unwrap a decoder layer's return value.

    transformers <5 returned a tuple (hidden, *extras); 5.x returns the bare
    tensor. Indexing [0] unconditionally silently slices the BATCH dimension on
    5.x -- no error, just a wrong-shaped tensor and a meaningless measurement.
    The box runs 5.18.0, so handle both.
    """
    if isinstance(ret, tuple):
        return ret[0]
    return ret


def _build_layer(device, n_layers=1):
    """Build n_layers decoder layers of the configured family on `device`."""
    config = load_arch()
    config._attn_implementation = "sdpa"  # same attention path as the pipeline
    DL, RE = resolve_classes()
    layers = [DL(config, layer_idx=i).half().to(device).eval() for i in range(n_layers)]
    rotary = RE(config=config, device=device)
    if n_layers == 1:
        return layers[0], rotary, config
    return layers, rotary, config


# ---------------------------------------------------------------------------
# 1. EFF_WEIGHT
#    How: allocate a tensor the size of one layer's weights (~400MB).
#         Time repeated .sum() calls which force the whole tensor through HBM.
#         achieved_bw = bytes_moved / time -> EFF_WEIGHT = achieved / peak.
# ---------------------------------------------------------------------------
def calculate_EFF_WEIGHT() -> float:
    device = torch.device("cuda:0")
    WEIGHT_BYTES = weight_bytes_per_layer()  # one layer, FP16, GQA-aware

    n_elements = WEIGHT_BYTES // 2  # float16 = 2 bytes
    tensor = torch.randn(n_elements, dtype=torch.float16, device=device)

    for _ in range(WARMUP):
        _ = tensor.sum()
    torch.cuda.synchronize()

    t0 = sync_time()
    for _ in range(REPEATS):
        _ = tensor.sum()
    t1 = sync_time()

    elapsed_s = (t1 - t0) / REPEATS
    achieved_bw = WEIGHT_BYTES / elapsed_s  # bytes/s
    return achieved_bw / PEAK_BW


# ---------------------------------------------------------------------------
# 2. EFF_ACT -- SWEPT over context length.
#    How: run ONE LlamaDecoderLayer decode step (1 new token) with the KV cache
#         pre-filled to `ctx` tokens. Subtract the weight-streaming time (which
#         is already accounted for by EFF_WEIGHT in the model) so what remains
#         is the activation/KV traffic time, then compare against the ideal.
#
#         EFF_ACT(ctx) = ideal_act_time / (measured_time - weight_time - launch)
# ---------------------------------------------------------------------------
def calculate_EFF_ACT_curve(eff_weight: float, launch_ms_per_layer: float):
    device = torch.device("cuda:0")
    layer, rotary, _ = _build_layer(device)
    WEIGHT_BYTES = weight_bytes_per_layer()

    weight_s = WEIGHT_BYTES / (PEAK_BW * eff_weight)
    launch_s = launch_ms_per_layer / 1000.0

    results = []
    for ctx in EFF_ACT_CTXS:
        # pre-fill a KV cache to exactly `ctx` tokens
        cache = DynamicCache()
        prefill = torch.randn(1, ctx, HIDDEN, dtype=torch.float16, device=device)
        pref_pos = torch.arange(ctx, device=device).unsqueeze(0)
        pref_emb = rotary(prefill, pref_pos)
        with torch.no_grad():
            layer(prefill, position_embeddings=pref_emb, past_key_values=cache)

        # one new token on top of that context
        hidden = torch.randn(1, 1, HIDDEN, dtype=torch.float16, device=device)
        pos = torch.tensor([[ctx]], dtype=torch.long, device=device)
        emb = rotary(hidden, pos)

        # The cache grows by one token per call, so a naive warmup+timing loop
        # measures a DRIFTING context while charging ideal bytes at the nominal
        # ctx. With WARMUP=20 and reps=50 the mean real context was ctx+45 --
        # +70% at ctx=64, exactly where the curve matters most, biasing EFF_ACT
        # low at the short end.
        #
        # Fix: rebuild the cache to exactly `ctx` before every timed call, and
        # time only the decode call. Rebuild cost is excluded from the timing by
        # synchronizing around each call individually.
        def _fresh_cache_at(n_ctx):
            c = DynamicCache()
            pf = torch.randn(1, n_ctx, HIDDEN, dtype=torch.float16, device=device)
            pp = torch.arange(n_ctx, device=device).unsqueeze(0)
            pe = rotary(pf, pp)
            with torch.no_grad():
                layer(pf, position_embeddings=pe, past_key_values=c)
            return c

        with torch.no_grad():
            warm = _fresh_cache_at(ctx)
            for _ in range(WARMUP):
                _ = layer(hidden, position_embeddings=emb, past_key_values=warm)
        torch.cuda.synchronize()

        reps = 30
        per_call = []
        with torch.no_grad():
            for _ in range(reps):
                c = _fresh_cache_at(ctx)          # exactly `ctx` every time
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = layer(hidden, position_embeddings=emb, past_key_values=c)
                torch.cuda.synchronize()
                per_call.append(time.perf_counter() - t0)
                del c

        # Median, not mean: each call now carries its own sync, so the
        # distribution has a right tail from allocator / scheduling noise.
        measured_s = statistics.median(per_call)
        act_only_s = measured_s - weight_s - launch_s
        ideal_act_s = act_bytes_per_token(ctx) / PEAK_BW

        eff = ideal_act_s / act_only_s if act_only_s > 0 else float("nan")
        results.append((ctx, eff, measured_s, act_only_s))

    return results


def weighted_EFF_ACT(curve) -> float:
    """Time-equivalent scalar efficiency over the real decode trajectory.

    The model turns EFF into TIME as  t = bytes / (PEAK * EFF). A single scalar
    is only valid if it reproduces the TOTAL time:

        sum_i t_i = sum_i bytes_i / (PEAK * eff_i)
                  = sum_i(bytes_i) / (PEAK * EFF_scalar)
        =>  EFF_scalar = sum(bytes) / sum(bytes / eff)      <- harmonic

    The previous version used a byte-weighted ARITHMETIC mean, which overstates
    efficiency (and so understates time) whenever efficiency varies, because the
    slow segments dominate total time but receive only byte-proportional weight.
    On a two-point example with equal bytes at eff 0.2 and 0.8 the arithmetic
    form gives 0.50 and the correct time-equivalent form gives 0.32 -- a 1.56x
    optimistic error in every decode cost the model computes.
    """
    import bisect

    ctxs = [c for c, _, _, _ in curve]
    effs = [e for _, e, _, _ in curve]

    num = 0.0   # total bytes
    den = 0.0   # total bytes/eff  (proportional to total time)
    for t in range(MAX_NEW_TOKENS):
        ctx = SEQ_LEN + t
        i = min(bisect.bisect_left(ctxs, ctx), len(ctxs) - 1)
        e = effs[i]
        if not (e == e) or e <= 0:   # NaN or non-positive: skip, cannot invert
            continue
        w = act_bytes_per_token(ctx)
        num += w
        den += w / e
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# 3. LAUNCH_US
#    How: launch an empty CUDA kernel many times and time the overhead.
#         This isolates the per-kernel launch + framework dispatch cost
#         with zero compute or memory work.
# ---------------------------------------------------------------------------
def calculate_LAUNCH_US() -> float:
    device = torch.device("cuda:0")
    tiny = torch.zeros(1, dtype=torch.float16, device=device)

    for _ in range(WARMUP):
        _ = tiny + 0
    torch.cuda.synchronize()

    t0 = sync_time()
    for _ in range(REPEATS * 10):
        _ = tiny + 0
    t1 = sync_time()

    return (t1 - t0) / (REPEATS * 10) * 1e6  # -> microseconds


# ---------------------------------------------------------------------------
# 4. KERNELS_PER_LAYER
#    How: profile ONE decode forward through a single LlamaDecoderLayer and
#         count distinct CUDA kernel launches. (kernel_measure.py does this for
#         the full pipeline including transport; this is the single-layer core.)
# ---------------------------------------------------------------------------
def calculate_KERNELS_PER_LAYER() -> float:
    from torch.autograd import DeviceType

    device = torch.device("cuda:0")
    layer, rotary, _ = _build_layer(device)

    cache = DynamicCache()
    prefill = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float16, device=device)
    pref_pos = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    pref_emb = rotary(prefill, pref_pos)
    with torch.no_grad():
        layer(prefill, position_embeddings=pref_emb, past_key_values=cache)

    hidden = torch.randn(1, 1, HIDDEN, dtype=torch.float16, device=device)
    pos = torch.tensor([[SEQ_LEN]], dtype=torch.long, device=device)
    emb = rotary(hidden, pos)

    with torch.no_grad():
        _ = layer(hidden, position_embeddings=emb, past_key_values=cache)
    torch.cuda.synchronize()

    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            _ = layer(hidden, position_embeddings=emb, past_key_values=cache)

    cuda_events = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
    # count launches, not distinct kernel names
    return float(sum(e.count for e in cuda_events))


# ---------------------------------------------------------------------------
# 5-7. TRANSPORT BANDWIDTHS
#    Mirrors mig_transport_pipeline_non_blocking.py exactly:
#      send: staging.copy_(tensor)        GPU -> pinned CPU   [PCIE_BW_PINNED]
#            np_slot[:n] = raw            pinned -> SHM       [MEMCPY_BW]
#      recv: peer_data.copy()             SHM -> heap         [MEMCPY_BW]
#            src_tensor.to(device)        heap -> GPU         [PCIE_BW_UNPINNED]
# ---------------------------------------------------------------------------
def _transport_payload_bytes(mb=TRANSPORT_MB, tokens=1):
    return mb * tokens * HIDDEN * 2  # FP16 hidden state


def calculate_PCIE_BW_PINNED() -> float:
    device = torch.device("cuda:0")
    nbytes = _transport_payload_bytes()
    numel = nbytes // 2

    gpu_t = torch.randn(numel, dtype=torch.float16, device=device)
    staging = torch.zeros(numel, dtype=torch.float16).pin_memory()

    for _ in range(WARMUP):
        staging.copy_(gpu_t, non_blocking=True)
    torch.cuda.synchronize()

    t0 = sync_time()
    for _ in range(REPEATS):
        staging.copy_(gpu_t, non_blocking=True)
        torch.cuda.synchronize()
    t1 = sync_time()

    return nbytes / ((t1 - t0) / REPEATS)


def calculate_PCIE_BW_UNPINNED() -> float:
    """STALE PATH -- kept only to explain why the model's constant is wrong.

    This measures pageable heap -> GPU plus an extra device-side copy, which is
    what the OLD receiver did. The current receiver
    (mig_transport_pipeline._queue_read_from_slot) does:

        SHM -> recv_pinned_staging (host memcpy, pinned)
        staging -> tensor          (PINNED H2D on recv_stream, one copy)

    There is no pageable transfer and no second GPU copy any more. So
    PCIE_BW_UNPINNED = 12e9 in pulp_layer_placement.py charges the receive leg
    at roughly half the real rate. Use calculate_RECV_H2D_BW() instead; this
    stays so the two numbers can be compared and the stale constant retired.
    """
    device = torch.device("cuda:0")
    nbytes = _transport_payload_bytes()
    numel = nbytes // 2

    heap = torch.from_numpy(np.zeros(numel, dtype=np.float16))  # NOT pinned
    dst = torch.zeros(numel, dtype=torch.float16, device=device)

    for _ in range(WARMUP):
        dst.copy_(heap.to(device))
    torch.cuda.synchronize()

    t0 = sync_time()
    for _ in range(REPEATS):
        dst.copy_(heap.to(device))
    t1 = sync_time()

    return nbytes / ((t1 - t0) / REPEATS)


# ---------------------------------------------------------------------------
# 6b. RECV_H2D_BW  (NEW -- matches the CURRENT receiver)
#     Mirrors _queue_read_from_slot exactly: a pinned staging buffer, H2D on a
#     dedicated stream, waited on via an event. One copy, pinned, not pageable.
# ---------------------------------------------------------------------------
def calculate_RECV_H2D_BW() -> float:
    device = torch.device("cuda:0")
    nbytes = _transport_payload_bytes()
    numel = nbytes // 2

    staging = torch.zeros(numel, dtype=torch.float16).pin_memory()
    dst = torch.zeros(numel, dtype=torch.float16, device=device)
    recv_stream = torch.cuda.Stream()

    def one_recv():
        with torch.cuda.stream(recv_stream):
            dst.copy_(staging, non_blocking=True)
        evt = torch.cuda.Event()
        evt.record(recv_stream)
        evt.synchronize()

    for _ in range(WARMUP):
        one_recv()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        one_recv()
    t1 = time.perf_counter()

    return nbytes / ((t1 - t0) / REPEATS)


# ---------------------------------------------------------------------------
# 7b. SHM_TO_PINNED_BW  (NEW -- the receiver's host-side leg, as written)
#     The current code does:
#         peer_view = np.frombuffer(shm_slot[:nbytes], dtype=np.float16)
#         staging[:numel].copy_(torch.from_numpy(peer_view))
#     i.e. SHM -> PINNED directly, with no intermediate heap buffer. The old
#     probe measured SHM -> heap (.copy()), a different and slower operation
#     that the code deliberately removed.
# ---------------------------------------------------------------------------
def calculate_SHM_TO_PINNED_BW() -> float:
    from multiprocessing.shared_memory import SharedMemory

    nbytes = _transport_payload_bytes()
    numel = nbytes // 2

    shm_name = "mig_measure_shm2pinned_probe"
    try:
        if os.path.exists(f"/dev/shm/{shm_name}"):
            os.unlink(f"/dev/shm/{shm_name}")
    except Exception:
        pass
    shm = SharedMemory(name=shm_name, create=True, size=nbytes)
    try:
        staging = torch.zeros(numel, dtype=torch.float16).pin_memory()

        def one_leg():
            peer_view = np.frombuffer(shm.buf[:nbytes], dtype=np.float16)
            staging[:numel].copy_(torch.from_numpy(peer_view))

        for _ in range(WARMUP):
            one_leg()

        t0 = time.perf_counter()
        for _ in range(REPEATS):
            one_leg()
        t1 = time.perf_counter()
        return nbytes / ((t1 - t0) / REPEATS)
    finally:
        shm.close()
        shm.unlink()


def calculate_MEMCPY_BW() -> float:
    """Host-to-host: the pinned->SHM write and the SHM->heap .copy()."""
    from multiprocessing.shared_memory import SharedMemory

    nbytes = _transport_payload_bytes()
    numel = nbytes // 2

    shm_name = "mig_measure_memcpy_probe"
    try:
        if os.path.exists(f"/dev/shm/{shm_name}"):
            os.unlink(f"/dev/shm/{shm_name}")
    except Exception:
        pass
    shm = SharedMemory(name=shm_name, create=True, size=nbytes)
    try:
        np_slot = np.ndarray((nbytes,), dtype=np.uint8, buffer=shm.buf)
        staging = torch.zeros(numel, dtype=torch.float16).pin_memory()
        raw = staging.numpy().view(np.uint8)

        for _ in range(WARMUP):
            np_slot[:nbytes] = raw[:nbytes]

        # leg 1: pinned -> SHM
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            np_slot[:nbytes] = raw[:nbytes]
        t1 = time.perf_counter()
        bw_write = nbytes / ((t1 - t0) / REPEATS)

        # leg 2: SHM -> heap (np .copy())
        view = np.frombuffer(np_slot[:nbytes], dtype=np.float16)
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            _ = view.copy()
        t1 = time.perf_counter()
        bw_read = nbytes / ((t1 - t0) / REPEATS)

        # the model uses ONE constant for both legs -- report the harmonic mean
        # (the two costs add, so the equivalent single bandwidth is harmonic)
        return 2.0 / (1.0 / bw_write + 1.0 / bw_read), bw_write, bw_read
    finally:
        shm.close()
        shm.unlink()


# ---------------------------------------------------------------------------
# 8. HANDOFF_FIXED_US
#    The cuda.synchronize() drain at line 356 plus handshake dispatch. Measured
#    as: full send-leg time minus the pure bandwidth terms. This is the residual
#    fixed cost -- and it is a LOWER BOUND, because cuda.synchronize() drains
#    whatever GPU work is outstanding, which in the real pipeline includes the
#    layer compute that was just issued.
# ---------------------------------------------------------------------------
def calculate_HANDOFF_FIXED_US(pcie_pinned: float, memcpy_bw: float) -> float:
    """Residual of the SEND leg only -- local copies plus a device-wide drain.

    What this is NOT: it contains no handshake and no peer. The real per-handoff
    fixed cost is this residual PLUS the protocol round-trip, which
    measure_handshake_ack_us() measures with an actual second rank. Prefer
    HANDSHAKE_ACK_US for the protocol term and treat this as the local-copy
    residual; adding both is correct, using this alone understates the handoff.

    Also note torch.cuda.synchronize() here is a FULL device drain, so under real
    load it absorbs whatever compute was outstanding -- a lower bound, not a
    constant.
    """
    from multiprocessing.shared_memory import SharedMemory

    device = torch.device("cuda:0")
    nbytes = _transport_payload_bytes()
    numel = nbytes // 2

    gpu_t = torch.randn(numel, dtype=torch.float16, device=device)
    staging = torch.zeros(numel, dtype=torch.float16).pin_memory()

    shm_name = "mig_measure_handoff_probe"
    try:
        if os.path.exists(f"/dev/shm/{shm_name}"):
            os.unlink(f"/dev/shm/{shm_name}")
    except Exception:
        pass
    shm = SharedMemory(name=shm_name, create=True, size=nbytes)
    try:
        np_slot = np.ndarray((nbytes,), dtype=np.uint8, buffer=shm.buf)

        def one_send():
            staging.copy_(gpu_t, non_blocking=True)
            torch.cuda.synchronize()
            raw = staging.numpy().view(np.uint8)
            np_slot[:nbytes] = raw[:nbytes]

        for _ in range(WARMUP):
            one_send()

        t0 = time.perf_counter()
        for _ in range(REPEATS):
            one_send()
        t1 = time.perf_counter()

        measured_s = (t1 - t0) / REPEATS
        bandwidth_s = nbytes / pcie_pinned + nbytes / memcpy_bw
        residual_s = measured_s - bandwidth_s
        return max(residual_s, 0.0) * 1e6
    finally:
        shm.close()
        shm.unlink()


# ---------------------------------------------------------------------------
# 9. TOK_ROUNDTRIP_US
#    The per-step next_tokens feedback: last rank -> rank 0, one small int
#    tensor, blocking send/recv over gloo. Without a second process we can only
#    measure the local dispatch+serialize cost of the same-size operation; the
#    true value needs two ranks. Run measure_tok_roundtrip_distributed() for
#    that (2-process version) when you have the cluster up.
# ---------------------------------------------------------------------------
def calculate_TOK_ROUNDTRIP_US_local(batch_size=32) -> float:
    """Lower bound: CPU-side cost of preparing/copying the token tensor."""
    tok = torch.zeros((batch_size, 1), dtype=torch.long)
    dst = torch.zeros((batch_size, 1), dtype=torch.long)

    for _ in range(WARMUP):
        dst.copy_(tok)

    t0 = time.perf_counter()
    for _ in range(REPEATS * 10):
        dst.copy_(tok)
    t1 = time.perf_counter()

    return (t1 - t0) / (REPEATS * 10) * 1e6


# NOTE ON THE DISTRIBUTED PROBES BELOW:
# The worker functions are at MODULE scope, not nested. mp.set_start_method
# ("spawn") pickles the target by qualified name, and a nested function is not
# picklable -- "Can't pickle local object", before any measurement runs. Spawn is
# required (not fork) because the parent may already hold a CUDA context.


def _tok_worker(rank, q, batch_size, port, warmup, repeats):
    """Module-scope so spawn can pickle it. One leg of the token round-trip."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    tok = torch.zeros((batch_size, 1), dtype=torch.long)

    for _ in range(warmup):
        if rank == 1:
            dist.send(tok, dst=0, tag=1)
        else:
            dist.recv(tok, src=1, tag=1)
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(repeats):
        if rank == 1:
            dist.send(tok, dst=0, tag=2)
        else:
            dist.recv(tok, src=1, tag=2)
    t1 = time.perf_counter()

    if rank == 0:
        q.put((t1 - t0) / repeats * 1e6)
    dist.destroy_process_group()


def measure_tok_roundtrip_distributed(batch_size=32, port=29800):
    """True 2-rank measurement of the blocking next_tokens round-trip.

    NOTE this is a ONE-WAY cost per iteration (rank1 send -> rank0 recv), which
    is what the pipeline actually does per step, not a full round-trip.
    """
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_tok_worker,
                    args=(r, q, batch_size, port, WARMUP, REPEATS))
        for r in range(2)
    ]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(timeout=120)
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
    return q.get() if not q.empty() else float("nan")


# ===========================================================================
# ADDED MEASUREMENTS -- gaps found against what pulp_layer_placement.py needs
# and what the 9/28+ sweeps showed actually dominates. Each notes why it exists.
# ===========================================================================


# ---------------------------------------------------------------------------
# 10. EFF_PREFILL + PREFILL_FLOP_EFF  (NEW -- prefill was never measured)
#     Why: the model's only efficiency terms are decode-shaped. Prefill is a
#     different regime -- mb_size*seq tokens at once, compute-bound rather than
#     weight-bound -- and the sweeps put it at 3-7% of total latency, which is
#     not negligible when the MILP is choosing a split. Measured as achieved
#     FLOP/s against the card's fp16 tensor-core peak, plus the same
#     bandwidth-efficiency framing as EFF_ACT for consistency.
# ---------------------------------------------------------------------------
def calculate_EFF_PREFILL(mb_sizes=(4, 8, 16, 32), seq_len=None):
    device = torch.device("cuda:0")
    seq_len = seq_len or SEQ_LEN
    layer, rotary, _ = _build_layer(device)
    wb = weight_bytes_per_layer()

    results = []
    for mb in mb_sizes:
        cache = DynamicCache()
        hidden = torch.randn(mb, seq_len, HIDDEN, dtype=torch.float16, device=device)
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        emb = rotary(hidden, pos)

        try:
            with torch.no_grad():
                for _ in range(5):
                    c = DynamicCache()
                    layer(hidden, position_embeddings=emb, past_key_values=c)
            torch.cuda.synchronize()

            reps = 20
            t0 = sync_time()
            with torch.no_grad():
                for _ in range(reps):
                    c = DynamicCache()
                    layer(hidden, position_embeddings=emb, past_key_values=c)
            t1 = sync_time()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append((mb, float("nan"), float("nan"), float("nan")))
            continue

        measured_s = (t1 - t0) / reps
        tokens = mb * seq_len
        # 2 FLOP per MAC; attention is O(seq^2) per head.
        q_out = N_HEADS * HEAD_DIM
        kvd = kv_dim()
        proj_flop = 2 * tokens * (HIDDEN * q_out + 2 * HIDDEN * kvd
                                  + q_out * HIDDEN + 3 * HIDDEN * INTER)
        attn_flop = 2 * mb * N_HEADS * seq_len * seq_len * HEAD_DIM * 2
        total_flop = proj_flop + attn_flop

        # Bandwidth view: weights read once, activations streamed per token.
        ideal_bw_s = (wb + tokens * act_bytes_per_token(seq_len // 2)) / PEAK_BW
        results.append((mb, measured_s, total_flop / measured_s,
                        ideal_bw_s / measured_s))
    return results


# ---------------------------------------------------------------------------
# 11. SLICE COMPUTE SCALING  (NEW -- the model only scales BANDWIDTH per slice)
#     Why: pulp_layer_placement.py derives every per-slice cost from
#     BW = A100_BW * SLICE_FRAC, i.e. it assumes compute and bandwidth scale
#     together. The 9/28 sweep contradicts that: rank0 (3g) ran 1.57 ms/layer
#     while rank3 (1g) ran 4.98 ms/layer -- 3.2x, where the GPC ratio is 3x and
#     the bandwidth ratio something else again. Compute and bandwidth scale
#     DIFFERENTLY across MIG geometries, so the model needs both fractions.
#     Run in slice mode; ratio against the full-GPU number.
# ---------------------------------------------------------------------------
def calculate_slice_compute() -> float:
    """Achieved fp16 FLOP/s on the visible device (big square GEMM)."""
    device = torch.device("cuda:0")
    n = 4096
    a = torch.randn(n, n, dtype=torch.float16, device=device)
    b = torch.randn(n, n, dtype=torch.float16, device=device)

    for _ in range(WARMUP):
        _ = a @ b
    torch.cuda.synchronize()

    reps = 50
    t0 = sync_time()
    for _ in range(reps):
        _ = a @ b
    t1 = sync_time()

    flop = 2 * n ** 3
    return flop / ((t1 - t0) / reps)


def calculate_slice_decode_layer_ms(n_layers=4, ctx=None) -> float:
    """End-to-end decode ms per layer on the visible device.

    The most directly comparable number to the benchmark's [B05], and the one
    the MILP's per-rank term should reproduce. Measuring it per slice gives the
    compute scaling empirically rather than inferring it from GPC counts.
    """
    device = torch.device("cuda:0")
    ctx = ctx if ctx is not None else SEQ_LEN
    layers, rotary, _ = _build_layer(device, n_layers=n_layers)

    hidden = torch.randn(1, 1, HIDDEN, dtype=torch.float16, device=device)
    pos = torch.tensor([[ctx]], dtype=torch.long, device=device)
    emb = rotary(hidden, pos)

    def _fresh_caches():
        """Caches holding exactly `ctx` tokens.

        Rebuilt per timed call: a single cache reused across warmup and timing
        grows one token per call, so the measured context drifts well above the
        nominal one (+70% at ctx=64 with warmup=10/reps=30) and the number is
        not the per-layer cost at `ctx` at all.
        """
        cs = [DynamicCache() for _ in range(n_layers)]
        pf = torch.randn(1, ctx, HIDDEN, dtype=torch.float16, device=device)
        pp = torch.arange(ctx, device=device).unsqueeze(0)
        pe = rotary(pf, pp)
        with torch.no_grad():
            h = pf
            for L, c in zip(layers, cs):
                h = _layer_out(L(h, position_embeddings=pe, past_key_values=c))
        return cs

    with torch.no_grad():
        warm = _fresh_caches()
        for _ in range(10):
            h = hidden
            for L, c in zip(layers, warm):
                h = _layer_out(L(h, position_embeddings=emb, past_key_values=c))
    torch.cuda.synchronize()

    reps = 20
    per_call = []
    with torch.no_grad():
        for _ in range(reps):
            cs = _fresh_caches()          # exactly `ctx` every time
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h = hidden
            for L, c in zip(layers, cs):
                h = _layer_out(L(h, position_embeddings=emb, past_key_values=c))
            torch.cuda.synchronize()
            per_call.append(time.perf_counter() - t0)
            del cs
    return (statistics.median(per_call) / n_layers) * 1e3


# ---------------------------------------------------------------------------
# 12. ACK / HANDSHAKE ROUND-TRIP  (NEW -- post-ACK-fix transport is unmeasured)
#     Why: HANDOFF_FIXED_US lumps everything into one residual measured without
#     a peer. After the ACK fix the real protocol is: handshake send, ACK irecv
#     posted early, ACK drained at wait(). [T25] handshake waits ran 6-53 ms in
#     the sweeps -- orders above the 30 us constant, because they include waiting
#     for the upstream rank. This measures the protocol's own cost with an idle
#     peer, which is the part that belongs in a fixed constant; the rest is
#     queueing the model should derive, not measure.
# ---------------------------------------------------------------------------
def _ack_worker(rank, q, port, reps):
    """Module-scope so spawn can pickle it (see note above _tok_worker)."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)

    handshake = torch.zeros(1, dtype=torch.int32)
    ack = torch.zeros(1, dtype=torch.int32)

    for _ in range(20):  # warmup
        if rank == 0:
            r = dist.irecv(ack, src=1, tag=10_000_001)
            dist.send(handshake, dst=1, tag=1)
            r.wait()
        else:
            dist.recv(handshake, src=0, tag=1)
            dist.send(ack, dst=0, tag=10_000_001)
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(reps):
        if rank == 0:
            # Exactly the post-fix order: post the ACK irecv BEFORE the
            # handshake send (async_handle_send.flush does this).
            r = dist.irecv(ack, src=1, tag=10_000_002)
            dist.send(handshake, dst=1, tag=2)
            r.wait()
        else:
            dist.recv(handshake, src=0, tag=2)
            dist.send(ack, dst=0, tag=10_000_002)
    t1 = time.perf_counter()

    if rank == 0:
        q.put((t1 - t0) / reps * 1e6)
    dist.destroy_process_group()


def measure_handshake_ack_us(port=29801, reps=200):
    """Two-process gloo: handshake + ACK with an idle peer.

    This is the protocol's OWN cost. It is not [T25]: that also contains waiting
    for the upstream rank's compute, which the model should derive from the
    schedule rather than fold into a constant.
    """
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ack_worker, args=(r, q, port, reps))
             for r in range(2)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(timeout=120)
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
    return q.get() if not q.empty() else float("nan")


# ---------------------------------------------------------------------------
# 13. D2H OVERLAP EFFICIENCY  (NEW)
#     Why: the transport stages activations D2H on a side stream so the copy
#     hides under the next microbatch's compute ([T29] reports 0.02-0.75 ms
#     residual block). The MILP charges the full copy time. This measures what
#     fraction actually hides, which is the difference between modelling
#     transport as additive or as overlapped.
# ---------------------------------------------------------------------------
def measure_d2h_overlap(payload_mb=TRANSPORT_MB) -> dict:
    device = torch.device("cuda:0")
    nbytes = payload_mb * 1 * HIDDEN * 2
    numel = nbytes // 2

    gpu_t = torch.randn(numel, dtype=torch.float16, device=device)
    staging = torch.zeros(numel, dtype=torch.float16).pin_memory()
    copy_stream = torch.cuda.Stream()

    # a compute kernel roughly the size of one layer's decode work
    a = torch.randn(2048, 2048, dtype=torch.float16, device=device)

    # All three legs use the SAME synchronization policy: one sync per
    # iteration, timed individually, median reported. Previously (a) and (b)
    # synced once for the whole loop (so per-iteration launch cost was
    # amortized/pipelined) while (c) synced every iteration -- the overlap
    # fraction then mixed a real overlap effect with a sync-policy artifact.
    def _timed(fn, reps=None):
        reps = reps or REPEATS
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    def _copy_only():
        with torch.cuda.stream(copy_stream):
            staging.copy_(gpu_t, non_blocking=True)
        copy_stream.synchronize()

    def _compute_only():
        _ = a @ a

    def _both():
        with torch.cuda.stream(copy_stream):
            staging.copy_(gpu_t, non_blocking=True)
        _ = a @ a
        copy_stream.synchronize()

    copy_alone = _timed(_copy_only)
    compute_alone = _timed(_compute_only)
    both = _timed(_both)

    serial = copy_alone + compute_alone
    hidden_frac = (serial - both) / copy_alone if copy_alone > 0 else float("nan")
    return {
        "copy_alone_us": copy_alone * 1e6,
        "compute_alone_us": compute_alone * 1e6,
        "overlapped_us": both * 1e6,
        "serial_us": serial * 1e6,
        "hidden_fraction": max(0.0, min(1.0, hidden_frac)),
    }


# ---------------------------------------------------------------------------
# 14. KV-CACHE GROWTH COST  (NEW)
#     Why: DynamicCache grows by concatenation, so each decode step reallocates
#     and copies the whole cache. That is an O(ctx) cost per step the model does
#     not charge at all, and it is on the critical path of every step.
# ---------------------------------------------------------------------------
def measure_kv_growth_us(ctxs=(64, 192, 320, 448, 576)) -> list:
    """Cost of DynamicCache's concat per layer per step: BOTH K and V.

    DO NOT ADD THIS TO THE EFF_ACT RESIDUAL. EFF_ACT is derived from a full
    decode call through a real layer, which already contains the cache update,
    so charging this on top double-counts it. It is reported separately to show
    how much of the per-step cost is cache reallocation rather than useful
    traffic -- i.e. to say whether a preallocated cache would be worth it.
    """
    device = torch.device("cuda:0")
    out = []
    for ctx in ctxs:
        k = torch.randn(1, N_KV_HEADS, ctx, HEAD_DIM, dtype=torch.float16, device=device)
        v = torch.randn(1, N_KV_HEADS, ctx, HEAD_DIM, dtype=torch.float16, device=device)
        nk = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=device)
        nv = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=device)

        def both():
            torch.cat([k, nk], dim=-2)
            torch.cat([v, nv], dim=-2)

        for _ in range(WARMUP):
            both()
        torch.cuda.synchronize()
        t0 = sync_time()
        for _ in range(REPEATS):
            both()
        t1 = sync_time()
        out.append((ctx, (t1 - t0) / REPEATS * 1e6))
    return out


# ---------------------------------------------------------------------------
# 15. EMBED / LM_HEAD BUILD PEAK + COST  (NEW)
#     Why: CLAUDE.md open item 6 -- these are built fp32 then .half()'d, so the
#     transient peak is 6 bytes/param, and that peak is what caps Qwen's last
#     rank. The MILP charges EMBED_BYTES/LM_HEAD_BYTES at fp16 only, so it
#     under-counts the binding memory constraint. Measures both the peak and the
#     per-step lm_head compute.
# ---------------------------------------------------------------------------
def measure_head_cost() -> dict:
    import torch.nn as nn
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    base = torch.cuda.memory_allocated(device)
    head = nn.Linear(HIDDEN, VOCAB, bias=False).to(device).half()
    peak_build = torch.cuda.max_memory_allocated(device) - base
    resident = torch.cuda.memory_allocated(device) - base

    x = torch.randn(1, 1, HIDDEN, dtype=torch.float16, device=device)
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = head(x)
    torch.cuda.synchronize()
    t0 = sync_time()
    with torch.no_grad():
        for _ in range(REPEATS):
            _ = head(x)
    t1 = sync_time()

    fp16_theory = VOCAB * HIDDEN * 2
    del head
    torch.cuda.empty_cache()
    return {
        "build_peak_mib": peak_build / 2**20,
        "resident_mib": resident / 2**20,
        "fp16_theory_mib": fp16_theory / 2**20,
        "peak_over_fp16": peak_build / fp16_theory if fp16_theory else float("nan"),
        "lm_head_us": (t1 - t0) / REPEATS * 1e6,
    }


# ---------------------------------------------------------------------------
# SLICE CALIBRATION -- the only thing slice mode measures.
#
# Runs the same large contiguous read probe as calculate_EFF_WEIGHT, but on a
# MIG slice, and reports achieved_bandwidth / A100_BW. Because the probe and the
# access pattern are identical to the full-GPU run, the workload efficiency
# cancels when you take the ratio:
#
#     frac = achieved_slice / achieved_full
#
# leaving the pure bandwidth scaling of the partition. That fraction is what
# SLICE_FRAC in pulp_layer_placement.py should hold.
#
# Pass --full-achieved <GB/s> (printed by full mode) to get the cancelled ratio.
# Without it, the raw achieved/A100_BW is reported, which still bounds the answer.
# ---------------------------------------------------------------------------
def calculate_slice_bandwidth() -> float:
    """Achieved bandwidth (bytes/s) on whatever device is visible."""
    device = torch.device("cuda:0")
    WEIGHT_BYTES = weight_bytes_per_layer()

    n_elements = WEIGHT_BYTES // 2
    tensor = torch.randn(n_elements, dtype=torch.float16, device=device)

    for _ in range(WARMUP):
        _ = tensor.sum()
    torch.cuda.synchronize()

    t0 = sync_time()
    for _ in range(REPEATS):
        _ = tensor.sum()
    t1 = sync_time()

    return WEIGHT_BYTES / ((t1 - t0) / REPEATS)


def run_slice_mode(full_achieved_bw=None):
    torch.cuda.set_device(0)
    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"Slice label: {SLICE_LABEL}")
    print(f"Full-GPU spec bandwidth: {A100_BW / 1e9:.1f} GB/s")
    print()

    print("Measuring achieved bandwidth on this slice (same probe as full mode)...")
    achieved = calculate_slice_bandwidth()
    print(f"  achieved = {achieved / 1e9:.1f} GB/s")
    print()

    frac_vs_spec = achieved / A100_BW
    print(f"  achieved / A100_BW(spec)  = {frac_vs_spec:.4f}")

    if full_achieved_bw:
        frac = achieved / full_achieved_bw
        print(f"  achieved / achieved_full  = {frac:.4f}   <-- USE THIS")
        print("      (workload efficiency cancels: this is pure bandwidth scaling)")
    else:
        frac = None
        print()
        print("  *** achieved/spec is NOT the scaling fraction. ***")
        print("  It still carries the workload's own efficiency (~0.85 for this")
        print("  probe), so pasting it into SLICE_FRAC applies efficiency TWICE:")
        print("    BW = A100_BW * (eff * true_frac), then costs divide by EFF again.")
        print("  Re-run with --full-achieved <bytes/s> from full mode so the")
        print("  efficiency cancels:  frac = achieved_slice / achieved_full.")
    print()

    # Compute scaling is a SEPARATE law from bandwidth scaling -- see section 11.
    print("Measuring compute scaling on this slice...")
    gflops = calculate_slice_compute()
    print(f"  achieved fp16 GEMM = {gflops / 1e12:.2f} TFLOP/s")
    try:
        dec_ms = calculate_slice_decode_layer_ms()
        print(f"  decode ms/layer    = {dec_ms:.3f} ms   (compare [B05])")
    except Exception as e:
        dec_ms = float("nan")
        print(f"  decode ms/layer    = n/a ({type(e).__name__})")
    print()

    print("=" * 62)
    print(f"Assumed fraction in the model : {SLICE_FRAC_ASSUMED:.4f}")
    if frac is not None:
        measured = frac
        print(f"Measured fraction             : {measured:.4f}")
    else:
        measured = frac_vs_spec
        print(f"achieved/spec (NOT the frac)  : {measured:.4f}  "
              f"<-- do not paste; see warning above")
    delta = (measured - SLICE_FRAC_ASSUMED) / SLICE_FRAC_ASSUMED * 100
    print(f"Difference                    : {delta:+.1f}%")
    print()
    # The two candidate scaling laws this distinguishes.
    print("Candidate laws for reference:")
    print(f"  GPC count      3/7 = {3 / 7:.4f}   (compute-slice proportional)")
    print(f"  HBM stacks     1/2 = {0.5:.4f}   (2 of 4 stacks on a 40GB A100)")
    print()
    print("Paste the measured fraction into SLICE_FRAC in pulp_layer_placement.py")
    print("for this slice geometry.")
    print()
    print("ALSO paste the compute numbers -- the model currently derives per-slice")
    print("compute from the BANDWIDTH fraction, which the 9/28 sweep contradicts")
    print("(rank0 1.57 ms/layer vs rank3 4.98 ms/layer). Two separate fractions:")
    if frac is not None:
        print(f"  SLICE_BW_FRAC   [{SLICE_LABEL}] = {measured:.4f}")
    else:
        print(f"  SLICE_BW_FRAC   [{SLICE_LABEL}] = <re-run with --full-achieved>")
    print(f"  SLICE_TFLOPS    [{SLICE_LABEL}] = {gflops / 1e12:.3f}")
    print(f"  SLICE_DECODE_MS [{SLICE_LABEL}] = {dec_ms:.3f}  # per layer")
    print("=" * 62)

    _emit_json({
        "mode": "slice",
        "slice_label": SLICE_LABEL,
        "model": MODEL_NAME,
        "card": _CARD,
        "card_peak_bw": PEAK_BW,
        "achieved_bw": achieved,
        "slice_bw_frac_vs_spec": frac_vs_spec,
        "slice_bw_frac_vs_full": frac,
        "slice_bw_frac_assumed": SLICE_FRAC_ASSUMED,
        "slice_tflops": gflops / 1e12,
        "slice_decode_ms_per_layer": dec_ms,
    })


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_full_mode():
    torch.cuda.set_device(0)
    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"Normalizing by FULL-GPU peak: {PEAK_BW / 1e9:.1f} GB/s")
    print("All efficiencies below are fractions of the WHOLE GPU's bandwidth,")
    print("so they are hardware properties independent of MIG partitioning.")
    print()

    print("[1/14] EFF_WEIGHT (large contiguous weight reads)...")
    eff_weight = calculate_EFF_WEIGHT()
    achieved_full = eff_weight * PEAK_BW
    print(f"      EFF_WEIGHT = {eff_weight:.3f}  ({eff_weight * 100:.1f}% of peak)")
    print(f"      achieved   = {achieved_full / 1e9:.1f} GB/s")
    print(f"      -> pass --full-achieved {achieved_full:.6e} to slice mode")
    print()

    print("[2/14] LAUNCH_US (per-kernel launch + dispatch overhead)...")
    launch_us = calculate_LAUNCH_US()
    print(f"      LAUNCH_US  = {launch_us:.1f} us/kernel")
    print()

    print("[3/14] KERNELS_PER_LAYER (profiler kernel count, one decode step)...")
    kernels = calculate_KERNELS_PER_LAYER()
    print(f"      KERNELS_PER_LAYER = {kernels:.0f}")
    print()

    launch_ms_per_layer = kernels * launch_us / 1000.0

    print("[4/14] EFF_ACT sweep (single-token decode, varying context)...")
    curve = calculate_EFF_ACT_curve(eff_weight, launch_ms_per_layer)
    print(f"      {'ctx':>6} {'EFF_ACT':>9} {'measured_ms':>12} {'act_only_ms':>12}")
    for ctx, eff, meas, act in curve:
        print(f"      {ctx:6d} {eff:9.3f} {meas * 1e3:12.3f} {act * 1e3:12.3f}")
    eff_act_w = weighted_EFF_ACT(curve)
    print(f"      traffic-weighted scalar EFF_ACT = {eff_act_w:.3f}")
    print()

    print("[5/14] PCIE_BW_PINNED (pinned GPU->host DMA)...")
    pcie_pinned = calculate_PCIE_BW_PINNED()
    print(f"      PCIE_BW_PINNED   = {pcie_pinned / 1e9:.1f} GB/s")
    print()

    print("[6/14] receive leg -- CURRENT path vs the model's stale constant...")
    recv_h2d = calculate_RECV_H2D_BW()
    pcie_unpinned = calculate_PCIE_BW_UNPINNED()
    shm2pin = calculate_SHM_TO_PINNED_BW()
    print(f"      RECV_H2D_BW      = {recv_h2d / 1e9:.1f} GB/s  <-- pinned, "
          f"what the code does now")
    print(f"      PCIE_BW_UNPINNED = {pcie_unpinned / 1e9:.1f} GB/s  (STALE: "
          f"pageable + extra GPU copy)")
    print(f"      SHM_TO_PINNED_BW = {shm2pin / 1e9:.1f} GB/s  <-- host leg, "
          f"SHM -> pinned directly")
    if pcie_unpinned > 0:
        print(f"      ratio            = {recv_h2d / pcie_unpinned:.2f}x -- the "
              f"model charges the receive leg at the stale rate")
    print()

    print("[7/14] MEMCPY_BW (pinned->SHM write, SHM->heap read)...")
    memcpy_bw, bw_w, bw_r = calculate_MEMCPY_BW()
    print(f"      write pinned->SHM = {bw_w / 1e9:.1f} GB/s")
    print(f"      read  SHM->heap   = {bw_r / 1e9:.1f} GB/s")
    print(f"      MEMCPY_BW (harmonic mean) = {memcpy_bw / 1e9:.1f} GB/s")
    print()

    print("[8/14] HANDOFF_FIXED_US (cuda.sync drain + handshake residual)...")
    handoff_us = calculate_HANDOFF_FIXED_US(pcie_pinned, memcpy_bw)
    tok_local = calculate_TOK_ROUNDTRIP_US_local()
    print(f"      HANDOFF_FIXED_US = {handoff_us:.1f} us")
    print(f"      TOK_ROUNDTRIP_US (local lower bound) = {tok_local:.1f} us")
    print("      NOTE: run measure_tok_roundtrip_distributed() on the cluster")
    print("            for the true 2-rank gloo round-trip.")
    print()

    print("[9/14] EFF_PREFILL (prefill regime -- NEVER measured before)...")
    pre = calculate_EFF_PREFILL()
    print(f"      {'mb':>4} {'ms':>9} {'TFLOP/s':>9} {'bw_eff':>8}")
    for mb, ms, fl, bweff in pre:
        if ms != ms:
            print(f"      {mb:4d} {'OOM':>9}")
            continue
        print(f"      {mb:4d} {ms * 1e3:9.3f} {fl / 1e12:9.2f} {bweff:8.3f}")
    print()

    print("[10/14] HANDSHAKE+ACK round-trip (2-rank gloo, idle peer)...")
    try:
        hs_ack_us = measure_handshake_ack_us()
        print(f"      HANDSHAKE_ACK_US = {hs_ack_us:.1f} us")
    except Exception as e:
        hs_ack_us = float("nan")
        print(f"      failed: {type(e).__name__}: {e}")
    print()

    print("[11/14] TOK_ROUNDTRIP_US (true 2-rank gloo)...")
    try:
        tok_dist = measure_tok_roundtrip_distributed()
        print(f"      TOK_ROUNDTRIP_US = {tok_dist:.1f} us")
    except Exception as e:
        tok_dist = float("nan")
        print(f"      failed: {type(e).__name__}: {e}")
    print()

    print("[12/14] D2H overlap (how much of the copy actually hides)...")
    ov = measure_d2h_overlap()
    print(f"      copy alone   = {ov['copy_alone_us']:.1f} us")
    print(f"      compute alone= {ov['compute_alone_us']:.1f} us")
    print(f"      overlapped   = {ov['overlapped_us']:.1f} us "
          f"(serial would be {ov['serial_us']:.1f})")
    print(f"      D2H_HIDDEN_FRACTION = {ov['hidden_fraction']:.3f}")
    print()

    print("[13/14] KV-cache growth cost per step (DynamicCache concat)...")
    kvg = measure_kv_growth_us()
    for ctx, us in kvg:
        print(f"      ctx={ctx:4d}  {us:8.2f} us/layer/step")
    print()

    print("[14/14] Embed / lm_head build peak (open item 6)...")
    hd = measure_head_cost()
    print(f"      build peak = {hd['build_peak_mib']:.0f} MiB "
          f"({hd['peak_over_fp16']:.2f}x the fp16 size)")
    print(f"      resident   = {hd['resident_mib']:.0f} MiB "
          f"(fp16 theory {hd['fp16_theory_mib']:.0f})")
    print(f"      lm_head    = {hd['lm_head_us']:.1f} us/step")
    print()

    print("=" * 62)
    print(f"MODEL = {MODEL_NAME}  ({MODEL_TYPE}, {LAYERS} layers, "
          f"hidden {HIDDEN}, kv_heads {N_KV_HEADS}/{N_HEADS})")
    print(f"CARD  = {_CARD}  peak {PEAK_BW / 1e9:.0f} GB/s")
    print("Paste these into pulp_layer_placement.py:")
    print(f"  EFF_WEIGHT        = {eff_weight:.3f}")
    print(
        f"  EFF_ACT           = {eff_act_w:.3f}   # traffic-weighted; see curve above"
    )
    print(f"  LAUNCH_US         = {launch_us:.1f}")
    print(f"  KERNELS_PER_LAYER = {kernels:.0f}")
    print(f"  PCIE_BW_PINNED    = {pcie_pinned:.3e}")
    print(f"  PCIE_BW_UNPINNED  = {pcie_unpinned:.3e}   # STALE, see below")
    print(f"  RECV_H2D_BW       = {recv_h2d:.3e}   # NEW: replaces "
          f"PCIE_BW_UNPINNED")
    print(f"  SHM_TO_PINNED_BW  = {shm2pin:.3e}   # NEW: receiver host leg")
    print(f"  MEMCPY_BW         = {memcpy_bw:.3e}")
    print(f"  HANDOFF_FIXED_US  = {handoff_us:.1f}")
    print(f"  TOK_ROUNDTRIP_US  = {tok_dist:.1f}")
    print(f"  HANDSHAKE_ACK_US  = {hs_ack_us:.1f}   # NEW: post-ACK-fix protocol")
    print(f"  D2H_HIDDEN_FRAC   = {ov['hidden_fraction']:.3f}   # NEW: charge "
          f"(1-frac) of the copy, not all of it")
    print(f"  KV_GROWTH_US      = {kvg[-1][1]:.2f}   # NEW: per layer per step "
          f"at ctx={kvg[-1][0]}")
    print(f"  HEAD_BUILD_PEAK_MIB = {hd['build_peak_mib']:.0f}   # NEW: fp32 "
          f"build peak, {hd['peak_over_fp16']:.1f}x fp16 -- the real mem cap")
    print(f"  WEIGHT_BYTES_PER_LAYER = {weight_bytes_per_layer()}   # GQA-aware")
    print()
    print("Prefill is NOT a single constant -- see the [9/14] table. The model")
    print("has no prefill term at all today; at 3-7% of latency it is worth one.")
    print()
    print("TRANSPORT PATH: the model's PCIE_BW_UNPINNED/MEMCPY_BW describe the OLD")
    print("receiver (pageable H2D + a second GPU copy, SHM->heap->pinned). The")
    print("current _queue_read_from_slot does SHM -> pinned staging -> one pinned")
    print("H2D on recv_stream. Use RECV_H2D_BW and SHM_TO_PINNED_BW instead.")
    print()

    _emit_json({
        "mode": "full",
        "model": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "card": _CARD,
        "card_peak_bw": PEAK_BW,
        "arch": {"hidden": HIDDEN, "inter": INTER, "vocab": VOCAB,
                 "layers": LAYERS, "n_heads": N_HEADS,
                 "n_kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM},
        "EFF_WEIGHT": eff_weight,
        "achieved_full_bw": achieved_full,
        "EFF_ACT_weighted": eff_act_w,
        "EFF_ACT_curve": [{"ctx": c, "eff": e} for c, e, _, _ in curve],
        "LAUNCH_US": launch_us,
        "KERNELS_PER_LAYER": kernels,
        "PCIE_BW_PINNED": pcie_pinned,
        "PCIE_BW_UNPINNED_stale": pcie_unpinned,
        "RECV_H2D_BW": recv_h2d,
        "SHM_TO_PINNED_BW": shm2pin,
        "MEMCPY_BW": memcpy_bw,
        "HANDOFF_FIXED_US": handoff_us,
        "TOK_ROUNDTRIP_US": tok_dist,
        "HANDSHAKE_ACK_US": hs_ack_us,
        "D2H_HIDDEN_FRACTION": ov["hidden_fraction"],
        "d2h_overlap": ov,
        "KV_GROWTH_US": [{"ctx": c, "us": u} for c, u in kvg],
        "head_cost": hd,
        "prefill": [{"mb": m, "ms": ms * 1e3 if ms == ms else None,
                     "tflops": f / 1e12 if f == f else None,
                     "bw_eff": b} for m, ms, f, b in pre],
        "WEIGHT_BYTES_PER_LAYER": weight_bytes_per_layer(),
        "KV_BYTES_PER_TOKEN_PER_LAYER": kv_bytes_per_token_per_layer(),
    })
    print("SLICE_FRAC is NOT set from this run -- it is a property of the MIG")
    print("partition, not of the hardware. Run slice mode once per geometry:")
    print(f"  CUDA_VISIBLE_DEVICES=<uuid> python3 measure_constants.py \\")
    print(f"      --mode slice --full-achieved {achieved_full:.6e}")
    print("=" * 62)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Measure the empirical constants for pulp_layer_placement.py"
    )
    ap.add_argument(
        "--mode",
        choices=["full", "slice"],
        default="full",
        help="full: hardware properties on the whole GPU (default). "
        "slice: calibrate one MIG slice's bandwidth fraction.",
    )
    ap.add_argument(
        "--full-achieved",
        type=float,
        default=None,
        help="slice mode: achieved bandwidth (bytes/s) from the "
        "full-mode run, so workload efficiency cancels.",
    )
    ap.add_argument("--model", default=None,
                    help="HF model id or local path (default: MODEL_NAME). "
                         "Architecture is read from its config.json, so "
                         "mistral/qwen2/13B/24B all work.")
    ap.add_argument("--card", choices=sorted(CARD_BW), default=None,
                    help="Card for peak-bandwidth normalization. Default: "
                         "detected from the device name. Getting this wrong "
                         "scales EVERY efficiency (40GB vs 80GB A100 = 1.31x).")
    ap.add_argument("--json", default=None,
                    help="Write measured constants to this JSON path so the "
                         "model can load them instead of hand-pasting.")
    ap.add_argument("--slice-label", default=None,
                    help="slice mode: label for this geometry, e.g. 1g.10gb.")
    ap.add_argument("--slice-frac-assumed", type=float, default=None,
                    help="slice mode: the fraction the model assumes today.")
    args = ap.parse_args()

    global A100_BW, PEAK_BW, _CARD, _JSON_PATH, SLICE_LABEL, SLICE_FRAC_ASSUMED
    _JSON_PATH = args.json
    if args.slice_label:
        SLICE_LABEL = args.slice_label
    if args.slice_frac_assumed is not None:
        SLICE_FRAC_ASSUMED = args.slice_frac_assumed

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. Every probe here times GPU work, so there "
            "is nothing to measure on a CPU-only host.\n"
            "  full mode : run on the box with the whole GPU visible\n"
            "  slice mode: CUDA_VISIBLE_DEVICES=<MIG-uuid> ..."
        )

    _CARD = args.card or detect_card()
    A100_BW = CARD_BW[_CARD]
    PEAK_BW = A100_BW
    print(f"[card] {_CARD}  peak {PEAK_BW / 1e9:.0f} GB/s"
          + ("" if args.card else "  (auto-detected; override with --card)"))

    load_arch(args.model)
    print(f"[model] {MODEL_NAME}  type={MODEL_TYPE} layers={LAYERS} "
          f"hidden={HIDDEN} inter={INTER} vocab={VOCAB} "
          f"heads={N_HEADS} kv_heads={N_KV_HEADS} head_dim={HEAD_DIM}")
    if N_KV_HEADS != N_HEADS:
        print(f"[model] GQA detected: KV traffic is "
              f"{N_HEADS / N_KV_HEADS:.0f}x smaller than the MHA formula assumes")
    print()

    if args.mode == "full":
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ[
            "CUDA_VISIBLE_DEVICES"
        ].startswith("MIG-"):
            print("WARNING: CUDA_VISIBLE_DEVICES points at a MIG slice, but full")
            print("         mode expects the whole GPU. Efficiencies will be")
            print("         normalized by A100_BW and come out ~slice_frac too low.")
            print()
        run_full_mode()
    else:
        run_slice_mode(full_achieved_bw=args.full_achieved)


if __name__ == "__main__":
    main()
