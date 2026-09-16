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
A100_BW    = 1555e9               # HBM2 bytes/s, full A100 40GB (spec sheet)

# Slice mode only: which slice you are calibrating, and what the model currently
# ASSUMES its bandwidth fraction is. Slice mode measures the real fraction and
# tells you whether this assumption holds.
MIG_UUID           = "MIG-98f93df6-d522-5c00-9923-4326839cef2e"  # 3g.20gb slice
SLICE_LABEL        = "3g.20gb"
SLICE_FRAC_ASSUMED = 3 / 7        # what pulp_layer_placement.py assumes today

MODEL_NAME = "lmsys/vicuna-7b-v1.5"   # must match your pipeline

# workload shape -- must match benchmark_pipeline_microbatching.py
SEQ_LEN        = 64
MAX_NEW_TOKENS = 512
MAX_SEQ        = SEQ_LEN + MAX_NEW_TOKENS

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
from transformers import LlamaConfig, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

HIDDEN, INTER, VOCAB = 4096, 11008, 32000
WARMUP  = 20
REPEATS = 200

# Normalization basis for every efficiency constant. Full mode runs on the whole
# GPU, so efficiencies are fractions of the FULL device's peak bandwidth. That is
# what makes them hardware properties rather than slice-specific numbers, and it
# is what lets the model predict a MIG geometry you have not run.
PEAK_BW = A100_BW


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def act_bytes_per_token(ctx: int) -> int:
    """Activation + KV traffic per token per layer (bytes), flash-attn aware.

    Must stay identical to act_bytes_per_token in pulp_layer_placement.py.
    """
    B = 2  # FP16
    qkv    = (HIDDEN + 3*HIDDEN) * B
    flash  = (HIDDEN + 2*ctx*HIDDEN + HIDDEN) * B   # K,V full ctx read
    oproj  = (HIDDEN + HIDDEN) * B
    gate   = (HIDDEN + INTER) * B
    up     = (HIDDEN + INTER) * B
    swiglu = (2*INTER + INTER) * B
    down   = (INTER + HIDDEN) * B
    misc   = (6*HIDDEN) * B
    return qkv + flash + oproj + gate + up + swiglu + down + misc


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _build_layer(device):
    config = LlamaConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "sdpa"   # flash attention, same as pipeline
    layer = LlamaDecoderLayer(config, layer_idx=0).half().to(device)
    layer.eval()
    rotary = LlamaRotaryEmbedding(config=config, device=device)
    return layer, rotary, config


# ---------------------------------------------------------------------------
# 1. EFF_WEIGHT
#    How: allocate a tensor the size of one layer's weights (~400MB).
#         Time repeated .sum() calls which force the whole tensor through HBM.
#         achieved_bw = bytes_moved / time -> EFF_WEIGHT = achieved / peak.
# ---------------------------------------------------------------------------
def calculate_EFF_WEIGHT() -> float:
    device = torch.device("cuda:0")
    WEIGHT_BYTES = (4*HIDDEN*HIDDEN + 3*HIDDEN*INTER) * 2   # one layer, FP16

    n_elements = WEIGHT_BYTES // 2   # float16 = 2 bytes
    tensor = torch.randn(n_elements, dtype=torch.float16, device=device)

    for _ in range(WARMUP):
        _ = tensor.sum()
    torch.cuda.synchronize()

    t0 = sync_time()
    for _ in range(REPEATS):
        _ = tensor.sum()
    t1 = sync_time()

    elapsed_s   = (t1 - t0) / REPEATS
    achieved_bw = WEIGHT_BYTES / elapsed_s          # bytes/s
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
    WEIGHT_BYTES = (4*HIDDEN*HIDDEN + 3*HIDDEN*INTER) * 2

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
        pos    = torch.tensor([[ctx]], dtype=torch.long, device=device)
        emb    = rotary(hidden, pos)

        with torch.no_grad():
            for _ in range(WARMUP):
                _ = layer(hidden, position_embeddings=emb, past_key_values=cache)
        torch.cuda.synchronize()

        # NOTE: the cache grows by one token per call, so keep the repeat count
        # small relative to ctx or the measured context drifts upward.
        reps = 50
        t0 = sync_time()
        with torch.no_grad():
            for _ in range(reps):
                _ = layer(hidden, position_embeddings=emb, past_key_values=cache)
        t1 = sync_time()

        measured_s = (t1 - t0) / reps
        act_only_s = measured_s - weight_s - launch_s
        ideal_act_s = act_bytes_per_token(ctx) / PEAK_BW

        eff = ideal_act_s / act_only_s if act_only_s > 0 else float("nan")
        results.append((ctx, eff, measured_s, act_only_s))

    return results


def weighted_EFF_ACT(curve) -> float:
    """Traffic-weighted average over the real decode trajectory.

    The model runs ctx = SEQ_LEN .. SEQ_LEN+MAX_NEW_TOKENS. Weight each sampled
    efficiency by how many bytes actually move at that context, so the scalar
    is representative of where the time is really spent.
    """
    import bisect
    ctxs  = [c for c, _, _, _ in curve]
    effs  = [e for _, e, _, _ in curve]

    num = 0.0
    den = 0.0
    for t in range(MAX_NEW_TOKENS):
        ctx = SEQ_LEN + t
        i = min(bisect.bisect_left(ctxs, ctx), len(ctxs) - 1)
        w = act_bytes_per_token(ctx)
        num += w * effs[i]
        den += w
    return num / den


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

    return (t1 - t0) / (REPEATS * 10) * 1e6   # -> microseconds


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
    pos    = torch.tensor([[SEQ_LEN]], dtype=torch.long, device=device)
    emb    = rotary(hidden, pos)

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
    return mb * tokens * HIDDEN * 2   # FP16 hidden state


def calculate_PCIE_BW_PINNED() -> float:
    device = torch.device("cuda:0")
    nbytes = _transport_payload_bytes()
    numel  = nbytes // 2

    gpu_t   = torch.randn(numel, dtype=torch.float16, device=device)
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
    """Pageable host->device, exactly as line 387 does it (heap numpy -> GPU)."""
    device = torch.device("cuda:0")
    nbytes = _transport_payload_bytes()
    numel  = nbytes // 2

    heap = torch.from_numpy(np.zeros(numel, dtype=np.float16))   # NOT pinned
    dst  = torch.zeros(numel, dtype=torch.float16, device=device)

    for _ in range(WARMUP):
        dst.copy_(heap.to(device))
    torch.cuda.synchronize()

    t0 = sync_time()
    for _ in range(REPEATS):
        dst.copy_(heap.to(device))
    t1 = sync_time()

    return nbytes / ((t1 - t0) / REPEATS)


def calculate_MEMCPY_BW() -> float:
    """Host-to-host: the pinned->SHM write and the SHM->heap .copy()."""
    from multiprocessing.shared_memory import SharedMemory

    nbytes = _transport_payload_bytes()
    numel  = nbytes // 2

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
    from multiprocessing.shared_memory import SharedMemory

    device = torch.device("cuda:0")
    nbytes = _transport_payload_bytes()
    numel  = nbytes // 2

    gpu_t   = torch.randn(numel, dtype=torch.float16, device=device)
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


def measure_tok_roundtrip_distributed(batch_size=32, port=29800):
    """True 2-rank measurement of the blocking next_tokens round-trip.

    Spawns two gloo processes on localhost and times send->recv->send->recv.
    Run this on the real cluster for a representative number.
    """
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def worker(rank, q):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=2)
        tok = torch.zeros((batch_size, 1), dtype=torch.long)

        for _ in range(WARMUP):
            if rank == 1:
                dist.send(tok, dst=0, tag=1)
            else:
                dist.recv(tok, src=1, tag=1)
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(REPEATS):
            if rank == 1:
                dist.send(tok, dst=0, tag=2)
            else:
                dist.recv(tok, src=1, tag=2)
        t1 = time.perf_counter()

        if rank == 0:
            q.put((t1 - t0) / REPEATS * 1e6)
        dist.destroy_process_group()

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(r, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    return q.get() if not q.empty() else float("nan")


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
    WEIGHT_BYTES = (4*HIDDEN*HIDDEN + 3*HIDDEN*INTER) * 2

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
    print(f"Full-GPU spec bandwidth: {A100_BW/1e9:.1f} GB/s")
    print()

    print("Measuring achieved bandwidth on this slice "
          "(same probe as full mode)...")
    achieved = calculate_slice_bandwidth()
    print(f"  achieved = {achieved/1e9:.1f} GB/s")
    print()

    frac_vs_spec = achieved / A100_BW
    print(f"  achieved / A100_BW(spec)  = {frac_vs_spec:.4f}")

    if full_achieved_bw:
        frac = achieved / full_achieved_bw
        print(f"  achieved / achieved_full  = {frac:.4f}   <-- USE THIS")
        print("      (workload efficiency cancels: this is pure bandwidth scaling)")
    else:
        frac = None
        print("  NOTE: pass --full-achieved <bytes/s> from the full-mode run to")
        print("        cancel workload efficiency and get the clean fraction.")
    print()

    print("=" * 62)
    print(f"Assumed fraction in the model : {SLICE_FRAC_ASSUMED:.4f}")
    measured = frac if frac is not None else frac_vs_spec
    print(f"Measured fraction             : {measured:.4f}")
    delta = (measured - SLICE_FRAC_ASSUMED) / SLICE_FRAC_ASSUMED * 100
    print(f"Difference                    : {delta:+.1f}%")
    print()
    # The two candidate scaling laws this distinguishes.
    print("Candidate laws for reference:")
    print(f"  GPC count      3/7 = {3/7:.4f}   (compute-slice proportional)")
    print(f"  HBM stacks     1/2 = {0.5:.4f}   (2 of 4 stacks on a 40GB A100)")
    print()
    print("Paste the measured fraction into SLICE_FRAC in pulp_layer_placement.py")
    print("for this slice geometry.")
    print("=" * 62)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_full_mode():
    torch.cuda.set_device(0)
    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"Normalizing by FULL-GPU peak: {PEAK_BW/1e9:.1f} GB/s")
    print("All efficiencies below are fractions of the WHOLE GPU's bandwidth,")
    print("so they are hardware properties independent of MIG partitioning.")
    print()

    print("[1/8] EFF_WEIGHT (large contiguous weight reads)...")
    eff_weight = calculate_EFF_WEIGHT()
    achieved_full = eff_weight * PEAK_BW
    print(f"      EFF_WEIGHT = {eff_weight:.3f}  ({eff_weight*100:.1f}% of peak)")
    print(f"      achieved   = {achieved_full/1e9:.1f} GB/s")
    print(f"      -> pass --full-achieved {achieved_full:.6e} to slice mode")
    print()

    print("[2/8] LAUNCH_US (per-kernel launch + dispatch overhead)...")
    launch_us = calculate_LAUNCH_US()
    print(f"      LAUNCH_US  = {launch_us:.1f} us/kernel")
    print()

    print("[3/8] KERNELS_PER_LAYER (profiler kernel count, one decode step)...")
    kernels = calculate_KERNELS_PER_LAYER()
    print(f"      KERNELS_PER_LAYER = {kernels:.0f}")
    print()

    launch_ms_per_layer = kernels * launch_us / 1000.0

    print("[4/8] EFF_ACT sweep (single-token decode, varying context)...")
    curve = calculate_EFF_ACT_curve(eff_weight, launch_ms_per_layer)
    print(f"      {'ctx':>6} {'EFF_ACT':>9} {'measured_ms':>12} {'act_only_ms':>12}")
    for ctx, eff, meas, act in curve:
        print(f"      {ctx:6d} {eff:9.3f} {meas*1e3:12.3f} {act*1e3:12.3f}")
    eff_act_w = weighted_EFF_ACT(curve)
    print(f"      traffic-weighted scalar EFF_ACT = {eff_act_w:.3f}")
    print()

    print("[5/8] PCIE_BW_PINNED (pinned GPU->host DMA)...")
    pcie_pinned = calculate_PCIE_BW_PINNED()
    print(f"      PCIE_BW_PINNED   = {pcie_pinned/1e9:.1f} GB/s")
    print()

    print("[6/8] PCIE_BW_UNPINNED (pageable host->GPU, bounce-buffered)...")
    pcie_unpinned = calculate_PCIE_BW_UNPINNED()
    print(f"      PCIE_BW_UNPINNED = {pcie_unpinned/1e9:.1f} GB/s")
    print()

    print("[7/8] MEMCPY_BW (pinned->SHM write, SHM->heap read)...")
    memcpy_bw, bw_w, bw_r = calculate_MEMCPY_BW()
    print(f"      write pinned->SHM = {bw_w/1e9:.1f} GB/s")
    print(f"      read  SHM->heap   = {bw_r/1e9:.1f} GB/s")
    print(f"      MEMCPY_BW (harmonic mean) = {memcpy_bw/1e9:.1f} GB/s")
    print()

    print("[8/8] HANDOFF_FIXED_US (cuda.sync drain + handshake residual)...")
    handoff_us = calculate_HANDOFF_FIXED_US(pcie_pinned, memcpy_bw)
    tok_local  = calculate_TOK_ROUNDTRIP_US_local()
    print(f"      HANDOFF_FIXED_US = {handoff_us:.1f} us")
    print(f"      TOK_ROUNDTRIP_US (local lower bound) = {tok_local:.1f} us")
    print("      NOTE: run measure_tok_roundtrip_distributed() on the cluster")
    print("            for the true 2-rank gloo round-trip.")
    print()

    print("=" * 62)
    print("Paste these into pulp_layer_placement.py:")
    print(f"  EFF_WEIGHT        = {eff_weight:.3f}")
    print(f"  EFF_ACT           = {eff_act_w:.3f}   # traffic-weighted; see curve above")
    print(f"  LAUNCH_US         = {launch_us:.1f}")
    print(f"  KERNELS_PER_LAYER = {kernels:.0f}")
    print(f"  PCIE_BW_PINNED    = {pcie_pinned:.3e}")
    print(f"  PCIE_BW_UNPINNED  = {pcie_unpinned:.3e}")
    print(f"  MEMCPY_BW         = {memcpy_bw:.3e}")
    print(f"  HANDOFF_FIXED_US  = {handoff_us:.1f}")
    print(f"  TOK_ROUNDTRIP_US  = <run distributed measurement>")
    print()
    print("SLICE_FRAC is NOT set from this run -- it is a property of the MIG")
    print("partition, not of the hardware. Run slice mode once per geometry:")
    print(f"  CUDA_VISIBLE_DEVICES=<uuid> python3 measure_constants.py \\")
    print(f"      --mode slice --full-achieved {achieved_full:.6e}")
    print("=" * 62)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Measure the empirical constants for pulp_layer_placement.py")
    ap.add_argument("--mode", choices=["full", "slice"], default="full",
                    help="full: hardware properties on the whole GPU (default). "
                         "slice: calibrate one MIG slice's bandwidth fraction.")
    ap.add_argument("--full-achieved", type=float, default=None,
                    help="slice mode: achieved bandwidth (bytes/s) from the "
                         "full-mode run, so workload efficiency cancels.")
    args = ap.parse_args()

    if args.mode == "full":
        if "CUDA_VISIBLE_DEVICES" in os.environ and \
                os.environ["CUDA_VISIBLE_DEVICES"].startswith("MIG-"):
            print("WARNING: CUDA_VISIBLE_DEVICES points at a MIG slice, but full")
            print("         mode expects the whole GPU. Efficiencies will be")
            print("         normalized by A100_BW and come out ~slice_frac too low.")
            print()
        run_full_mode()
    else:
        run_slice_mode(full_achieved_bw=args.full_achieved)


if __name__ == "__main__":
    main()
