"""
Layer-placement optimizer, v2 -- 4 MIG slices (20/10/5/5), OVERLAPPED pipeline.

Decision:   how many of the LAYERS transformer blocks go on each of 4 slices.
Objective:  minimize generation latency under a bottleneck (min-max) law.
Constraints: per-slice memory (no OOM), contiguity, each layer placed once.

============================================================================
THIS MODEL IS NOT FITTED TO ANY LATENCY DATA.
============================================================================
Every number below comes from one of three places:

  (1) ARCHITECTURE   -- byte counts derived from the model's own dimensions
                        (HIDDEN, INTER, VOCAB, LAYERS). Exact, not estimated.
  (2) HARDWARE SPEC  -- A100 HBM2 peak bandwidth, MIG compute fractions, PCIe
                        generation limits. Published numbers.
  (3) PLACEHOLDERS   -- efficiency factors and fixed overheads that require a
                        microbenchmark to pin down. Each is marked
                        "MEASURE:" with the exact measurement that settles it
                        and the reasoning behind the current guess.

No constant is chosen by minimizing error against mig_benchmark_results_9_28.csv
or any other latency sweep. That matters: a model tuned on a sweep and then
scored on the same sweep reports its own fit, not its predictive power. The
comparison in validate_v2.py is therefore a genuine out-of-sample test of the
THEORY, and its residual is information -- it tells us which placeholder in
group (3) is wrong and by how much.

If a placeholder turns out to be off, fix THAT constant from its own
microbenchmark (measure_constants.py), not by back-solving from the latency CSV.

============================================================================
WHY V2 EXISTS -- THE PIPELINE LAW CHANGED
============================================================================
v1 (pulp_layer_placement.py) minimized TOTAL work -- a pure sum objective --
and justified it with a structural claim about the harness code: the transport
did not overlap, so there was no bottleneck stage to balance, and the makespan
was just the total. That claim was correct for the harness v1 was written
against. It is no longer correct. Three serializing defects have been removed:

  1. isend() called a full torch.cuda.synchronize() inline before returning, so
     the "async" D2H copy blocked the entire device. The copy is now queued on
     a side stream (copy_stream) and waited per-slot on a CUDA event.
  2. The receiver ACKed with a blocking dist.send while the sender posted its
     matching recv only at final drain -- so the receiver could not begin
     computing until the sender had finished every microbatch. The ACK recv is
     now posted before the handshake (irecv) and drained at wait().
  3. TokenExchange called a full-device sync 512x per run, draining copy_stream
     as a side effect. Narrowed to current_stream().

v1's own docstring named the condition for revisiting the objective:

    "Do not reinstate [min-max] unless the transport is changed to actually
     overlap -- i.e. the ACK drain and the inline cuda.synchronize() are
     removed."

Both are removed. So the law changes from

    v1:  T_step = N * SUM_s E_s          (every microbatch pays every stage)
    v2:  T_step = SUM_s E_s + (N-1) * MAX_s E_s      (fill, then bottleneck)

and the objective changes from "minimize total" to "minimize the slowest
stage", because with real overlap the pipeline retires one microbatch per
MAX_s E_s in steady state. This is the classic GPipe law; v2 is simply the
standard theory applied now that the harness satisfies its assumptions.

CONSEQUENCE FOR PLACEMENT: v1's sum objective piles layers onto the fastest
slice (more total throughput, no bottleneck to worry about). v2's min-max
objective EQUALIZES per-stage time -- slow slices get fewer layers, because
each layer costs them more. Equal time, not equal layers.

============================================================================
DERIVATION OF T_step
============================================================================
Let C_s be the time slice s spends on ONE microbatch (its service time), and
N the number of microbatches in a step. Work flows 0 -> 1 -> ... -> n-1.

FILL: microbatch 0 must traverse the whole chain before the pipeline is full.
      Cost = SUM_s E_s.
STEADY: once full, stages run concurrently. Stage s cannot start microbatch
      k+1 until it finishes k, so the pipeline emits one microbatch every
      MAX_s E_s. The remaining N-1 microbatches each cost that.
      Cost = (N-1) * MAX_s E_s.

      T_step = SUM_s E_s + (N-1) * MAX_s E_s + tok_roundtrip

where E_s = (layers on s) * layer_compute
          + endpoint_compute      (embed on stage 0, lm_head on the last)
          + transport_out         (stages 0..n-2)

PREFILL is ONE step, not N: the harness runs a single prefill forward over the
whole prompt before the decode loop starts. It is charged once.

============================================================================
"""

import pulp  # type: ignore

# ===========================================================================
# GROUP (2): HARDWARE SPEC -- published A100 numbers, not measured here.
# ===========================================================================
A100_BW = 1555e9  # HBM2e peak bandwidth, bytes/s, full A100-40GB (spec sheet)

# MIG compute/bandwidth fractions. A100 exposes 7 GPU slices; a Ng.Mgb profile
# gets N/7 of the SMs and, to first order, N/7 of the HBM bandwidth (each slice
# is bound to its own memory channels). Layout here: 3g.20gb + 2g.10gb +
# 1g.5gb + 1g.5gb, which is what the 9/28 sweep ran.
NUM_SLICES = 4
SLICE_FRAC = [3 / 7, 2 / 7, 1 / 7, 1 / 7]
SLICE_NAME = ["20gb", "10gb", "5gb", "5gb"]
SLICE_BYTES = [20e9, 10e9, 5e9, 5e9]
BW = [A100_BW * f for f in SLICE_FRAC]

# PCIe gen4 x16 = 31.5 GB/s theoretical per direction (16 GT/s * 16 lanes,
# 128b/130b encoding). Used as the ceiling for the host<->device hops below.
PCIE_GEN4_X16_BW = 31.5e9

# ===========================================================================
# GROUP (3): PLACEHOLDERS -- each needs one microbenchmark. Guesses justified
# from first principles + published profiling, NOT fitted to latency data.
# ===========================================================================

# EFF_WEIGHT: fraction of peak HBM bandwidth achieved when streaming weights.
#
# MEASURE: measure_constants.py calculate_EFF_WEIGHT -- time a large
# contiguous read (>=32 MB, the size of q_proj) on one MIG slice and divide by
# BW[s].
#
# WHY 0.85: weight streaming is the regime HBM is designed for. One large
# contiguous read per projection matrix, fully coalesced, no reuse, no
# dependency stalls. A STREAM-style read is the standard way to approach peak.
# Published A100 transformer-inference profiling lands 80-90% here.
EFF_WEIGHT = 0.85

# EFF_ACT: fraction of peak achieved on per-token activation + KV traffic.
#
# MEASURE: measure_constants.py calculate_EFF_ACT_curve -- it SWEEPS context
# length, because this regime is not obviously a constant (see below).
#
# WHY 0.25: decode activation traffic is dominated by the KV-cache read, which
# is scattered -- stored per layer, read per head with a stride, and grown by
# concatenation in DynamicCache, so it is not a single contiguous stream even
# at long context. That is the latency-bound regime, where published decode
# profiling puts achieved bandwidth at 0.20-0.30 of peak. 0.25 is the midpoint.
#
# KNOWN RISK: the KV share of total traffic grows with context (79% at ctx=64,
# 97% at ctx=576 by the byte counts below), so the right value may drift along
# the decode trajectory. Held flat deliberately -- the sweep settles whether a
# scalar suffices. DO NOT hand-tune this to close a latency gap.
EFF_ACT = 0.25

# LAUNCH_US: per-kernel launch + framework dispatch, microseconds.
#
# MEASURE: measure_constants.py calculate_LAUNCH_US -- time a long sequence of
# trivial kernels and divide.
#
# WHY 12.0: a raw CUDA launch on one stream is ~3-10 us. PyTorch eager adds
# Python dispatch, arg marshalling and bookkeeping, which op-level benchmarks
# put at ~5-15 us total. The harness runs eager (no CUDA graphs, no
# torch.compile -- see forward_through_layers), so the full eager path applies.
LAUNCH_US = 12.0

# KERNELS_PER_LAYER: CUDA kernels in one LlamaDecoderLayer forward.
#
# This one is GROUP (1)-ish -- counted from the ops, not guessed:
#   2 RMSNorm (reduce + scale)  -> 4     q/k/v/o projection GEMMs    -> 4
#   RoPE apply to q and k       -> 2     SDPA (one fused flash)      -> 1
#   2 residual adds             -> 2     gate/up/down MLP GEMMs      -> 3
#   SiLU + elementwise mul      -> 2     KV cache concat/update      -> 2
#                                                              total = 20
# Some fuse in practice (RMSNorm often one kernel, SiLU+mul often fuses), so
# the achieved count is ~14-16. 16 is the conservative end.
#
# NOTE: only the PRODUCT KERNELS_PER_LAYER * LAUNCH_US enters the model, so a
# measurement that moves one without the other is what actually shifts things.
KERNELS_PER_LAYER = 16

# ---- transport: the cross-slice activation handoff ----
# Path (one hop): sender GPU -> pinned host -> SHM -> heap -> receiver GPU.
#
# The two PCIe crossings are NOT symmetric. The send side copies out of a
# pre-allocated pinned staging buffer, so the DMA engine goes straight over the
# bus. The receive side copies from an ordinary heap buffer, which the driver
# must stage through an internal pinned bounce buffer -- an extra host-to-host
# copy hidden inside the transfer. Separate constants.
#
# MEASURE: measure_constants.py sections 5-8.

# WHY 0.80 of gen4 peak: pinned host<->device DMA typically achieves 75-85% of
# theoretical on a healthy gen4 x16 link.
# RISK: assumes gen4. On gen3 x16 the ceiling halves. Confirm with
# `nvidia-smi -q | grep -i pcie` before trusting any transport cost.
PCIE_BW_PINNED = 0.80 * PCIE_GEN4_X16_BW  # ~25.2 GB/s

# WHY half of pinned: the bounce-buffer staging adds a host-to-host copy on the
# same data, so the observed rate is roughly halved. This is the standard
# pageable-transfer penalty.
PCIE_BW_UNPINNED = 0.5 * PCIE_BW_PINNED  # ~12.6 GB/s

# MEMCPY_BW: host-to-host, for the pinned->SHM write and the SHM->heap read.
# These are numpy slice-assign / .copy(), single-threaded, not a tuned memcpy.
# Server DDR4 single-threaded memcpy lands ~8-15 GB/s; numpy sits low in that.
MEMCPY_BW = 10e9

# HANDOFF_FIXED_US: per-handoff fixed cost -- the CUDA event wait plus the gloo
# handshake dispatch.
#
# WHY 15.0, AND WHY IT DROPPED FROM v1's 30.0: v1 charged a full
# torch.cuda.synchronize() here and noted in its own comment that 30 us was
# therefore a LOWER BOUND, since a device drain blocks until the layer compute
# just issued retires. That sync is gone. What remains is an event wait on one
# slot's copy -- a few us once the copy has overlapped -- plus the gloo
# handshake for a 4-byte tensor (~10-30 us of dispatch and socket work). 15 us
# keeps the handshake and drops the drain.
HANDOFF_FIXED_US = 15.0

# TOK_ROUNDTRIP_US: per-step next_tokens feedback, last rank -> rank 0. A
# (batch,1) int64 tensor over gloo TCP on localhost -- 256 bytes at batch 32,
# so pure latency, no bandwidth. Loopback TCP round-trip is ~20-50 us; gloo
# dispatch and rendezvous add on top. ~60 us is a reasonable midpoint.
TOK_ROUNDTRIP_US = 60.0

# ===========================================================================
# GROUP (1): ARCHITECTURE -- exact, derived from the model's dimensions.
# ===========================================================================
# Vicuna-7B (LLaMA-7B)
HIDDEN, INTER, LAYERS, VOCAB = 4096, 11008, 32, 32000

# One decoder layer, FP16: 4 attention projections [HIDDEN,HIDDEN] + 3 MLP
# matrices [HIDDEN,INTER].
WEIGHT_BYTES_PER_LAYER = (4 * HIDDEN * HIDDEN + 3 * HIDDEN * INTER) * 2

EMBED_BYTES = VOCAB * HIDDEN * 2
LM_HEAD_BYTES = VOCAB * HIDDEN * 2
NORM_BYTES = HIDDEN * 2

# workload
SEQ_LEN, MAX_NEW_TOKENS = 64, 512
MAX_SEQ = SEQ_LEN + MAX_NEW_TOKENS

USABLE_FRAC = 0.9  # slice memory left for weights+KV after allocator overhead

LAUNCH_MS_PER_LAYER = KERNELS_PER_LAYER * LAUNCH_US / 1000.0


def activation_bytes_per_token(ctx):
    """Activation + KV traffic for one token through one layer, bytes.

    Flash attention (SDPA): the [ctx, ctx] score matrix is never materialized
    to HBM, so it is excluded. K and V for the full context ARE read every
    step -- that is the KV cache, and it is what dominates at long context.
    """
    B = 2  # FP16
    qkv = (HIDDEN + 3 * HIDDEN) * B  # read input, write Q,K,V
    flash = (HIDDEN + 2 * ctx * HIDDEN + HIDDEN) * B  # Q + KV(ctx) + out
    oproj = (HIDDEN + HIDDEN) * B
    gate = (HIDDEN + INTER) * B
    up = (HIDDEN + INTER) * B
    swiglu = (2 * INTER + INTER) * B
    down = (INTER + HIDDEN) * B
    misc = (6 * HIDDEN) * B  # ~2 rmsnorm + 2 residual, read+write
    return qkv + flash + oproj + gate + up + swiglu + down + misc


def layer_compute_in_seconds(
    slice_bandwidth_bytes_per_second, microbatch_size, context_length, is_prefill
):
    """One layer, one microbatch, one step -- seconds.

    prefill: all context_length tokens at once for every sequence. Weights and
             activations both stream in large contiguous chunks, so both are
             charged at EFF_WEIGHT.
    decode : one new token per sequence. Weights still stream efficiently; the
             per-token activation + KV traffic is scattered, so it is charged
             at EFF_ACT.
    """
    if is_prefill:
        activation_bytes = (
            activation_bytes_per_token(context_length)
            * microbatch_size
            * context_length
        )
        time_seconds = (WEIGHT_BYTES_PER_LAYER + activation_bytes) / (
            slice_bandwidth_bytes_per_second * EFF_WEIGHT
        )
        return time_seconds + LAUNCH_MS_PER_LAYER / 1000.0

    activation_bytes = activation_bytes_per_token(context_length) * microbatch_size
    time_seconds = WEIGHT_BYTES_PER_LAYER / (
        slice_bandwidth_bytes_per_second * EFF_WEIGHT
    )
    time_seconds += activation_bytes / (slice_bandwidth_bytes_per_second * EFF_ACT)
    time_seconds += LAUNCH_MS_PER_LAYER / 1000.0
    return time_seconds


def endpoint_compute_in_seconds(
    stage_index,
    num_stages,
    slice_bandwidth_bytes_per_second,
    microbatch_size,
    is_prefill,
):
    """Non-transformer work pinned to the first and last stages.

    Stage 0: embedding row gather out of a [VOCAB, HIDDEN] table. Only
    microbatch_size rows are touched, so the traffic is the gathered rows, not
    the whole table -- but the ids are arbitrary, so the gather is scattered
    (EFF_ACT). The .half() then writes a fresh tensor.

    Last stage: norm, then lm_head, then argmax. lm_head is a [HIDDEN, VOCAB]
    matmul against one token per sequence, so it is weight-bound exactly like a
    projection: the whole 262 MB weight streams to produce microbatch_size rows
    of logits. It runs INSIDE the per-microbatch loop, so a step with N
    microbatches streams it N times.

    WHY THIS MATTERS MORE UNDER MIN-MAX THAN IT DID UNDER v1's SUM: these costs
    land on one specific stage. In a sum objective a per-stage constant cannot
    change the argmin. Under min-max it can and does -- a stage carrying
    lm_head is a bottleneck candidate and should be given fewer layers.
    """
    time_seconds = 0.0

    if stage_index == 0:
        rows = microbatch_size * (SEQ_LEN if is_prefill else 1)
        gathered_bytes = rows * HIDDEN * 2
        # scattered gather + the .half() copy: read + write, latency-bound
        time_seconds += 2 * gathered_bytes / (
            slice_bandwidth_bytes_per_second * EFF_ACT
        )
        time_seconds += LAUNCH_US / 1e6

    if stage_index == num_stages - 1:
        norm_bytes = 2 * microbatch_size * HIDDEN * 2  # read + write
        time_seconds += norm_bytes / (slice_bandwidth_bytes_per_second * EFF_ACT)
        logit_bytes = microbatch_size * VOCAB * 2
        time_seconds += (LM_HEAD_BYTES + logit_bytes) / (
            slice_bandwidth_bytes_per_second * EFF_WEIGHT
        )
        time_seconds += logit_bytes / (slice_bandwidth_bytes_per_second * EFF_ACT)
        time_seconds += 3 * LAUNCH_US / 1e6  # norm + lm_head + argmax

    return time_seconds


def transport_time_for_activation_transfer_seconds(mb, ctx, is_prefill, dst_slice=None):
    """One cross-slice handoff of the activation tensor, seconds.

    Payload is (mb, tokens, HIDDEN) FP16, with tokens = ctx in prefill and 1 in
    decode. Four byte-movement terms plus a fixed cost.

    NOTE ON OVERLAP: with the copy now on copy_stream, the sender's D2H no
    longer blocks its own compute. But the RECEIVER still cannot start until
    the data lands, so the handoff remains on the critical path of the chain
    and is charged into E_s. What changed is HANDOFF_FIXED_US (no device
    drain), not the byte counts.
    """
    tokens = ctx if is_prefill else 1
    payload = mb * tokens * HIDDEN * 2  # bytes, FP16

    t = payload / PCIE_BW_PINNED  # sender GPU -> pinned host
    t += payload / MEMCPY_BW  # pinned -> SHM
    t += payload / MEMCPY_BW  # SHM -> heap
    t += payload / PCIE_BW_UNPINNED  # heap -> receiver GPU (pageable)

    if dst_slice is not None:
        # The arrived tensor sits in a temporary buffer; the GPU then moves it
        # into the layer's input slot (read + write on the receiver's HBM).
        #
        # KNOWN SMALL DOUBLE-COUNT: layer_compute also charges reading that
        # same input. ~0.7 us out of ~1370 us of layer time (0.05%) -- left in
        # as negligible. Revisit only if transport grows to dominate.
        t += 2 * payload / (BW[dst_slice] * EFF_WEIGHT)

    t += HANDOFF_FIXED_US / 1e6
    return t


def kv_bytes_per_layer(batch_size):
    """K+V for the full sequence, FP16, one layer."""
    return 2 * batch_size * MAX_SEQ * HIDDEN * 2


def layer_limits(batch_size):
    """Theoretical per-slice layer capacity: weights + KV within usable HBM.

    Embed sits on slice 0 and lm_head+norm on the LAST slice (index 3 here, not
    2 as in v1's 3-slice layout), so the two endpoints consume room on
    different slices than before.
    """
    kv_per_layer = kv_bytes_per_layer(batch_size)
    mem_per_layer = WEIGHT_BYTES_PER_LAYER + kv_per_layer

    usable = [USABLE_FRAC * b for b in SLICE_BYTES]
    usable[0] -= EMBED_BYTES
    usable[NUM_SLICES - 1] -= LM_HEAD_BYTES + NORM_BYTES

    return [max(0, int(usable[s] // mem_per_layer)) for s in range(NUM_SLICES)]


# ---------------------------------------------------------------------------
# LATENCY MODEL
# ---------------------------------------------------------------------------
def _stage_intervals_s(counts, mb, ctx, is_prefill):
    """E_s for every slice at one step: its layers' compute, its endpoint work,
    and its one outbound handoff.

    Transport is charged on stages 0..n-2. The last stage has no outbound
    activation handoff -- it emits a token instead, and that round-trip is
    added once per step in _step_latency_s.
    """
    n = len(counts)
    E = []
    for s in range(n):
        e = counts[s] * layer_compute_in_seconds(BW[s], mb, ctx, is_prefill)
        e += endpoint_compute_in_seconds(s, n, BW[s], mb, is_prefill)
        if s < n - 1:
            e += transport_time_for_activation_transfer_seconds(
                mb, ctx, is_prefill, dst_slice=s + 1
            )
        E.append(e)
    return E


def _step_latency_s(counts, mb, N, ctx, is_prefill):
    """One step under the overlapped (GPipe / min-max) law:

        T_step = SUM_s E_s + (N-1) * MAX_s E_s + tok_roundtrip
                 |-- fill --|  |-- steady state --|

    FILL: microbatch 0 traverses every stage before the pipeline is full.
    STEADY: thereafter the pipeline emits one microbatch per MAX_s E_s, because
            the slowest stage gates throughput; the other N-1 cost that each.

    No fitted correction factor. This is the textbook law with the model's own
    E_s terms substituted in. Whatever it gets wrong is a statement about the
    constants in GROUP (3) above, and should be fixed there by measuring them.

    (v1 used T_step = N * SUM_s E_s, correct only for a pipeline with zero
    overlap -- which the harness no longer is. See the module docstring.)
    """
    E = _stage_intervals_s(counts, mb, ctx, is_prefill)
    tok_rt = 0.0 if is_prefill else TOK_ROUNDTRIP_US / 1e6
    return sum(E) + (N - 1) * max(E) + tok_rt


def predict_latency_ms(counts, num_microbatches, batch_size):
    """End-to-end generation latency for a split + batch config, ms.

    counts = per-slice layer counts, length NUM_SLICES.
    """
    mb = batch_size // num_microbatches
    N = num_microbatches

    # prefill: ONE step over the whole prompt, no token round-trip
    total_s = _step_latency_s(counts, mb, N, SEQ_LEN, is_prefill=True)

    # decode: MAX_NEW_TOKENS serial steps, context grows by one each step
    for t in range(MAX_NEW_TOKENS):
        total_s += _step_latency_s(counts, mb, N, SEQ_LEN + t, is_prefill=False)

    return total_s * 1000.0


# ---------------------------------------------------------------------------
# OPTIMIZER -- min-max (bottleneck) objective over 4 slices.
# ---------------------------------------------------------------------------
def _per_layer_rank_cost_ms(bw, mb=1):
    """One layer's compute summed over prefill + every decode step, ms."""
    total_s = layer_compute_in_seconds(bw, mb, SEQ_LEN, is_prefill=True)
    for t in range(MAX_NEW_TOKENS):
        total_s += layer_compute_in_seconds(bw, mb, SEQ_LEN + t, is_prefill=False)
    return total_s * 1000.0


def _endpoint_rank_cost_ms(stage_index, num_stages, bw, mb=1):
    """Endpoint cost on this stage over the whole run, ms. A per-STAGE
    constant -- independent of layer count, so it enters as a fixed offset."""
    total_s = endpoint_compute_in_seconds(
        stage_index, num_stages, bw, mb, is_prefill=True
    )
    total_s += MAX_NEW_TOKENS * endpoint_compute_in_seconds(
        stage_index, num_stages, bw, mb, is_prefill=False
    )
    return total_s * 1000.0


def _transport_rank_cost_ms(stage_index, num_stages, mb=1):
    """Outbound handoff cost charged to this stage over the whole run, ms.
    Zero on the last stage."""
    if stage_index >= num_stages - 1:
        return 0.0
    total_s = transport_time_for_activation_transfer_seconds(
        mb, SEQ_LEN, True, dst_slice=stage_index + 1
    )
    for t in range(MAX_NEW_TOKENS):
        total_s += transport_time_for_activation_transfer_seconds(
            mb, SEQ_LEN + t, False, dst_slice=stage_index + 1
        )
    return total_s * 1000.0


def optimize_placement(total_layers=LAYERS, batch_size=32, layer_caps=None):
    """Solve for the split that minimizes the pipeline BOTTLENECK.

        minimize  z
        s.t.      z >= cost_s   for every slice s
                  cost_s = per_layer_cost[s] * (layers on s)
                           + endpoint_cost[s] + transport_cost[s]

    max() is not linear, but "z at least as large as every stage, then minimize
    z" is -- and at the optimum z equals the true max. Standard min-max
    linearization.

    WHY MIN-MAX AND NOT v1's SUM: with real overlap the pipeline retires one
    microbatch per MAX_s E_s, so the slowest stage sets throughput and the
    right split equalizes stage TIME (not layer count -- a slow slice should
    hold fewer layers because each costs it more).

    layer_caps: optional hard per-slice caps (e.g. layer_limits.py's measured
    values), intersected with the theoretical layer_limits().
    """
    limits = layer_limits(batch_size)
    if layer_caps is not None:
        limits = [min(limits[s], layer_caps[s]) for s in range(NUM_SLICES)]

    per_layer_cost = [_per_layer_rank_cost_ms(BW[s]) for s in range(NUM_SLICES)]
    endpoint_cost = [
        _endpoint_rank_cost_ms(s, NUM_SLICES, BW[s]) for s in range(NUM_SLICES)
    ]
    transport_cost = [_transport_rank_cost_ms(s, NUM_SLICES) for s in range(NUM_SLICES)]

    prob = pulp.LpProblem("layer_placement_v2", pulp.LpMinimize)

    # x[i][s] = 1 if layer i is on slice s.
    x = [
        [pulp.LpVariable(f"x_{i}_{s}", cat="Binary") for s in range(NUM_SLICES)]
        for i in range(total_layers)
    ]

    # z: the bottleneck stage cost.
    z = pulp.LpVariable("z", lowBound=0, cat="Continuous")

    # CONSTRAINT: each layer on exactly one slice.
    for i in range(total_layers):
        prob += pulp.lpSum(x[i][s] for s in range(NUM_SLICES)) == 1

    # CONSTRAINT: contiguity -- slice s holds a contiguous block, because the
    # pipeline forwards layer i to i+1 in order.
    #
    # GENERALIZED FROM v1. v1 constrained only slices 0 and 2 and relied on the
    # middle slice's contiguity falling out automatically. That works for
    # exactly 3 slices and FAILS for 4, where two middle slices could
    # interleave. General form: the slice index assigned to layer i is
    # non-decreasing in i, encoded via prefix sums -- for each s, "layer i is
    # on a slice < s" must be non-increasing as i grows.
    for s in range(1, NUM_SLICES):
        for i in range(total_layers - 1):
            prefix_i = pulp.lpSum(x[i][t] for t in range(s))
            prefix_next = pulp.lpSum(x[i + 1][t] for t in range(s))
            prob += prefix_i >= prefix_next

    for s in range(NUM_SLICES):
        layers_on_s = pulp.lpSum(x[i][s] for i in range(total_layers))
        # CONSTRAINT: memory ceiling.
        prob += layers_on_s <= limits[s]
        # CONSTRAINT: no empty stage. (Relax for Qwen-style head-only last
        # ranks -- see layer_limits.HEAD_ONLY_LAST_RANK.)
        prob += layers_on_s >= 1
        # CONSTRAINT: z bounds every stage -- the min-max linearization.
        prob += z >= (
            per_layer_cost[s] * layers_on_s + endpoint_cost[s] + transport_cost[s]
        )

    prob += z  # OBJECTIVE: minimize the bottleneck

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    counts = [
        int(sum(pulp.value(x[i][s]) for i in range(total_layers)))
        for s in range(NUM_SLICES)
    ]
    return counts, limits


if __name__ == "__main__":
    print("v2: 4 slices (20/10/5/5), min-max objective, overlapped pipeline")
    print("    THEORY ONLY -- no constant fitted to latency data.")
    print(f"  slice bandwidths (GB/s): {[round(b / 1e9) for b in BW]}")
    print(
        f"  placeholders: EFF_WEIGHT={EFF_WEIGHT} EFF_ACT={EFF_ACT} "
        f"LAUNCH_US={LAUNCH_US} HANDOFF_FIXED_US={HANDOFF_FIXED_US}"
    )
    print()
    for batch in [8, 16, 32, 64]:
        counts, limits = optimize_placement(batch_size=batch)
        if sum(counts) != LAYERS:
            print(f"batch={batch:3d}: limits={limits} -> INFEASIBLE")
            continue
        lat = predict_latency_ms(counts, num_microbatches=2, batch_size=batch)
        print(
            f"batch={batch:3d}: limits={limits} -> split {counts}"
            f"  predicted(N=2) {lat:.0f} ms"
        )
