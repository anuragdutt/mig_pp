"""
Layer-placement optimizer for pipeline-parallel LLM inference on MIG slices.

Decision:   how many of the LAYERS transformer blocks go on each slice.
Objective:  minimize total generation latency (prefill + decode).
Constraints: per-slice memory (no OOM), contiguity, each layer placed once.

============================================================================
PIPELINE LATENCY MODEL -- derived from the harness code, NOT fit to data.
See benchmark_pipeline_microbatching.py and mig_transport_pipeline_non_blocking.py.

The harness runs one autoregressive step at a time. Within a step it pushes
N = num_microbatches microbatches through a 3-stage pipeline (one stage per
slice). Across steps it is strictly serial: the last rank ships the sampled
token back to rank 0 (blocking dist.send/recv) before the next step starts.

  ==> steps are SERIAL, microbatches within a step are PIPELINED.

Per-microbatch cost of stage s (the pipeline's per-slot service time):

    E_s(step) = counts[s] * layer_compute(s, step)          # forward compute
              + endpoint_compute(s, step)                   # embed / lm_head
              + transport(step)                             # SHM handoff out

  layer_compute : weights + activation/KV HBM traffic for counts[s] layers,
                  one microbatch, at this step's context length.
  endpoint_compute : the non-transformer ops the harness runs inside the same
                  per-microbatch loop -- the embedding gather on stage 0, and
                  norm + lm_head + argmax on the last stage. lm_head streams a
                  full [HIDDEN, VOCAB] weight ONCE PER MICROBATCH, so a step with
                  N microbatches pays it N times. Asymmetric: it makes the last
                  stage slower, which the optimizer must offset with fewer layers.
  transport     : the cross-slice handoff of the activation tensor. In the
                  harness this is NOT free -- isend() does GPU->pinned-CPU copy
                  + torch.cuda.synchronize() + CPU->SHM memcpy INLINE, and the
                  receiver does SHM->CPU->GPU inline in recv.wait(). So each
                  microbatch pays a real, serialized transport cost per stage
                  boundary it crosses. (mig_transport_pipeline_non_blocking.py
                  lines 320-346 write path, 108-129 read path.)

One step is a SERIAL chain, not an overlapped pipeline:

    T_step = N * SUM_s E_s        # every microbatch pays every stage
           + tok_roundtrip        # blocking next_tokens send last->first rank

Total = T_prefill (1 step, full-sequence activations)
      + SUM over MAX_NEW_TOKENS decode steps of T_step(ctx = SEQ_LEN + t)

WHY NOT THE GPipe LAW (SUM_s E_s + (N-1)*MAX_s E_s). That law is right for
stages that run as independent servers. This harness does not achieve overlap:
isend() does its GPU->CPU copy, a full cuda.synchronize() and the SHM memcpy
INLINE before returning, irecv() defers all real work to wait(), and every
send handle is drained against a receiver ACK at each step boundary -- so no
rank can leave a step until its downstream neighbour has consumed every
microbatch. See _step_latency_s for the line-by-line mechanism.

This is not a modelling preference. On the 453-row oracle sweep the GPipe law
ranks splits INVERTED against measurement (Kendall tau -0.48 to -1.00); the
serial law gives +0.94 to +1.00, and simultaneously cuts median error from
-44.5% to -12.8% and the N=2->8 drift from -13.9 pp to +2.9 pp.

CONSEQUENCE: with no overlap there is no bottleneck stage to balance, so the
optimizer should load the FASTEST slice as heavily as memory allows. The
measured data agrees -- sweeping the fast slice from 8 to 24 layers moves
latency 23.3%, while permuting the remaining layers between the two slow
slices moves it 0.7%.

WHAT IS FIRST-PRINCIPLES (architecture + A100 spec sheet):
    - all byte counts (weights, activations, KV cache, transport payload)
    - peak bandwidth per slice (A100 HBM * MIG fraction)
    - kernel count per layer
    - the fill+steady pipeline structure (read off the harness schedule)
WHAT IS EMPIRICAL (measure once on ONE MIG slice -- see measure_constants.py):
    - EFF_WEIGHT, EFF_ACT : achieved/peak bandwidth in the two access regimes
    - LAUNCH_US           : per-kernel launch + framework dispatch overhead
    - PCIE_BW_PINNED, PCIE_BW_UNPINNED, MEMCPY_BW : the three bandwidths that
                            gate the cross-slice activation handoff
    NONE of these are derived from the pipeline latency CSV. They are hardware
    properties. The defaults below are literature-typical placeholders;
    replace them with measured values for a clean, non-circular model.
============================================================================
"""
import pulp  # type: ignore

# ================= EMPIRICAL CONSTANTS (measure once, then freeze) =========
# EFF_WEIGHT answers the question: of the peak memory bandwidth this MIG slice is
# spec'd for, what fraction does it actually achieve when reading the model weights?
#
# ESTIMATE (measure with measure_constants.py calculate_EFF_WEIGHT):
# Weight streaming is the regime HBM is designed for -- one large contiguous read
# per projection matrix (the smallest here is q_proj at 32MB), fully coalesced,
# no reuse. Published A100 profiling of transformer inference consistently lands
# 80-90% of peak in this regime; a large-tensor STREAM-style read is the standard
# way to hit the upper end. 0.85 is mid-range and the least uncertain constant here.
EFF_WEIGHT = 0.85

# EFF_ACT: what fraction of peak does the slice achieve on the per-token
# activation + KV traffic during decode?
#
# ESTIMATE (measure with calculate_EFF_ACT_curve -- note it SWEEPS context):
# Decode activation traffic is dominated by the KV-cache read, whose share of
# the total grows with context:
#       ctx= 64   1.32 MB total,  79% is KV read
#       ctx=128   2.37 MB total,  88% is KV read
#       ctx=320   5.51 MB total,  95% is KV read
#       ctx=576   9.71 MB total,  97% is KV read
# So this constant is a scalar standing in for a regime that may shift across
# the decode trajectory. But the direction and size of that shift are NOT
# derivable from the byte counts alone: the KV cache is stored per layer and
# read per head with a stride, and DynamicCache grows by concatenation -- it is
# not a single contiguous stream even at ctx=576, so it should not be assumed to
# approach the weight-streaming regime just because it gets larger.
# Published decode-phase profiling puts achieved bandwidth in this scattered,
# latency-bound regime at roughly 0.20-0.30 of peak. 0.25 is the midpoint of
# that range and is held deliberately flat across all contexts.
# DO NOT hand-tune this value. If it looks wrong, that is what the ctx sweep in
# measure_constants.py is for -- it reports the whole curve plus a traffic-
# weighted scalar, and it settles both the level and whether a scalar suffices.
# TODO: replace with the measured sweep; if the curve is steep, make this a
#       function of ctx rather than picking a compromise scalar.
EFF_ACT    = 0.25

# LAUNCH_US: per-kernel launch + framework dispatch overhead.
#
# ESTIMATE (measure with calculate_LAUNCH_US):
# A raw CUDA kernel launch on the same stream is ~3-10 us. PyTorch eager mode
# adds Python dispatch, autograd bookkeeping and arg marshalling on top, which
# published op-level benchmarks put at roughly 5-15 us. The harness runs eager
# (no CUDA graphs, no torch.compile -- see forward_through_layers), so the full
# eager path applies. 12 us is a mid-range eager estimate; the prior 20.0 sat at
# the pessimistic end of plausible.
LAUNCH_US  = 12.0

# KERNELS_PER_LAYER: CUDA kernel launches in one LlamaDecoderLayer forward.
#
# ESTIMATE (measure with calculate_KERNELS_PER_LAYER, or kernel_measure.py for
# the full pipeline including transport):
# Counted directly from the ops in forward_through_layers:
#   2 RMSNorm (reduce + scale)          -> 4
#   q/k/v/o projection GEMMs            -> 4
#   RoPE apply to q and k               -> 2
#   SDPA (single fused flash kernel)    -> 1
#   2 residual adds                     -> 2
#   gate/up/down MLP GEMMs              -> 3
#   SiLU + elementwise mul              -> 2
#   KV cache concat/update              -> 2
#                                       ------
#                                         20
# Some of these fuse in practice (RMSNorm is often one kernel, SiLU+mul often
# fuses), so the achieved count is realistically 14-16. 16 is the conservative
# end of that. The prior 10 was below the op count and cannot be right.
KERNELS_PER_LAYER = 16

# NOTE ON THESE TWO TOGETHER: only their PRODUCT enters the model, via
# LAUNCH_MS_PER_LAYER = KERNELS_PER_LAYER * LAUNCH_US. The previous pair
# (10 x 20.0us) gave 200us/layer; this pair (16 x 12.0us) gives 192us/layer --
# a 4% change. The individual numbers are better justified now, but the model's
# behaviour is essentially unchanged by the edit. Do not read the more detailed
# reasoning as a more accurate result: BOTH still need measuring, and only the
# product matters, so a measurement that moves one without the other is what
# actually shifts the launch term.

# ---- transport constants (cross-slice activation handoff) ----
# Traced op-by-op against mig_transport_pipeline_non_blocking.py.
#
#   SEND  _write_tensor_to_slot (fp16 fast path, lines 336-363):
#     line 351  staging[:n].copy_(tensor)        GPU -> PINNED cpu   [PCIe, pinned]
#     line 356  torch.cuda.synchronize()         full device drain   [fixed]
#     line 359  staging.numpy().view(uint8)      view, free
#     line 363  my_np_slots[slot][:n] = raw      pinned -> SHM       [host memcpy]
#
#   RECV  _read_tensor_from_slot (lines 371-387):
#     line 381  peer_slots[src][slot][:n]        view, free
#     line 382  np.frombuffer(...)               view, free
#     line 385  peer_data.copy()                 SHM -> heap         [host memcpy]
#     line 387  src_tensor.to(tensor.device)     heap -> GPU         [PCIe, UNPINNED]
#     line 387  tensor.copy_(...)                device-local copy   [HBM r+w]
#
# The two PCIe crossings are NOT symmetric: the send side copies out of a
# pre-allocated pinned staging buffer (DMA straight over the bus), while the
# receive side copies from an ordinary heap buffer produced by peer_data.copy(),
# which PyTorch must stage through an internal pinned bounce buffer -- roughly
# half the achieved bandwidth. They get separate constants.
#
# ESTIMATES (measure with measure_constants.py sections 5-8):
#
# PCIE_BW_PINNED: an A100 sits on PCIe gen4 x16 = 32 GB/s theoretical per
# direction. Pinned DMA typically achieves 75-85% of that, so ~24-27 GB/s.
# NOTE: this assumes gen4. If the instance is gen3 x16 the ceiling halves to
# ~16 GB/s theoretical and this constant should drop to ~12e9. Confirm the
# link generation (nvidia-smi -q | grep -i pcie) before trusting transport costs.
PCIE_BW_PINNED   = 25e9   # bytes/s, pinned host<->device DMA
#
# PCIE_BW_UNPINNED: line 387 copies from an ordinary heap buffer produced by
# peer_data.copy(), so the driver stages it through an internal pinned bounce
# buffer -- an extra host-to-host copy hidden inside the transfer. The standard
# observed penalty is roughly half the pinned rate.
PCIE_BW_UNPINNED = 12e9   # bytes/s, pageable host->device
#
# MEMCPY_BW: host-to-host, used for both the pinned->SHM write (line 363) and
# the SHM->heap read (line 385). These are numpy slice-assign / .copy(), not a
# hand-tuned memcpy, and they are single-threaded. Server DDR4 single-threaded
# memcpy lands ~8-15 GB/s; numpy's path sits in the lower half of that.
MEMCPY_BW        = 10e9   # bytes/s, host-to-host memcpy
#
# HANDOFF_FIXED_US: the residual per-handoff cost -- the cuda.synchronize() at
# line 356 plus the gloo handshake dispatch. A bare cudaDeviceSynchronize with
# an already-idle queue costs ~10-20 us; the gloo handshake for a 4-byte tensor
# adds ~10-30 us of dispatch and socket work. So ~25-40 us with an idle queue.
# 30 us is mid-range. IMPORTANT: line 356 is a FULL device drain, not a fixed
# cost -- in the real pipeline it blocks until the layer compute just issued
# retires, so this constant is a LOWER BOUND. Treating it as fixed is consistent
# with E_s = compute + transport (the two never overlap on the sender), but the
# true cost under load will be higher. The prior 50.0 over-weighted it at short
# context, where it was 54% of total decode transport.
HANDOFF_FIXED_US = 30.0
#
# TOK_ROUNDTRIP_US: per-step next_tokens feedback, last rank -> rank 0, a
# (batch,1) int64 tensor over gloo TCP on localhost. That is 256 bytes at
# batch=32 -- pure latency, no bandwidth. Loopback TCP round-trip is ~20-50 us;
# gloo's dispatch and rendezvous add on top. ~60 us is a reasonable midpoint.
# The prior 100.0 was at the pessimistic end.
TOK_ROUNDTRIP_US = 60.0
# ===========================================================================

# ---------- model architecture: Vicuna-7B (LLaMA-7B) ----------
HIDDEN, INTER, LAYERS, VOCAB = 4096, 11008, 32, 32000
WEIGHT_BYTES_PER_LAYER = (4*HIDDEN*HIDDEN + 3*HIDDEN*INTER) * 2   # FP16

# fixed-position components: embedding on first slice, lm_head+norm on last.
# Their PLACEMENT is not a decision (the harness hardcodes it), but their COST is
# charged two ways: memory in layer_limits(), and TIME in endpoint_compute_in_seconds().
# The time half matters for placement even though the position is fixed -- lm_head
# makes the last stage slower, so the optimizer should give that stage fewer
# transformer layers to compensate.
EMBED_BYTES   = VOCAB * HIDDEN * 2
LM_HEAD_BYTES = VOCAB * HIDDEN * 2
NORM_BYTES    = HIDDEN * 2

# workload
SEQ_LEN, MAX_NEW_TOKENS = 64, 512
MAX_SEQ = SEQ_LEN + MAX_NEW_TOKENS
S = 1 + MAX_NEW_TOKENS               # total steps (kept for reference)

# ---------- hardware: A100 40GB, MIG slices ----------
A100_BW     = 1555e9                  # HBM2 bytes/s, full GPU
SLICE_FRAC  = [3/7, 2/7, 2/7]        # 3g.20gb, 2g.10gb, 2g.10gb
SLICE_NAME  = ["20gb", "10gb", "10gb"]
SLICE_BYTES = [20e9, 10e9, 10e9]
USABLE_FRAC = 0.9
BW = [A100_BW * f for f in SLICE_FRAC]

LAUNCH_MS_PER_LAYER = KERNELS_PER_LAYER * LAUNCH_US / 1000.0


# ---------------------------------------------------------------------------
# activation + KV traffic per token per layer (bytes). Flash attention (SDPA):
# attention scores are NOT materialized to HBM, so they are excluded. K,V for
# the full context ARE read each step (the KV cache).
# ---------------------------------------------------------------------------
def activation_bytes_per_token(ctx):
    B = 2  # FP16
    qkv    = (HIDDEN + 3*HIDDEN) * B          # read input, write Q,K,V
    flash  = (HIDDEN + 2*ctx*HIDDEN + HIDDEN) * B   # Q + K,V(ctx read) + out
    oproj  = (HIDDEN + HIDDEN) * B
    gate   = (HIDDEN + INTER) * B
    up     = (HIDDEN + INTER) * B
    swiglu = (2*INTER + INTER) * B
    down   = (INTER + HIDDEN) * B
    misc   = (6*HIDDEN) * B                   # ~2 rmsnorm + 2 residual r/w
    return qkv + flash + oproj + gate + up + swiglu + down + misc


# ---------------------------------------------------------------------------
# ONE LAYER, ONE MICROBATCH, ONE STEP -- seconds. No step loop here: the step
# loop lives in predict_latency_ms so the pipeline math (fill + steady state)
# can wrap it correctly. This is the compute term C in the derivation.
#
#   prefill : mb sequences * SEQ_LEN tokens, weights + activations stream in the
#             efficient (bandwidth-bound) regime.
#   decode  : one token per sequence; weights efficient, per-token activation/KV
#             traffic is the scattered latency-bound regime.
# ---------------------------------------------------------------------------
def layer_compute_in_seconds(slice_bandwidth_bytes_per_second, microbatch_size, context_length, is_prefill):
    if is_prefill:
        # prefill processes all context_length tokens at once for every sequence in the microbatch
        activation_bytes = activation_bytes_per_token(context_length) * microbatch_size * context_length
        # weights and activations both stream in large contiguous chunks during prefill, so both use EFF_WEIGHT
        time_seconds = (WEIGHT_BYTES_PER_LAYER + activation_bytes) / (slice_bandwidth_bytes_per_second * EFF_WEIGHT)
        return time_seconds + LAUNCH_MS_PER_LAYER / 1000.0

    # decode: only 1 new token per sequence (not the full context)
    activation_bytes = activation_bytes_per_token(context_length) * microbatch_size
    # weights are read as one large contiguous block -- efficient streaming
    time_seconds  = WEIGHT_BYTES_PER_LAYER / (slice_bandwidth_bytes_per_second * EFF_WEIGHT)
    # KV cache reads are scattered across memory (one entry per previous token per head) -- slow, latency-bound
    time_seconds += activation_bytes / (slice_bandwidth_bytes_per_second * EFF_ACT)
    time_seconds += LAUNCH_MS_PER_LAYER / 1000.0

    return time_seconds


# ---------------------------------------------------------------------------
# ENDPOINT WORK: the non-transformer ops the first and last stages run, once per
# microbatch per step, in the same serial loop as the layer forwards.
#
#   FIRST stage (rank 0), benchmark_pipeline_microbatching.py L530-532:
#       embed(next_tokens[start:end]).half()
#     A row gather out of a [VOCAB, HIDDEN] table. Only microbatch_size rows are
#     touched, so the traffic is the gathered rows, NOT the whole table. The rows
#     are scattered (token ids are arbitrary), so this is the latency-bound
#     regime. The .half() then allocates and writes a fresh tensor.
#
#   LAST stage, L545-548:
#       normed = norm(current_hidden)
#       logits = lm_head(normed)
#       argmax(logits)
#     lm_head is a [HIDDEN, VOCAB] matmul against a single token per sequence, so
#     it is weight-bound exactly like a transformer layer projection: the whole
#     262 MB weight is streamed to produce microbatch_size rows of logits. This
#     runs INSIDE the microbatch loop, so a step with N microbatches streams that
#     weight N times.
#
# WHY THIS MATTERS FOR PLACEMENT: both terms are asymmetric -- they land on one
# specific stage. Charging only their memory (which layer_limits already does)
# tells the optimizer those stages have less ROOM, but not that they are SLOWER.
# Without the time term the optimizer balances layers as if all stages were bare
# transformer stacks, and systematically overloads the two endpoints.
# ---------------------------------------------------------------------------
def endpoint_compute_in_seconds(stage_index, num_stages, slice_bandwidth_bytes_per_second,
                                microbatch_size, is_prefill):
    time_seconds = 0.0

    if stage_index == 0:
        # embedding gather: read microbatch_size rows, write them out again.
        # prefill gathers the whole prompt, decode gathers one token per sequence.
        rows = microbatch_size * (SEQ_LEN if is_prefill else 1)
        gathered_bytes = rows * HIDDEN * 2
        # scattered row gather + the .half() copy -> read + write, latency-bound
        time_seconds += 2 * gathered_bytes / (slice_bandwidth_bytes_per_second * EFF_ACT)
        time_seconds += LAUNCH_US / 1e6            # embed + .half() dispatch

    if stage_index == num_stages - 1:
        # final norm over the hidden state being classified (one token per
        # sequence in both phases -- prefill slices normed[:, -1:, :] before
        # lm_head, L475-478).
        norm_bytes = 2 * microbatch_size * HIDDEN * 2      # read + write
        time_seconds += norm_bytes / (slice_bandwidth_bytes_per_second * EFF_ACT)

        # lm_head: stream the full [HIDDEN, VOCAB] weight, write the logits.
        logit_bytes = microbatch_size * VOCAB * 2
        time_seconds += (LM_HEAD_BYTES + logit_bytes) / (slice_bandwidth_bytes_per_second * EFF_WEIGHT)

        # argmax reduces the logits -- one more full read of them.
        time_seconds += logit_bytes / (slice_bandwidth_bytes_per_second * EFF_ACT)

        time_seconds += 3 * LAUNCH_US / 1e6        # norm + lm_head + argmax

    return time_seconds


# ---------------------------------------------------------------------------
# TRANSPORT: computes the cost of moving activattions of microbatch N to N+1
# Both sides are inline/blocking on their rank, 
# so this cost serializes with compute rather than overlapping it.
# ie its not E_s = max(compute, transport) but E_s = compute + transport

# Rank 0 GPU → CPU RAM → shared memory → CPU RAM → Rank 1 GPU

# Payload is the hidden-state tensor: (mb, tokens, HIDDEN) FP16.
#   prefill handoff: tokens = ctx (full sequence)
#   decode  handoff: tokens = 1 (single new token)

# dst_slice is the RECEIVING slice: the final device-local tensor.copy_()
# (line 387) runs on the receiver's HBM, so it is charged at that slice's
# bandwidth. Pass None to skip that term.
# ---------------------------------------------------------------------------
def transport_time_for_activation_transfer_seconds(mb, ctx, is_prefill, dst_slice=None):
    tokens  = ctx if is_prefill else 1
    payload = mb * tokens * HIDDEN * 2                 # bytes, FP16
    
    # --- sender rank N---
    t  = payload / PCIE_BW_PINNED          # line 351: GPU -> pinned CPU
    t += payload / MEMCPY_BW               # line 363: pinned -> SHM

    # --- receiver rank N+1 ---
    t += payload / MEMCPY_BW               # line 385: SHM -> heap (.copy())
    t += payload / PCIE_BW_UNPINNED        # line 387: heap -> GPU (pageable)

    if dst_slice is not None:
        # After the data arrives on the receiving GPU it sits in a temporary
        # buffer. The GPU then moves it into the layer's actual input slot --
        # that internal GPU-to-GPU move is what this line charges (read + write,
        # so factor of 2).
        #
        # KNOWN SMALL DOUBLE-COUNT: layer_compute_s on the receiving rank also
        # charges reading that same input slot as the first thing it does. So
        # one copy of the data gets counted twice. We leave it in because the
        # error is ~0.7 us out of ~1370 us total layer time (0.05%) -- too small
        # to matter. Only revisit if transport costs ever grow to dominate.
        t += 2 * payload / (BW[dst_slice] * EFF_WEIGHT)

    t += HANDOFF_FIXED_US / 1e6            # line 356 cuda.sync + gloo handshake
    return t


def kv_bytes_per_layer(batch_size):
    return 2 * batch_size * MAX_SEQ * HIDDEN * 2   # K+V, FP16


def layer_limits(batch_size):
    kv_per_layer  = kv_bytes_per_layer(batch_size)
    mem_per_layer = WEIGHT_BYTES_PER_LAYER + kv_per_layer

    usable = [USABLE_FRAC * b for b in SLICE_BYTES]
    usable[0] -= EMBED_BYTES
    usable[2] -= (LM_HEAD_BYTES + NORM_BYTES)

    return [int(usable[s] // mem_per_layer) for s in range(3)]


# ---------------------------------------------------------------------------
# OPTIMIZER. Objective is now the accurate per-layer time (which differs per
# slice via bandwidth). Round-trip/launch are per-layer additive, so they DO
# shift the balance slightly vs the old bandwidth-only model.
# ---------------------------------------------------------------------------
def _per_layer_rank_cost_ms(bw, mb=1):
    """Representative per-layer cost (ms) for RANKING splits in the MILP.

    Sums one layer's compute over prefill + all decode steps for one microbatch.
    This is a monotone proxy: it does not include the fill/steady MAX term (that
    is nonlinear and belongs only in predict_latency_ms), but per-layer cost is
    monotincreasing in slice bandwidth, so it ranks splits correctly. Exact
    latency for a given (batch, mb, N) is computed by predict_latency_ms.
    """
    total_s = layer_compute_in_seconds(bw, mb, SEQ_LEN, is_prefill=True)
    for t in range(MAX_NEW_TOKENS):
        total_s += layer_compute_in_seconds(bw, mb, SEQ_LEN + t, is_prefill=False)
    return total_s * 1000.0


def _endpoint_rank_cost_ms(stage_index, num_stages, bw, mb=1):
    """Endpoint (embed / lm_head) cost in ms over the whole run, for the MILP.

    A per-STAGE constant, not per-layer: it does not scale with how many layers
    the stage holds. In the objective it therefore enters as a fixed offset on
    that stage rather than multiplying the layer count. It still changes the
    chosen split, because a min-max objective balances TOTAL stage cost -- a
    stage carrying lm_head should be given fewer layers.
    """
    total_s = endpoint_compute_in_seconds(stage_index, num_stages, bw, mb, is_prefill=True)
    total_s += MAX_NEW_TOKENS * endpoint_compute_in_seconds(
        stage_index, num_stages, bw, mb, is_prefill=False)
    return total_s * 1000.0


def optimize_placement(total_layers=LAYERS, batch_size=32):
    limits = layer_limits(batch_size)

    # microbatch size doesn't change which SPLIT is optimal for a fixed
    # batch/mb, but the per-layer weight matters; use mb=1 unit cost for the
    # objective ranking (per-layer cost is monotonic in bandwidth regardless).
    # For ranking splits we use a representative mb; the true latency for a
    # given (batch, mb) is computed by predict_latency_ms.
    per_layer_cost = [_per_layer_rank_cost_ms(BW[s]) for s in range(3)]
    # per-STAGE constant: embedding on stage 0, norm+lm_head+argmax on stage 2.
    # Independent of how many layers the stage holds, so it cannot be folded into
    # per_layer_cost -- it enters the objective as a fixed offset per stage.
    endpoint_cost = [_endpoint_rank_cost_ms(s, 3, BW[s]) for s in range(3)]

    prob = pulp.LpProblem("layer_placement", pulp.LpMinimize)

    # decision variable x[i][s] = 1 if layer i is placed on slice s, else 0.
    # one binary variable per (layer, slice) pair -- total_layers * 3 of them.
    x = [[pulp.LpVariable(f"x_{i}_{s}", cat="Binary") for s in range(3)]
         for i in range(total_layers)]

    # CONSTRAINT: each layer placed on exactly one slice (not zero, not two).
    for i in range(total_layers):
        prob += pulp.lpSum(x[i][s] for s in range(3)) == 1

    # CONSTRAINT: contiguity. Layers are placed in order 0..total_layers-1 and
    # slice s must hold a contiguous block (the pipeline forwards layer i to
    # i+1 in sequence, so splitting a slice's layers into two ranges would
    # break the forward pass). Encoded as: once a later layer leaves slice 0,
    # no earlier layer can come back to it (x[i][0] >= x[i+1][0], monotone
    # non-increasing membership in slice 0); symmetric for slice 2, which only
    # ever gains layers as i increases (x[i][2] <= x[i+1][2], monotone
    # non-decreasing). Slice 1 (the middle) is left unconstrained here -- its
    # contiguity falls out automatically once slice 0 and slice 2's blocks are
    # fixed to the front and back.
    for i in range(total_layers - 1):
        prob += x[i][0] >= x[i+1][0]
        prob += x[i][2] <= x[i+1][2]

    for s in range(3):
        # total layers assigned to slice s (sum of its binary indicators)
        layers_on_s = pulp.lpSum(x[i][s] for i in range(total_layers))
        # CONSTRAINT: memory ceiling -- slice s cannot hold more layers than
        # layer_limits() says fit in its usable HBM (weights + KV cache).
        prob += layers_on_s <= limits[s]
        # CONSTRAINT: every slice must host at least one layer (no empty
        # stage -- a slice with zero layers isn't part of the pipeline).
        prob += layers_on_s >= 1

    # OBJECTIVE: minimize TOTAL work across stages.
    #
    # This mirrors _step_latency_s, which is N * SUM_s E_s -- the stages do not
    # overlap (see that function for the mechanism and the measured evidence), so
    # there is no bottleneck to balance and the makespan is just the total. The
    # right split therefore puts as many layers as memory allows on the fastest
    # slice, rather than equalizing per-stage cost.
    #
    # A min-max/bottleneck objective would be correct for a truly overlapped
    # GPipe pipeline and is the intuitive choice here, but it ranks splits
    # BACKWARDS against this harness (Kendall tau -0.48 to -1.00 on the oracle
    # sweep). Do not reinstate it unless the transport is changed to actually
    # overlap -- i.e. the ACK drain and the inline cuda.synchronize() are removed.
    #
    # endpoint_cost is a per-stage constant, so it cannot change which split wins
    # a pure sum comparison. It is included for the objective VALUE to be a
    # meaningful latency estimate, not because it affects the argmin.
    prob += pulp.lpSum(
        per_layer_cost[s] * pulp.lpSum(x[i][s] for i in range(total_layers))
        + endpoint_cost[s]
        for s in range(3)
    )

    # hand the fully-built LP/MILP to the CBC solver (msg=False: suppress its
    # console spam) and let it search for the assignment that minimizes the
    # objective subject to every constraint added above.
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # read back the solved values: for each slice, count how many of its
    # binary indicators came back 1 (pulp.value returns floats like 1.0/0.0,
    # so sum + int() collapses that to a clean layer count).
    counts = [int(sum(pulp.value(x[i][s]) for i in range(total_layers)))
              for s in range(3)]
    return counts, limits


# ---------------------------------------------------------------------------
# PREDICTION: full generation latency for a given split + batch config.
# counts = [layers_on_slice0, slice1, slice2]
# ---------------------------------------------------------------------------
def _stage_intervals_s(counts, mb, ctx, is_prefill):
    """E_s for each slice at one step: compute of its layers + one transport hop.

    Transport is charged once per stage: every microbatch that leaves stage s
    pays the handoff to s+1. The last stage has no outbound activation handoff
    (it emits the token instead), but the per-step token round-trip is added
    separately in _step_latency_s, so here we charge transport on stages 0..n-2.
    The handoff out of stage s lands on stage s+1, so the receiver-side
    device-local copy is charged at slice s+1's bandwidth.

    Endpoint work (embedding on stage 0, norm+lm_head+argmax on the last) is
    charged here too: the harness runs it inside the same per-microbatch loop as
    the layer forwards (L530-532 and L545-548), so it is part of that stage's
    per-microbatch service time, not a once-per-step cost.
    """
    n = len(counts)
    E = []
    for s in range(n):
        e = counts[s] * layer_compute_in_seconds(BW[s], mb, ctx, is_prefill)
        e += endpoint_compute_in_seconds(s, n, BW[s], mb, is_prefill)
        if s < n - 1:
            e += transport_time_for_activation_transfer_seconds(mb, ctx, is_prefill, dst_slice=s + 1)
        E.append(e)
    return E


def _step_latency_s(counts, mb, N, ctx, is_prefill):
    """One step: every microbatch pays every stage. NO cross-stage overlap.

        T_step = N * SUM_s E_s + token_roundtrip

    WHY NOT THE GPipe fill + (N-1)*max LAW. That law assumes the stages run as
    independent servers, so while stage 1 works on microbatch k, stage 0 is
    already working on k+1. The harness does not achieve that. Two mechanisms in
    mig_transport_pipeline_non_blocking.py serialize the stages:

      1. ACK backpressure. isend() returns a handle whose wait() blocks until the
         RECEIVER has finished reading the SHM slot and sent an ACK back
         (AsyncHandle.wait, transport L70-88; the ACK is sent from the receiver's
         wait at L120-126). The harness drains all send handles at the end of
         every step (benchmark L554-555), so no rank can leave a step until its
         downstream neighbour has consumed every microbatch of that step.

      2. isend is not actually asynchronous. It does the GPU->pinned copy, a FULL
         torch.cuda.synchronize(), and the CPU->SHM memcpy inline before it
         returns (transport L289-302 -> L320-364). Likewise irecv only posts a
         handshake; the real SHM->GPU movement happens inside wait()
         (transport L304-318, L108-127). So the data movement never overlaps
         anything -- posting recvs up front buys nothing.

    Together these collapse the pipeline into a serial chain: each microbatch
    traverses all stages before the next makes progress, so the cost is N copies
    of the full depth rather than a fill plus a bottleneck-gated stream.

    EVIDENCE (oracle sweep, 453 rows): under fill+steady the model's split
    ranking was INVERTED against measurement, Kendall tau -0.48 to -1.00, median
    error -44.5%, and error drifted -13.9 pp from N=2 to N=8. Under this serial
    law tau is +0.94 to +1.00, median error -12.8%, spread narrows 34.7 -> 14.7 pp,
    and the N-drift falls to +2.9 pp. The measured data also shows latency is
    monotone in the layer count of the FAST slice and nearly flat in how the rest
    is balanced (holding c0=24 and permuting the other 8 layers moves latency
    0.7%, while sweeping c0 from 8 to 24 moves it 23.3%) -- the signature of a
    serial chain, not of a bottleneck-gated pipeline.

    CONSEQUENCE FOR PLACEMENT: with no overlap there is no bottleneck to balance.
    Total work is what matters, so layers belong on the fastest slice. See
    optimize_placement, whose objective follows this same law.
    """
    E = _stage_intervals_s(counts, mb, ctx, is_prefill)
    tok_rt = 0.0 if is_prefill else TOK_ROUNDTRIP_US / 1e6
    return N * sum(E) + tok_rt


# ---------------------------------------------------------------------------
# PREDICTION: full generation latency for a given split + batch config.
# counts = [layers_on_slice0, slice1, slice2]
# ---------------------------------------------------------------------------
def predict_latency_ms(counts, num_microbatches, batch_size):
    mb = batch_size // num_microbatches
    N  = num_microbatches

    # prefill: one step, full-sequence activations, no token round-trip
    total_s = _step_latency_s(counts, mb, N, SEQ_LEN, is_prefill=True)

    # decode: MAX_NEW_TOKENS serial steps, context grows each step
    for t in range(MAX_NEW_TOKENS):
        total_s += _step_latency_s(counts, mb, N, SEQ_LEN + t, is_prefill=False)

    return total_s * 1000.0  # -> ms


if __name__ == "__main__":
    print("Empirical constants (measure to make non-circular):")
    print(f"  EFF_WEIGHT={EFF_WEIGHT}  EFF_ACT={EFF_ACT}  LAUNCH_US={LAUNCH_US}")
    print()
    for batch in [8, 16, 32, 64]:
        counts, limits = optimize_placement(batch_size=batch)
        print(f"batch={batch:3d}: limits={limits} -> optimal split {counts}")