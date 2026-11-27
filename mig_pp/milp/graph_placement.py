"""
Some terms to know before reading this

MEMORY BANDWDITH: Its the speed (in bytes per second) at which a GPU slice can read and write data 
between its own onboard memory (VRAM / HBM) and its math cores.
RANK: one pipeline stage = one process on one MIG slice, holding a block of consecutive decoder layers.
SPLIT: how many layers each rank holds, e.g. [22, 11, 6, 6]. The MILP picks it.
MIG SLICE: a hardware partition of the A100: memory slices of 8 (capacity + bandwidth), compute (GPC) slices of 7.
MICROBATCH: the batch cut into batch // mb equal pieces that flow through the ranks in a pipeline.
PREFILL / DECODE: prefill runs the whole prompt once; decode then generates one token per step.
KV CACHE: keys/values of earlier tokens kept on the GPU so decode need not recompute them.
ACTIVATION: the tensor a rank hands to the next rank (hidden size per token).
PINNED / SHM: page-locked host buffer (fast GPU copies) / shared-memory slot the ranks exchange data through.
D2H / H2D: copy device (GPU) to host / host to device.
HANDSHAKE / ACK: gloo messages saying "data is in SHM" / "I have read it, slot is free".
EVENT / LONGEST PATH: each step is a graph of timed events; the step time is its longest chain.


MILP layer placement for the MIG pipeline: how many decoder layers each slice holds.

Edit CONFIG in main() and run:  python3 pulp_layer_placement.py
Needs PuLP only to solve; predict_ms() is plain Python. Torch-free.

MODEL (read off the harness code, not fit to data)
--------------------------------------------------
A run is one prefill step and then MAX_NEW_TOKENS decode steps. Steps are serial:
rank 0 cannot embed step t+1 before the last rank's next_tokens arrive. Within a step
the N microbatches pipeline across the R ranks. Each rank is one CPU thread, in this
order per step:

    rank 0  :          for k: LAUNCH(k); FLUSH(k-1)     then FLUSH(N-1), drain ACKs
    middle  : for k: RECV(k); LAUNCH(k); FLUSH(k-1)     then FLUSH(N-1), drain ACKs
    last    : for k: RECV(k); LAUNCH(k)                 then send next_tokens

    LAUNCH(k): queue rotary/embed, the layers and the clone (last rank: norm, lm_head,
               argmax) on the GPU; isend queues the D2H copy on a side stream.
    FLUSH(k) : wait for D2H(k); pinned -> SHM memcpy; gloo handshake send.
    RECV(k)  : wait for handshake k; SHM -> pinned memcpy; H2D, wait; gloo ACK send.

Publishing is ONE BEHIND: FLUSH(k-1) runs after RECV(k), so a rank hands k-1 on only
once k has arrived from upstream. That is neither the old serial chain nor GPipe.

As a longest path (t_v >= t_u + w), with A = CPU holds input k, F = GPU done with k,
X = D2H(k) landed, P = FLUSH(k) done (handshake k delivered to the next rank):

    A_0(0) >= STEP;  A_0(1) >= A_0(0) + Lc_0;  A_0(k) >= P_0(k-2)
    A_s(k) >= P_{s-1}(k) + RECV_s;  A_s(k) >= cpu_s(k) + RECV_s      (s >= 1)
    F_s(k) >= A_s(k) + Cg_s;  F_s(k) >= F_s(k-1) + Cg_s;  F_s(k) >= A_s(k) + Lc_s
    X_s(k) >= F_s(k) + D_s;   X_s(k) >= X_s(k-1) + D_s
    P_s(k) >= X_s(k) + M_s + HS;  P_s(k) >= A_s(k+1) + Lc_s + M_s + HS
    END >= F_last(N-1) + TOK;  END >= A_last(N-1) + Lc_last + TOK

    Cg_s = GPU time of one microbatch on rank s  = n_s * layer_s + fixed_s
    Lc_s = CPU time to queue it                  = n_s * KERNELS * launch + fixed
    D, M = D2H and pinned->SHM of one activation; RECV = SHM->pinned + H2D + ACK
    HS = handshake send; TOK = next_tokens exchange; STEP = position_ids sync

The last decode step has no token exchange: rank 0's window closes at the final barrier,
which does not wait for the last rank's queued GPU work (END >= A_last(N-1) + Lc + BARRIER).

n_s are the integer decisions; the MILP has one variable per event and one >= per edge,
so minimizing END is the longest path for any split. Decode is sampled at a few
contexts to keep it small; predict_ms() evaluates every step.

GPU TIME FROM PROBE PRIMITIVES (constants sets printed by probe_constants.py, e.g. "mig1")
------------------------------------------------------------------------------------------
When a slice's constants hold the probe's generic primitives (GEMM, attention, KV append,
elementwise, device copy, D2H / H2D, each fixed + size / rate), layer_s and fixed_s are composed:
every kernel of the layer (layer_ops.layer_ops_from_shape: the model code's ops written out from
the shape; tests/test_layer_ops_torch.py checks them against the traced code) at its primitive's
time, and the same for the per-microbatch work outside the layers (rotary, embed, clone, norm +
lm_head + argmax). KERNELS and the launches outside the layers are those lists' kernel counts.
Nothing is measured or fitted per model. Such sets have no clock model yet: clock="fast" only.

CPU CLOCK (CONFIG clock="schedutil"; "fast", the default, keeps every rank at full clock)
--------------------------------------------------------------------------------------
Under the schedutil governor a rank whose CPU mostly waits is clocked down, and its launches
(Lc) take clock_slow_factor times as long (1.56 vs 0.69 ms per layer on mig1/mig2). Which ranks
are slow depends on the split: rank s is slow when its GPU busy fraction,
N x Cg_s summed over the run / the run's step time, is below clock_busy_threshold[s].
Slow ranks lengthen the step, which can push more ranks under, so the slow set is grown until
nothing changes (slow_ranks()). The MILP has one binary per rank for it (see optimize_placement).
Unlike the rest of the model, the slow factor and thresholds are fitted to measured data
(run 3 events data, schedutil; held-out Qwen-7B processes: 69-99% of levels right by rank).
Scored on mig1/mig2 Vicuna-7B with that box's constants (hardware_constants "a100_40gb_pcie_mig",
2026-11-14): clock="fast" picks the measured winner of the full-clock, one-thread runs at all 9
pairs (Kendall tau 0.92); against the schedutil runs, clock="schedutil" picks splits 0.87% slower
than the best on average vs 1.19% for "fast" (predicted vs measured slowdown per split: r = 0.86).

With a probe set (probe_constants.py: clock_threshold_by_period present) the rule comes from the
governor probe and the harness code instead, nothing fitted:
  - busy = each rank's CPU busy time in the step (cpu_busy_ms(), read off the code): launches, host
    copies, the receive (memcpy, pooled copy, H2D wait, ACK) and the spins on its own GPU (CUDA
    events and stream syncs with default flags spin); waits for gloo messages (handshake, ACK,
    next_tokens) sleep. Rank 0 never sleeps from the step's start to its last flush.
  - a rank still at full clock is slowed when its busy time over the run is below the governor's
    threshold x the run's time, the threshold interpolated at the run's mean step time between the
    probe's periods (40 ms: 40% busy, 120 ms: 30% on mig1/mig2); a slow rank's launches take
    clock_slow_factor x as long (1 / the probe's slow clock level, ~2.25-2.29).
  - the split is then picked by exact search over every split in memory (brute_force): the rule
    depends on when each rank waits, which a MILP cannot hold without being free to game it.

Left out: copy engines / PCIe shared between slices, CPU cores shared between lanes,
cold start of each config (fresh processes), PyTorch's CPU thread pool on the receive copy
(mb x hidden > 32768, problems.md "Run 8"), and anything but MIG_LOG_LEVEL=summary.
"""

import math
import os
import sys
import warnings
from functools import lru_cache
import pulp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_limits  # noqa: E402
import hardware_constants  # noqa: E402
import layer_ops  # noqa: E402  (torch-free unless tracing)

warnings.filterwarnings("ignore", message=".*deprecated.*")

# measured constants, card bandwidths and MIG memory slices live in hardware_constants.py;
# pick a set with CONFIG["constants"] = hardware_constants.CONSTANTS["<name>"]
DEFAULT_CONSTANTS = hardware_constants.CONSTANTS[hardware_constants.DEFAULT_NAME]
CARD_BW = hardware_constants.CARD_BW
MEMORY_SLICES = hardware_constants.MEMORY_SLICES

# Matching harness setup
SWEEP_PAIRS = [(8, 4), (8, 2), (16, 8), (16, 4), (16, 2), (32, 16), (32, 8), (32, 4),
               (32, 2), (64, 32), (64, 16), (64, 8), (64, 4), (64, 2)]


# ---------------------------------------------------------------------------
# bytes (same counts as measure_constants.py, which its efficiencies are defined on)
# ---------------------------------------------------------------------------
def weight_bytes(model_shape):
    q_width = model_shape["heads"] * model_shape["hd"]
    kv_width = model_shape["kv"] * model_shape["hd"]
    hidden, inter = model_shape["H"], model_shape["I"]
    return (hidden * q_width + 2 * hidden * kv_width + q_width * hidden + 3 * hidden * inter) * 2


def act_bytes(model_shape, context_len):
    """Activation + KV traffic of one token through one layer (flash attention: no scores)."""
    q_width = model_shape["heads"] * model_shape["hd"]
    kv_width = model_shape["kv"] * model_shape["hd"]
    hidden, inter = model_shape["H"], model_shape["I"]
    return 2 * ((hidden + q_width + 2 * kv_width) + (2 * q_width + 2 * context_len * kv_width)
                + (q_width + hidden) + 2 * (hidden + inter)
                + 3 * inter + (inter + hidden) + 6 * hidden)


def slice_peak_memory_bandwidths(slice_gb, card):
    """Peak bytes/s of each slice; raises if the card has no MIG profile of that size."""
    bad = [g for g in slice_gb if g not in MEMORY_SLICES[card]]
    if bad:
        raise KeyError(f"no MIG profile of {bad} GB on {card}")
    return [CARD_BW[card] * MEMORY_SLICES[card][g] / 8 for g in slice_gb]


def terms(model_shape, slice_bandwidths, constants, microbatch_size, context_len, is_prefill, seq_len,
          rank_slices=None):
    """Durations of one step's events, in ms: term -> (ms per layer on that rank, fixed ms).

    rank_slices: per rank, the measured constants of its MIG slice (constants["slices"][slice GB],
    see hardware_constants.py), or None / {} for the whole-GPU efficiencies x slice bandwidth."""

    NUM_OF_RANKS = len(slice_bandwidths)
    HIDDEN_SIZE = model_shape["H"]
    VOCAB_SIZE = model_shape["V"]

    MICROSECONDS_TO_SECONDS = 1e-6
    SECONDS_TO_MILLISECONDS = 1e3
    MICROSECONDS_TO_MILLISECONDS = 1e-3
    # handshake_ack_us is a round trip; one direction is half
    HANDSHAKE_ACK_HALF = 0.5  

    kernel_launch_time_in_seconds = constants["launch_us"] * MICROSECONDS_TO_SECONDS
    handshake_time_one_way_in_seconds = constants["handshake_ack_us"] * HANDSHAKE_ACK_HALF * MICROSECONDS_TO_SECONDS

    # Decode generates one taok
    tokens = microbatch_size * (seq_len if is_prefill else 1)

    bytes_per_fp16_value = 2
    activation_bytes_to_next_rank = tokens * HIDDEN_SIZE * bytes_per_fp16_value

    # PyTorch's CPU thread pool (sets from probe_constants.py): a receive copy of more values than
    # pool_min_values runs on the intra-op pool. It then costs recv_copy_pool_us more (waking the pool
    # next to busy ranks; the probe's one-copier case) and every rank's launches run
    # pool_launch_slowdown x slower. Sets without these keys: no pool term (as before).
    pooled = ("recv_copy_pool_us" in constants
              and tokens * HIDDEN_SIZE > constants.get("pool_min_values", 32768))
    if pooled:
        kernel_launch_time_in_seconds *= constants.get("pool_launch_slowdown", 1.0)
    pool_seconds = constants["recv_copy_pool_us"] * MICROSECONDS_TO_SECONDS if pooled else 0.0

    # event name -> (ms per layer, fixed ms)
    # fixed ms - cost for communication tasks
    # ms per layer - cost for computation
    output: dict[str, tuple[float, float]] = {}  
    
    for rank in range(NUM_OF_RANKS):
        memory_bandwidth_for_current_rank = slice_bandwidths[rank]
        slice_constants = rank_slices[rank] if rank_slices else {}

        is_first_rank = rank == 0
        is_last_rank = rank == NUM_OF_RANKS - 1

        if "gemm_decode_rate" in slice_constants:
            # Probe primitives (a set printed by probe_constants.py, e.g. hardware_constants "mig1"):
            # every kernel of the layer, and of the work outside it, is composed from this slice's
            # generic primitives (layer_ops). Nothing is measured or fitted per model.
            prims = slice_constants
            prefill_seq = seq_len if is_prefill else None
            layer_kernels = layer_ops.layer_ops_from_shape(model_shape, microbatch_size, context_len, prefill_seq)
            layer_us, _, kernels_per_layer, _ = layer_ops.layer_us(layer_kernels, prims, loaded=True)
            gpu_seconds_per_layer = layer_us * MICROSECONDS_TO_SECONDS

            # per microbatch outside the layers: rotary on every rank, embed on the first, the clone
            # before isend on all but the last, norm + lm_head + argmax on the last
            extras = layer_ops.extras_from_shape(model_shape, microbatch_size, context_len, prefill_seq)
            roles = (["rotary"] + ["embed"] * is_first_rank + ["clone"] * (not is_last_rank)
                     + ["head"] * is_last_rank)
            fixed_us = 0.0
            extra_kernel_launches = 0
            for role in roles:
                role_us, _, role_kernels, _ = layer_ops.layer_us(extras[role], prims, loaded=True)
                fixed_us += role_us
                extra_kernel_launches += role_kernels
            fixed = fixed_us * MICROSECONDS_TO_SECONDS
            # Probe v7 on: the GPU's idle gap between consecutive kernels, per kernel, from this
            # slice's boundary_gap ("queued": GPU side only; "live": launched as the pipeline does),
            # interpolated in log(microbatch size) between the probe's mb 1 / 8 / 32. Only with
            # constants["gap_mode"] set; sets without it compose as before.
            gap_mode = constants.get("gap_mode")
            if gap_mode and "boundary_gap" in prims:
                pts = sorted((int(m), r[f"gap_{gap_mode}_us_per_kernel"]) for m, r in prims["boundary_gap"].items())
                x = math.log(max(pts[0][0], min(pts[-1][0], microbatch_size)))
                gap_us = pts[-1][1]
                for (m0, g0), (m1, g1) in zip(pts, pts[1:]):
                    if math.log(m0) <= x <= math.log(m1):
                        gap_us = g0 + (g1 - g0) * (x - math.log(m0)) / (math.log(m1) - math.log(m0))
                        break
                gpu_seconds_per_layer += kernels_per_layer * gap_us * MICROSECONDS_TO_SECONDS
                fixed += extra_kernel_launches * gap_us * MICROSECONDS_TO_SECONDS

            # the activation's copies off and onto this slice (pinned host buffer)
            d2h_seconds = (prims["d2h_fixed_us"]
                           + activation_bytes_to_next_rank / prims["d2h_rate"] * 1e6) * MICROSECONDS_TO_SECONDS
            h2d_seconds = (prims["h2d_fixed_us"]
                           + activation_bytes_to_next_rank / prims["h2d_rate"] * 1e6) * MICROSECONDS_TO_SECONDS
        else:
            # achieved rates of this rank's slice: measured per slice when constants has them,
            # else the whole-GPU efficiencies x the slice's peak bandwidth
            weight_rate = slice_constants.get("weight_rate", memory_bandwidth_for_current_rank * constants["eff_weight"])
            act_rate = slice_constants.get("act_rate", memory_bandwidth_for_current_rank * constants["eff_act"])
            layer_fixed_seconds = slice_constants.get("layer_fixed_us", 0.0) * MICROSECONDS_TO_SECONDS
            pcie_bw = slice_constants.get("pcie_bw", constants["pcie_bw"])
            recv_h2d_bw = slice_constants.get("recv_h2d_bw", constants["recv_h2d_bw"])

            # gpu_seconds_per_layer: estimated GPU time (s) to run ONE decoder layer on this rank
            # for one microbatch = bytes moved / effective bandwidth.
            # Cg{rank} = gpu_seconds_per_layer * layers_on_rank + fixed, so this is the per-layer slope.
            # Time to move data between this slice's HBM and its compute units while one layer runs
            # (on-GPU memory traffic, not the transfer to the next rank). The layer is treated as
            # memory-bound: bytes / (slice bandwidth * measured efficiency), plus the slice's measured
            # fixed time per layer when there is one.
            # Compute (FLOP) time is not a separate term; it is absorbed in eff_weight / eff_act,
            # except in prefill on a slice with a measured prefill FLOP rate (2 FLOPs per parameter
            # per token = weight_bytes FLOPs per token): the slower of streaming the weights and the FLOPs.
            if is_prefill and "prefill_flops" in slice_constants:
                gpu_seconds_per_layer = max(weight_bytes(model_shape) / weight_rate + layer_fixed_seconds,
                                            tokens * weight_bytes(model_shape) / slice_constants["prefill_flops"])
            elif is_prefill:
                gpu_seconds_per_layer = ((weight_bytes(model_shape) + tokens * act_bytes(model_shape, seq_len))
                                         / weight_rate)
            else:
                gpu_seconds_per_layer = (weight_bytes(model_shape) / weight_rate + layer_fixed_seconds
                                         + microbatch_size * act_bytes(model_shape, context_len) / act_rate)
            # No launch time here: launches are CPU work, already in Lc (F >= A + Lc). Adding them
            # to GPU time too counted them twice.

            # launches besides the layers: rotary everywhere, embed / clone / norm+lm_head+argmax
            extra_kernel_launches = constants["rotary_kernels"] + is_first_rank + (not is_last_rank) + 4 * is_last_rank
            kernels_per_layer = constants["kernels_per_layer"]

            # their launches are in Lc only (as for the layers); GPU time below is bytes moved
            fixed = 0.0
            if is_first_rank:  # embedding gather, read + write
                fixed += 2 * tokens * HIDDEN_SIZE * bytes_per_fp16_value / act_rate
            if is_last_rank:  # norm; lm_head streams the whole weight once per microbatch; argmax
                fixed += 2 * tokens * HIDDEN_SIZE * bytes_per_fp16_value / act_rate
                fixed += ((VOCAB_SIZE * HIDDEN_SIZE * bytes_per_fp16_value
                           + microbatch_size * VOCAB_SIZE * bytes_per_fp16_value)
                          / weight_rate)
                fixed += microbatch_size * VOCAB_SIZE * bytes_per_fp16_value / act_rate
            if not is_last_rank:
                fixed += 2 * activation_bytes_to_next_rank / weight_rate  # the clone before isend

            d2h_seconds = activation_bytes_to_next_rank / pcie_bw
            h2d_seconds = activation_bytes_to_next_rank / recv_h2d_bw

        cpu_calls_outside_layers = (extra_kernel_launches + constants["timing_event_calls"]
                                    + constants["send_queue_calls"] * (not is_last_rank))

        if not is_last_rank:
            output[f"D{rank}"] = (0.0, d2h_seconds * SECONDS_TO_MILLISECONDS)
            # M: CPU memcpy of the activation from the pinned host buffer into the shared-memory slot
            output[f"M{rank}"] = (0.0, activation_bytes_to_next_rank / constants["memcpy_bw"] * SECONDS_TO_MILLISECONDS)

        if rank > 0:
            recv = (activation_bytes_to_next_rank / constants["shm_to_pinned_bw"] + pool_seconds
                    + h2d_seconds
                    + kernel_launch_time_in_seconds + handshake_time_one_way_in_seconds)
            output[f"RECV{rank}"] = (0.0, recv * SECONDS_TO_MILLISECONDS)

        # Cg = GPU compute of this rank: scales with the layers it holds, plus work done once per step
        output[f"Cg{rank}"] = (gpu_seconds_per_layer * SECONDS_TO_MILLISECONDS, fixed * SECONDS_TO_MILLISECONDS)

        # Lc = CPU time to launch this rank's kernels: per layer, plus the launches outside the layers
        output[f"Lc{rank}"] = (kernels_per_layer * kernel_launch_time_in_seconds * SECONDS_TO_MILLISECONDS,
                               cpu_calls_outside_layers * kernel_launch_time_in_seconds * SECONDS_TO_MILLISECONDS)

    # Events that exist once per step and do not depend on how many layers a rank holds,
    # so the per-layer slot is 0.0.

    # HS = the sender's half of the gloo handshake/ACK that tells the next rank the data is in SHM
    output["HS"] = (0.0, handshake_time_one_way_in_seconds * SECONDS_TO_MILLISECONDS)

    # TOK = next_tokens exchange from the last rank back to rank 0: gloo send/recv plus its D2H and H2D
    token_exchange_ms = (constants["tok_roundtrip_us"] + constants["token_staging_us"]) * MICROSECONDS_TO_MILLISECONDS
    output["TOK"] = (0.0, token_exchange_ms)

    # BARRIER = gloo barrier, once after prefill (the final decode step ends at the barrier inside END)
    output["BARRIER"] = (0.0, constants["barrier_us"] * MICROSECONDS_TO_MILLISECONDS)

    # STEP = position_ids sync at the start of a decode step (prefill has none)
    step_start_ms = 0.0 if is_prefill else constants["step_start_us"] * MICROSECONDS_TO_MILLISECONDS
    output["STEP"] = (0.0, step_start_ms)

    return output


def rank_of(term):
    """Cg3 -> 3, Lc0 -> 0; None for terms that do not scale with a layer count."""
    return int(term[2:]) if term[:2] in ("Cg", "Lc") else None


# ---------------------------------------------------------------------------
# one step as a longest-path graph
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def step_graph(num_ranks, num_microbatches, sink):
    """(edges, order): edges are (source, target, edge_terms) meaning
    t_target >= t_source + sum(edge_terms); order is a topological order of the nodes
    ('S' is time 0, 'END' is the step's end).
    Node tags: A = CPU holds input k, F = GPU done with k, X = D2H(k) landed,
    P = FLUSH(k) done (handshake k delivered to the next rank)."""
    edges = []

    def add_edge(source, target, *edge_terms):
        edges.append((source, target, edge_terms))

    for rank in range(num_ranks):
        is_last_rank = rank == num_ranks - 1
        cpu_launch_term, gpu_compute_term = f"Lc{rank}", f"Cg{rank}"
        for k in range(num_microbatches):
            cpu_node, gpu_node = ("A", rank, k), ("F", rank, k)
            if rank == 0:
                if k == 0:
                    add_edge("S", cpu_node, "STEP")
                elif k == 1:
                    add_edge(("A", 0, 0), cpu_node, cpu_launch_term)  # no FLUSH between LAUNCH(0) and LAUNCH(1)
                else:
                    add_edge(("P", 0, k - 2), cpu_node)  # LAUNCH(k) follows FLUSH(k-2)
            else:
                recv_term = f"RECV{rank}"
                add_edge(("P", rank - 1, k), cpu_node, recv_term)  # handshake k arrives, then memcpy + H2D + ACK
                if k == 0:
                    add_edge("S", cpu_node, recv_term)
                elif k == 1 or is_last_rank:
                    add_edge(("A", rank, k - 1), cpu_node, cpu_launch_term, recv_term)
                else:
                    add_edge(("P", rank, k - 2), cpu_node, recv_term)
            add_edge(cpu_node, gpu_node, gpu_compute_term)  # GPU computes k
            add_edge(cpu_node, gpu_node, cpu_launch_term)  # cannot finish before its last kernel is queued
            if k > 0:
                add_edge(("F", rank, k - 1), gpu_node, gpu_compute_term)  # in-order stream
            if is_last_rank:
                continue
            d2h_node, flush_node = ("X", rank, k), ("P", rank, k)
            add_edge(gpu_node, d2h_node, f"D{rank}")  # D2H waits for compute k
            if k > 0:
                add_edge(("X", rank, k - 1), d2h_node, f"D{rank}")
            add_edge(d2h_node, flush_node, f"M{rank}", "HS")  # FLUSH(k): wait for D2H, memcpy, handshake
            if k > 0:
                add_edge(("P", rank, k - 1), flush_node, f"M{rank}", "HS")  # flushes run in order
            if k < num_microbatches - 1:
                add_edge(("A", rank, k + 1), flush_node, cpu_launch_term, f"M{rank}", "HS")  # FLUSH(k) runs after LAUNCH(k+1)
            else:
                add_edge(cpu_node, flush_node, cpu_launch_term, f"M{rank}", "HS")
    last_cpu_node = ("A", num_ranks - 1, num_microbatches - 1)
    last_gpu_node = ("F", num_ranks - 1, num_microbatches - 1)
    if sink == "token":
        add_edge(last_gpu_node, "END", "TOK")  # the token D2H waits for the GPU
        add_edge(last_cpu_node, "END", f"Lc{num_ranks - 1}", "TOK")
    else:  # final decode step: straight to the barrier
        add_edge(last_cpu_node, "END", f"Lc{num_ranks - 1}", "BARRIER")

    phase = {"A": 0.0, "F": 0.3, "X": 0.6, "P": 2.5}
    nodes = {target for _, target, _ in edges}
    order = sorted(nodes, key=lambda node: (num_ranks, 0.0) if node == "END"
                   else (node[1], 2 * node[2] + phase[node[0]]))
    return tuple(edges), tuple(order)


@lru_cache(maxsize=None)
def _compiled(edges, order):
    """Per node in topological order, its (source, edge terms) predecessors."""
    predecessors = {}
    for source, target, edge_terms in edges:
        predecessors.setdefault(target, []).append((source, edge_terms))
    return tuple((node, tuple(predecessors[node])) for node in order)


def event_times(edges, order, term_ms):
    """Every event's earliest time (ms) given term_ms[term]: the schedule the program order runs,
    each event as soon as all its predecessors allow."""
    event_time = {"S": 0.0}
    for node, predecessors in _compiled(edges, order):
        event_time[node] = max(event_time[source] + sum(term_ms[term] for term in edge_terms)
                               for source, edge_terms in predecessors)
    return event_time


def longest_path(edges, order, term_ms):
    """Time of END given term_ms[term] in ms."""
    return event_times(edges, order, term_ms)["END"]


def step_groups(prompt_len, max_new_tokens, num_decode_samples):
    """(weight, context_len, is_prefill, sink) per step group. Decode step t sees
    prompt_len + t - 1 cached tokens. num_decode_samples=None keeps every step; else the
    max_new_tokens-1 regular decode steps become that many equal ranges, each at its
    midpoint context."""
    groups = [(1.0, prompt_len, True, "token")]
    regular_steps = list(range(1, max_new_tokens))
    if regular_steps:
        num_ranges = (len(regular_steps) if num_decode_samples is None
                      else min(num_decode_samples, len(regular_steps)))
        for j in range(num_ranges):
            first_step = regular_steps[len(regular_steps) * j // num_ranges]
            last_step = regular_steps[len(regular_steps) * (j + 1) // num_ranges - 1]
            groups.append((last_step - first_step + 1.0,
                           prompt_len + (first_step + last_step) / 2 - 1, False, "token"))
    groups.append((1.0, prompt_len + max_new_tokens - 1, False, "last"))
    return groups


def prepare(config, batch_size, microbatch_size, num_decode_samples):
    """Everything about a (batch, microbatch) pair that does not depend on the split."""
    if config.get("clock", "fast") == "schedutil" and "clock_slow_factor" not in config["constants"]:
        raise ValueError('this constants set has no CPU clock model (clock_slow_factor, '
                         'clock_busy_threshold): use clock="fast"')
    peak_memory_bandwidths = slice_peak_memory_bandwidths(config["slice_gb"], config["card"])

    model_shape = memory_limits.REFERENCE_SHAPES[config["model"]]

    num_microbatches = batch_size // microbatch_size

    groups = step_groups(config["seq_len"], config["max_new_tokens"], num_decode_samples)

    # measured constants of each rank's slice, when the constants set has them (keyed by slice GB)
    rank_slices = [config["constants"].get("slices", {}).get(gb, {}) for gb in config["slice_gb"]]

    steps = []
    for weight, context_len, is_prefill, sink in groups:
        # weight = how many real steps this group stands for; sink = which event ends the step
        step_terms = terms(model_shape, peak_memory_bandwidths, config["constants"],
                           microbatch_size, context_len, is_prefill, config["seq_len"], rank_slices)

        graph = step_graph(len(peak_memory_bandwidths), num_microbatches, sink)

        steps.append((weight, step_terms, graph))
    return steps


def num_microbatches_of(order):
    """N, read off a step graph: rank 0 has one input node per microbatch."""
    return sum(1 for node in order if node != "END" and node[0] == "A" and node[1] == 0)


def term_ms_for(layer_split, step_terms, slow=(), slow_factor=1.0):
    """Every term of one step in ms for this split; the launches (Lc) of slow ranks take
    slow_factor times as long."""
    term_ms = {}
    for name, (per_layer_ms, fixed_ms) in step_terms.items():
        rank = rank_of(name)
        ms = per_layer_ms * layer_split[rank] + fixed_ms if rank is not None else fixed_ms
        if name.startswith("Lc") and rank in slow:
            ms *= slow_factor
        term_ms[name] = ms
    return term_ms


@lru_cache(maxsize=None)
def _ends_with_token(edges):
    """True for a step that ends with the next_tokens exchange (TOK), False for the final step."""
    return any("TOK" in edge_terms for _, _, edge_terms in edges)


def cpu_busy_ms(event_time, term_ms, num_ranks, num_microbatches, constants, token):
    """Each rank's CPU busy time (ms) in one step of the given schedule: time its CPU runs (works
    or spins) rather than sleeps, read off the harness code (benchmark_pipeline_microbatching.py,
    the transport, token_exchange.py). Spins: CUDA event / stream syncs on its own GPU work
    (default flags). Sleeps: gloo waits (handshake k before RECV(k), the ACK drain, next_tokens).
      rank 0:  never sleeps from the step's start to its last flush: P_0(N-1), plus its half of the
               token copy (an H2D + sync) when the step ends with the token
      middle:  STEP + N x (RECV + Lc) + every FLUSH(k): from its start c(k) (after LAUNCH(k+1), or
               after FLUSH(N-2) for the last one) to P(k): the spin on D2H(k), memcpy, handshake
      last:    STEP + N x (RECV + Lc) + the spin on its GPU in the token's D2H + its half of the
               token copy (none in the final step, which goes straight to the barrier)
    STEP (the position_ids sync) runs on every rank; the graph has it on rank 0's path only."""
    n = num_microbatches
    step = constants["step_start_us"] * 1e-3
    tok_half = constants["token_staging_us"] * 0.5e-3 if token else 0.0
    busy = []
    for rank in range(num_ranks):
        lc = term_ms[f"Lc{rank}"]
        if rank == 0:
            busy.append(event_time[("P", 0, n - 1)] + tok_half)
            continue
        b = step + n * (term_ms[f"RECV{rank}"] + lc)
        if rank < num_ranks - 1:
            for k in range(n):
                if k < n - 1:
                    start = event_time[("A", rank, k + 1)] + lc
                else:
                    start = event_time[("P", rank, n - 2)] if n > 1 else event_time[("A", rank, 0)] + lc
                b += event_time[("P", rank, k)] - start
        elif token:
            b += max(0.0, event_time[("F", rank, n - 1)] - event_time[("A", rank, n - 1)] - lc) + tok_half
        busy.append(b)
    return busy


def busy_threshold(constants, period_ms):
    """The governor probe's busy threshold at a step period: linear between the measured periods,
    held flat outside them."""
    table = constants["clock_threshold_by_period"]
    periods = sorted(table)
    if period_ms <= periods[0]:
        return table[periods[0]]
    for low, high in zip(periods, periods[1:]):
        if period_ms <= high:
            return table[low] + (table[high] - table[low]) * (period_ms - low) / (high - low)
    return table[periods[-1]]


def probe_clock(config):
    """True when the clock rule comes from the governor probe (a probe set and clock="schedutil")."""
    return config.get("clock", "fast") == "schedutil" and "clock_threshold_by_period" in config["constants"]


def slow_ranks(layer_split, steps, config):
    """Ranks schedutil clocks down (empty unless CONFIG clock="schedutil").

    Rank s is slow when N x Cg_s, summed over the run, is below clock_busy_threshold[s] x the
    run's step time. Start with every rank fast; slow ranks lengthen the steps, so repeat with
    them slow until no new rank falls under. Ranks only ever join, so this ends within one pass
    per rank, at the smallest consistent slow set -- the one the MILP picks too."""
    if config.get("clock", "fast") != "schedutil":
        return set()
    constants = config["constants"]
    num_ranks = len(layer_split)
    if probe_clock(config):
        # a rank at full clock is slowed when its CPU busy time over the run (cpu_busy_ms, in this
        # schedule) is below the governor's threshold at the run's mean step time x the run's time
        slow = set()
        while True:
            busy, total, steps_run = [0.0] * num_ranks, 0.0, 0.0
            for weight, step_terms, (edges, order) in steps:
                term_ms = term_ms_for(layer_split, step_terms, slow, constants["clock_slow_factor"])
                times = event_times(edges, order, term_ms)
                total += weight * times["END"]
                steps_run += weight
                for rank, b in enumerate(cpu_busy_ms(times, term_ms, num_ranks, num_microbatches_of(order),
                                                     constants, _ends_with_token(edges))):
                    busy[rank] += weight * b
            threshold = busy_threshold(constants, total / steps_run)
            under = {rank for rank in range(num_ranks) if rank not in slow and busy[rank] < threshold * total}
            if not under:
                return slow
            slow |= under
    busy = [0.0] * num_ranks  # GPU time per run on each rank; does not depend on the clock
    for weight, step_terms, (_, order) in steps:
        term_ms = term_ms_for(layer_split, step_terms)
        for rank in range(num_ranks):
            busy[rank] += weight * num_microbatches_of(order) * term_ms[f"Cg{rank}"]
    slow = set()
    while True:
        total = sum(weight * longest_path(edges, order, term_ms_for(layer_split, step_terms, slow,
                                                                    constants["clock_slow_factor"]))
                    for weight, step_terms, (edges, order) in steps)
        under = {rank for rank in range(num_ranks)
                 if busy[rank] < constants["clock_busy_threshold"][rank] * total}
        if under <= slow:
            return slow
        slow |= under


def evaluate(layer_split, steps, config):
    """Latency of a split in ms, summed over the prepared steps.

    The question the whole file answers: which layer split runs fastest? This function is
    the scorer. It takes one candidate layer_split (layers per rank, e.g. [22, 11, 6, 6])
    and the steps from prepare() (the durations and graph, which do not depend on the split)
    and returns the predicted latency of the full run in ms.

    For each step group it turns the per-layer and fixed durations into milliseconds for this
    split, takes the longest path through the graph for the step time, and adds it to the
    total weighted by how many real steps the group stands for.

    Callers: predict_ms (every step), brute_force (scores every split, to check the MILP),
    validate.py / compare.py (model vs measured). The MILP in optimize_placement builds the
    same sum as constraints so the solver finds the best split without scoring each one.
    """

    # Initial Overhead: the barrier after prefill (dist.barrier())
    total = config["constants"]["barrier_us"] * 1e-3

    # ranks schedutil clocks down (none unless CONFIG clock="schedutil")
    slow = slow_ranks(layer_split, steps, config)
    slow_factor = config["constants"].get("clock_slow_factor", 1.0)

    for weight, step_terms, (edges, order) in steps:
        term_ms = term_ms_for(layer_split, step_terms, slow, slow_factor)
        total += weight * longest_path(edges, order, term_ms)

    return total


def predict_ms(config, layer_split, batch_size, microbatch_size):
    """total_latency_ms the model predicts for a split, every decode step evaluated."""
    return evaluate(layer_split, prepare(config, batch_size, microbatch_size, None), config)

def layer_bounds(config, batch_size, microbatch_size):
    """(lower, upper) layers per rank: >= 1 on every rank but the last (config
    min_last_rank_layers on the last), at most what fits in the slice's memory."""

    model_shape = memory_limits.REFERENCE_SHAPES[config["model"]]
    num_ranks = len(config["slice_gb"])

    memory_model = memory_limits.MemoryModel(model_shape, config["seq_len"], config["max_new_tokens"])

    available_memory_per_rank_mib = memory_limits.slice_mibs(config["slice_gb"])

    upper_bounds_of_layers_per_rank = []
    for rank, cap_mib in enumerate(available_memory_per_rank_mib):
        fits = memory_model.max_layers_that_can_fit(rank, num_ranks, batch_size, microbatch_size, cap_mib)
        upper_bounds_of_layers_per_rank.append(min(model_shape["L"], fits))

    
    lower_bounds_of_layers_per_rank = [1] * (num_ranks - 1) + [config["min_last_rank_layers"]]

    if (any(lower_bound > upper_bound for lower_bound, upper_bound
            in zip(lower_bounds_of_layers_per_rank, upper_bounds_of_layers_per_rank))
            or not sum(lower_bounds_of_layers_per_rank) <= model_shape["L"] <= sum(upper_bounds_of_layers_per_rank)):
        raise ValueError(f"no split of {model_shape['L']} layers fits: "
                         f"{lower_bounds_of_layers_per_rank}..{upper_bounds_of_layers_per_rank}")
    
    return lower_bounds_of_layers_per_rank, upper_bounds_of_layers_per_rank


def all_splits(total_layers, lower, upper):
    """Every split of total_layers layers within the bounds."""
    splits = []

    def place_layers(rank, layers_so_far, layers_left):
        if rank == len(lower) - 1:
            if lower[rank] <= layers_left <= upper[rank]:
                splits.append(layers_so_far + [layers_left])
            return
        for layers in range(lower[rank], min(upper[rank], layers_left) + 1):
            place_layers(rank + 1, layers_so_far + [layers], layers_left - layers)

    place_layers(0, [], total_layers)
    return splits


def optimize_placement(config, batch_size, microbatch_size):
    """(split, objective_ms) minimizing latency at one (batch, microbatch) pair. The best
    split depends on microbatch_size: more microbatches shift it towards balanced stages."""

    if probe_clock(config):  # the probe clock rule needs each split's schedule: exact search
        best, best_ms, _ = brute_force(config, batch_size, microbatch_size)
        return best, best_ms

    model_shape = memory_limits.REFERENCE_SHAPES[config["model"]]
    lower, upper = layer_bounds(config, batch_size, microbatch_size)

    steps = prepare(config, batch_size, microbatch_size, config["samples"])
    num_ranks = len(lower)

    problem = pulp.LpProblem("layer_placement", pulp.LpMinimize)
    layers_on_rank = [pulp.LpVariable(f"n{rank}", lower[rank], upper[rank], "Integer")
                      for rank in range(num_ranks)]
    problem += pulp.lpSum(layers_on_rank) == model_shape["L"]

    # CPU clock (CONFIG clock="schedutil"): slow[rank] = 1 when the rank is clocked down;
    # slow_layers[rank] = slow[rank] * layers_on_rank[rank], linearised, since a slow rank's
    # launch time is clock_slow_factor x (per layer x layers + fixed)
    clock = config.get("clock", "fast") == "schedutil"
    if clock:
        slow_factor = config["constants"]["clock_slow_factor"]
        slow = [pulp.LpVariable(f"slow{rank}", cat="Binary") for rank in range(num_ranks)]
        slow_layers = [pulp.LpVariable(f"slow_layers{rank}", 0, upper[rank]) for rank in range(num_ranks)]
        for rank in range(num_ranks):
            problem += slow_layers[rank] <= upper[rank] * slow[rank]
            problem += slow_layers[rank] <= layers_on_rank[rank]
            problem += slow_layers[rank] >= layers_on_rank[rank] - upper[rank] * (1 - slow[rank])
        busy = [[] for _ in range(num_ranks)]  # N x Cg_s per step group, weighted
        run_time = []  # step times, weighted (the objective without the barrier)
        run_time_bound = 0.0  # >= any run_time: the big M of the clock constraints

    objective = [config["constants"]["barrier_us"] * 1e-3]

    for group_index, (weight, step_terms, (edges, order)) in enumerate(steps):
        event_time = {node: pulp.LpVariable(f"t{group_index}_{i}", 0) for i, node in enumerate(order)}

        for source, target, edge_terms in edges:
            duration = sum(step_terms[term][1] for term in edge_terms)  # fixed part
            duration += pulp.lpSum(step_terms[term][0] * layers_on_rank[rank_of(term)]
                                   for term in edge_terms if rank_of(term) is not None)
            if clock:  # the extra launch time of a slow rank
                duration += pulp.lpSum((slow_factor - 1) * (step_terms[term][0] * slow_layers[rank_of(term)]
                                                           + step_terms[term][1] * slow[rank_of(term)])
                                       for term in edge_terms if term.startswith("Lc"))
            problem += event_time[target] >= duration + (event_time[source] if source != "S" else 0)

        objective.append(weight * event_time["END"])

        if clock:
            num_microbatches = num_microbatches_of(order)
            for rank in range(num_ranks):
                per_layer, fixed = step_terms[f"Cg{rank}"]
                busy[rank].append(weight * num_microbatches * (per_layer * layers_on_rank[rank] + fixed))
            run_time.append(weight * event_time["END"])
            # a path enters each node once: the sum of every node's longest incoming edge, with
            # every rank at its most layers and clocked down, bounds END for any split
            term_max = {name: (per_layer * upper[rank_of(name)] + fixed if rank_of(name) is not None else fixed)
                        * (slow_factor if name.startswith("Lc") else 1.0)
                        for name, (per_layer, fixed) in step_terms.items()}
            longest_in = {}
            for _, target, edge_terms in edges:
                longest_in[target] = max(longest_in.get(target, 0.0), sum(term_max[term] for term in edge_terms))
            run_time_bound += weight * sum(longest_in.values())

    if clock:
        # a rank may run fast only if its GPU busy time >= threshold x the run's step time
        for rank in range(num_ranks):
            threshold = config["constants"]["clock_busy_threshold"][rank]
            problem += (pulp.lpSum(busy[rank])
                        >= threshold * pulp.lpSum(run_time) - threshold * run_time_bound * slow[rank])

    problem += pulp.lpSum(objective)

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[problem.status] != "Optimal":
        raise RuntimeError(f"solver status {pulp.LpStatus[problem.status]}")
    
    return [int(round(variable.value())) for variable in layers_on_rank], pulp.value(problem.objective)

def brute_force(config, batch_size, microbatch_size):
    """(split, objective_ms, splits tried): the same objective over every split. The check
    that the MILP is exact."""
    lower, upper = layer_bounds(config, batch_size, microbatch_size)
    steps = prepare(config, batch_size, microbatch_size, config["samples"])
    splits = all_splits(memory_limits.REFERENCE_SHAPES[config["model"]]["L"], lower, upper)
    best = min(splits, key=lambda candidate: evaluate(candidate, steps, config))
    return best, evaluate(best, steps, config), len(splits)


# ---------------------------------------------------------------------------
def main():
    CONFIG = dict(
        model="vicuna_7b",  # key of memory_limits.REFERENCE_SHAPES
        slice_gb=[20, 10, 5, 5],  # rank order
        card="A100-40GB",  # or "A100-80GB"
        pairs=SWEEP_PAIRS,  # (batch, microbatch)
        seq_len=64,
        max_new_tokens=512,
        samples=8,  # decode contexts inside the MILP; predict_ms uses every step
        min_last_rank_layers=1,  # 0 allows a last rank with only norm + lm_head
        check=False,  # also run the brute force over every split (slow)
        constants=hardware_constants.CONSTANTS["a100_40gb_pcie_mig"],  # mig1/mig2; see hardware_constants.py
        clock="schedutil",  # mig1/mig2's governor: model which ranks get clocked down;
                            # "fast" keeps every rank at full clock
    )

    print(f"{CONFIG['model']} on {CONFIG['slice_gb']} ({CONFIG['card']}), clock {CONFIG['clock']}")

    for batch_size, microbatch_size in CONFIG["pairs"]:
        layer_split, _ = optimize_placement(CONFIG, batch_size, microbatch_size)
        
        line = (f"B{batch_size:<3d} mb{microbatch_size:<3d} N={batch_size // microbatch_size:<3d} -> {layer_split}  "
                f"predicted {predict_ms(CONFIG, layer_split, batch_size, microbatch_size) / 1000:8.2f} s")
        
        if CONFIG["check"]:
            best, _, tried = brute_force(CONFIG, batch_size, microbatch_size)
            line += f"  | brute force over {tried}: {best}"
        
        print(line)


if __name__ == "__main__":
    main()
