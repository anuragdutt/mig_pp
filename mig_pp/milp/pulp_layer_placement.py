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

Left out: copy engines / PCIe shared between slices, CPU cores shared between lanes,
cold start of each config (fresh processes), and anything but MIG_LOG_LEVEL=summary.
"""

import os
import sys
import warnings
from functools import lru_cache
import pulp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import memory_limits  # noqa: E402

warnings.filterwarnings("ignore", message=".*deprecated.*")

# Measured by measure_constants.py on a whole A100 80GB PCIe (a100-05, job 2302084,
# Vicuna-7B config, torch 2.14.1+cu126). eff_act is the launch-free value, matching terms(),
# which keeps launches out of a layer's GPU time.
DEFAULT_CONSTANTS = dict(
    eff_weight=0.769,  # achieved / peak bandwidth streaming weights
    eff_act=0.125,  # ... on per-token activation + KV traffic (launch-free form)
    launch_us=22.6,  # CPU dispatch per kernel inside a decode layer call
    kernels_per_layer=40,
    pcie_bw=1.33e9,  # pinned D2H, bytes/s, at the mb-8 decode activation (64 KiB)
    recv_h2d_bw=2.11e9,  # pinned H2D, same size
    shm_to_pinned_bw=2.18e10,  # receiver host memcpy
    memcpy_bw=2.64e10,  # sender pinned -> SHM
    handshake_ack_us=70.5,  # one handshake + one ACK over gloo (half charged to each side)
    tok_roundtrip_us=37.1,  # next_tokens gloo send/recv
    token_staging_us=47.6,  # its D2H + H2D
    step_start_us=73.4,  # position_ids sync at the start of a decode step
    barrier_us=99.9,  # gloo barrier, once after prefill (the final one is in END)
    rotary_kernels=9,  # extra launches per microbatch on every rank
    timing_event_calls=2,  # CPU-only CUDA calls per microbatch on every rank (not measured)
    send_queue_calls=2,  # ... and on sending ranks (not measured)
)

# peak HBM bytes/s
CARD_BW = {"A100-40GB": 1555e9, "A100-80GB": 2039e9}  

# slice GB -> memory slices of 8 on each card. NVIDIA MIG docs: each memory slice is 1/8 of
# the card's memory capacity AND bandwidth (3g.40gb = 4/8 memory, 3/7 SMs), so bandwidth
# follows memory slices, not GPCs.
MEMORY_SLICES = {
    "A100-40GB": {5: 1, 10: 2, 20: 4, 40: 8},
    "A100-80GB": {10: 1, 20: 2, 40: 4, 80: 8},
}

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


def terms(model_shape, slice_bandwidths, constants, microbatch_size, context_len, is_prefill, seq_len):
    """Durations of one step's events, in ms: term -> (ms per layer on that rank, fixed ms).     """

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

    # event name -> (ms per layer, fixed ms)
    # fixed ms - cost for communication tasks
    # ms per layer - cost for computation
    output: dict[str, tuple[float, float]] = {}  
    
    for rank in range(NUM_OF_RANKS):
        memory_bandwidth_for_current_rank = slice_bandwidths[rank]

        is_first_rank = rank == 0
        is_last_rank = rank == NUM_OF_RANKS - 1

        # gpu_seconds_per_layer: estimated GPU time (s) to run ONE decoder layer on this rank
        # for one microbatch = bytes moved / effective bandwidth.
        # Cg{rank} = gpu_seconds_per_layer * layers_on_rank + fixed, so this is the per-layer slope.]
        # Time to move data between this slice's HBM and its compute units while one layer runs
        # (on-GPU memory traffic, not the transfer to the next rank). The layer is treated as
        # memory-bound: bytes / (slice bandwidth * measured efficiency). 
        # Compute (FLOP) time is not a separate term; it is absorbed in eff_weight / eff_act.
        if is_prefill:
            gpu_seconds_per_layer = ((weight_bytes(model_shape) + tokens * act_bytes(model_shape, seq_len))
                     / (memory_bandwidth_for_current_rank * constants["eff_weight"]))
        else:
            gpu_seconds_per_layer = (weight_bytes(model_shape) / (memory_bandwidth_for_current_rank * constants["eff_weight"])
                     + microbatch_size * act_bytes(model_shape, context_len) / (memory_bandwidth_for_current_rank * constants["eff_act"]))
        # No launch time here: launches are CPU work, already in Lc (F >= A + Lc). Adding them
        # to GPU time too counted them twice.

        # launches besides the layers: rotary everywhere, embed / clone / norm+lm_head+argmax
        extra_kernel_launches = constants["rotary_kernels"] + is_first_rank + (not is_last_rank) + 4 * is_last_rank
        
        cpu_calls_outside_layers = (extra_kernel_launches + constants["timing_event_calls"]
                 + constants["send_queue_calls"] * (not is_last_rank))
        
        # their launches are in Lc only (as for the layers); GPU time below is bytes moved
        fixed = 0.0
        
        if is_first_rank:  # embedding gather, read + write
            fixed += 2 * tokens * HIDDEN_SIZE * bytes_per_fp16_value / (memory_bandwidth_for_current_rank * constants["eff_act"])
            
        if is_last_rank:  # norm; lm_head streams the whole weight once per microbatch; argmax
            fixed += 2 * tokens * HIDDEN_SIZE * bytes_per_fp16_value / (memory_bandwidth_for_current_rank * constants["eff_act"])
            fixed += ((VOCAB_SIZE * HIDDEN_SIZE * bytes_per_fp16_value
                       + microbatch_size * VOCAB_SIZE * bytes_per_fp16_value)
                      / (memory_bandwidth_for_current_rank * constants["eff_weight"]))
            fixed += microbatch_size * VOCAB_SIZE * bytes_per_fp16_value / (memory_bandwidth_for_current_rank * constants["eff_act"])

        if not is_last_rank:
            fixed += 2 * activation_bytes_to_next_rank / (memory_bandwidth_for_current_rank * constants["eff_weight"])  # the clone before isend
            
            output[f"D{rank}"] = (0.0, activation_bytes_to_next_rank / constants["pcie_bw"] * SECONDS_TO_MILLISECONDS)
            # M: CPU memcpy of the activation from the pinned host buffer into the shared-memory slot
            output[f"M{rank}"] = (0.0, activation_bytes_to_next_rank / constants["memcpy_bw"] * SECONDS_TO_MILLISECONDS)

        if rank > 0:
            recv = (activation_bytes_to_next_rank / constants["shm_to_pinned_bw"]
                    + activation_bytes_to_next_rank / constants["recv_h2d_bw"]
                    + kernel_launch_time_in_seconds + handshake_time_one_way_in_seconds)
            
            output[f"RECV{rank}"] = (0.0, recv * SECONDS_TO_MILLISECONDS)

        # Cg = GPU compute of this rank: scales with the layers it holds, plus work done once per step
        gpu_ms_per_layer = gpu_seconds_per_layer * SECONDS_TO_MILLISECONDS
        gpu_fixed_ms = fixed * SECONDS_TO_MILLISECONDS
        output[f"Cg{rank}"] = (gpu_ms_per_layer, gpu_fixed_ms)

        # Lc = CPU time to launch this rank's kernels: per layer, plus the launches outside the layers
        cpu_launch_ms_per_layer = (constants["kernels_per_layer"] * kernel_launch_time_in_seconds
                                   * SECONDS_TO_MILLISECONDS)
        cpu_launch_fixed_ms = cpu_calls_outside_layers * kernel_launch_time_in_seconds * SECONDS_TO_MILLISECONDS
        output[f"Lc{rank}"] = (cpu_launch_ms_per_layer, cpu_launch_fixed_ms)

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


def longest_path(edges, order, term_ms):
    """Time of END given term_ms[term] in ms."""
    predecessors = {}
    for source, target, edge_terms in edges:
        predecessors.setdefault(target, []).append((source, sum(term_ms[term] for term in edge_terms)))
    event_time = {"S": 0.0}
    for node in order:
        event_time[node] = max(event_time[source] + duration for source, duration in predecessors[node])
    return event_time["END"]


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
    peak_memory_bandwidths = slice_peak_memory_bandwidths(config["slice_gb"], config["card"])

    model_shape = memory_limits.REFERENCE_SHAPES[config["model"]]

    num_microbatches = batch_size // microbatch_size

    groups = step_groups(config["seq_len"], config["max_new_tokens"], num_decode_samples)

    steps = []
    for weight, context_len, is_prefill, sink in groups:
        # weight = how many real steps this group stands for; sink = which event ends the step
        step_terms = terms(model_shape, peak_memory_bandwidths, config["constants"],
                           microbatch_size, context_len, is_prefill, config["seq_len"])

        graph = step_graph(len(peak_memory_bandwidths), num_microbatches, sink)

        steps.append((weight, step_terms, graph))
    return steps


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

    for weight, step_terms, (edges, order) in steps:
        term_ms = {}

        for name, (per_layer_ms, fixed_ms) in step_terms.items():
            rank = rank_of(name)
            term_ms[name] = per_layer_ms * layer_split[rank] + fixed_ms if rank is not None else fixed_ms
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

    model_shape = memory_limits.REFERENCE_SHAPES[config["model"]]
    lower, upper = layer_bounds(config, batch_size, microbatch_size)

    steps = prepare(config, batch_size, microbatch_size, config["samples"])
    num_ranks = len(lower)

    problem = pulp.LpProblem("layer_placement", pulp.LpMinimize)
    layers_on_rank = [pulp.LpVariable(f"n{rank}", lower[rank], upper[rank], "Integer")
                      for rank in range(num_ranks)]
    problem += pulp.lpSum(layers_on_rank) == model_shape["L"]

    objective = [config["constants"]["barrier_us"] * 1e-3]

    for group_index, (weight, step_terms, (edges, order)) in enumerate(steps):
        event_time = {node: pulp.LpVariable(f"t{group_index}_{i}", 0) for i, node in enumerate(order)}

        for source, target, edge_terms in edges:
            duration = sum(step_terms[term][1] for term in edge_terms)  # fixed part
            duration += pulp.lpSum(step_terms[term][0] * layers_on_rank[rank_of(term)]
                                   for term in edge_terms if rank_of(term) is not None)
            problem += event_time[target] >= duration + (event_time[source] if source != "S" else 0)

        objective.append(weight * event_time["END"])

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
        constants=DEFAULT_CONSTANTS,
    )

    print(f"{CONFIG['model']} on {CONFIG['slice_gb']} ({CONFIG['card']})")

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
