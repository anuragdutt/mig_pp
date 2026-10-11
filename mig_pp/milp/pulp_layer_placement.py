"""
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import memory_limits  # noqa: E402

warnings.filterwarnings("ignore", message=".*deprecated.*")

# Placeholders until measure_constants.py has measured them on the box.
DEFAULT_CONSTANTS = dict(
    eff_weight=0.85,  # achieved / peak bandwidth streaming weights
    eff_act=0.25,  # ... on per-token activation + KV traffic
    launch_us=12.0,  # per kernel launch + dispatch
    kernels_per_layer=16,
    pcie_bw=25e9,  # pinned D2H, bytes/s
    recv_h2d_bw=25e9,  # pinned H2D
    shm_to_pinned_bw=10e9,  # receiver host memcpy
    memcpy_bw=10e9,  # sender pinned -> SHM
    handshake_ack_us=60.0,  # one handshake + one ACK over gloo (half charged to each side)
    tok_roundtrip_us=60.0,  # next_tokens gloo send/recv
    token_staging_us=40.0,  # its D2H + H2D
    step_start_us=30.0,  # position_ids sync at the start of a decode step
    barrier_us=100.0,  # gloo barrier, once after prefill (the final one is in END)
    rotary_kernels=9,  # extra launches per microbatch on every rank
    timing_event_calls=2,  # CPU-only CUDA calls per microbatch on every rank
    send_queue_calls=2,  # ... and on sending ranks
)

CARD_BW = {"A100-40GB": 1555e9, "A100-80GB": 2039e9}  # peak HBM bytes/s
# slice GB -> GPU slices (GPCs) of 7 on each card; bandwidth is assumed to scale with them
GPCS = {
    "A100-40GB": {5: 1, 10: 2, 20: 3, 40: 7},
    "A100-80GB": {10: 1, 20: 2, 40: 3, 80: 7},
}

SWEEP_PAIRS = [(8, 4), (8, 2), (16, 8), (16, 4), (16, 2), (32, 16), (32, 8), (32, 4),
               (32, 2), (64, 32), (64, 16), (64, 8), (64, 4), (64, 2)]


# ---------------------------------------------------------------------------
# bytes (same counts as measure_constants.py, which its efficiencies are defined on)
# ---------------------------------------------------------------------------
def weight_bytes(s):
    q, kv = s["heads"] * s["hd"], s["kv"] * s["hd"]
    return (s["H"] * q + 2 * s["H"] * kv + q * s["H"] + 3 * s["H"] * s["I"]) * 2


def act_bytes(s, ctx):
    """Activation + KV traffic of one token through one layer (flash attention: no scores)."""
    q, kv, H, I = s["heads"] * s["hd"], s["kv"] * s["hd"], s["H"], s["I"]
    return 2 * ((H + q + 2 * kv) + (2 * q + 2 * ctx * kv) + (q + H) + 2 * (H + I)
                + 3 * I + (I + H) + 6 * H)


def validate_layout(slice_gb, card):
    """Peak bytes/s of each slice; raises if the card has no MIG profile of that size."""
    bad = [g for g in slice_gb if g not in GPCS[card]]
    if bad:
        raise KeyError(f"no MIG profile of {bad} GB on {card}")
    return [CARD_BW[card] * GPCS[card][g] / 7 for g in slice_gb]


# ---------------------------------------------------------------------------
# durations of one step's events, in ms: term -> (ms per layer on that rank, fixed ms)
# ---------------------------------------------------------------------------
def terms(shape, bw, k, mb, ctx, prefill, seq):
    R, H, V = len(bw), shape["H"], shape["V"]
    lu = k["launch_us"] * 1e-6
    half = k["handshake_ack_us"] * 0.5e-6
    tokens = mb * (seq if prefill else 1)
    payload = tokens * H * 2
    out = {}
    for s in range(R):
        b = bw[s]
        if prefill:
            layer = (weight_bytes(shape) + tokens * act_bytes(shape, seq)) / (b * k["eff_weight"])
        else:
            layer = (weight_bytes(shape) / (b * k["eff_weight"])
                     + mb * act_bytes(shape, ctx) / (b * k["eff_act"]))
        layer += k["kernels_per_layer"] * lu
        # launches besides the layers: rotary everywhere, embed / clone / norm+lm_head+argmax
        kernels = k["rotary_kernels"] + (s == 0) + (s < R - 1) + 4 * (s == R - 1)
        calls = kernels + k["timing_event_calls"] + k["send_queue_calls"] * (s < R - 1)
        fixed = kernels * lu
        if s == 0:  # embedding gather, read + write
            fixed += 2 * tokens * H * 2 / (b * k["eff_act"])
        if s == R - 1:  # norm; lm_head streams the whole weight once per microbatch; argmax
            fixed += 2 * tokens * H * 2 / (b * k["eff_act"])
            fixed += (V * H * 2 + mb * V * 2) / (b * k["eff_weight"])
            fixed += mb * V * 2 / (b * k["eff_act"])
        if s < R - 1:
            fixed += 2 * payload / (b * k["eff_weight"])  # the clone before isend
            out[f"D{s}"] = (0.0, payload / k["pcie_bw"] * 1e3)
            out[f"M{s}"] = (0.0, payload / k["memcpy_bw"] * 1e3)
        if s > 0:
            recv = payload / k["shm_to_pinned_bw"] + payload / k["recv_h2d_bw"] + lu + half
            out[f"RECV{s}"] = (0.0, recv * 1e3)
        out[f"Cg{s}"] = (layer * 1e3, fixed * 1e3)
        out[f"Lc{s}"] = (k["kernels_per_layer"] * lu * 1e3, calls * lu * 1e3)
    out["HS"] = (0.0, half * 1e3)
    out["TOK"] = (0.0, (k["tok_roundtrip_us"] + k["token_staging_us"]) * 1e-3)
    out["BARRIER"] = (0.0, k["barrier_us"] * 1e-3)
    out["STEP"] = (0.0, 0.0 if prefill else k["step_start_us"] * 1e-3)
    return out


def rank_of(term):
    """Cg3 -> 3, Lc0 -> 0; None for terms that do not scale with a layer count."""
    return int(term[2:]) if term[:2] in ("Cg", "Lc") else None


# ---------------------------------------------------------------------------
# one step as a longest-path graph
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def step_graph(R, N, sink):
    """(edges, order): edges are (u, v, terms) meaning t_v >= t_u + sum(terms); order is a
    topological order of the nodes ('S' is time 0, 'END' is the step's end)."""
    edges = []

    def e(u, v, *ts):
        edges.append((u, v, ts))

    for s in range(R):
        last = s == R - 1
        Lc, Cg = f"Lc{s}", f"Cg{s}"
        for k in range(N):
            A, F = ("A", s, k), ("F", s, k)
            if s == 0:
                if k == 0:
                    e("S", A, "STEP")
                elif k == 1:
                    e(("A", 0, 0), A, Lc)  # no FLUSH between LAUNCH(0) and LAUNCH(1)
                else:
                    e(("P", 0, k - 2), A)  # LAUNCH(k) follows FLUSH(k-2)
            else:
                RECV = f"RECV{s}"
                e(("P", s - 1, k), A, RECV)  # handshake k arrives, then memcpy + H2D + ACK
                if k == 0:
                    e("S", A, RECV)
                elif k == 1 or last:
                    e(("A", s, k - 1), A, Lc, RECV)
                else:
                    e(("P", s, k - 2), A, RECV)
            e(A, F, Cg)  # GPU computes k
            e(A, F, Lc)  # cannot finish before its last kernel is queued
            if k > 0:
                e(("F", s, k - 1), F, Cg)  # in-order stream
            if last:
                continue
            X, P = ("X", s, k), ("P", s, k)
            e(F, X, f"D{s}")  # D2H waits for compute k
            if k > 0:
                e(("X", s, k - 1), X, f"D{s}")
            e(X, P, f"M{s}", "HS")  # FLUSH(k): wait for D2H, memcpy, handshake
            if k > 0:
                e(("P", s, k - 1), P, f"M{s}", "HS")  # flushes run in order
            if k < N - 1:
                e(("A", s, k + 1), P, Lc, f"M{s}", "HS")  # FLUSH(k) runs after LAUNCH(k+1)
            else:
                e(A, P, Lc, f"M{s}", "HS")
    a_end, f_end = ("A", R - 1, N - 1), ("F", R - 1, N - 1)
    if sink == "token":
        e(f_end, "END", "TOK")  # the token D2H waits for the GPU
        e(a_end, "END", f"Lc{R - 1}", "TOK")
    else:  # final decode step: straight to the barrier
        e(a_end, "END", f"Lc{R - 1}", "BARRIER")

    phase = {"A": 0.0, "F": 0.3, "X": 0.6, "P": 2.5}
    nodes = {v for _, v, _ in edges}
    order = sorted(nodes, key=lambda v: (R, 0.0) if v == "END" else (v[1], 2 * v[2] + phase[v[0]]))
    return tuple(edges), tuple(order)


def longest_path(edges, order, value):
    """Time of END given value[term] in ms."""
    preds = {}
    for u, v, ts in edges:
        preds.setdefault(v, []).append((u, sum(value[t] for t in ts)))
    t = {"S": 0.0}
    for v in order:
        t[v] = max(t[u] + w for u, w in preds[v])
    return t["END"]


def step_groups(seq, T, samples):
    """(weight, ctx, prefill, sink) per step group. Decode step t sees seq + t - 1 cached
    tokens. samples=None keeps every step; else the T-1 regular decode steps become that
    many equal ranges, each at its midpoint context."""
    groups = [(1.0, seq, True, "token")]
    regular = list(range(1, T))
    if regular:
        m = len(regular) if samples is None else min(samples, len(regular))
        for j in range(m):
            lo = regular[len(regular) * j // m]
            hi = regular[len(regular) * (j + 1) // m - 1]
            groups.append((hi - lo + 1.0, seq + (lo + hi) / 2 - 1, False, "token"))
    groups.append((1.0, seq + T - 1, False, "last"))
    return groups


def prepare(cfg, batch, mb, samples):
    """Everything about a (batch, microbatch) pair that does not depend on the split."""
    bw = validate_layout(cfg["slice_gb"], cfg["card"])
    shape = memory_limits.REFERENCE_SHAPES[cfg["model"]]
    N = batch // mb
    return [(w, terms(shape, bw, cfg["constants"], mb, ctx, pre, cfg["seq_len"]),
             step_graph(len(bw), N, sink))
            for w, ctx, pre, sink in step_groups(cfg["seq_len"], cfg["max_new_tokens"], samples)]


def evaluate(split, steps, cfg):
    """Latency of a split in ms, summed over the prepared steps."""
    total = cfg["constants"]["barrier_us"] * 1e-3  # the barrier after prefill
    for w, tm, (edges, order) in steps:
        value = {}
        for name, (per_layer, fixed) in tm.items():
            r = rank_of(name)
            value[name] = per_layer * split[r] + fixed if r is not None else fixed
        total += w * longest_path(edges, order, value)
    return total


def predict_ms(cfg, split, batch, mb):
    """total_latency_ms the model predicts for a split, every decode step evaluated."""
    return evaluate(split, prepare(cfg, batch, mb, None), cfg)


# ---------------------------------------------------------------------------
# split space and optimizer
# ---------------------------------------------------------------------------
def layer_bounds(cfg, batch, mb):
    """(lower, upper) layers per rank: >= 1 on every rank but the last (cfg
    min_last_rank_layers on the last), at most what fits in the slice's memory."""
    shape = memory_limits.REFERENCE_SHAPES[cfg["model"]]
    R = len(cfg["slice_gb"])
    mm = memory_limits.MemoryModel(shape, cfg["seq_len"], cfg["max_new_tokens"])
    upper = [min(shape["L"], mm.max_layers(r, R, batch, mb, cap))
             for r, cap in enumerate(memory_limits.slice_mibs(cfg["slice_gb"]))]
    lower = [1] * (R - 1) + [cfg["min_last_rank_layers"]]
    if any(lo > hi for lo, hi in zip(lower, upper)) or not sum(lower) <= shape["L"] <= sum(upper):
        raise ValueError(f"no split of {shape['L']} layers fits: {lower}..{upper}")
    return lower, upper


def all_splits(L, lower, upper):
    """Every split of L layers within the bounds."""
    out = []

    def rec(s, acc, left):
        if s == len(lower) - 1:
            if lower[s] <= left <= upper[s]:
                out.append(acc + [left])
            return
        for n in range(lower[s], min(upper[s], left) + 1):
            rec(s + 1, acc + [n], left - n)

    rec(0, [], L)
    return out


def optimize_placement(cfg, batch, mb):
    """(split, objective_ms) minimizing latency at one (batch, microbatch) pair. The best
    split depends on mb: more microbatches shift it towards balanced stages."""
    import pulp

    shape = memory_limits.REFERENCE_SHAPES[cfg["model"]]
    lower, upper = layer_bounds(cfg, batch, mb)
    steps = prepare(cfg, batch, mb, cfg["samples"])
    R = len(lower)

    prob = pulp.LpProblem("layer_placement", pulp.LpMinimize)
    n = [pulp.LpVariable(f"n{s}", lower[s], upper[s], "Integer") for s in range(R)]
    prob += pulp.lpSum(n) == shape["L"]

    objective = [cfg["constants"]["barrier_us"] * 1e-3]
    for g, (w, tm, (edges, order)) in enumerate(steps):
        t = {v: pulp.LpVariable(f"t{g}_{i}", 0) for i, v in enumerate(order)}
        for u, v, ts in edges:
            rhs = sum(tm[x][1] for x in ts)  # fixed part
            rhs += pulp.lpSum(tm[x][0] * n[rank_of(x)] for x in ts if rank_of(x) is not None)
            prob += t[v] >= rhs + (t[u] if u != "S" else 0)
        objective.append(w * t["END"])
    prob += pulp.lpSum(objective)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"solver status {pulp.LpStatus[prob.status]}")
    return [int(round(v.value())) for v in n], pulp.value(prob.objective)


def brute_force(cfg, batch, mb):
    """(split, objective_ms, splits tried): the same objective over every split. The check
    that the MILP is exact."""
    lower, upper = layer_bounds(cfg, batch, mb)
    steps = prepare(cfg, batch, mb, cfg["samples"])
    splits = all_splits(memory_limits.REFERENCE_SHAPES[cfg["model"]]["L"], lower, upper)
    best = min(splits, key=lambda sp: evaluate(sp, steps, cfg))
    return best, evaluate(best, steps, cfg), len(splits)


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
    for batch, mb in CONFIG["pairs"]:
        split, _ = optimize_placement(CONFIG, batch, mb)
        line = (f"B{batch:<3d} mb{mb:<3d} N={batch // mb:<3d} -> {split}  "
                f"predicted {predict_ms(CONFIG, split, batch, mb) / 1000:8.2f} s")
        if CONFIG["check"]:
            best, _, tried = brute_force(CONFIG, batch, mb)
            line += f"  | brute force over {tried}: {best}"
        print(line)


if __name__ == "__main__":
    main()
