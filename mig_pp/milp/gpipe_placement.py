"""
GPipe layer placement: the same question as pulp_layer_placement.py (how many layers per
MIG slice), answered with the textbook GPipe pipeline law instead of the harness graph.

Edit CONFIG in main() and run:  python3 gpipe_placement.py

Ported from pulp_layer_placement_v2.py (test/measuring-constants, a5bd130). Kept from v2:
the GPipe step law. Changed: every input comes from pulp_layer_placement.py -- the same
measured constants, slice bandwidths, per-stage terms and memory bounds -- so the two
files differ ONLY in how stage times are combined into a step time. v2's own placeholder
constants and its "minimise the slowest stage at mb=1" objective (one split per batch,
whatever the microbatch count) are gone; this minimises the full GPipe time at the
exact (batch, microbatch) pair.

GPipe LAW, per step, with C_s = time stage s spends on one microbatch:

    T_step = SUM_s C_s + (N - 1) * MAX_s C_s + TOK
             |-- fill --|  |-- steady state --|

    C_s = max(Cg_s, Lc_s) + outbound transport (D + M + HS) + inbound RECV
          (GPU time and CPU queueing overlap, so a stage takes the longer of the two)

What the law assumes and the harness does not do: a stage hands microbatch k on the
moment it is done (the harness publishes one behind, after k+1 arrives), and the pipeline
spends most of its time in steady flow (with N = 2 it is all fill and drain).

MILP: integer layers n_s; per step group one variable C_s per stage and one z >= every
C_s (the max); minimise the weighted sum of SUM_s C_s + (N-1) z + TOK. The objective only
grows with C_s and z, so at the optimum they equal the max()es. Equals brute force.
"""

import pulp

import mig_pp.milp.graph_placement as plp
import mig_pp.milp.hardware_constants as hardware_constants


def stage_costs(layer_split, step_terms):
    """C_s in ms for every stage of one step."""
    num_ranks = len(layer_split)

    def ms(name):
        per_layer, fixed = step_terms[name]
        rank = plp.rank_of(name)
        return per_layer * layer_split[rank] + fixed if rank is not None else fixed

    costs = []
    for rank in range(num_ranks):
        cost = max(ms(f"Cg{rank}"), ms(f"Lc{rank}"))
        if rank < num_ranks - 1:
            cost += ms(f"D{rank}") + ms(f"M{rank}") + ms("HS")
        if rank > 0:
            cost += ms(f"RECV{rank}")
        costs.append(cost)
    return costs


def evaluate(layer_split, steps, num_microbatches, config):
    """GPipe latency of a split in ms, summed over the prepared steps."""
    total = config["constants"]["barrier_us"] * 1e-3
    for weight, step_terms, _ in steps:
        costs = stage_costs(layer_split, step_terms)
        total += weight * (sum(costs) + (num_microbatches - 1) * max(costs) + step_terms["TOK"][1])
    return total


def predict_ms(config, layer_split, batch_size, microbatch_size):
    """total_latency_ms GPipe predicts for a split, every decode step evaluated."""
    steps = plp.prepare(config, batch_size, microbatch_size, None)
    return evaluate(layer_split, steps, batch_size // microbatch_size, config)


def optimize_placement(config, batch_size, microbatch_size):
    """(split, objective_ms) minimising the GPipe latency at one (batch, microbatch) pair."""
    model_shape = plp.memory_limits.REFERENCE_SHAPES[config["model"]]
    lower, upper = plp.layer_bounds(config, batch_size, microbatch_size)
    steps = plp.prepare(config, batch_size, microbatch_size, config["samples"])
    num_ranks = len(lower)
    num_microbatches = batch_size // microbatch_size

    problem = pulp.LpProblem("gpipe_placement", pulp.LpMinimize)
    layers_on_rank = [pulp.LpVariable(f"n{rank}", lower[rank], upper[rank], "Integer")
                      for rank in range(num_ranks)]
    problem += pulp.lpSum(layers_on_rank) == model_shape["L"]

    def linear(name, step_terms):  # a term's ms as a linear expression in the layer counts
        per_layer, fixed = step_terms[name]
        rank = plp.rank_of(name)
        return per_layer * layers_on_rank[rank] + fixed if rank is not None else fixed

    objective = [config["constants"]["barrier_us"] * 1e-3]
    for group_index, (weight, step_terms, _) in enumerate(steps):
        stage = [pulp.LpVariable(f"c{group_index}_{rank}", 0) for rank in range(num_ranks)]
        slowest = pulp.LpVariable(f"z{group_index}", 0)
        for rank in range(num_ranks):
            transport = 0.0
            if rank < num_ranks - 1:
                transport += step_terms[f"D{rank}"][1] + step_terms[f"M{rank}"][1] + step_terms["HS"][1]
            if rank > 0:
                transport += step_terms[f"RECV{rank}"][1]
            problem += stage[rank] >= linear(f"Cg{rank}", step_terms) + transport  # C_s >= GPU + transport
            problem += stage[rank] >= linear(f"Lc{rank}", step_terms) + transport  # C_s >= CPU + transport
            problem += slowest >= stage[rank]  # z >= every C_s
        objective.append(weight * (pulp.lpSum(stage) + (num_microbatches - 1) * slowest
                                   + step_terms["TOK"][1]))
    problem += pulp.lpSum(objective)

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[problem.status] != "Optimal":
        raise RuntimeError(f"solver status {pulp.LpStatus[problem.status]}")
    return [int(round(variable.value())) for variable in layers_on_rank], pulp.value(problem.objective)


def brute_force(config, batch_size, microbatch_size):
    """(split, objective_ms, splits tried): the same objective over every split."""
    lower, upper = plp.layer_bounds(config, batch_size, microbatch_size)
    steps = plp.prepare(config, batch_size, microbatch_size, config["samples"])
    num_microbatches = batch_size // microbatch_size
    splits = plp.all_splits(plp.memory_limits.REFERENCE_SHAPES[config["model"]]["L"], lower, upper)
    best = min(splits, key=lambda candidate: evaluate(candidate, steps, num_microbatches, config))
    return best, evaluate(best, steps, num_microbatches, config), len(splits)


# ---------------------------------------------------------------------------
def main():
    CONFIG = dict(
        model="vicuna_7b",  # key of memory_limits.REFERENCE_SHAPES
        slice_gb=[20, 10, 5, 5],  # rank order
        card="A100-40GB",  # or "A100-80GB"
        pairs=plp.SWEEP_PAIRS,  # (batch, microbatch)
        seq_len=64,
        max_new_tokens=512,
        samples=8,  # decode contexts inside the MILP; predict_ms uses every step
        min_last_rank_layers=1,  # 0 allows a last rank with only norm + lm_head
        check=False,  # also run the brute force over every split (slow)
        constants=hardware_constants.CONSTANTS["a100_80gb_pcie_whole_gpu"],  # see hardware_constants.py
    )

    print(f"GPipe: {CONFIG['model']} on {CONFIG['slice_gb']} ({CONFIG['card']})")

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
