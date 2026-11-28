"""
P3 Run B (Mistral-7B on mig5): the split lists for step 2 (refinement) and step 3 (confirmation),
built from measured data by the rules fixed in TO-DO.md on 2026-11-27, before any Run B data was
opened. Writes a split-list module (ALL = {pair: [splits]}) for clock_study.sh (SPLIT_MODULE,
VALIDATE_KIND="all") and prints, per pair, counts and the split lists only: never a latency.

Step 2, per pair, candidates not measured yet that fit in memory (each unique split once):
  (a) every one-layer move and the rank 2 <-> rank 3 swap of each split within 3% of the pair's best
      in step 1;
  (b) every one-layer move of the frozen graph ILP pick (k = 1);
  (a) + (b) capped at 25, sorted by (fastest step-1 latency among the candidate's parents, then the
  split as a tuple); then (c) the frozen baselines not measured yet, always.
Step 3, per pair: every split within 3% of the best after steps 1-2 (median over its runs), plus the
frozen graph pick, the GPipe pick and the rule baselines that were measured. The same set is the
3rd-run set when runb_noise_check.py flags a pair.

Edit CONFIG in main(), then:  .venv/bin/python runb_lists.py
"""

import csv
import glob
import json
import os
import statistics

import memory_limits

PAIRS = [(8, 4), (16, 8), (32, 16), (8, 2), (16, 4), (32, 8), (16, 2), (32, 4), (32, 2)]
SLICE_GB = [20, 10, 5, 5]
LAYERS = 32

# Frozen 2026-11-26 (mig_pp/milp/SCORING.md, "Pre-registered: Mistral-7B"; tests/test_frozen_picks.py).
GRAPH_PICK = {(8, 4): (27, 3, 1, 1), (16, 8): (27, 3, 1, 1), (32, 16): (26, 4, 1, 1),
              (8, 2): (19, 9, 3, 1), (16, 4): (19, 9, 3, 1), (32, 8): (20, 8, 3, 1),
              (16, 2): (16, 9, 4, 3), (32, 4): (17, 9, 4, 2), (32, 2): (16, 9, 4, 3)}
GPIPE_PICK = {(8, 4): (20, 10, 1, 1), (16, 8): (20, 10, 1, 1), (32, 16): (20, 10, 1, 1),
              (8, 2): (16, 9, 3, 4), (16, 4): (18, 10, 1, 3), (32, 8): (18, 10, 3, 1),
              (16, 2): (16, 9, 3, 4), (32, 4): (16, 9, 3, 4), (32, 2): (16, 9, 3, 4)}
RULE_BASELINES = [(8, 8, 8, 8), (14, 9, 5, 4), (16, 8, 4, 4)]  # even, GPC 3:2:1:1, memory 4:2:1:1


def last_run_block(export):
    """[(step, run dir)] of the last '# run' block in an export's investigation_runs.txt."""
    blocks = []
    with open(os.path.join(os.path.expanduser(export), "investigation_runs.txt")) as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("# run"):
            blocks.append([])
        elif line.strip() and not line.startswith("#") and blocks:
            step, run_dir = line.split()[:2]
            blocks[-1].append((step, run_dir))
    if not blocks or not blocks[-1]:
        raise SystemExit(f"{export}: no steps after the last '# run' line")
    return blocks[-1]


def read_rows(export, run_dir):
    """All result rows of one oracle (any model folder under gpu0)."""
    files = glob.glob(os.path.join(os.path.expanduser(export), run_dir, "gpu*", "*", "mig_benchmark_results.csv"))
    if len(files) != 1:
        raise SystemExit(f"{export}/{run_dir}: expected one mig_benchmark_results.csv, found {len(files)}")
    with open(files[0]) as f:
        return list(csv.DictReader(f))


def prefix_problems(export, prefix):
    """Step names of the export's last run block that do not start with prefix (wrong phase)."""
    return [f"{export}: step {step} does not start with {prefix}"
            for step, _ in last_run_block(export) if not step.startswith(prefix)]


def manifest_problems(export, expected_pairs, prefix, passes=1):
    """Problems with an export's last run block: every step name starts with prefix (e.g.
    "mistral7b_s3_"), no run directory twice, exactly `passes` oracles per expected pair (1 for the
    Run B steps, 2 for the Vicuna top-up)."""
    problems, dirs, per_pair = [], [], {}
    for step, run_dir in last_run_block(export):
        if not step.startswith(prefix):
            problems.append(f"step {step} does not start with {prefix}")
        dirs.append(run_dir)
        pairs = {(int(r["batch_size"]), int(r["microbatch_size"])) for r in read_rows(export, run_dir)}
        for pair in pairs:
            per_pair[pair] = per_pair.get(pair, 0) + 1
    problems += [f"run directory {d} listed {dirs.count(d)} times" for d in sorted(set(dirs)) if dirs.count(d) > 1]
    problems += [f"pair {p}: {per_pair.get(p, 0)} oracles, want {passes}" for p in expected_pairs
                 if per_pair.get(p, 0) != passes]
    problems += [f"pair {p}: not expected" for p in per_pair if p not in expected_pairs]
    return problems


def load(export):
    """{pair: {split: [latency ms]}} from the 'ok' rows of the export's last run block."""
    out = {}
    for _, run_dir in last_run_block(export):
        for r in read_rows(export, run_dir):
            if r["status"] != "ok":
                continue
            pair = (int(r["batch_size"]), int(r["microbatch_size"]))
            out.setdefault(pair, {}).setdefault(tuple(json.loads(r["split"])), []).append(float(r["total_latency_ms"]))
    return out


def merge(*datasets):
    out = {}
    for d in datasets:
        for pair, splits in d.items():
            for s, lats in splits.items():
                out.setdefault(pair, {}).setdefault(s, []).extend(lats)
    return out


def one_layer_moves(split):
    """Every split one layer away (one layer from rank i to rank j), every rank >= 1 layer."""
    out = []
    for i in range(len(split)):
        for j in range(len(split)):
            if i != j and split[i] > 1:
                s = list(split)
                s[i] -= 1
                s[j] += 1
                out.append(tuple(s))
    return out


def swap_23(split):
    return (split[0], split[1], split[3], split[2])


def fits(mm, split, pair):
    return all(x >= 1 for x in split) and sum(split) == LAYERS and not mm.oom_ranks(list(split), *pair, SLICE_GB)


def group(med, pct):
    """Splits within pct% of the best median."""
    best = min(med.values())
    return [s for s, v in med.items() if v <= best * (1 + pct / 100)]


def step2_list(mm, measured, pair, pct, cap):
    med = {s: statistics.median(v) for s, v in measured.items()}
    parents = {}  # candidate -> fastest parent latency
    for p in group(med, pct):
        for c in one_layer_moves(p) + [swap_23(p)]:
            parents[c] = min(parents.get(c, float("inf")), med[p])
    pick = GRAPH_PICK[pair]
    for c in one_layer_moves(pick):
        parents[c] = min(parents.get(c, float("inf")), med.get(pick, float("inf")))
    cands = sorted((v, c) for c, v in parents.items() if c not in med and fits(mm, c, pair))
    chosen = [c for _, c in cands[:cap]]
    baselines = [s for s in RULE_BASELINES + [GPIPE_PICK[pair]]
                 if s not in med and s not in chosen and fits(mm, s, pair)]
    for s in baselines:
        if s not in chosen:
            chosen.append(s)
    return chosen, len(cands), baselines


def step3_list(mm, measured, pair, pct):
    med = {s: statistics.median(v) for s, v in measured.items()}
    out = sorted(group(med, pct))
    for s in [GRAPH_PICK[pair], GPIPE_PICK[pair]] + RULE_BASELINES:
        if s in med and s not in out:
            out.append(s)
    return out


def write_module(path, lists, header):
    with open(path, "w") as f:
        f.write(f'"""{header}"""\n\n')
        f.write("ALL = {\n")
        for pair in PAIRS:
            f.write(f"    {pair}: {[list(s) for s in lists[pair]]},\n")
        f.write("}\n")


def main(config=None):
    CONFIG = config or dict(
        step=2,  # 2: refinement list from step 1; 3: confirmation list from steps 1 + 2
        step1="~/Desktop/AP/logs/mig5_p3_runB_step1",  # export of Run B step 1
        step2="~/Desktop/AP/logs/mig5_p3_runB_step2",  # export of Run B step 2 (step 3 only)
        step1_prefix="mistral7b_clock_",  # step names of each phase: the exports must be the right ones
        step2_prefix="mistral7b_s2_",
        group_pct=3.0,
        cap=25,
        out_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    c = CONFIG
    mm = memory_limits.MemoryModel(memory_limits.REFERENCE_SHAPES["mistral_7b"])
    # No list from the wrong inputs: step 1 (and step 2) must be their own phase, one oracle per pair,
    # different folders; for step 3 every frozen pick and every baseline that fits must be measured.
    problems = manifest_problems(c["step1"], PAIRS, c["step1_prefix"])
    if c["step"] == 3:
        problems += manifest_problems(c["step2"], PAIRS, c["step2_prefix"])
        if os.path.realpath(os.path.expanduser(c["step1"])) == os.path.realpath(os.path.expanduser(c["step2"])):
            problems.append("step 1 and step 2 are the same export folder")
    data = load(c["step1"]) if c["step"] == 2 else merge(load(c["step1"]), load(c["step2"]))
    problems += [f"pair {p}: no ok rows" for p in PAIRS if p not in data]
    if c["step"] == 3:
        for pair in PAIRS:
            need = [s for s in [GRAPH_PICK[pair], GPIPE_PICK[pair]] + RULE_BASELINES if fits(mm, s, pair)]
            absent = [list(s) for s in need if s not in data.get(pair, {})]
            if absent:
                problems.append(f"pair {pair}: frozen picks / baselines never measured: {absent}")
    if problems:
        raise SystemExit("NO LIST WRITTEN, inputs are not right:\n  " + "\n  ".join(problems))
    lists = {}
    for pair in PAIRS:
        if c["step"] == 2:
            lists[pair], n_cands, baselines = step2_list(mm, data[pair], pair, c["group_pct"], c["cap"])
            print(f"{pair}: {len(lists[pair])} splits ({min(n_cands, c['cap'])} of {n_cands} refinement "
                  f"candidates, {len(baselines)} baselines): {[list(s) for s in lists[pair]]}")
        else:
            lists[pair] = step3_list(mm, data[pair], pair, c["group_pct"])
            print(f"{pair}: {len(lists[pair])} splits: {[list(s) for s in lists[pair]]}")
    name = f"mistral7b_step{c['step']}_split_order"
    path = os.path.join(c["out_dir"], name + ".py")
    write_module(path, lists, f"Run B step {c['step']} split lists, written by runb_lists.py (rules: TO-DO.md).")
    total = sum(len(v) for v in lists.values())
    print(f"\nwrote {path}: {total} splits over {len(PAIRS)} pairs. clock_study.sh: SPLIT_MODULE=\"{name}\", "
          f"VALIDATE_KIND=\"all\", EXPECT_SPLITS=\"\" (lists differ per pair), EXPECT_LAYERS=32")


if __name__ == "__main__":
    main()
