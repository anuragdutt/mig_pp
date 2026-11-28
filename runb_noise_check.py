"""
Noise check after Run B step 3: per pair, the repeat difference of every split in the confirmation set
between its earlier run (step 1 or 2) and its step-3 run, as aggregates only: no split names, no
latencies. Repeat difference = 100 x |t2 - t1| / ((t1 + t2) / 2) (order-free). A pair gets a 3rd run
of the same confirmation set if its median difference > 0.5% or any split > 2% (TO-DO.md, 2026-11-27).

Writes mistral7b_step3b_split_order.py (only the flagged pairs, each with its step-3 confirmation
set) for clock_study.sh's runB_step3_rerun preset; nothing when no pair is flagged.

Edit CONFIG in main(), then:  .venv/bin/python runb_noise_check.py
"""

import importlib
import os
import statistics

import runb_lists as rl


def repeat_diff(t1, t2):
    return 100 * abs(t2 - t1) / ((t1 + t2) / 2)


def pair_stats(first, second):
    """first / second: {split: latency}. Aggregates over the splits measured in both."""
    d = sorted(repeat_diff(first[s], second[s]) for s in second if s in first)
    if not d:
        return None
    return dict(n=len(d), median=statistics.median(d), p90=d[min(len(d) - 1, int(0.9 * len(d)))], worst=d[-1],
                over1=sum(x > 1 for x in d), over2=sum(x > 2 for x in d))


def needs_third(st, median_limit=0.5, any_limit=2.0):
    return st["median"] > median_limit or st["worst"] > any_limit


def main(config=None):
    CONFIG = config or dict(
        earlier=[("~/Desktop/AP/logs/mig5_p3_runB_step1", "mistral7b_clock_"),  # (export, its step-name prefix)
                 ("~/Desktop/AP/logs/mig5_p3_runB_step2", "mistral7b_s2_")],
        step3="~/Desktop/AP/logs/mig5_p3_runB_step3",
        median_limit=0.5,
        any_limit=2.0,
        step3_module="mistral7b_step3_split_order",  # the confirmation sets step 3 ran
        step3_prefix="mistral7b_s3_",  # step names in the step-3 export
        out_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    c = CONFIG
    sets = importlib.import_module(c["step3_module"]).ALL
    # No verdict on the wrong inputs: the step-3 export must be step 3 (its prefix, one oracle per pair,
    # no repeated directory), must not be one of the earlier exports, and every split of every
    # confirmation set needs an earlier and a step-3 measurement.
    problems = rl.manifest_problems(c["step3"], list(sets), c["step3_prefix"])
    for export, prefix in c["earlier"]:  # each earlier export must be its own phase (step 1, step 2)
        problems += rl.prefix_problems(export, prefix)
    folders = [os.path.realpath(os.path.expanduser(e)) for e, _ in c["earlier"]] + \
        [os.path.realpath(os.path.expanduser(c["step3"]))]
    problems += [f"export folder given twice: {f}" for f in sorted(set(folders)) if folders.count(f) > 1]
    first = rl.merge(*[rl.load(e) for e, _ in c["earlier"]])
    second = rl.load(c["step3"])
    for pair, splits in sets.items():
        no1 = [s for s in splits if tuple(s) not in first.get(pair, {})]
        no2 = [s for s in splits if tuple(s) not in second.get(pair, {})]
        if no1 or no2:
            problems.append(f"pair {pair}: {len(no1)} splits without an earlier ok run, {len(no2)} without a step-3 ok run")
    if problems:
        print("NO VERDICT, inputs are not a complete step 3:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    third = []
    print(f"{'pair':9s} {'n':>3s} {'median%':>8s} {'p90%':>6s} {'worst%':>7s} {'>1%':>4s} {'>2%':>4s}  verdict")
    for pair in sets:
        a = {tuple(s): first[pair][tuple(s)][0] for s in sets[pair]}
        b = {tuple(s): second[pair][tuple(s)][0] for s in sets[pair]}
        st = pair_stats(a, b)
        if st is None:
            print(f"{str(pair):9s} no splits measured in both")
            continue
        flag = needs_third(st, c["median_limit"], c["any_limit"])
        third += [pair] if flag else []
        print(f"{str(pair):9s} {st['n']:3d} {st['median']:8.2f} {st['p90']:6.2f} {st['worst']:7.2f} {st['over1']:4d} "
              f"{st['over2']:4d}  {'3rd run' if flag else 'ok'}")
    print(f"\n3rd run (same confirmation set) for: {third or 'no pair'}")
    if third:
        path = os.path.join(c["out_dir"], "mistral7b_step3b_split_order.py")
        with open(path, "w") as f:
            f.write('"""Run B 3rd run: the pairs runb_noise_check.py flagged, each with its step-3 set."""\n\nALL = {\n')
            for pair in third:
                f.write(f"    {pair}: {[list(s) for s in sets[pair]]},\n")
            f.write("}\n")
        print(f"wrote {path}: clock_study.sh preset runB_step3_rerun")
    return third


if __name__ == "__main__":
    main()
