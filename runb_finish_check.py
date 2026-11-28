"""
Finish check for one Run B step: is every expected split there with status ok? Counts only, never a
latency. First the manifest: exactly one oracle per expected pair, no run directory twice, every step
name with the step's prefix. Then per step (oracle) of the export's last '# run' block: rows, distinct splits, ok, expected
splits missing, unexpected splits, and every non-ok row by status (split names are not results).
run_parallel.sh counts processes; a configuration can still record OOM / crash / timeout in the CSV.

Edit CONFIG in main(), then:  .venv/bin/python runb_finish_check.py   (exit 1 if anything is off)
"""

import collections
import importlib
import json
import sys

import runb_lists as rl


def check(export, module, prefix, passes=1):
    expected = importlib.import_module(module).ALL
    problems = 0
    for p in rl.manifest_problems(export, list(expected), prefix, passes):
        print(f"manifest: {p}")
        problems += 1
    for step, run_dir in rl.last_run_block(export):
        rows = rl.read_rows(export, run_dir)
        pairs = {(int(r["batch_size"]), int(r["microbatch_size"])) for r in rows}
        if len(pairs) != 1:
            print(f"{step}: rows for {len(pairs)} pairs {sorted(pairs)}, want 1")
            problems += 1
            continue
        pair = pairs.pop()
        want = {tuple(s) for s in expected.get(pair, [])}
        splits = [tuple(json.loads(r["split"])) for r in rows]
        ok = {s for s, r in zip(splits, rows) if r["status"] == "ok"}
        bad = collections.Counter(r["status"] for r in rows if r["status"] != "ok")
        missing = want - set(splits)
        extra = set(splits) - want
        not_ok = sorted(set(splits) - ok)
        good = not missing and not extra and not bad and len(ok) == len(want) == len(rows)
        problems += not good
        print(f"{step} {pair}: rows {len(rows)}, distinct {len(set(splits))}, ok {len(ok)} of {len(want)} expected"
              + (f", missing {len(missing)}" if missing else "") + (f", unexpected {len(extra)}" if extra else "")
              + (f", not ok {dict(bad)}: {[list(s) for s in not_ok]}" if bad else "") + ("  OK" if good else "  CHECK"))
    print("\nALL OK" if not problems else f"\n{problems} problem(s)")
    return problems


def main(config=None):
    CONFIG = config or dict(
        export="~/Desktop/AP/logs/mig5_p3_runB_step1",
        module="mistral7b_40gb_split_order",  # the list the step ran: step 1; mistral7b_step2/3_split_order later
        prefix="mistral7b_clock_",  # step names of that step: step 1; mistral7b_s2_ / s3_ / s3b_ / post_, vicuna7b_vtop_
        passes=1,  # oracles per pair: 1 for the Run B steps, 2 for vicuna_topup
    )
    sys.exit(1 if check(CONFIG["export"], CONFIG["module"], CONFIG["prefix"], CONFIG["passes"]) else 0)


if __name__ == "__main__":
    main()
