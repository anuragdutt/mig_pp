"""
Model vs a measured sweep CSV (oracle files or a run's mig_benchmark_results.csv).
Edit CONFIG in main() and run:  python3 validate.py

Q1  accuracy: predict each row's own split, error% = (predicted - measured) / measured.
Q2  split choice, per (batch, microbatch): rank of the model's best MEASURED split by
    measured latency, Kendall tau of predicted vs measured order, and the MILP's free pick.

Nothing here feeds back into the model. Predictions use the MILP's sampled decode.
"""

import ast
import csv
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pulp_layer_placement as plp  # noqa: E402


def load_rows(path):
    """(rows with status ok, slice GB per rank from the header). Read positionally: oracle
    headers repeat a slice size, which csv.DictReader would collapse."""
    with open(path, newline="") as f:
        table = list(csv.reader(f))
    header, body = table[0], [r for r in table[1:] if r]
    col = {name: i for i, name in enumerate(header)}
    peaks = sorted((int(m.group(1)), int(m.group(2)))
                   for m in (re.match(r"^peak_rank(\d+)_(\d+)gb_mb$", h) for h in header) if m)
    if "split" in col:
        split_of = lambda r: tuple(ast.literal_eval(r[col["split"]]))  # noqa: E731
        slice_gb = [gb for _, gb in peaks] or None
    else:  # oracle: one column per rank before batch_size
        n = col["batch_size"]
        split_of = lambda r: tuple(int(x) for x in r[:n])  # noqa: E731
        slice_gb = [gb for _, gb in peaks] or [int(h[:-2]) for h in header[:n]]
    rows = []
    for r in body:
        if r[col["status"]] != "ok":
            continue
        batch, mb = int(r[col["batch_size"]]), int(r[col["microbatch_size"]])
        rows.append(dict(split=split_of(r), batch=batch, mb=mb, meas=float(r[col["total_latency_ms"]])))
    return rows, slice_gb


def model_from_path(path):
    """vicuna_7B.csv -> vicuna_7b, or <run>/gpu2/01_qwen_7b/results.csv -> qwen_7b."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    job = re.sub(r"^\d+_", "", os.path.basename(os.path.dirname(os.path.abspath(path)))).lower()
    return next((m for m in (stem, job) if m in plp.memory_limits.REFERENCE_SHAPES), None)


def kendall_tau(a, b):
    con = dis = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            x = (a[i] - a[j]) * (b[i] - b[j])
            con += x > 0
            dis += x < 0
    return (con - dis) / (con + dis) if con + dis else float("nan")


def main():
    CONFIG = dict(
        csv="../../oracle/40gb/20_10_5_5/vicuna_7B.csv",
        card="A100-40GB",  # the CSV cannot tell
        model=None,  # None: from the file name / job folder
        slice_gb=None,  # None: from the CSV header
        seq_len=64,
        max_new_tokens=512,
        samples=8,
        min_last_rank_layers=1,
        constants=plp.DEFAULT_CONSTANTS,
    )

    rows, header_gb = load_rows(CONFIG["csv"])
    cfg = dict(CONFIG, model=CONFIG["model"] or model_from_path(CONFIG["csv"]),
               slice_gb=CONFIG["slice_gb"] or header_gb)
    print(f"{CONFIG['csv']}: {len(rows)} ok rows, {cfg['model']} on {cfg['slice_gb']} ({cfg['card']})")

    # per (batch, mb): the split-independent part is built once
    pairs = {}
    for r in rows:
        pairs.setdefault((r["batch"], r["mb"]), []).append(r)
    for (batch, mb), group in pairs.items():
        steps = plp.prepare(cfg, batch, mb, cfg["samples"])
        for r in group:
            r["pred"] = plp.evaluate(r["split"], steps, cfg)
            r["err"] = (r["pred"] - r["meas"]) / r["meas"] * 100

    print("\nQ1  error% = (predicted - measured) / measured; negative = model too fast")
    errs = [r["err"] for r in rows]
    print(f"  all: median {statistics.median(errs):.1f}  min {min(errs):.1f}  max {max(errs):.1f}")
    for key, name in (("mb", "mb"), ("batch", "batch")):
        for v in sorted({r[key] for r in rows}):
            sub = [r["err"] for r in rows if r[key] == v]
            print(f"  {name}={v:<3d} rows={len(sub):4d} median {statistics.median(sub):7.1f}%")

    print("\nQ2  per (batch, mb): best predicted measured split, its measured rank, tau, MILP pick")
    for (batch, mb), group in sorted(pairs.items()):
        measured = {}
        for r in group:  # fastest repeat of a split
            if r["split"] not in measured or r["meas"] < measured[r["split"]][0]:
                measured[r["split"]] = (r["meas"], r["pred"])
        by_meas = sorted(measured, key=lambda s: measured[s][0])
        pick = min(measured, key=lambda s: measured[s][1])
        tau = kendall_tau([measured[s][1] for s in by_meas], [measured[s][0] for s in by_meas])
        try:
            free, _ = plp.optimize_placement(cfg, batch, mb)
            free_txt = f"MILP {free}{' (measured)' if tuple(free) in measured else ''}"
        except ValueError as e:
            free_txt = f"MILP: {e}"
        best = measured[by_meas[0]][0]
        print(f"  B{batch:<3d} mb{mb:<3d} picks {list(pick)} rank {by_meas.index(pick) + 1}/{len(by_meas)} "
              f"{(measured[pick][0] - best) / best * 100:+5.1f}% vs best {list(by_meas[0])} | "
              f"tau {tau:+.2f} | {free_txt}")


if __name__ == "__main__":
    main()
