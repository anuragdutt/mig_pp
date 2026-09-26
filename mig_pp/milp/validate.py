"""
Validates pulp_layer_placement.py against an oracle sweep CSV.

This is the ONE place where the model meets measured data. It is a validation
step, never an input to the model: nothing here feeds constants back into
pulp_layer_placement.py. Derive the model from first principles, then run this
to see how far off it is.

Two independent questions, deliberately separated:

  Q1  IS THE LATENCY MODEL ACCURATE?
      Predict each oracle row using the split THAT ROW ACTUALLY RAN, and
      compare against its measured latency. This tests the equation alone.

  Q2  DOES THE MILP PICK A GOOD SPLIT?
      Run the optimizer, then look up its chosen split in the oracle sweep and
      see where it ranks among the splits that were actually measured. This
      tests the objective, scored against ground truth rather than against the
      model's own prediction.

Keeping them apart matters: if you only ran Q2 and the split looked bad, you
could not tell whether the objective is wrong or the latency model underneath
it is wrong. Q1 isolates the model; Q2 then isolates the objective.

Usage:
    python3 validate.py
    python3 validate.py --csv ../../oracle/40gb/20_10_10/vicuna_7B.csv
    python3 validate.py --out comparison_results.csv     # also write per-row CSV
"""

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulp_layer_placement import (  # noqa: E402
    optimize_placement,
    predict_latency_ms,
    EFF_WEIGHT,
    EFF_ACT,
    LAUNCH_US,
    KERNELS_PER_LAYER,
    PCIE_BW_PINNED,
    PCIE_BW_UNPINNED,
    MEMCPY_BW,
    HANDOFF_FIXED_US,
    TOK_ROUNDTRIP_US,
)

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "oracle",
    "40gb",
    "20_10_10",
    "vicuna_7B.csv",
)

# Oracle CSV column positions. The header has two columns both named "10gb",
# so csv.DictReader collapses them -- read positionally instead.
COL_SPLIT0, COL_SPLIT1, COL_SPLIT2 = 0, 1, 2
COL_BATCH, COL_MBSIZE, COL_NMB = 3, 4, 5
COL_LATENCY = 9
COL_STATUS = 10


# ---------------------------------------------------------------------------
def load_oracle(path):
    """Read the oracle sweep, keeping only rows that ran to completion."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            if r[COL_STATUS] != "ok":
                continue
            rows.append(
                {
                    "split": (
                        int(r[COL_SPLIT0]),
                        int(r[COL_SPLIT1]),
                        int(r[COL_SPLIT2]),
                    ),
                    "batch": int(r[COL_BATCH]),
                    "mb": int(r[COL_MBSIZE]),
                    "nmb": int(r[COL_NMB]),
                    "meas": float(r[COL_LATENCY]),
                }
            )
    return rows


def pct(a, b):
    """Signed percent difference of a relative to b."""
    return (a - b) / b * 100.0


def summarize(values):
    v = sorted(values)
    n = len(v)
    return {
        "n": n,
        "min": v[0],
        "p25": v[n // 4],
        "median": statistics.median(v),
        "p75": v[3 * n // 4],
        "max": v[-1],
        "mean": statistics.mean(v),
        "mae": statistics.mean([abs(x) for x in v]),
    }


# ---------------------------------------------------------------------------
# Q1 -- latency model accuracy
# ---------------------------------------------------------------------------
def run_q1(rows, out_path=None):
    print("=" * 74)
    print("Q1  LATENCY MODEL ACCURACY")
    print("    Each oracle row predicted on the split it actually ran.")
    print("    error% = (predicted - measured) / measured * 100")
    print("    negative => model under-predicts (says it is faster than reality)")
    print("=" * 74)

    for row in rows:
        row["pred"] = predict_latency_ms(list(row["split"]), row["nmb"], row["batch"])
        row["err"] = pct(row["pred"], row["meas"])

    s = summarize([r["err"] for r in rows])
    print(f"\n  rows: {s['n']}")
    print(
        f"  error%:  min={s['min']:7.1f}  p25={s['p25']:7.1f}  "
        f"median={s['median']:7.1f}  p75={s['p75']:7.1f}  max={s['max']:7.1f}"
    )
    print(f"  mean={s['mean']:7.1f}   mean|error|={s['mae']:7.1f}")

    # A tight spread with a large offset means a systematic SCALE error (one or
    # more constants off by a common factor), not model noise. A wide spread
    # would instead point at the structure being wrong.
    spread = s["max"] - s["min"]
    print(f"  spread (max-min) = {spread:.1f} pp", end="")
    if spread < 40:
        print("  -> tight: systematic scale error, not structural noise")
    else:
        print("  -> wide: structure may be wrong, not just constants")

    print("\n  by num_microbatches (tests the fill + (N-1)*max law):")
    for n in sorted({r["nmb"] for r in rows}):
        sub = [r["err"] for r in rows if r["nmb"] == n]
        print(
            f"    N={n:2d}  rows={len(sub):4d}  median={statistics.median(sub):7.1f}%"
        )

    print("\n  by batch_size:")
    for b in sorted({r["batch"] for r in rows}):
        sub = [r["err"] for r in rows if r["batch"] == b]
        print(
            f"    batch={b:3d} rows={len(sub):4d}  median={statistics.median(sub):7.1f}%"
        )

    # Drift across N tells you whether the overlap law itself is off, separately
    # from the overall scale.
    meds = [
        statistics.median([r["err"] for r in rows if r["nmb"] == n])
        for n in sorted({r["nmb"] for r in rows})
    ]
    if len(meds) > 1:
        drift = meds[-1] - meds[0]
        print(f"\n  drift across N: {drift:+.1f} pp", end="")
        if abs(drift) < 3:
            print("  -> N-law looks right")
        elif drift < 0:
            print(
                "  -> model too OPTIMISTIC at high N (real overlap is worse than ideal GPipe)"
            )
        else:
            print(
                "  -> model too PESSIMISTIC at high N (real overlap is better than modelled)"
            )

    if out_path:
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "split",
                    "batch_size",
                    "microbatch_size",
                    "num_microbatches",
                    "predicted_ms",
                    "measured_ms",
                    "error_pct",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        "/".join(str(x) for x in r["split"]),
                        r["batch"],
                        r["mb"],
                        r["nmb"],
                        round(r["pred"], 1),
                        round(r["meas"], 1),
                        round(r["err"], 1),
                    ]
                )
        print(f"\n  per-row results written to {out_path}")

    return rows


# ---------------------------------------------------------------------------
# Q2 -- optimizer split quality, scored against measured ground truth
# ---------------------------------------------------------------------------
def run_q2(rows):
    print()
    print("=" * 74)
    print("Q2  MILP SPLIT QUALITY")
    print("    PuLP's chosen split looked up in the oracle sweep and ranked by")
    print("    MEASURED latency against every other split that was swept.")
    print("=" * 74)

    for batch in sorted({r["batch"] for r in rows}):
        counts, limits = optimize_placement(batch_size=batch)
        chosen = tuple(counts)
        print(f"\n  batch={batch}: PuLP picks {chosen}   (memory limits {limits})")

        # Flag when the optimizer is pinned against a memory limit -- then its
        # "choice" is nearly forced and says little about the objective.
        pinned = [s for s in range(3) if counts[s] == limits[s]]
        if pinned:
            print(
                f"    NOTE: at the memory limit on slice(s) {pinned} -- "
                f"choice is partly forced, weak test of the objective"
            )

        for n in sorted({r["nmb"] for r in rows if r["batch"] == batch}):
            sub = [r for r in rows if r["batch"] == batch and r["nmb"] == n]
            ranked = sorted(sub, key=lambda r: r["meas"])
            best = ranked[0]

            hit = [r for r in sub if r["split"] == chosen]
            if not hit:
                span = pct(ranked[-1]["meas"], best["meas"])
                print(
                    f"    N={n}: NOT in sweep. {len(sub)} splits measured; "
                    f"best {best['meas']:.0f} ms {best['split']}, "
                    f"worst {ranked[-1]['meas']:.0f} ms (spread {span:.0f}%)"
                )
                continue

            h = hit[0]
            rank = ranked.index(h) + 1
            penalty = pct(h["meas"], best["meas"])
            print(
                f"    N={n}: measured {h['meas']:9.0f} ms | rank {rank:3d}/{len(sub)} | "
                f"{penalty:+5.1f}% vs best {best['meas']:.0f} ms {best['split']}"
            )


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="oracle sweep CSV")
    ap.add_argument("--out", default=None, help="write per-row Q1 results here")
    ap.add_argument("--q1-only", action="store_true")
    ap.add_argument("--q2-only", action="store_true")
    args = ap.parse_args()

    print("Constants currently in pulp_layer_placement.py:")
    print(
        f"  EFF_WEIGHT={EFF_WEIGHT}  EFF_ACT={EFF_ACT}  "
        f"LAUNCH_US={LAUNCH_US}  KERNELS_PER_LAYER={KERNELS_PER_LAYER}"
    )
    print(
        f"  PCIE_BW_PINNED={PCIE_BW_PINNED:.2e}  "
        f"PCIE_BW_UNPINNED={PCIE_BW_UNPINNED:.2e}  MEMCPY_BW={MEMCPY_BW:.2e}"
    )
    print(f"  HANDOFF_FIXED_US={HANDOFF_FIXED_US}  TOK_ROUNDTRIP_US={TOK_ROUNDTRIP_US}")
    print()

    rows = load_oracle(args.csv)
    print(f"oracle: {args.csv}")
    print(f"        {len(rows)} rows with status=ok")
    print()

    if not args.q2_only:
        run_q1(rows, out_path=args.out)
    if not args.q1_only:
        run_q2(rows)
    print()


if __name__ == "__main__":
    main()
