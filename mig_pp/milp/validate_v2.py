"""
Out-of-sample test of pulp_layer_placement_v2's theory against a measured sweep.

This is a genuine held-out comparison: v2 contains no constant fitted to any
latency CSV, so the residuals below are predictive error, not goodness of fit.

Usage:
    python3 validate_v2.py [path/to/results.csv]

Default CSV: ../../mig_benchmark_results_9_28.csv (938 rows, 20/10/5/5, 4 ranks,
Vicuna-7B, 663 of them status=ok).

Read the output in this order:
  Q1  accuracy       -- is the level right?
  Q2  N-drift        -- is the LAW right? (a drift means wrong structure, not
                        wrong constants)
  Q3  ranking        -- does it order splits correctly? This is what the
                        optimizer actually needs; it can be right even when the
                        level is off.
  Q4  decomposition  -- which TERM carries the error, using the CSV's per-rank
                        compute columns as ground truth for E_s.
  Q5  choice quality -- how far off the measured best is v2's pick?
"""

import ast
import collections
import csv
import itertools
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pulp_layer_placement_v2 as v2  # noqa: E402

try:
    import pulp_layer_placement as v1  # noqa: E402

    HAVE_V1 = True
except Exception:
    HAVE_V1 = False

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "mig_benchmark_results_9_28.csv"
NR = v2.NUM_SLICES


def load(path):
    rows = list(csv.DictReader(open(path)))
    ok = [r for r in rows if r["status"] == "ok"]
    return rows, ok


def err_pct(pred, meas):
    return (pred - meas) / meas * 100.0


def describe(vals):
    v = sorted(vals)
    n = len(v)
    return dict(
        n=n,
        min=v[0],
        p25=v[n // 4],
        median=st.median(v),
        p75=v[3 * n // 4],
        max=v[-1],
        mae=st.mean(abs(x) for x in v),
    )


def kendall_tau(a, b):
    con = dis = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0:
            con += 1
        elif s < 0:
            dis += 1
    return (con - dis) / (con + dis) if con + dis else float("nan")


def main(csv_path):
    all_rows, ok = load(csv_path)
    status = collections.Counter(r["status"] for r in all_rows)
    print("=" * 78)
    print(f"OUT-OF-SAMPLE VALIDATION -- {Path(csv_path).name}")
    print(f"  {len(all_rows)} rows, {len(ok)} ok   statuses: {dict(status)}")
    print("  v2 has NO constant fitted to this (or any) latency data.")
    print("=" * 78)

    preds = []
    for r in ok:
        counts = ast.literal_eval(r["split"])
        N = int(r["num_microbatches"])
        B = int(r["batch_size"])
        meas = float(r["total_latency_ms"])
        p2 = v2.predict_latency_ms(counts, N, B)
        row = dict(r=r, counts=counts, N=N, B=B, meas=meas, p2=p2,
                   e2=err_pct(p2, meas))
        if HAVE_V1:
            # v1 is 3-slice; collapse the two 5GB slices so it can be scored.
            p1 = v1.predict_latency_ms(
                [counts[0], counts[1], counts[2] + counts[3]], N, B
            )
            row["p1"] = p1
            row["e1"] = err_pct(p1, meas)
        preds.append(row)

    # ---- Q1 accuracy -------------------------------------------------------
    print("\nQ1  ACCURACY on total_latency_ms")
    print("    error% = (predicted - measured) / measured * 100")
    print("    negative => predicts faster than reality")
    hdr = f"    {'model':>12} {'min':>7} {'p25':>7} {'med':>7} {'p75':>7} {'max':>7} {'MAE':>7}"
    print(hdr)
    s2 = describe([p["e2"] for p in preds])
    print(
        f"    {'v2 min-max':>12} {s2['min']:7.1f} {s2['p25']:7.1f} "
        f"{s2['median']:7.1f} {s2['p75']:7.1f} {s2['max']:7.1f} {s2['mae']:7.1f}"
    )
    if HAVE_V1:
        s1 = describe([p["e1"] for p in preds])
        print(
            f"    {'v1 serial':>12} {s1['min']:7.1f} {s1['p25']:7.1f} "
            f"{s1['median']:7.1f} {s1['p75']:7.1f} {s1['max']:7.1f} {s1['mae']:7.1f}"
        )

    spread = s2["max"] - s2["min"]
    print(f"\n    v2 spread (max-min) = {spread:.1f} pp", end="")
    if spread < 40:
        print(" -> tight: a systematic SCALE error (one or more GROUP (3)")
        print("       constants is off by a factor), not a structural one.")
    else:
        print(" -> wide: the STRUCTURE may be wrong, not just the constants.")

    # ---- Q2 N-drift --------------------------------------------------------
    print("\nQ2  N-DRIFT -- tests the LAW itself")
    print("    A law with the right structure has error roughly FLAT in N.")
    print("    Drift means the fill/steady split is wrong.")
    by_n = collections.defaultdict(list)
    for p in preds:
        by_n[p["N"]].append(p)
    print(f"    {'N':>4} {'rows':>5} {'v2 med':>9} {'v2 MAE':>8}", end="")
    print(f" {'v1 med':>9}" if HAVE_V1 else "")
    for N in sorted(by_n):
        sub = by_n[N]
        m2 = st.median(p["e2"] for p in sub)
        a2 = st.mean(abs(p["e2"]) for p in sub)
        line = f"    {N:4d} {len(sub):5d} {m2:+8.1f}% {a2:8.1f}"
        if HAVE_V1:
            line += f" {st.median(p['e1'] for p in sub):+8.1f}%"
        print(line)
    ns = sorted(by_n)
    drift2 = st.median(p["e2"] for p in by_n[ns[-1]]) - st.median(
        p["e2"] for p in by_n[ns[0]]
    )
    print(f"\n    v2 N-drift (N={ns[0]} -> N={ns[-1]}): {drift2:+.1f} pp")
    if HAVE_V1:
        d1 = st.median(p["e1"] for p in by_n[ns[-1]]) - st.median(
            p["e1"] for p in by_n[ns[0]]
        )
        print(f"    v1 N-drift: {d1:+.1f} pp")

    print("\n    by batch_size:")
    by_b = collections.defaultdict(list)
    for p in preds:
        by_b[p["B"]].append(p)
    for B in sorted(by_b):
        sub = by_b[B]
        print(
            f"      B={B:3d} rows={len(sub):4d} "
            f"median={st.median(p['e2'] for p in sub):+7.1f}% "
            f"MAE={st.mean(abs(p['e2']) for p in sub):6.1f}"
        )

    # ---- Q3 ranking --------------------------------------------------------
    print("\nQ3  RANKING -- Kendall tau within each (batch, N) group")
    print("    This is what the optimizer needs. +1 = perfect order, 0 = noise,")
    print("    negative = orders splits BACKWARDS.")
    groups = collections.defaultdict(list)
    for p in preds:
        groups[(p["B"], p["N"])].append(p)
    print(f"    {'batch/N':>9} {'rows':>5} {'v2':>7}", end="")
    print(f" {'v1':>7}" if HAVE_V1 else "")
    t2s, t1s = [], []
    for k in sorted(groups):
        sub = groups[k]
        if len(sub) < 5:
            continue
        meas = [p["meas"] for p in sub]
        t2 = kendall_tau(meas, [p["p2"] for p in sub])
        t2s.append(t2)
        line = f"    {f'{k[0]}/{k[1]}':>9} {len(sub):5d} {t2:7.3f}"
        if HAVE_V1:
            t1 = kendall_tau(meas, [p["p1"] for p in sub])
            t1s.append(t1)
            line += f" {t1:7.3f}"
        print(line)
    line = f"    {'median':>9} {'':>5} {st.median(t2s):7.3f}"
    if HAVE_V1 and t1s:
        line += f" {st.median(t1s):7.3f}"
    print(line)

    # ---- Q4 decomposition --------------------------------------------------
    # The CSV carries per-rank per-microbatch compute (decode_compute_mean_ms_*)
    # and per-rank wall time. That lets us separate "are the E_s terms right?"
    # from "is the law combining them right?" -- the single most useful
    # diagnostic here, because it tells us WHICH constant to go measure.
    have_compute = all(
        f"decode_compute_mean_ms_rank{i}" in (ok[0] or {}) for i in range(NR)
    )
    if have_compute:
        print("\nQ4  DECOMPOSITION -- where does the error live?")
        print("    Uses the CSV's own per-rank compute means as ground truth for")
        print("    E_s, so the LAW and the E_s MODEL can be scored separately.")

        ratio_es, law_err = [], []
        for p in preds:
            r = p["r"]
            N, T = p["N"], int(r["max_new_tokens"])
            C_meas = [
                float(r[f"decode_compute_mean_ms_rank{i}"]) for i in range(NR)
            ]
            step_meas = max(
                float(r[f"decode_ms_rank{i}"]) for i in range(NR)
            ) / T

            # model's own E_s at mid-trajectory context
            mb = p["B"] // N
            ctx = v2.SEQ_LEN + v2.MAX_NEW_TOKENS // 2
            E_model = [
                e * 1000.0
                for e in v2._stage_intervals_s(p["counts"], mb, ctx, False)
            ]
            if sum(C_meas) > 0:
                ratio_es.append(sum(E_model) / sum(C_meas))

            # the LAW, fed MEASURED E_s -- isolates structure from constants
            law_pred = sum(C_meas) + (N - 1) * max(C_meas)
            law_err.append(err_pct(law_pred, step_meas))

        rs = describe(ratio_es)
        print(
            f"\n    (a) E_s MODEL vs measured compute:  sum(E_model)/sum(C_meas)"
        )
        print(
            f"        median={rs['median']:.3f}  p25={rs['p25']:.3f}  "
            f"p75={rs['p75']:.3f}"
        )
        print("        1.0 = the per-stage cost model is right.")
        if rs["median"] < 0.9:
            print(f"        -> UNDER-estimates stage cost by ~{(1-rs['median'])*100:.0f}%.")
            print("           Suspects: EFF_ACT too high (KV reads slower than")
            print("           0.25 of peak), or LAUNCH_US*KERNELS too low.")
        elif rs["median"] > 1.1:
            print(f"        -> OVER-estimates stage cost by ~{(rs['median']-1)*100:.0f}%.")
            print("           Suspects: EFF_ACT too low, or EFF_WEIGHT too low.")
        else:
            print("        -> stage cost model is within 10%.")

        ls = describe(law_err)
        print(f"\n    (b) THE LAW, fed MEASURED E_s (structure only):")
        print(
            f"        median={ls['median']:+.1f}%  p25={ls['p25']:+.1f}%  "
            f"p75={ls['p75']:+.1f}%  MAE={ls['mae']:.1f}"
        )
        print("        Near 0 = fill + (N-1)*max is the right structure and any")
        print("        remaining error is in the E_s constants, not the law.")

        by_n_law = collections.defaultdict(list)
        for p, e in zip(preds, law_err):
            by_n_law[p["N"]].append(e)
        print("        by N:", end="")
        for N in sorted(by_n_law):
            print(f"  N={N}:{st.median(by_n_law[N]):+.0f}%", end="")
        print()

    # ---- Q5 choice quality -------------------------------------------------
    print("\nQ5  CHOICE QUALITY -- how good is the split v2 would pick?")
    print("    Restricted to splits the sweep actually measured.")
    gaps = []
    for k in sorted(groups):
        sub = groups[k]
        if len(sub) < 10:
            continue
        B, N = k
        best_meas = min(sub, key=lambda p: p["meas"])
        pick = min(sub, key=lambda p: p["p2"])
        gap = (pick["meas"] / best_meas["meas"] - 1) * 100
        gaps.append(gap)
        print(
            f"      B={B:3d} N={N:2d}: best {str(best_meas['counts']):16s}"
            f" {best_meas['meas']/1000:7.1f}s | v2 picks "
            f"{str(pick['counts']):16s} -> {gap:+5.1f}% off best"
        )
    if gaps:
        print(
            f"\n    median gap from optimal: {st.median(gaps):+.1f}%   "
            f"worst: {max(gaps):+.1f}%"
        )

    print("\n" + "=" * 78)
    print("HOW TO ACT ON THIS")
    print("  Q2 flat + Q3 positive  -> law is right; close the level gap by")
    print("                            MEASURING the GROUP (3) constants.")
    print("  Q2 drifting            -> law is wrong; constants cannot fix it.")
    print("  Q4(b) near 0           -> structure confirmed; Q4(a) names the")
    print("                            term to go measure.")
    print("  Do NOT add a fudge factor to close Q1. That is what v1 did with")
    print("  its serial law and it hid a structural error for a whole sweep.")
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)
