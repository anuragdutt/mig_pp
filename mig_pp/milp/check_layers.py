"""
Step 3 check: does one decoder layer's GPU time, composed from the generic primitives
(layer_ops.layer_us), match the layer measured directly on the same slice (the probe's validation
layers)? Validation only: a miss is fixed in the op list or the primitives, never by using the
measured layer as an input. Edit CONFIG in main(), then (needs torch + transformers 5.18.0, the
boxes' version; on the Mac a scratch install):

    python3 mig_pp/milp/check_layers.py

Prints, per box, slice and model, every validation point: measured, composed, error, and the
composed time split by op kind; then how many points are within the bar.
"""

import json
import os
import re
import statistics

import layer_ops


def measured_us(v):
    """A validation point: a number (probe v1) or {median, spread, windows} (probe v2)."""
    return v["median"] if isinstance(v, dict) else v


def main():
    CONFIG = dict(
        probes={  # box -> the measure_constants_mig.py JSON from that box
            "mig1": "~/Desktop/AP/logs/mig1_probes_v6/constants_mig.json",
            "mig2": "~/Desktop/AP/logs/mig2_probes_v6/constants_mig.json",
            "mig5": "~/Desktop/AP/logs/mig5_probes_v71/probes_v71_mig5/constants_mig.json",  # probe v7.1, main condition
        },
        bar=0.05,  # agreed pass bar: composed within 5% of measured at each point
        show_points=True,
    )
    c = CONFIG

    traces = {}  # (model, mb, ctx, prefill seq) -> classified ops (shapes only, the same on every slice)
    total = passed = 0
    worst = []
    for box, path in c["probes"].items():
        slices = json.load(open(os.path.expanduser(path)))
        for prims in slices:
            for model, points in prims.get("validation", {}).items():
                cfg = layer_ops.MODELS[model]
                rows = []
                for label, v in points.items():
                    m = re.match(r"(decode|prefill) mb(\d+) (?:ctx|seq)(\d+)", label)
                    phase, mb, n = m[1], int(m[2]), int(m[3])
                    key = (model, mb, n if phase == "decode" else 0, n if phase == "prefill" else None)
                    if key not in traces:
                        traces[key] = layer_ops.classify(layer_ops.trace_ops(cfg, mb, key[2], key[3]))
                    comp, parts, kernels, approx = layer_ops.layer_us(traces[key], prims)
                    meas = measured_us(v)
                    err = comp / meas - 1
                    total += 1
                    passed += abs(err) <= c["bar"]
                    worst.append((abs(err), box, prims["profile"], model.split("/")[-1], label, err))
                    rows.append((label, meas, comp, err, parts, kernels, approx))
                ok = sum(abs(r[3]) <= c["bar"] for r in rows)
                errs = [r[3] for r in rows]
                print(f"\n{box} {prims['profile']} {model.split('/')[-1]}: {ok}/{len(rows)} within "
                      f"{100 * c['bar']:.0f}%, median error {100 * statistics.median(errs):+.1f}%")
                if c["show_points"]:
                    for label, meas, comp, err, parts, kernels, approx in rows:
                        split = " ".join(f"{k} {us:.0f}" for k, us in sorted(parts.items()))
                        flag = "  [approx: " + "; ".join(sorted(set(approx))) + "]" if approx else ""
                        print(f"  {label:<20} measured {meas:8.0f} us  composed {comp:8.0f} us  {100 * err:+6.1f}%"
                              f"   ({kernels} kernels: {split}){flag}")
    print(f"\nall: {passed}/{total} points within {100 * c['bar']:.0f}%")
    print("worst 5: " + "; ".join(f"{b} {p} {m} {l} {100 * e:+.1f}%" for _, b, p, m, l, e in sorted(worst, reverse=True)[:5]))


if __name__ == "__main__":
    main()
