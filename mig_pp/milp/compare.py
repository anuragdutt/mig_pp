"""
Per-row CSV: the model's prediction for each measured split next to its measured
latency, and the MILP's own pick for the row's (batch, microbatch).
Edit the lines below and run:  python3 compare.py
"""

import csv
import sys

sys.path.insert(0, ".")
import pulp_layer_placement as plp  # noqa: E402
from validate import load_rows, model_from_path  # noqa: E402

CSV_PATH = "../../oracle/40gb/20_10_5_5/vicuna_7B.csv"
OUT_PATH = "comparison_results.csv"
CARD = "A100-40GB"

rows, slice_gb = load_rows(CSV_PATH)
cfg = dict(model=model_from_path(CSV_PATH), slice_gb=slice_gb, card=CARD, seq_len=64,
           max_new_tokens=512, samples=8, min_last_rank_layers=1,
           constants=plp.DEFAULT_CONSTANTS)

out, steps, pick = [], {}, {}
for r in rows:
    key = (r["batch"], r["mb"])
    if key not in steps:
        steps[key] = plp.prepare(cfg, *key, cfg["samples"])
        pick[key] = plp.optimize_placement(cfg, *key)[0]
    pred = plp.evaluate(r["split"], steps[key], cfg)
    out.append(dict(split=list(r["split"]), batch_size=r["batch"], microbatch_size=r["mb"],
                    predicted_ms=round(pred, 1), measured_ms=round(r["meas"], 1),
                    error_pct=round((pred - r["meas"]) / r["meas"] * 100, 1),
                    milp_pick=pick[key]))

with open(OUT_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(f"Saved {len(out)} rows to {OUT_PATH}")
