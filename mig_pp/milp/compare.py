"""
Runs the PuLP optimizer and compares predicted vs measured latency.
Exports results to a CSV file. No terminal output.

EDIT THESE THREE LINES:
    CSV_PATH        -- path to your benchmark CSV
    PULP_MODULE_DIR -- folder containing pulp_layer_placement.py
    OUT_PATH        -- where to save the results CSV
"""

import sys
import os
import pandas as pd

# ============================================================
CSV_PATH        = "../../oracle/40gb/20_10_10/vicuna_7B.csv"
PULP_MODULE_DIR = "."
OUT_PATH = "comparison_results.csv"
# ============================================================

sys.path.insert(0, PULP_MODULE_DIR)
from pulp_layer_placement import optimize_placement, predict_latency_ms

# ---- load data, skip non-ok rows ----
df = pd.read_csv(CSV_PATH)
valid = df[df["status"] == "ok"].copy()
cols = list(valid.columns)
valid.columns = ["layers_20gb", "layers_10gb_b", "layers_10gb_c"] + cols[3:]

results = []
cache = {}   # avoid re-solving same batch_size twice

for _, row in valid.iterrows():
    bs  = int(row["batch_size"])
    mb  = int(row["microbatch_size"])
    nmb = int(row["num_microbatches"])

    if bs not in cache:
        counts, limits = optimize_placement(batch_size=bs)
        cache[bs] = (counts, limits)

    counts, limits = cache[bs]
    pred_ms = predict_latency_ms(counts, nmb, bs)
    meas_ms = row["total_latency_ms"]
    err_pct = (pred_ms - meas_ms) / meas_ms * 100

    results.append({
        "batch_size":       bs,
        "microbatch_size":  mb,
        "num_microbatches": nmb,
        "pulp_split":       f"{counts[0]}/{counts[1]}/{counts[2]}",
        "mem_limits":       f"{limits[0]}/{limits[1]}/{limits[2]}",
        "data_split":       f"{int(row['layers_20gb'])}/{int(row['layers_10gb_b'])}/{int(row['layers_10gb_c'])}",
        "predicted_ms":     round(pred_ms, 1),
        "measured_ms":      round(meas_ms, 1),
        "error_pct":        round(err_pct, 1),
    })

out = pd.DataFrame(results)
os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
out.to_csv(OUT_PATH, index=False)
print(f"Saved {len(out)} rows to {OUT_PATH}")