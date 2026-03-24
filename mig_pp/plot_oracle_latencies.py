import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("oracle_data") / "mig_real_weights_final.csv"
df = pd.read_csv(csv_path)

# Parse split strings like "[4, 20, 8]" into python lists (optional, but useful)
df["split_list"] = df["split"].apply(ast.literal_eval)

# Sort by latency ascending
df_sorted = df.sort_values("latency_ms", ascending=True).reset_index(drop=True)

# Plot latency sorted ascending (rank on x-axis)
plt.figure()
plt.plot(df_sorted.index, df_sorted["latency_ms"])
plt.xlabel("Rank (sorted by latency, ascending)")
plt.ylabel("Latency (ms)")
plt.title("Oracle latencies (sorted ascending)")
plt.tight_layout()

out_path = Path("oracle_data") / "latencies_sorted.png"
plt.savefig(out_path, dpi=200)
plt.show()

# (Optional) print top-10 best splits
print("\nTop-10 lowest latency splits:")
print(df_sorted.loc[:9, ["split", "latency_ms"]].to_string(index=False))
print(f"\nSaved plot to: {out_path}")
