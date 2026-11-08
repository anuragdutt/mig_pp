"""Merge the mig_benchmark_results_*.csv sweeps of each model dir into one CSV per model.

Key = (split, batch_size, microbatch_size). Rows are sorted by that key and get a
`source_file` column. A key present in two files is an error.
"""
import ast
import csv
import glob
import os


def unify(in_dir, out_path):
    rows = {}
    fields = None
    for path in sorted(glob.glob(os.path.join(in_dir, "mig_benchmark_results_*.csv"))):
        if os.path.abspath(path) == out_path:
            continue
        name = os.path.basename(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = reader.fieldnames
            elif reader.fieldnames != fields:
                raise SystemExit(f"{name}: header differs from first file")
            for r in reader:
                key = (r["split"], int(r["batch_size"]), int(r["microbatch_size"]))
                if key in rows:
                    old = rows[key]
                    same = all(old[c] == r[c] for c in fields)
                    raise SystemExit(
                        f"duplicate {key}: {old['source_file']} vs {name} "
                        f"({'identical' if same else 'DIFFERENT'})")
                r["source_file"] = name
                rows[key] = r

    order = sorted(rows, key=lambda k: (ast.literal_eval(k[0]), k[1], k[2]))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["source_file"])
        w.writeheader()
        for k in order:
            w.writerow(rows[k])

    splits = {k[0] for k in rows}
    pairs = {(k[1], k[2]) for k in rows}
    print(f"{in_dir}: {len(rows)} rows, {len(splits)} splits, {len(pairs)} (batch, mb) pairs -> {out_path}")


def main():
    CONFIG = {
        "root": "80gb/40_20_10_10",
        "models": {"qwen": "qwen_14b", "mistral": "mistral_24b", "llama": "llama_13b"},
    }
    base = os.path.dirname(os.path.abspath(__file__))
    for m, out in CONFIG["models"].items():
        d = os.path.join(base, CONFIG["root"], m)
        unify(d, os.path.join(d, f"{out}.csv"))


if __name__ == "__main__":
    main()
