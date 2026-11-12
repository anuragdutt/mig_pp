"""
Configs finished so far in a running sweep, read from benchmark.log (the results CSV is only
written when the whole job ends). Prints them and writes partial_results.csv next to the log:
latency and status, and per rank prefill/decode time, decode compute per microbatch and barrier
idle; memory only for the tightest slice; transport waits (T23-T26 totals per rank, from the
transport log). Status follows the parent benchmark's outcome;
rank 0 finishing alone does not confirm success. Per-step latencies exist only in the final files.

    python3 partial_results.py runs/<run_dir>
"""

import csv
import glob
import os
import re
import sys

START = re.compile(r"\[(\d+)/(\d+)\] Split (\[[^\]]*\]) \| Batch: (\d+) \| Microbatch: (\d+)")
DONE = re.compile(r"\[Rank 0\] Finished\. Latency: (\d+) ms")
SUCCESS = re.compile(r"Total latency:\s+(\d+) ms")
OOM = re.compile(r"OOM on Rank (\d+) — skipping\.")
PHASE = re.compile(r"\[B03\]\[Rank (\d+)\] phase split: prefill=([\d.]+)ms .* decode=([\d.]+)ms")
COMPUTE = re.compile(r"\[B05\]\[Rank (\d+)\] decode compute/mb: mean=([\d.]+)ms")
IDLE = re.compile(r"\[B06\]\[Rank (\d+)\] barrier idle: post_prefill=([\d.]+)ms final=([\d.]+)ms")
# Only the tightest slice's memory is logged per config; all ranks appear in the final CSV.
MEMORY = re.compile(r"(Peak|Avg) +(\d+)GB memory \(rank (\d+)\): (\d+) MB")
RANKS = 4
FIELDS = (["n", "of", "split", "batch", "mb", "latency_ms", "status"]
          + [f"{name}_rank{r}" for name in ("prefill_ms", "decode_ms", "decode_compute_mean_ms",
                                            "barrier_idle_ms") for r in range(RANKS)]
          + ["tight_rank", "tight_peak_mb", "tight_avg_mb"]
          + [f"{tag}_total_ms_rank{r}" for tag in ("t23", "t24", "t25", "t26") for r in range(RANKS)])

# Transport log summaries, written by every rank as each config ends: T22 names the config, then
# T23 copy blocking, T24 H2D blocking, T25 wait for upstream, T26 ACK wait (totals over the config).
SECTION = re.compile(r"\[T22\]\[rank(\d)\].*split=(\[[^\]]*\]) mb=(\d+) latency=(\d+)ms")
TAG = re.compile(r"\[(T2[3-6])\]\[rank(\d)\].*?mean=([\d.]+)ms(?: total=([\d.]+)ms)? over (\d+)")


def add_transport(job_dir, rows):
    """Fill the t2x columns; a config is matched by split, microbatch and the closest latency."""
    current = {}
    for path in glob.glob(os.path.join(job_dir, "logs", "transport_*.log")):
        with open(path, errors="ignore") as f:
            for line in f:
                m = SECTION.search(line)
                if m:
                    rank, split, mb, latency = int(m[1]), m[2], int(m[3]), int(m[4])
                    same = [r for r in rows if r["split"] == split and r["mb"] == mb and r["latency_ms"] != ""]
                    current[rank] = min(same, key=lambda r: abs(r["latency_ms"] - latency)) if same else None
                    continue
                m = TAG.search(line)
                if m and current.get(int(m[2])) is not None:
                    total = float(m[4]) if m[4] else float(m[3]) * int(m[5])
                    current[int(m[2])][f"{m[1].lower()}_total_ms_rank{m[2]}"] = round(total, 1)


def parse(log_path):
    rows, current = [], None
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = START.search(line)
            if m:
                current = {name: "" for name in FIELDS}
                current.update(n=int(m[1]), of=int(m[2]), split=m[3], batch=int(m[4]),
                               mb=int(m[5]), status="running")
                rows.append(current)
                continue
            if current is None:
                continue
            m = DONE.search(line)
            if m and current["status"] == "running":
                current["latency_ms"] = int(m[1])
            # Only the parent can confirm that every rank completed successfully.
            m = SUCCESS.search(line)
            if m:
                current.update(latency_ms=int(m[1]), status="ok")
            elif "Hang detected: ranks still alive after timeout:" in line:
                current["status"] = "hang"
            elif "Crashed. Exit codes:" in line:
                current["status"] = "crash"
            elif "Timeout — no results received." in line:
                current["status"] = "timeout"
            m = OOM.search(line)
            if m:
                current["status"] = f"OOM_rank{m[1]}"
            m = PHASE.search(line)
            if m:
                current[f"prefill_ms_rank{m[1]}"] = float(m[2])
                current[f"decode_ms_rank{m[1]}"] = float(m[3])
            m = COMPUTE.search(line)
            if m:
                current[f"decode_compute_mean_ms_rank{m[1]}"] = float(m[2])
            m = IDLE.search(line)
            if m:
                current[f"barrier_idle_ms_rank{m[1]}"] = round(float(m[2]) + float(m[3]), 1)
            m = MEMORY.search(line)
            if m:
                current["tight_rank"] = int(m[3])
                current["tight_peak_mb" if m[1] == "Peak" else "tight_avg_mb"] = int(m[4])
    return rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    for log_path in sorted(glob.glob(os.path.join(sys.argv[1], "gpu*", "*", "benchmark.log"))):
        rows = parse(log_path)
        add_transport(os.path.dirname(log_path), rows)
        out = os.path.join(os.path.dirname(log_path), "partial_results.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        done = [r for r in rows if r["status"] != "running"]
        print(f"{log_path}: {len(done)} of {rows[-1]['of'] if rows else '?'} configs done -> {out}")
        best = {}
        for r in done:
            if r["status"] == "ok":
                k = (r["batch"], r["mb"])
                if k not in best or r["latency_ms"] < best[k]["latency_ms"]:
                    best[k] = r
        for (b, mb), r in sorted(best.items()):
            print(f"  ({b},{mb}) fastest so far: {r['split']} {r['latency_ms'] / 1000:.1f} s")


if __name__ == "__main__":
    main()
