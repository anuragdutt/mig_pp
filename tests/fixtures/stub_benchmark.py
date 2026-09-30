"""
Stand-in for benchmark_pipeline_microbatching.py in the run_parallel.sh tests.

Stdlib only. Behaves like the real benchmark where the manager can see it:
reads MIG_EXP_CONFIG, runs in the job directory, holds /dev/shm segments
named <MIG_SHM_PREFIX>_<rank>_slot0 (under MIG_SHM_DIR), spawns one child
process per rank (so killing the process group must reach grandchildren),
and writes the files parallel_plan.py's report / verdict / merge read, in
the real formats: benchmark.log ([B08], [k/N] Split), logs/transport_*.log
(T28/T29 VERDICT), mig_benchmark_results.csv, mig_memory_trace.csv.

Knobs, from job.json (set via MODELS[key]["stub"] in a test config):
  stub_sleep       seconds to run (default 1)
  stub_rc          exit code (default 0)
  stub_wrong_prefix  log a different shm_prefix in [T01] (as a transport that
                   ignored MIG_SHM_PREFIX would)
  stub_leak_shm    leave its segments behind on exit (default false)
  stub_ignore_term ignore SIGTERM, so stop has to escalate to KILL
"""

import csv
import json
import os
import signal
import subprocess
import sys
import time


def main() -> int:
    job = json.load(open(os.environ["MIG_EXP_CONFIG"]))
    world = len(job["mig_uuids"])
    prefix = os.environ.get("MIG_SHM_PREFIX", "mig_pipe_shm")
    shm_dir = os.environ.get("MIG_SHM_DIR", "/dev/shm")
    sleep_s = float(job.get("stub_sleep", 1))
    if job.get("stub_ignore_term"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    started = time.time()
    record = {
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "cwd": os.getcwd(),
        "prefix": prefix,
        "port": job["master_port"],
        "job_id": job.get("job_id"),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "env": {k: os.environ.get(k) for k in ("MIG_EXP_CONFIG", "MIG_SHM_PREFIX", "PYTHONUNBUFFERED", "TQDM_DISABLE")},
        "started": started,
    }

    segs = [os.path.join(shm_dir, f"{prefix}_{r}_slot0") for r in range(world)]
    for p in segs:
        with open(p, "wb") as f:
            f.write(b"\0" * 16)

    # One child per rank, like mp.Process in the real parent. They sleep a
    # little longer than the parent so an early kill must reach them too.
    children = [
        subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({sleep_s + 30})"])
        for _ in range(world)
    ]
    record["children"] = [c.pid for c in children]
    with open("stub_record.json", "w") as f:
        json.dump(record, f)

    os.makedirs("logs", exist_ok=True)
    with open("benchmark.log", "a") as log:
        log.write(f"Parallel job {job.get('job_id')}: stub\n")
        log.write(f"[1/1] Split [x] | Batch: 8 | Microbatch: 4 | Microbatches: 2\n")
        for r in range(world):
            log.write(f"[B08][Rank {r}] weights: loaded=10/10 missing=0 format=safetensors files=1\n")
    time.sleep(sleep_s)

    logged = "mig_pipe_shm" if job.get("stub_wrong_prefix") else prefix
    with open(os.path.join("logs", "transport_stub.log"), "a") as t:
        for r in range(world):
            t.write(f"12:00:00.000 [INFO] [T01][rank{r}] init transport: world_size={world} "
                    f"num_slots=10 slot_size=1MB (host cost: stub) shm_prefix={logged}\n")
    slice_gb = job["slice_gb"]
    peaks = [int(g * 1024 * 0.5) for g in slice_gb]
    with open("mig_benchmark_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "batch_size", "microbatch_size", "num_microbatches", "max_new_tokens"]
                   + [f"peak_rank{r}_{slice_gb[r]}gb_mb" for r in range(world)]
                   + ["total_latency_ms", "status"])
        w.writerow([str(job.get("splits") or [[1] * world][0]), 8, 4, 2, job["max_new_tokens"]]
                   + peaks + [1234.5, "ok"])
    with open("mig_memory_trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label", "gpu_mb"] + [f"rank{r}_{slice_gb[r]}gb_mb" for r in range(world)])
        w.writerow(["12:00:00.000", "stub", sum(peaks)] + peaks)
    with open("benchmark.log", "a") as log:
        log.write("[Rank 0] Finished. Latency: 1234 ms\nDone.\n")

    for c in children:
        c.kill()
        c.wait()
    if not job.get("stub_leak_shm"):
        for p in segs:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
    record["finished"] = time.time()
    with open("stub_record.json", "w") as f:
        json.dump(record, f)
    return int(job.get("stub_rc", 0))


if __name__ == "__main__":
    sys.exit(main())
