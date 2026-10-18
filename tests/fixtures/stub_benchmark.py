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
  stub_rc_node     two-node job: only this node exits with stub_rc (the other exits 0)
  stub_wrong_prefix  log a different shm_prefix in [T01] (as a transport that
                   ignored MIG_SHM_PREFIX would)
  stub_leak_shm    leave its segments behind on exit (default false)
  stub_ignore_term ignore SIGTERM, so stop has to escalate to KILL

Two-node job (job.json has rank_nodes): spawns only this node's ranks and runs
the real two_node_control protocol with the other node's stub, the way the
benchmark parent will: hello ([B09] line), one ready per config, the non-lead
ships its ranks' weights and peaks in a result, bye. The lead logs the remote
ranks' B08 lines and writes the results CSV for every rank plus a placement
column; the non-lead writes no results CSV. Memory trace columns carry global
rank numbers.
"""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import two_node_control as tc  # noqa: E402  (stdlib only)


def connect(job):
    """The control socket, after a matching hello; the [B09] line goes to benchmark.log."""
    lead = job["rank_nodes"][0]
    addr, port, wait = job["node_addrs"][lead], job["control_port"], job["peer_timeout_s"]
    if job["node"] == lead:
        sock = tc.listen_for_peer(addr, port, wait)
    else:
        sock = tc.connect_to_lead(addr, port, wait)
    mine = tc.make_hello(job["node"], job["run_id"], str(REPO), job)
    theirs = tc.exchange_hello(sock, mine)
    with open("benchmark.log", "a") as log:
        log.write(tc.hello_line(mine, theirs) + "\n")
    return sock


def main() -> int:
    job = json.load(open(os.environ["MIG_EXP_CONFIG"]))
    world = len(job["mig_uuids"])
    rank_nodes = job.get("rank_nodes")
    local = [r for r in range(world) if rank_nodes is None or rank_nodes[r] == job["node"]]
    lead = rank_nodes is None or rank_nodes[0] == job["node"]
    sock = connect(job) if rank_nodes else None
    if sock is not None:
        already_ok = os.environ.get("MIG_JOB_ALREADY_OK") == "1"
        if tc.agree_on_resume(sock, already_ok, job["peer_timeout_s"]):
            with open("benchmark.log", "a") as log:
                log.write("[B10] both nodes finished this job already: skipped\n")
            sock.close()
            return 0
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

    segs = [os.path.join(shm_dir, f"{prefix}_{r}_slot0") for r in local]
    for p in segs:
        with open(p, "wb") as f:
            f.write(b"\0" * 16)

    split = (job.get("splits") or [[1] * world])[0]
    if sock is not None:  # lockstep: both nodes start config 0 together
        tc.sync_ready(sock, 0, f"s{'_'.join(map(str, split))}_b8_mb4", job["peer_timeout_s"])

    # One child per local rank, like mp.Process in the real parent. They sleep
    # a little longer than the parent so an early kill must reach them too.
    children = [
        subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({sleep_s + 30})"])
        for _ in local
    ]
    record["children"] = [c.pid for c in children]
    with open("stub_record.json", "w") as f:
        json.dump(record, f)

    os.makedirs("logs", exist_ok=True)
    with open("benchmark.log", "a") as log:
        log.write(f"Parallel job {job.get('job_id')}: stub\n")
        log.write("[1/1] Split [x] | Batch: 8 | Microbatch: 4 | Microbatches: 2\n")
        for r in local:
            log.write(f"[B08][Rank {r}] weights: loaded=10/10 missing=0 format=safetensors files=1\n")
    time.sleep(sleep_s)

    logged = "mig_pipe_shm" if job.get("stub_wrong_prefix") else prefix
    with open(os.path.join("logs", "transport_stub.log"), "a") as t:
        for r in local:
            t.write(f"12:00:00.000 [INFO] [T01][rank{r}] init transport: world_size={world} "
                    f"num_slots=10 slot_size=1MB (host cost: stub) shm_prefix={logged}\n")
    slice_gb = job["slice_gb"]
    peaks = {r: int(slice_gb[r] * 1024 * 0.5) for r in local}

    if sock is not None:  # the non-lead ships its ranks' numbers; the lead merges them
        if lead:
            remote = tc.recv_result(sock, job["peer_timeout_s"])
            peaks.update({int(r): mb for r, mb in remote["peaks"].items()})
            with open("benchmark.log", "a") as log:
                for r in remote["weights"]:
                    log.write(f"[B08][Rank {r}] weights: loaded=10/10 missing=0 "
                              f"format=safetensors files=1 (node {remote['node']})\n")
        else:
            tc.send_result(sock, {"node": job["node"], "weights": local,
                                  "peaks": {str(r): mb for r, mb in peaks.items()}})

    if lead:
        extra = ["placement"] if rank_nodes else []
        with open("mig_benchmark_results.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "batch_size", "microbatch_size", "num_microbatches", "max_new_tokens"]
                       + [f"peak_rank{r}_{slice_gb[r]}gb_mb" for r in range(world)]
                       + ["total_latency_ms", "status"] + extra)
            w.writerow([str(job.get("splits") or [[1] * world][0]), 8, 4, 2, job["max_new_tokens"]]
                       + [peaks[r] for r in range(world)] + [1234.5, "ok"]
                       + ([job["placement"]] if rank_nodes else []))
    with open("mig_memory_trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label", "gpu_mb"] + [f"rank{r}_{slice_gb[r]}gb_mb" for r in local])
        w.writerow(["12:00:00.000", "stub", sum(peaks[r] for r in local)] + [peaks[r] for r in local])
    rc = int(job.get("stub_rc", 0))
    if job.get("stub_rc_node") not in (None, job.get("node")):
        rc = 0  # stub_rc is meant for the other node
    if sock is not None:  # the job is ok only if both nodes finished cleanly
        if not tc.finish(sock, rc == 0, job["peer_timeout_s"]):
            rc = rc or 1
        sock.close()
    with open("benchmark.log", "a") as log:
        log.write("[Rank 0] Finished. Latency: 1234 ms\nDone.\n" if lead else "Done.\n")

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
    return rc


if __name__ == "__main__":
    sys.exit(main())
