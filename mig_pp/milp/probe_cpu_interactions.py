"""
Two generic CPU-interaction probes (no model, nothing workload-specific), current.md step 3:

  A. Thread-pool interference: one process dispatches small GPU ops in a loop (like a rank
     launching kernels) while `copiers` other processes do the transport's receive copy (torch
     copy_ from a numpy view of SHM) every `gap_ms`, like ranks 1-3. Cases: no copiers; copies
     of 65536 values (PyTorch's thread pool); the same with one thread; 32768 values (one thread
     by PyTorch's own rule). Reports the dispatcher's CPU time per op: how much the pool slows a
     neighbour's launches.
  B. Governor response: one process alternates busy (a fixed Python work chunk, timed) and asleep,
     at busy fractions 10..90% and several periods. Speed relative to a 100% busy reference
     ~ the clock it gets (time per chunk ~ 1/frequency). Then 4 processes at once with different
     fractions, as 4 ranks. Also prints the box's cpufreq settings.

Run it as the harness runs: same CPU pinning, governor and OMP_NUM_THREADS as the runs it should
describe (schedutil, OMP_NUM_THREADS unset), nothing else running, one MIG slice for probe A:

    CUDA_VISIBLE_DEVICES=MIG-<any slice uuid> taskset -c 64-127,192-255 python3 mig_pp/milp/probe_cpu_interactions.py

~6 min. Prints tables and writes CONFIG["json"].
"""

import glob
import json
import os
import statistics
import time

import numpy as np
import torch

import measure_common as mc
from measure_constants_cpu import _pool_worker


# ---------------------------------------------------------------------------
# A. thread-pool interference on a neighbour's op dispatch
# ---------------------------------------------------------------------------
def dispatch_us_per_op(seconds, ops=80):
    """CPU time per op to dispatch small GPU ops (an elementwise op and a small matmul,
    alternating), the GPU idle at each burst start. (median, p90) us."""
    x = torch.randn(256, device="cuda")
    w = torch.randn(64, 64, device="cuda")
    v = torch.randn(64, 64, device="cuda")
    per, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ops // 2):
            x.mul_(1.0)
            torch.mm(w, v)
        per.append((time.perf_counter() - t0) / ops * 1e6)
    per.sort()
    return statistics.median(per), per[int(0.9 * len(per))]


def interference(c):
    import multiprocessing as mp
    from multiprocessing.shared_memory import SharedMemory

    dispatch_us_per_op(2.0)  # warm up CUDA and the clock
    cases = [("no copiers", None, None), ("65536 values, default pool", 65536, None),
             ("65536 values, 1 thread", 65536, 1), ("32768 values, default pool", 32768, None)]
    out = {}
    print(f"\nA. op dispatch with {c['copiers']} receive copiers, one copy every {c['gap_ms']} ms each "
          f"(median / p90 us per op; copy median us):")
    for label, numel, threads in cases:
        if numel is None:
            med, p90 = dispatch_us_per_op(c["case_s"])
            out[label] = dict(dispatch_us=med, dispatch_p90_us=p90)
            print(f"  {label:<28} {med:7.2f} / {p90:7.2f}")
            continue
        shm = SharedMemory(create=True, size=numel * 2)
        try:
            ctx = mp.get_context("spawn")
            barrier, q = ctx.Barrier(c["copiers"] + 1), ctx.Queue()
            ps = [ctx.Process(target=_pool_worker, args=(shm.name, numel, threads, c["case_s"],
                                                         c["gap_ms"] / 1000, barrier, q))
                  for _ in range(c["copiers"])]
            for p in ps:
                p.start()
            barrier.wait()  # the copiers start their loop now
            med, p90 = dispatch_us_per_op(c["case_s"])
            res = [q.get(timeout=120) for _ in ps]
            for p in ps:
                p.join(timeout=60)
        finally:
            shm.close()
            shm.unlink()
        copy_us = statistics.median(x for _, ts in res for x in ts) * 1e6
        out[label] = dict(dispatch_us=med, dispatch_p90_us=p90, copy_us=copy_us, copier_threads=res[0][0])
        print(f"  {label:<28} {med:7.2f} / {p90:7.2f}   copies {copy_us:8.1f} us ({res[0][0]} threads)")
    base = out["no copiers"]["dispatch_us"]
    for label in out:
        out[label]["slowdown"] = out[label]["dispatch_us"] / base
    print("  dispatch slowdown vs no copiers: " + ", ".join(f"{k} x{v['slowdown']:.2f}" for k, v in out.items()))
    return out


# ---------------------------------------------------------------------------
# B. governor response to busy / asleep patterns
# ---------------------------------------------------------------------------
def _chunk(n):
    s = 0
    for i in range(n):
        s += i
    return s


def _duty_worker(frac, period_s, seconds, chunk_n, barrier, q):
    """Busy for frac x period (timed work chunks), asleep for the rest; median chunk time
    after the first quarter (the governor's adaptation). frac >= 1: never sleeps."""
    if barrier is not None:
        barrier.wait()
    times, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        start = time.perf_counter()
        busy_until = start + frac * period_s
        while time.perf_counter() < busy_until:
            t0 = time.perf_counter()
            _chunk(chunk_n)
            times.append(time.perf_counter() - t0)
        rest = start + period_s - time.perf_counter()
        if frac < 1 and rest > 0:
            time.sleep(rest)
    k = len(times) // 4
    q.put((frac, statistics.median(times[k:]), len(times)))


def run_duty(cases, seconds, chunk_n):
    """cases: [(frac, period_s)] run at the same time, one process each -> [(frac, chunk s)]."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(len(cases)), ctx.Queue()
    ps = [ctx.Process(target=_duty_worker, args=(f, p, seconds, chunk_n, barrier, q)) for f, p in cases]
    for p in ps:
        p.start()
    res = [q.get(timeout=seconds + 120) for _ in ps]
    for p in ps:
        p.join(timeout=60)
    return sorted(res)


def cpufreq_facts():
    def read(path):
        try:
            return open(path).read().strip()
        except OSError:
            return None
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    base = f"/sys/devices/system/cpu/cpu{cpus[0]}/cpufreq"
    facts = {k: read(f"{base}/{k}") for k in ("scaling_driver", "scaling_governor", "scaling_available_frequencies",
                                              "cpuinfo_min_freq", "cpuinfo_max_freq", "scaling_max_freq")}
    facts["boost"] = read("/sys/devices/system/cpu/cpufreq/boost")
    rl = glob.glob(f"{base}/schedutil/rate_limit_us") + glob.glob("/sys/devices/system/cpu/cpufreq/schedutil/rate_limit_us")
    facts["schedutil_rate_limit_us"] = read(rl[0]) if rl else None
    return facts


def governor(c):
    facts = cpufreq_facts()
    print("\nB. cpufreq: " + ", ".join(f"{k} {v}" for k, v in facts.items()))
    ref = run_duty([(1.0, 0.1)], c["duty_s"], c["chunk_n"])[0][1]
    print(f"   100% busy reference: {ref * 1e6:.1f} us per chunk (= relative speed 1.00)")
    table = {}
    for period in c["periods_ms"]:
        row = []
        for frac in c["fracs"]:
            t = run_duty([(frac, period / 1000)], c["duty_s"], c["chunk_n"])[0][1]
            row.append(ref / t)
        table[period] = dict(zip(c["fracs"], row))
        print(f"   period {period:4d} ms, busy " + "  ".join(f"{int(f * 100):2d}%:{r:.2f}" for f, r in zip(c["fracs"], row)))
    together = run_duty([(f, c["periods_ms"][0] / 1000) for f in c["together"]], c["duty_s"], c["chunk_n"])
    print(f"   {len(together)} processes at once (period {c['periods_ms'][0]} ms): "
          + "  ".join(f"busy {int(f * 100)}%:{ref / t:.2f}" for f, t, _ in together))
    print("   relative speed ~ clock / boost clock: 1.0 boost; ~0.7 at 2.45 GHz, ~0.57 at 2.0, ~0.43 at 1.5"
          " if boost ~3.5 GHz (sysfs above)")
    return dict(facts=facts, ref_chunk_us=ref * 1e6, speed=table,
                together={f"{int(f * 100)}%": ref / t for f, t, _ in together})


# ---------------------------------------------------------------------------
def main():
    CONFIG = dict(
        copiers=3,  # receive copiers, like ranks 1-3
        gap_ms=25,  # one receive copy every gap_ms per copier
        case_s=15.0,  # seconds per interference case
        fracs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        periods_ms=[40, 120],  # duty-cycle periods (decode steps are ~40-140 ms)
        duty_s=8.0,  # seconds per duty-cycle case
        chunk_n=5000,  # Python loop iterations per timed work chunk (~0.2-0.5 ms)
        together=[0.15, 0.3, 0.55, 0.8],  # 4 processes at once, like 4 ranks
        json="probe_cpu_interactions.json",
    )
    c = CONFIG
    host = mc.host_info()
    print("host: " + ", ".join(f"{k} {v}" for k, v in host.items()))
    if host["governor"] != "schedutil" or host["omp_num_threads"]:
        print("NOTE: not the runs' conditions (schedutil, OMP_NUM_THREADS unset); results describe these settings")
    a = interference(c)
    b = governor(c)
    mc.save_json(c["json"], dict(host=host, config=c, interference=a, governor=b))


if __name__ == "__main__":
    main()
