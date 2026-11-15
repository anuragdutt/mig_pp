"""
Two generic CPU-interaction probes (no model, nothing workload-specific), current.md step 3:

  A. Thread-pool interference: one process dispatches small GPU ops (like a rank launching
     kernels; back to back, or in bursts with gaps as in a decode) while copier processes do the
     transport's receive copy (torch copy_ from a numpy view of SHM) every few ms, like ranks 1-3.
     Sweep: receive size (32768 = one thread by PyTorch's rule, 65536 / 131072 = the pool), copier
     count, copy interval, and a one-thread control. Two rounds, the second in reverse order, so
     clock or temperature drift does not favour one case. Reports the dispatcher's CPU time per op
     (median / p90) and each copier's median / p90.
  B. Governor response: one process alternates busy (a fixed Python work chunk, timed) and asleep,
     at busy fractions 10..90% and several periods. Speed relative to a 100% busy reference
     ~ the clock it gets (time per chunk ~ 1/frequency); the first quarter of each window's elapsed
     time is dropped (adjustment). Then 4 processes at once, as 4 ranks. Prints the cpufreq settings.

All times are wall time. Run it as the harness runs: same CPU pinning, governor and
OMP_NUM_THREADS as the runs it should describe (schedutil, OMP_NUM_THREADS unset), nothing else
running, one MIG slice for probe A:

    CUDA_VISIBLE_DEVICES=MIG-<any slice uuid> taskset -c 64-127,192-255 python3 mig_pp/milp/probe_cpu_interactions.py

~12 min. Prints tables and writes CONFIG["json"].
"""

import glob
import os
import random
import statistics
import time

import torch

import measure_common as mc
from measure_constants_cpu import _pool_worker


def quantiles(values):
    v = sorted(values)
    return dict(median=statistics.median(v), p90=v[int(0.9 * len(v))], n=len(v))


# ---------------------------------------------------------------------------
# A. thread-pool interference on a neighbour's op dispatch
# ---------------------------------------------------------------------------
def dispatch_us_per_op(seconds, gap_ms=0, ops=80):
    """CPU time per op to dispatch small GPU ops (an elementwise op and a small matmul,
    alternating) in bursts of `ops`, asleep gap_ms between bursts (0 = back to back), the GPU idle
    at each burst start. Bursts in the first quarter of the elapsed time are dropped."""
    x = torch.randn(256, device="cuda")
    w = torch.randn(64, 64, device="cuda")
    v = torch.randn(64, 64, device="cuda")
    per, start = [], time.perf_counter()
    while time.perf_counter() < start + seconds:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ops // 2):
            x.mul_(1.0)
            torch.mm(w, v)
        if t0 - start >= 0.25 * seconds:
            per.append((time.perf_counter() - t0) / ops * 1e6)
        if gap_ms:
            time.sleep(gap_ms / 1000)
    return quantiles(per)


def one_case(c, dispatch_gap, copier):
    """copier: None, or (numel, copiers, copy interval ms, threads or None)."""
    import multiprocessing as mp
    from multiprocessing.shared_memory import SharedMemory

    if copier is None:
        return dict(dispatch=dispatch_us_per_op(c["case_s"], dispatch_gap))
    numel, n, gap_ms, threads = copier
    shm = SharedMemory(create=True, size=numel * 2)
    try:
        ctx = mp.get_context("spawn")
        barrier, q = ctx.Barrier(n + 1), ctx.Queue()
        ps = [ctx.Process(target=_pool_worker, args=(shm.name, numel, threads, c["case_s"], gap_ms / 1000, barrier, q))
              for _ in range(n)]
        for p in ps:
            p.start()
        barrier.wait()  # the copiers start their loop now
        d = dispatch_us_per_op(c["case_s"], dispatch_gap)
        res = [q.get(timeout=120) for _ in ps]
        for p in ps:
            p.join(timeout=60)
    finally:
        shm.close()
        shm.unlink()
    copiers = [dict(quantiles([t * 1e6 for t in ts]), achieved_hz=hz, target_hz=1000 / gap_ms) for _, ts, hz in res if ts]
    return dict(dispatch=d, copier_threads=res[0][0], copiers=copiers)


def interference(c):
    copier_cases = [None] + [tuple(x) for x in c["copier_cases"]]
    cases = [(g, cc) for g in c["dispatch_gaps_ms"] for cc in copier_cases]
    random.Random(1).shuffle(cases)
    order = cases + cases[::-1]  # round 2 in reverse order
    dispatch_us_per_op(2.0)  # warm up CUDA and the clock
    results = {}
    print(f"\nA. op dispatch next to receive copiers ({len(cases)} cases, 2 rounds, the second reversed; "
          f"{c['case_s']:g} s each)  [wall time]", flush=True)
    for i, (gap, cc) in enumerate(order):
        r = one_case(c, gap, cc)
        results.setdefault((gap, cc), []).append(r)
        print(f"  [{i + 1:2d}/{len(order)}] dispatch every {gap:2d} ms, copiers {cc}: "
              f"{r['dispatch']['median']:.2f} / {r['dispatch']['p90']:.2f} us per op", flush=True)
    out = []
    print("\n  summary: dispatch us per op, median of the 2 rounds (vs no copiers at the same dispatch gap);"
          " copies median / p90 us (worst copier)")
    for gap in c["dispatch_gaps_ms"]:
        base = statistics.median(r["dispatch"]["median"] for r in results[(gap, None)])
        for cc in copier_cases:
            rs = results[(gap, cc)]
            med = statistics.median(r["dispatch"]["median"] for r in rs)
            row = dict(dispatch_gap_ms=gap, copier=cc, dispatch_us=med, slowdown=med / base,
                       rounds=[r["dispatch"] for r in rs])
            text = f"    gap {gap:2d} ms, copiers {str(cc):<28} {med:7.2f} us (x{med / base:.2f})"
            if cc is not None:
                worst = [max(rs_["copiers"], key=lambda q_: q_["median"]) for rs_ in rs]
                row.update(copier_threads=rs[0]["copier_threads"], copiers=[r["copiers"] for r in rs])
                text += (f"   copies {statistics.median(w['median'] for w in worst):9.1f} / {statistics.median(w['p90'] for w in worst):9.1f}"
                         f"   achieved {statistics.median(w['achieved_hz'] for w in worst):5.1f} of {worst[0]['target_hz']:.1f} copies/s")
            out.append(row)
            print(text)
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
    """Busy for frac x period (timed work chunks), asleep for the rest; median chunk time over the
    window after its first quarter of elapsed time (the governor adjusting). frac >= 1: never sleeps."""
    if barrier is not None:
        barrier.wait()
    times, start = [], time.perf_counter()
    while time.perf_counter() < start + seconds:
        p0 = time.perf_counter()
        busy_until = p0 + frac * period_s
        while time.perf_counter() < busy_until:
            t0 = time.perf_counter()
            _chunk(chunk_n)
            if t0 - start >= 0.25 * seconds:
                times.append(time.perf_counter() - t0)
        rest = p0 + period_s - time.perf_counter()
        if frac < 1 and rest > 0:
            time.sleep(rest)
    q.put((frac, statistics.median(times), len(times)))


def run_duty(cases, seconds, chunk_n):
    """cases: [(frac, period_s)] run at the same time, one process each -> [(frac, chunk s, n)]."""
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
    print(f"   100% busy reference: {ref * 1e6:.1f} us per chunk (= relative speed 1.00)  [wall time]")
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
        dispatch_gaps_ms=[0, 25],  # dispatcher: back to back, or bursts with 25 ms asleep between
        copier_cases=[  # (receive copy values, copier processes, start-to-start copy interval ms, threads; None = default pool)
            (65536, 3, 25, None), (65536, 3, 25, 1), (32768, 3, 25, None), (131072, 3, 25, None),
            (65536, 1, 25, None), (65536, 3, 10, None), (65536, 3, 55, None)],
        case_s=10.0,  # seconds per interference case
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
    mc.save_json(c["json"], dict(host=host, config=c, interference=a, governor=b, timing="all wall time"))


if __name__ == "__main__":
    main()
