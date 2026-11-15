"""
CPU / host constants of the box the ranks run on: everything graph_placement.py charges to the
CPU (kernel launch, host copies, gloo messages, token staging, step start), plus the cost of
PyTorch's thread pool on the receive copy (problems.md, "Run 8"). Run it as the harness runs:
same CPU pinning, same governor and OMP_NUM_THREADS as the runs it should describe (for
oracle_v2-like runs: schedutil, OMP_NUM_THREADS unset). Needs one CUDA device (a MIG slice is
fine; only the launch / staging probes touch it). Edit CONFIG in main(), then on the box:

    CUDA_VISIBLE_DEVICES=MIG-<any slice uuid> taskset -c 64-127,192-255 python3 mig_pp/milp/measure_constants_cpu.py

Prints a dict to paste into hardware_constants.py and writes CONFIG["json"]. ~5 min.
"""

import os
import statistics
import time

import numpy as np
import torch

import measure_common as mc


# ---------------------------------------------------------------------------
# host copies (the transport's two memcpys)
# ---------------------------------------------------------------------------
def host_copies(nbytes):
    """(pinned -> SHM, SHM -> pinned) bytes/s: _flush_staging_to_shm (numpy slice
    assignment) and _queue_read_from_slot (torch copy_ from a numpy view of SHM)."""
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(create=True, size=nbytes)
    try:
        slot = np.ndarray((nbytes,), dtype=np.uint8, buffer=shm.buf)
        pinned = torch.empty(nbytes // 2, dtype=torch.float16).pin_memory()
        raw = pinned.numpy().view(np.uint8)
        peer = torch.from_numpy(slot.view(np.float16))

        def send():
            slot[:nbytes] = raw[:nbytes]

        def recv():
            pinned.copy_(peer)

        out = nbytes / mc.wall_s(send), nbytes / mc.wall_s(recv)
        del slot, raw, peer  # drop the views of SHM before closing it
        return out
    finally:
        shm.close()
        shm.unlink()


def _pool_worker(shm_name, numel, threads, seconds, gap_s, barrier, q):
    """One 'rank': the receive copy (torch copy_ from a numpy view of SHM) started every gap_s
    (start to start), for `seconds`, all workers at once. No CUDA: a pageable destination, same CPU
    copy path. Drops copies in the first quarter of the elapsed time (adjustment). Returns
    (threads, copy durations, achieved copies per second)."""
    from multiprocessing.shared_memory import SharedMemory

    if threads:
        torch.set_num_threads(threads)
    shm = SharedMemory(name=shm_name)
    peer = torch.from_numpy(np.ndarray((numel,), dtype=np.float16, buffer=shm.buf))
    dst = torch.empty(numel, dtype=torch.float16)
    for _ in range(20):
        dst.copy_(peer)
    barrier.wait()
    times, start = [], time.perf_counter()
    next_t = start
    while time.perf_counter() < start + seconds:
        t0 = time.perf_counter()
        dst.copy_(peer)
        times.append((t0 - start, time.perf_counter() - t0))
        next_t += gap_s  # start-to-start interval; a copy longer than it starts the next one late
        time.sleep(max(0.0, next_t - time.perf_counter()))
    del peer
    shm.close()
    kept = [d for at, d in times if at >= 0.25 * seconds]
    achieved_hz = sum(1 for at, _ in times if at >= 0.25 * seconds) / (0.75 * seconds)
    q.put((torch.get_num_threads(), kept, achieved_hz))


def pool_copy_us(numel, procs, threads, seconds=3.0, gap_s=0.02):
    """(median, p90, threads, per-process [(median, p90, achieved copies/s)]) us of one receive copy of numel fp16
    values, `procs` processes copying at once on this process's CPUs; threads=None keeps PyTorch's
    default pool."""
    import multiprocessing as mp
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(create=True, size=numel * 2)
    try:
        ctx = mp.get_context("spawn")
        barrier, q = ctx.Barrier(procs), ctx.Queue()
        ps = [ctx.Process(target=_pool_worker, args=(shm.name, numel, threads, seconds, gap_s, barrier, q))
              for _ in range(procs)]
        for p in ps:
            p.start()
        res = [q.get(timeout=120) for _ in ps]
        for p in ps:
            p.join(timeout=60)
    finally:
        shm.close()
        shm.unlink()
    per = [(statistics.median(ts) * 1e6, sorted(ts)[int(0.9 * len(ts))] * 1e6, hz) for _, ts, hz in res if ts]
    t = sorted(x for _, ts, _ in res for x in ts)
    return statistics.median(t) * 1e6, t[int(0.9 * len(t))] * 1e6, res[0][0], per


def summary(values):
    """Repeated windows of one measurement -> median, relative spread (max - min) / min, all values."""
    return dict(median=statistics.median(values), spread=(max(values) - min(values)) / min(values), windows=list(values))


def recv_copy_gap_us(numel, gap_ms, seconds=4.0):
    """One process doing the transport's receive copy (torch copy_ from a numpy view of SHM into
    pinned memory) started every gap_ms (start to start), asleep in between (0 = back to back): what the governor and the
    thread pool do to it when the CPU idles between copies, as in a decode. (median, p90) us."""
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(create=True, size=numel * 2)
    try:
        peer = torch.from_numpy(np.ndarray((numel,), dtype=np.float16, buffer=shm.buf))
        dst = torch.empty(numel, dtype=torch.float16).pin_memory()
        for _ in range(20):
            dst.copy_(peer)
        times, start = [], time.perf_counter()
        next_t = start
        while time.perf_counter() < start + seconds:
            t0 = time.perf_counter()
            dst.copy_(peer)
            times.append((t0 - start, time.perf_counter() - t0))
            if gap_ms:  # start-to-start interval
                next_t += gap_ms / 1000
                time.sleep(max(0.0, next_t - time.perf_counter()))
        del peer
    finally:
        shm.close()
        shm.unlink()
    t = sorted(d for at, d in times if at >= 0.25 * seconds)  # drop the first quarter of the elapsed time
    return statistics.median(t) * 1e6, t[int(0.9 * len(t))] * 1e6


# ---------------------------------------------------------------------------
# small CUDA-side CPU costs and gloo
# ---------------------------------------------------------------------------
def token_staging_us(batch):
    """TokenExchange: D2H of next_tokens + stream sync, then H2D + stream sync."""
    dev_tok = torch.zeros((batch, 1), dtype=torch.long, device="cuda")
    host = torch.empty((batch, 1), dtype=torch.long).pin_memory()

    def exchange():
        host.copy_(dev_tok)
        torch.cuda.current_stream().synchronize()
        dev_tok.copy_(host)
        torch.cuda.current_stream().synchronize()

    return mc.wall_s(exchange) * 1e6


def step_start_us(mb, seq):
    """Rank 0's work at a decode step start: position_ids built on the device (a pageable
    H2D that waits for itself), expand().contiguous(), the step event's first record."""
    def step():
        torch.tensor([[seq]], dtype=torch.long, device="cuda").expand(mb, -1).contiguous()
        torch.cuda.Event(enable_timing=True).record()

    return mc.wall_s(step) * 1e6


def _gloo_worker(rank, world, port, batch, reps, windows, q):
    """handshake + ACK (ranks 0 and 1, as FLUSH/RECV), next_tokens (last rank -> 0),
    barrier (all). Times are rank 0's. One warm-up pass, then `windows` passes, each kept."""
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    hs, ack = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    tok = torch.zeros((batch, 1), dtype=torch.long)
    out = {"handshake_ack_us": [], "tok_roundtrip_us": [], "barrier_us": []}
    for w in range(windows + 1):
        dist.barrier()
        t0 = time.perf_counter()
        for i in range(reps):
            if rank == 0:  # FLUSH: post the ACK irecv, send the handshake ... wait() later
                r = dist.irecv(ack, src=1, tag=2 * i + 1)
                dist.send(hs, dst=1, tag=2 * i)
                r.wait()
            elif rank == 1:  # RECV: handshake in, ACK out
                dist.recv(hs, src=0, tag=2 * i)
                dist.send(ack, dst=0, tag=2 * i + 1)
        hs_us = (time.perf_counter() - t0) / reps * 1e6
        dist.barrier()
        t0 = time.perf_counter()
        for i in range(reps):
            if rank == world - 1:
                dist.send(tok, dst=0, tag=100_000 + i)
            elif rank == 0:
                dist.recv(tok, src=world - 1, tag=100_000 + i)
        tok_us = (time.perf_counter() - t0) / reps * 1e6
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(reps):
            dist.barrier()
        bar_us = (time.perf_counter() - t0) / reps * 1e6
        if w > 0:  # window 0 is the warm-up
            out["handshake_ack_us"].append(hs_us)
            out["tok_roundtrip_us"].append(tok_us)
            out["barrier_us"].append(bar_us)
    if rank == 0:
        q.put(out)
    dist.destroy_process_group()


def gloo_probes(world, port, batch, windows, reps=200):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_gloo_worker, args=(r, world, port, batch, reps, windows, q)) for r in range(world)]
    for p in procs:
        p.start()
    out = q.get(timeout=600)
    for p in procs:
        p.join(timeout=60)
    return {k: summary(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
def main():
    CONFIG = dict(
        model_path="lmsys/vicuna-7b-v1.5",  # for the layer launch probe (only config.json is fetched)
        transport_mb=8,  # decode activation for the host copy probe
        seq_len=64,
        batch=32,  # next_tokens rows
        world=4,  # ranks for the gloo probes and the thread-pool probe
        port=29910,
        pool_mbs=[8, 16],  # receive copy of mb x hidden values: 32768 (one thread) and 65536 (the pool) at hidden 4096
        copy_gaps_ms=[0, 10, 50],  # receive copy every N ms, asleep between (0 = back to back)
        windows=5,  # repeated windows per number (median and spread kept)
        json="constants_cpu.json",
    )
    c = CONFIG

    host = mc.host_info()
    print("host: " + ", ".join(f"{k} {v}" for k, v in host.items()))
    if host["governor"] != "schedutil" or host["omp_num_threads"]:
        print("NOTE: not oracle_v2's conditions (schedutil, OMP_NUM_THREADS unset); the numbers describe these settings")

    W = c["windows"]
    # kernel launch: CPU time to queue one decode layer call / its kernels (GPU idle at each start)
    layer, rotary = mc.load_model(c["model_path"])
    kpl, _ = mc.layer_kernel_us(layer, rotary, 1, c["seq_len"], reps=1)
    launch = summary([mc.layer_dispatch_us(layer, rotary, c["transport_mb"], c["seq_len"]) / kpl for _ in range(W)])
    with torch.no_grad():
        h, _, pos = mc.decode_input(rotary, c["transport_mb"], c["seq_len"])
        rotary_kernels, _ = mc.kernels(lambda: rotary(h, pos))
    print(f"\nlaunch: {kpl} kernels per decode layer x {launch['median']:.1f} us (median of {W} windows, spread "
          f"{100 * launch['spread']:.1f}%); rotary {rotary_kernels} kernels  [wall time, CPU only]")

    # host copies of one decode activation, back to back, W windows
    nbytes = c["transport_mb"] * mc.HIDDEN * 2
    rates = [host_copies(nbytes) for _ in range(W)]
    send, recv = summary([r[0] for r in rates]), summary([r[1] for r in rates])
    print(f"host copies, {nbytes // 1024} KiB, back to back: pinned->SHM {send['median'] / 1e9:.2f} GB/s "
          f"({100 * send['spread']:.1f}%), SHM->pinned {recv['median'] / 1e9:.2f} GB/s ({100 * recv['spread']:.1f}%)  [wall time]")

    # the receive copy with gaps: one process, asleep between copies, as in a decode
    print("\nreceive copy, one process, asleep between copies (median / p90 us per copy)  [wall time]:")
    gaps = {}
    for mb in c["pool_mbs"]:
        numel = mb * mc.HIDDEN
        for gap in c["copy_gaps_ms"]:
            med, p90 = recv_copy_gap_us(numel, gap)
            gaps[f"mb{mb} gap{gap}ms"] = dict(median_us=med, p90_us=p90)
            print(f"  mb {mb:3d} ({numel} values{', pool' if numel > 32768 else ''}), every {gap:3d} ms: {med:9.1f} / {p90:9.1f}")

    # the receive copy with PyTorch's thread pool, `world` ranks copying at once on these CPUs
    print(f"\nreceive copy, {c['world']} processes at once on {host['allowed_cpus']} CPUs, one copy every 20 ms each "
          f"(median / p90 us per copy)  [wall time]:")
    pool = {}
    for mb in c["pool_mbs"]:
        numel = mb * mc.HIDDEN
        for label, threads in (("default threads", None), ("1 thread", 1)):
            med, p90, n, per = pool_copy_us(numel, c["world"], threads)
            pool[f"mb{mb} {label}"] = dict(median_us=med, p90_us=p90, threads=n, per_process=per)
            print(f"  mb {mb:3d} ({numel} values{', over 32768: pool' if numel > 32768 else ''}), {label} ({n}): "
                  f"{med:8.1f} / {p90:8.1f}")

    staging = summary([token_staging_us(c["batch"]) for _ in range(W)])
    step = summary([step_start_us(c["transport_mb"], c["seq_len"]) for _ in range(W)])
    g = gloo_probes(c["world"], c["port"], c["batch"], W)
    print(f"\nmedian of {W} windows (spread): token_staging_us {staging['median']:.1f} ({100 * staging['spread']:.0f}%), "
          f"step_start_us {step['median']:.1f} ({100 * step['spread']:.0f}%), "
          + ", ".join(f"{k} {v['median']:.1f} ({100 * v['spread']:.0f}%)" for k, v in g.items()) + "  [wall time]")

    big = max(c["pool_mbs"])
    print(f"""
# measured on {host['cpu']}, governor {host['governor']}, {host['allowed_cpus']} CPUs,
# OMP_NUM_THREADS={host['omp_num_threads'] or 'unset'} (torch threads {host['torch_threads']}); medians of {W} windows
CPU_CONSTANTS["<box>"] = dict(
    launch_us={launch['median']:.1f},
    kernels_per_layer={kpl},
    rotary_kernels={rotary_kernels},
    memcpy_bw={send['median']:.3g},  # pinned -> SHM, back to back
    shm_to_pinned_bw={recv['median']:.3g},  # SHM -> pinned, back to back
    handshake_ack_us={g['handshake_ack_us']['median']:.1f},
    tok_roundtrip_us={g['tok_roundtrip_us']['median']:.1f},
    token_staging_us={staging['median']:.1f},
    step_start_us={step['median']:.1f},
    barrier_us={g['barrier_us']['median']:.1f},
    recv_copy_pool_us={pool[f'mb{big} default threads']['median_us']:.1f},  # over 32768 values, {c['world']} ranks at once
    recv_copy_one_thread_us={pool[f'mb{big} 1 thread']['median_us']:.1f},  # same copy, OMP_NUM_THREADS=1
)""")
    mc.save_json(c["json"], dict(host=host, config=c, kernels_per_layer=kpl, rotary_kernels=rotary_kernels,
                                 launch_us=launch, memcpy_bw=send, shm_to_pinned_bw=recv, token_staging_us=staging,
                                 step_start_us=step, recv_copy_gaps=gaps, pool_copy=pool, **g,
                                 timing="all wall time (CPU, and GPU waits where the transport waits)"))

if __name__ == "__main__":
    main()
