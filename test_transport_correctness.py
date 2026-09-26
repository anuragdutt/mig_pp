"""
Correctness + overlap self-test for the async SHM transport.

Run this BEFORE the benchmark sweep. It is cheap (seconds) and catches the
failure modes that would otherwise corrupt a whole sweep:

  1. Data corruption from the async D2H copy (missing record_stream, or
     reading pinned staging before the copy landed).
  2. Handshake/ACK protocol deadlock.
  3. Slot exhaustion under the batched ACK drain.
  4. No actual overlap (i.e. the fix did not take effect).

Usage (on the MIG box, same UUIDs as the benchmark):
    python test_transport_correctness.py
"""

import os
import datetime
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import mig_transport_pipeline as mig_transport

# Keep this file standalone (no transformers/datasets import) so it can run
# before the model is even downloaded. Paste the SAME UUIDs you put in
# benchmark_pipeline_microbatching.py.
MIG_UUIDS = [
    "MIG-cbf6f13f-88d6-550a-95b3-259a93afe90f",  # Rank 0: 20GB (3g.20gb)
    "MIG-3551cc21-290c-58ef-936e-50bc04135d53",  # Rank 1: 10GB (2g.10gb)
    "MIG-1e5ad904-ba2b-5830-9639-2ded2002e3a7",  # Rank 2: 10GB (2g.10gb)
]
WORLD_SIZE = len(MIG_UUIDS)

MB_SIZE = 4
SEQ_LEN = 64
HIDDEN = 4096  # vicuna-7b
NUM_MICROBATCHES = 4
TAG_BASE = 1000


def worker(rank, world_size, result_queue, device_uuid):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29777"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=3),
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Transport logging -> logs/transport_<stamp>_selftest.log, shared by all
    # ranks. This is where the real evidence lands: per-send T05 queue times,
    # T07 flush block times, and the T29/T30 verdict per rank.
    log_path = mig_transport.setup_transport_logging(run_label="selftest")
    if rank == 0:
        print(f"[transport log] {log_path}", flush=True)

    mig_transport.register_hooks(
        mb_size=MB_SIZE,
        seq_len=SEQ_LEN,
        hidden_size=HIDDEN,
        num_slots=NUM_MICROBATCHES + 8,
    )

    shape = (MB_SIZE, SEQ_LEN, HIDDEN)
    failures = []

    # ---- TEST 1: data integrity through the pipeline ----------------------
    # Each microbatch carries a distinct known constant. Rank 0 originates,
    # every other rank verifies what it received then forwards it on.
    # Corruption from a missing record_stream or a premature staging read
    # shows up here as a value mismatch.
    recv_bufs = [
        torch.zeros(shape, dtype=torch.float16, device=device)
        for _ in range(NUM_MICROBATCHES)
    ]

    dist.barrier()

    if rank > 0:
        handles = [
            dist.irecv(recv_bufs[i], src=rank - 1, tag=TAG_BASE + i)
            for i in range(NUM_MICROBATCHES)
        ]

    send_handles = []
    for mb in range(NUM_MICROBATCHES):
        expected = float(mb + 1) * 1.5

        if rank == 0:
            payload = torch.full(shape, expected, dtype=torch.float16, device=device)
        else:
            handles[mb].wait()
            payload = recv_bufs[mb]

            got = payload.float().mean().item()
            if abs(got - expected) > 1e-3:
                failures.append(
                    f"mb{mb}: expected {expected}, got {got} (data corruption)"
                )
            # also check no element deviates (catches partial-copy bugs that
            # a mean could average away)
            if not torch.allclose(
                payload.float(),
                torch.full(shape, expected, device=device),
                atol=1e-3,
            ):
                failures.append(f"mb{mb}: elementwise mismatch (partial copy)")

        if rank < world_size - 1:
            h = dist.isend(payload.clone(), dst=rank + 1, tag=TAG_BASE + mb)
            send_handles.append(h)
            if len(send_handles) > 1:
                send_handles[-2].flush()

    if send_handles:
        send_handles[-1].flush()
    for h in send_handles:
        h.wait()

    dist.barrier()

    # ---- TEST 2: isend must NOT block on the copy -------------------------
    # If isend() still synchronized the device, queueing would take about as
    # long as the copy itself. We queue a send behind a deliberately slow
    # matmul: with the fix, isend() returns while that matmul is still
    # running, so queue time stays far below the compute time.
    overlap_ms = None
    if rank == 0:
        big = torch.randn(4096, 4096, dtype=torch.float16, device=device)
        payload = torch.full(shape, 7.0, dtype=torch.float16, device=device)

        # Warm up: first matmul triggers cuBLAS handle creation + autotuning,
        # which would otherwise land inside the measured region.
        for _ in range(3):
            big = big @ big.T * 0.0001
        torch.cuda.synchronize()

        # Queue enough compute to keep the GPU busy well past isend().
        t_compute_start = time.perf_counter()
        for _ in range(20):
            big = big @ big.T * 0.0001
        launch_ms = (time.perf_counter() - t_compute_start) * 1000.0

        # Measure isend() ALONE. The previous version timed the matmul loop
        # and isend() together, so matmul kernel-launch throttling dominated
        # the number and isend()'s own cost was unmeasurable.
        t_isend = time.perf_counter()
        h = dist.isend(payload.clone(), dst=1, tag=TAG_BASE + 500)
        isend_ms = (time.perf_counter() - t_isend) * 1000.0

        # How much GPU work was still outstanding when isend() returned.
        t_drain = time.perf_counter()
        torch.cuda.synchronize()
        drain_ms = (time.perf_counter() - t_drain) * 1000.0

        overlap_ms = (isend_ms, drain_ms, launch_ms)

        h.flush()
        h.wait()
    elif rank == 1:
        buf = torch.zeros(shape, dtype=torch.float16, device=device)
        hh = dist.irecv(buf, src=0, tag=TAG_BASE + 500)
        hh.wait()
        if abs(buf.float().mean().item() - 7.0) > 1e-3:
            failures.append("overlap test: payload corrupted")

    dist.barrier()

    # Per-rank aggregate stats + the T29/T30 verdict line.
    stats = mig_transport.log_summary(label="selftest")

    result_queue.put((rank, failures, overlap_ms, stats))
    dist.destroy_process_group()

    # Release SHM so repeated runs don't accumulate segments in /dev/shm.
    # Best-effort: cleanup is a teardown nicety, and failing it must not
    # fail a rank whose actual checks all passed.
    try:
        mig_transport.cleanup()
    except AttributeError:
        print(
            f"[rank {rank}] transport has no cleanup(); SHM will leak "
            f"(update mig_transport_pipeline_non_blocking.py)",
            flush=True,
        )
    except Exception as e:
        print(f"[rank {rank}] cleanup failed: {e}", flush=True)


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    q = mp.Queue()
    procs = []
    for rank in range(WORLD_SIZE):
        p = mp.Process(target=worker, args=(rank, WORLD_SIZE, q, MIG_UUIDS[rank]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=300)

    hung = [i for i, p in enumerate(procs) if p.is_alive()]
    if hung:
        print(f"FAIL: ranks {hung} hung (deadlock in handshake/ACK protocol)")
        for p in procs:
            if p.is_alive():
                p.terminate()
        return 1

    # Non-zero exit code means the rank crashed (CUDA init failure, OOM,
    # exception) rather than completing its checks. Without this, a run where
    # every rank died would report PASS on an empty result set.
    exit_codes = [p.exitcode for p in procs]
    crashed = [i for i, c in enumerate(exit_codes) if c != 0]

    all_failures = []
    overlap = None
    reported = set()
    all_stats = {}
    while not q.empty():
        rank, failures, ov, stats = q.get()
        reported.add(rank)
        for f in failures:
            all_failures.append(f"[rank {rank}] {f}")
        if ov:
            overlap = ov
        if stats:
            all_stats[rank] = stats

    silent = [r for r in range(WORLD_SIZE) if r not in reported]

    print("=" * 60)
    if crashed:
        print(
            f"FAIL — ranks {crashed} exited non-zero: "
            f"{[exit_codes[i] for i in crashed]}"
        )
        print("       Scroll up for the traceback. Nothing was verified.")
    if silent:
        print(f"FAIL — ranks {silent} never reported results")
    if all_failures:
        print("FAIL — correctness problems found:")
        for f in all_failures:
            print("  ", f)

    if not crashed and not silent and not all_failures:
        print(f"PASS — data integrity OK across all {WORLD_SIZE} ranks/microbatches")

    if overlap:
        isend_ms, drain_ms, launch_ms = overlap
        print()
        print(f"matmul launch loop : {launch_ms:7.2f} ms  (CPU-side kernel launches)")
        print(f"isend() alone      : {isend_ms:7.2f} ms  <-- the measurement")
        print(f"drain after isend  : {drain_ms:7.2f} ms  (GPU work still pending)")
        print()
        # The claim being tested: isend() queues a D2H copy on a side stream
        # and returns, rather than calling torch.cuda.synchronize(). If it
        # still synced the device, isend() could not return while the matmuls
        # were outstanding — so drain would be ~0 and isend would absorb it.
        if drain_ms < 1.0:
            print("INCONCLUSIVE — no GPU work was left pending after isend();")
            print("               the compute finished too early to test overlap.")
        elif isend_ms < drain_ms * 0.25:
            print("PASS — isend() returned in a fraction of the outstanding GPU time,")
            print(
                f"       so it did not wait for the device ({isend_ms:.2f}ms vs "
                f"{drain_ms:.2f}ms still pending)."
            )
        else:
            print("FAIL — isend() blocked for a large share of the outstanding")
            print("       GPU time; it is still synchronizing rather than queueing.")

    # Aggregated per-rank stats from the transport's own instrumentation.
    # More trustworthy than the single-sample timing above: averaged over
    # every send/recv the test performed.
    if all_stats:
        print()
        print("--- transport stats (from instrumentation, all sends) ---")
        print(
            f"{'rank':>4} {'sends':>6} {'D2H block':>11} {'recvs':>6} "
            f"{'H2D block':>11} {'spins':>6}"
        )
        for r in sorted(all_stats):
            s = all_stats[r]
            print(
                f"{r:>4} {s['flush_count']:>6} "
                f"{s['flush_block_mean_ms']:>9.2f}ms {s['h2d_count']:>6} "
                f"{s['h2d_block_mean_ms']:>9.2f}ms {s['slot_spins']:>6}"
            )

        senders = [s for s in all_stats.values() if s["flush_count"] > 0]
        if senders:
            worst = max(s["flush_block_mean_ms"] for s in senders)
            print()
            if worst < 1.0:
                print(
                    f"PASS — D2H copies hid under compute on every rank "
                    f"(worst mean block {worst:.2f}ms)"
                )
            else:
                print(
                    f"WARN — worst mean D2H block {worst:.2f}ms. At this "
                    f"payload size the copy may simply be"
                )
                print(
                    f"       shorter than the compute available to hide it; "
                    f"check the same number in a real"
                )
                print(f"       benchmark run before concluding the fix failed.")
    print("=" * 60)

    return 1 if (all_failures or crashed or silent) else 0


if __name__ == "__main__":
    raise SystemExit(main())
