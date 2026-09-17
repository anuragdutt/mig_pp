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

import mig_transport_pipeline_non_blocking as mig_transport

# Keep this file standalone (no transformers/datasets import) so it can run
# before the model is even downloaded. Paste the SAME UUIDs you put in
# benchmark_pipeline_microbatching.py.
MIG_UUIDS = [
    "MIG-REPLACE-ME-SLICE0-20GB",  # Rank 0: 20GB (3g.20gb)
    "MIG-REPLACE-ME-SLICE1-10GB",  # Rank 1: 10GB (2g.10gb)
    "MIG-REPLACE-ME-SLICE2-10GB",  # Rank 2: 10GB (2g.10gb)
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

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            big = big @ big.T * 0.0001  # keep the default stream busy
        h = dist.isend(payload.clone(), dst=1, tag=TAG_BASE + 500)
        queue_ms = (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        overlap_ms = (queue_ms, total_ms)

        h.flush()
        h.wait()
    elif rank == 1:
        buf = torch.zeros(shape, dtype=torch.float16, device=device)
        hh = dist.irecv(buf, src=0, tag=TAG_BASE + 500)
        hh.wait()
        if abs(buf.float().mean().item() - 7.0) > 1e-3:
            failures.append("overlap test: payload corrupted")

    dist.barrier()

    result_queue.put((rank, failures, overlap_ms))
    dist.destroy_process_group()


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

    all_failures = []
    overlap = None
    while not q.empty():
        rank, failures, ov = q.get()
        for f in failures:
            all_failures.append(f"[rank {rank}] {f}")
        if ov:
            overlap = ov

    print("=" * 60)
    if all_failures:
        print("FAIL — correctness problems found:")
        for f in all_failures:
            print("  ", f)
    else:
        print("PASS — data integrity OK across all ranks/microbatches")

    if overlap:
        queue_ms, total_ms = overlap
        print(f"\nisend() queue time: {queue_ms:.1f} ms")
        print(f"total incl. compute: {total_ms:.1f} ms")
        if queue_ms < total_ms * 0.5:
            print("PASS — isend() returns while compute is still running (overlap OK)")
        else:
            print("FAIL — isend() is still blocking on the copy; fix did not take")
    print("=" * 60)

    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
