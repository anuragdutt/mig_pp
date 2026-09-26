#!/usr/bin/env python3
"""
Acceptance gate for token_exchange.TokenExchange.

Every test here MUST pass. A single FAIL means the staged next_tokens
exchange is not safe to put in the benchmark, and the benchmark stays
untouched.

This exercises the SAME module the benchmark imports — not a
reimplementation of it — so a green run here is evidence about the real
code path, not about a lookalike.

History: this file began as a diagnostic, sending the same (32,1) int64
tensor over gloo from CPU and from CUDA to find out why the harness died
with "writev ...: Bad address". The verdict was unambiguous — CPU passed,
CUDA failed with EFAULT, and a CUDA fp16 activation failed the same way, so
it is the device pointer and not the dtype or the size. gloo is CPU-only:
it hands tensor.data_ptr() to writev(2), and a device pointer there is not
host-readable.

That diagnosis is finished, so the file has been repurposed as the gate for
the fix. It no longer sends a raw CUDA tensor: doing so tears down the
process group and every later collective throws "Connection closed by
peer", which would poison all the tests after it. If you need the original
diagnostic again, git history has it.

Run:  python3 probe_gloo_send.py
Log:  logs/probe_gloo_send_<stamp>.log  (plus stdout)
Exit: 0 if every test passed, 1 otherwise.
"""

import datetime
import logging
import os
import sys
import time
import traceback
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from token_exchange import TokenExchange

# Mirror the benchmark.
WORLD_SIZE = 3
BATCH_SIZE = 32
SEQ_LEN = 64
MAX_NEW_TOKENS = 512
TOKENS_TAG_BASE = 9000

MIG_UUIDS: List[str] = [
    "MIG-ecf4f88a-f84e-52b9-8214-5682aa2cd17b",  # Rank 0: 20GB (3g.20gb)
    "MIG-c6c45ff8-b1dd-5c62-9265-67b7cd538e31",  # Rank 1: 10GB (2g.10gb)
    "MIG-d14c9a69-f89d-5f67-ab42-bf8435443f09",  # Rank 2: 10GB (2g.10gb)
]

LOG_DIR = "logs"


def setup_probe_logging(rank: int, stamp: str) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"probe_gloo_send_{stamp}.log")

    logger = logging.getLogger("probe")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [rank" + str(rank) + "] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return path


class Gate:
    """Runs tests, records PASS/FAIL, keeps ranks in step."""

    def __init__(self, rank: int, log: logging.Logger):
        self.rank = rank
        self.log = log
        self.results: List[tuple] = []

    def run(self, name: str, fn, sender: int, receiver: int) -> None:
        self.log.info("=" * 70)
        self.log.info("TEST: %s", name)
        try:
            dist.barrier()
        except Exception as e:
            self.log.error("  barrier BEFORE failed: %r", e)
            self.results.append((name, False, f"pre-barrier failed: {e!r}"))
            return

        if self.rank not in (sender, receiver):
            self.log.info("  role=idle")
            self.results.append((name, None, "idle rank"))
            try:
                dist.barrier()
            except Exception as e:
                self.log.error("  barrier AFTER failed: %r", e)
            return

        role = "sender" if self.rank == sender else "receiver"
        self.log.info("  role=%s", role)
        t0 = time.perf_counter()
        try:
            detail = fn(role)
            dt = (time.perf_counter() - t0) * 1000
            self.log.info("  ---> PASS (%.3fms) %s", dt, detail or "")
            self.results.append((name, True, f"{dt:.3f}ms {detail or ''}"))
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            self.log.error("  ---> FAIL (%.3fms) %s: %s", dt, type(e).__name__, e)
            self.log.error("  traceback:\n%s", traceback.format_exc())
            self.results.append((name, False, f"{type(e).__name__}: {e}"))

        try:
            dist.barrier()
        except Exception as e:
            self.log.error("  barrier AFTER failed: %r", e)


def run_probe(rank: int, world_size: int, device_uuid: Optional[str], stamp: str, result_q) -> None:
    path = setup_probe_logging(rank, stamp)
    log = logging.getLogger("probe")
    results = []

    try:
        if device_uuid:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        log.info("probe log -> %s", path)
        log.info("torch=%s cuda=%s cuda_version=%s",
                 torch.__version__, torch.cuda.is_available(), torch.version.cuda)
        log.info("CUDA_VISIBLE_DEVICES=%r", os.environ.get("CUDA_VISIBLE_DEVICES"))

        dist.init_process_group(
            backend="gloo", rank=rank, world_size=world_size,
            timeout=datetime.timedelta(minutes=2),
        )
        log.info("process group up: rank=%d world_size=%d backend=%s",
                 dist.get_rank(), dist.get_world_size(), dist.get_backend())

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        log.info("device=%s name=%r mem=%.1fGiB", device, props.name,
                 props.total_memory / 2**30)

        gate = Gate(rank, log)
        LAST = world_size - 1

        # =================================================================
        # T1 — construction: pinned host buffer, matching shape and dtype.
        # =================================================================
        def t1(role):
            nt = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
            ex = TokenExchange(nt)
            assert ex.host_buffer.is_pinned(), "host buffer is not pinned"
            assert ex.host_buffer.shape == nt.shape, "shape mismatch"
            assert ex.host_buffer.dtype == nt.dtype, "dtype mismatch"
            assert ex.host_buffer.device.type == "cpu", "buffer not on cpu"
            assert ex.stats["n"] == 0, "counter not zeroed"
            log.info("  host_buffer: shape=%s dtype=%s pinned=%s data_ptr=0x%x",
                     tuple(ex.host_buffer.shape), ex.host_buffer.dtype,
                     ex.host_buffer.is_pinned(), ex.host_buffer.data_ptr())
            return "pinned host buffer allocated, shape/dtype match"

        gate.run("T1  TokenExchange construction", t1, sender=LAST, receiver=0)

        # =================================================================
        # T2 — one exchange, values verified end to end.
        # =================================================================
        def t2(role):
            nt = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
            ex = TokenExchange(nt)
            tag = TOKENS_TAG_BASE + 0
            if role == "sender":
                nt.copy_(torch.arange(BATCH_SIZE, device=device).unsqueeze(1))
                ex.send(dst=0, tag=tag)
                return f"sent arange(32), {ex.summary()}"
            ex.recv(src=LAST, tag=tag)
            expect = torch.arange(BATCH_SIZE, device=device).unsqueeze(1)
            assert torch.equal(nt, expect), f"value mismatch: got {nt.flatten()[:8]}"
            return f"received and verified, {ex.summary()}"

        gate.run("T2  single exchange, values verified", t2, sender=LAST, receiver=0)

        # =================================================================
        # T3 — the benchmark's prefill topology: rank2 -> rank0, rank1 in
        # neither branch, then a barrier (benchmark lines 472-484).
        # =================================================================
        log.info("=" * 70)
        log.info("TEST: T3  prefill topology (rank1 idle) + barrier")
        try:
            dist.barrier()
            nt = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
            ex = TokenExchange(nt)
            tag = TOKENS_TAG_BASE + 0
            t0 = time.perf_counter()
            if rank == LAST:
                nt.fill_(19)
                ex.send(dst=0, tag=tag)
                log.info("  rank%d staged send returned", rank)
            elif rank == 0:
                ex.recv(src=LAST, tag=tag)
                assert bool((nt == 19).all()), "value mismatch after recv"
                log.info("  rank0 staged recv verified")
            else:
                log.info("  rank%d: neither branch — straight to barrier", rank)
            dist.barrier()
            dt = (time.perf_counter() - t0) * 1000
            log.info("  ---> PASS (%.3fms) barrier cleared", dt)
            results.append(("T3  prefill topology + barrier", True, f"{dt:.3f}ms"))
        except Exception as e:
            log.error("  ---> FAIL %s: %s", type(e).__name__, e)
            log.error("  traceback:\n%s", traceback.format_exc())
            results.append(("T3  prefill topology + barrier", False, f"{type(e).__name__}: {e}"))

        # =================================================================
        # T4 — full decode loop: MAX_NEW_TOKENS exchanges with the real tag
        # arithmetic (TOKENS_TAG_BASE + step), every value checked. This is
        # the load the benchmark actually puts on the path.
        # =================================================================
        def t4(role):
            nt = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
            ex = TokenExchange(nt)
            for step in range(1, MAX_NEW_TOKENS + 1):
                tag = TOKENS_TAG_BASE + step
                if role == "sender":
                    nt.fill_(step % 1000)
                    ex.send(dst=0, tag=tag)
                else:
                    ex.recv(src=LAST, tag=tag)
                    if not bool((nt == step % 1000).all()):
                        raise AssertionError(
                            f"value mismatch at step {step}: expected "
                            f"{step % 1000}, got {int(nt.flatten()[0])}"
                        )
            assert ex.stats["n"] == MAX_NEW_TOKENS, (
                f"counter drift: {ex.stats['n']} != {MAX_NEW_TOKENS}"
            )
            log.info("  %s", ex.summary())
            return f"{MAX_NEW_TOKENS} exchanges verified, {ex.summary()}"

        gate.run(f"T4  full decode loop ({MAX_NEW_TOKENS} exchanges)", t4,
                 sender=LAST, receiver=0)

        # =================================================================
        # T5 — buffer reuse: the host buffer must be allocated once, not
        # per exchange. Catches an accidental per-step allocation.
        # =================================================================
        def t5(role):
            nt = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
            ex = TokenExchange(nt)
            ptr_before = ex.host_buffer.data_ptr()
            tag0 = TOKENS_TAG_BASE + 600
            for i in range(16):
                if role == "sender":
                    nt.fill_(i)
                    ex.send(dst=0, tag=tag0 + i)
                else:
                    ex.recv(src=LAST, tag=tag0 + i)
            ptr_after = ex.host_buffer.data_ptr()
            assert ptr_before == ptr_after, (
                f"host buffer was reallocated: 0x{ptr_before:x} -> 0x{ptr_after:x}"
            )
            return f"buffer stable across 16 exchanges at 0x{ptr_after:x}"

        gate.run("T5  host buffer reused, not reallocated", t5, sender=LAST, receiver=0)

        # =================================================================
        # T6 — stats and reporting fields are well formed, so the benchmark
        # can put them straight into its results row.
        # =================================================================
        def t6(role):
            nt = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
            ex = TokenExchange(nt)
            tag0 = TOKENS_TAG_BASE + 700
            for i in range(8):
                if role == "sender":
                    nt.fill_(i)
                    ex.send(dst=0, tag=tag0 + i)
                else:
                    ex.recv(src=LAST, tag=tag0 + i)
            assert ex.stats["n"] == 8, f"count wrong: {ex.stats['n']}"
            assert ex.total_ms() >= 0.0, "negative total"
            assert ex.mean_ms() >= 0.0, "negative mean"
            # summary() is the only reporting surface — the staging cost goes
            # to the log and deliberately not into the results CSV.
            s = ex.summary()
            for frag in ("d2h=", "h2d=", "total=", "mean="):
                assert frag in s, f"summary() missing {frag!r}: {s}"
            assert "8 exchanges" in s, f"summary() count wrong: {s}"
            log.info("  summary()=%s", s)
            return f"stats well formed: {s}"

        gate.run("T6  stats and log summary", t6, sender=LAST, receiver=0)

        # =================================================================
        # T7 — guard: constructing on a CPU tensor is a programming error
        # and must raise rather than silently doing nothing useful.
        # =================================================================
        def t7(role):
            cpu_t = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device="cpu")
            try:
                TokenExchange(cpu_t)
            except ValueError as e:
                return f"correctly rejected CPU tensor: {e}"
            raise AssertionError("TokenExchange accepted a CPU tensor; it must not")

        gate.run("T7  rejects a CPU tensor", t7, sender=LAST, receiver=0)

        results = gate.results + results

    except Exception as e:
        log.error("FATAL: %s", traceback.format_exc())
        results.append(("SETUP", False, f"{type(e).__name__}: {e}"))

    finally:
        try:
            result_q.put((rank, results))
        except Exception:
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                log.info("process group destroyed")
        except Exception as e:
            log.error("teardown error: %r", e)


def main() -> int:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"token_exchange acceptance gate  —  world_size={WORLD_SIZE}")
    for r, u in enumerate(MIG_UUIDS):
        print(f"  rank {r}: {u}")
    print(f"log: {LOG_DIR}/probe_gloo_send_{stamp}.log")
    print("=" * 70)

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    procs = []
    for r in range(WORLD_SIZE):
        p = mp.Process(target=run_probe, args=(r, WORLD_SIZE, MIG_UUIDS[r], stamp, q))
        p.start()
        procs.append(p)

    collected = {}
    deadline = time.time() + 600
    while len(collected) < WORLD_SIZE and time.time() < deadline:
        try:
            rank, res = q.get(timeout=5)
            collected[rank] = res
        except Exception:
            if not any(p.is_alive() for p in procs):
                break

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    if not collected:
        print("No results collected — every rank died before reporting.")
        print(f"Read {LOG_DIR}/probe_gloo_send_{stamp}.log")
        return 1

    names = []
    for res in collected.values():
        for name, _, _ in res:
            if name not in names:
                names.append(name)

    failures = 0
    for name in names:
        marks, details = [], []
        for rank in sorted(collected):
            for n, ok, detail in collected[rank]:
                if n == name:
                    marks.append(f"r{rank}:{'PASS' if ok else 'SKIP' if ok is None else 'FAIL'}")
                    if ok is False:
                        failures += 1
                        details.append(f"        r{rank} -> {detail}")
                    elif ok:
                        details.append(f"        r{rank} -> {detail}")
        print(f"  {name}")
        print(f"        {' '.join(marks)}")
        for d in details:
            print(d)

    print()
    print("=" * 70)
    if failures == 0:
        print("ALL TESTS PASSED")
        print("token_exchange.TokenExchange is safe to wire into the benchmark.")
        verdict = 0
    else:
        print(f"{failures} FAILURE(S) — DO NOT PATCH THE BENCHMARK")
        print("Send the log and we look again before touching anything.")
        verdict = 1
    print("=" * 70)
    print(f"Full log: {LOG_DIR}/probe_gloo_send_{stamp}.log")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
