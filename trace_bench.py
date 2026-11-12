"""
Runs the unmodified benchmark and adds per-layer GPU timing.

Used instead of the benchmark through the manager's existing hook:
    MIG_PARALLEL_BENCH=$PWD/trace_bench.py ./run_parallel.sh start --detach

What it adds, per rank, in the job directory:
  layer_latency_rank<r>.csv  one row per layer, for prefill and every SAMPLE_EVERY-th decode step
  cpu_time_rank<r>.csv       CPU seconds used during inference, thread count, CPU affinity size
  token_wait_rank0.csv       rank 0's wait for each step's next token from the last rank

Timing is the harness's own method: torch.cuda.Event.record() between layers, read back only
after the benchmark's final torch.cuda.synchronize(), which is outside the timed region. This
file adds no synchronize. The events are allocated before the timed region.

Nothing in the benchmark changes on disk: the three functions below are replaced at import, in
the parent and in every spawned rank (spawn re-imports this file as __mp_main__).
"""

import csv
import functools
import os
import resource
import time

import torch

import benchmark_pipeline_microbatching as bench
import mig_transport_pipeline as transport
import token_exchange

SAMPLE_EVERY = 16  # decode steps 1, 17, 33, ...; prefill is always sampled

# Timeline mode (INV_TIMELINE=1, set by nsys_bench.py): no layer events. Instead every rank
# labels each layer with an NVTX range and tells nsys to capture only decode steps
# TIMELINE_STEPS[0] .. TIMELINE_STEPS[1]-1, via cudaProfilerStart/Stop.
TIMELINE = os.environ.get("INV_TIMELINE") == "1"
TIMELINE_STEPS = (256, 260)

_orig_forward = bench.forward_through_layers
_orig_run_pipeline = bench.run_pipeline
_orig_register_hooks = transport.register_hooks
_orig_token_recv = token_exchange.TokenExchange.recv

STATE = {}  # one config per rank process


def _sampled_steps():
    if TIMELINE:
        return set()
    return set(range(1, bench.MAX_NEW_TOKENS + 1, SAMPLE_EVERY)) | {0}


def timeline_forward(layers, current_hidden, position_embeddings, mask, cache, step, mb_idx):
    if mb_idx == 0 and step == TIMELINE_STEPS[0]:
        torch.cuda.profiler.start()
    if mb_idx == 0 and step == TIMELINE_STEPS[1]:
        torch.cuda.profiler.stop()
    if not TIMELINE_STEPS[0] <= step < TIMELINE_STEPS[1]:
        return _orig_forward(layers, current_hidden, position_embeddings, mask, cache)
    first = sum(STATE["split"][:STATE["rank"]])
    torch.cuda.nvtx.range_push(f"rank{STATE['rank']} step{step} mb{mb_idx}")
    for i, layer in enumerate(layers):
        torch.cuda.nvtx.range_push(f"L{first + i}")
        current_hidden = _orig_forward([layer], current_hidden, position_embeddings, mask, cache)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    return current_hidden


def traced_register_hooks(*args, **kwargs):
    """Runs after the rank has selected its device and before the timed region."""
    out = _orig_register_hooks(*args, **kwargs)
    n_layers = STATE["split"][STATE["rank"]]
    count = len(_sampled_steps()) * STATE["n_mb"] * (n_layers + 1)
    STATE["pool"] = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    return out


def _event():
    pool = STATE["pool"]
    return pool.pop() if pool else torch.cuda.Event(enable_timing=True)


def traced_forward(layers, current_hidden, position_embeddings, mask, cache):
    if STATE.get("cpu_start") is None:
        STATE["cpu_start"] = resource.getrusage(resource.RUSAGE_SELF)
        STATE["wall_start"] = time.perf_counter()
    call = STATE["call"]
    STATE["call"] = call + 1
    n_mb = STATE["n_mb"]
    # Prefill is calls 0..n-1 (step 0); decode step s is calls s*n .. s*n+n-1.
    step = call // n_mb
    if TIMELINE:
        return timeline_forward(layers, current_hidden, position_embeddings, mask, cache, step,
                                call % n_mb)
    if step not in STATE["sampled"]:
        return _orig_forward(layers, current_hidden, position_embeddings, mask, cache)

    events = []
    for layer in layers:
        event = _event()
        event.record()
        events.append(event)
        current_hidden = _orig_forward([layer], current_hidden, position_embeddings, mask, cache)
    event = _event()
    event.record()
    events.append(event)
    STATE["records"].append((step, call % n_mb, events))
    return current_hidden


def traced_token_recv(self, src, tag):
    """Rank 0's wait for the next token from the last rank, per step: the whole recv, so the
    time the token takes to arrive plus its ~0.1 ms copy to the GPU. CPU clock, no GPU sync."""
    t0 = time.perf_counter()
    _orig_token_recv(self, src, tag)
    STATE["token_waits"].append((tag - bench.TOKENS_TAG_BASE, (time.perf_counter() - t0) * 1000.0))


def _append(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def _save():
    rank = STATE["rank"]
    split = STATE["split"]
    key = ["_".join(map(str, split)), STATE["batch"], STATE["mb"], rank]
    first_layer = sum(split[:rank])
    rows = []
    for step, mb_idx, events in STATE["records"]:
        for i in range(len(events) - 1):
            rows.append(key + [step, mb_idx, first_layer + i, events[i].elapsed_time(events[i + 1])])
    _append(f"layer_latency_rank{rank}.csv",
            ["split", "batch", "mb", "rank", "step", "mb_idx", "layer", "layer_ms"], rows)

    if STATE["token_waits"]:
        _append(f"token_wait_rank{rank}.csv", ["split", "batch", "mb", "rank", "step", "wait_ms"],
                [key + [step, round(ms, 4)] for step, ms in STATE["token_waits"]])

    now = resource.getrusage(resource.RUSAGE_SELF)
    start = STATE["cpu_start"]
    _append(f"cpu_time_rank{rank}.csv",
            ["split", "batch", "mb", "rank", "wall_s", "cpu_user_s", "cpu_sys_s", "torch_threads",
             "affinity_cpus"],
            [key + [round(time.perf_counter() - STATE["wall_start"], 3),
                    round(now.ru_utime - start.ru_utime, 3),
                    round(now.ru_stime - start.ru_stime, 3),
                    torch.get_num_threads(), len(os.sched_getaffinity(0))]])


@functools.wraps(_orig_run_pipeline)
def traced_run_pipeline(rank, world_size, split_config, result_queue, device_uuid, input_ids_seed,
                        mb_size):
    batch = input_ids_seed.shape[0]
    STATE.update(rank=rank, split=list(split_config), batch=batch, mb=mb_size, n_mb=batch // mb_size,
                 call=0, records=[], sampled=_sampled_steps(), cpu_start=None, token_waits=[])
    _orig_run_pipeline(rank, world_size, split_config, result_queue, device_uuid, input_ids_seed,
                       mb_size)
    try:
        if (STATE["records"] or STATE["token_waits"]) and not TIMELINE:
            _save()
    except Exception as e:  # a failed config has unfinished events; its row just goes missing
        print(f"[trace_bench] rank {rank}: layer timing not saved: {type(e).__name__}: {e}",
              flush=True)


bench.forward_through_layers = traced_forward
bench.run_pipeline = traced_run_pipeline
transport.register_hooks = traced_register_hooks
token_exchange.TokenExchange.recv = traced_token_recv

if __name__ == "__main__":
    bench.main()
