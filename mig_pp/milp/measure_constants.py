"""
Measures DEFAULT_CONSTANTS of pulp_layer_placement.py and prints them as a dict to
paste over it. Edit CONFIG in main(), then on the box:

    CUDA_VISIBLE_DEVICES=<gpu index> taskset -c <that GPU's NUMA CPUs> python3 measure_constants.py

CONFIG["mig"] picks what to measure:

  "off"         (default) one whole GPU with MIG disabled and nothing else on it: every
                DEFAULT_CONSTANTS entry. Refuses to run on a MIG slice.
  "slice"       ONE MIG slice, MIG enabled. Run once per slice type (on a 40GB card:
                3g.20gb, 2g.10gb, 1g.5gb), nothing else on that slice:
                    CUDA_VISIBLE_DEVICES=MIG-<uuid> python3 measure_constants.py
                Measures what may differ on a slice: weight streaming (eff_weight),
                activation/KV traffic (eff_act), D2H (pcie_bw) and H2D (recv_h2d_bw),
                prefill layer time, and the lm_head / endpoint kernels. Efficiencies are
                printed against three peaks, because the ILP multiplies the card peak by
                a slice share and a slice number must not be divided by it twice:
                    vs card   achieved / card peak            (whole-GPU convention)
                    vs GPC    achieved / (card peak x g/7)    (the ILP's current share)
                    vs mem    achieved / (card peak x GB/card GB)  (NVIDIA's 4/8, 2/8, 1/8)
                For a model's lm_head on the last slice, set CONFIG["model_path"] to it
                (e.g. Qwen) and rerun on that slice type.
  "contention"  the slices of ONE card copying at the same time, MIG enabled, nothing
                else running. List that card's slices, comma-separated:
                    CUDA_VISIBLE_DEVICES=MIG-<a>,MIG-<b> python3 measure_constants.py
                Each slice copies alone first, then all together (D2H, then H2D), one
                process per slice; the ratio is the copy-engine / PCIe sharing a
                single-slice probe cannot see. Two nodes: run it on each node's card.

No sudo: the probes are ordinary CUDA, gloo on localhost, host memcpy and
torch.profiler kernel tracing (tracing needs no admin rights; only hardware counters,
e.g. ncu metrics, do).

HOW EACH CONSTANT IS MEASURED, matching pulp_layer_placement.terms():
  eff_weight         one layer's weight bytes / its decode GEMMs' time / card peak
  eff_act            activation + KV bytes / (layer kernel time - weight time - launch
                     term); the ILP adds kernels_per_layer x launch_us to a layer's GPU time
  launch_us          CPU time to dispatch one decode layer call / its kernel count
  kernels_per_layer, rotary_kernels   kernels counted by torch.profiler
  pcie_bw, recv_h2d_bw                one activation's D2H / H2D as the transport does it
  memcpy_bw, shm_to_pinned_bw         the transport's two host copies
  handshake_ack_us, tok_roundtrip_us, barrier_us   gloo, CONFIG["world"] processes
  token_staging_us, step_start_us     TokenExchange's two copies; the position_ids sync
Layer kernel times are the profiler's summed kernel durations, so launch gaps are out.
"""

import os
import re
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import DeviceType

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CARD_BW = {"A100-40GB": 1555e9, "A100-80GB": 2039e9,  # peak HBM bytes/s, as in the ILP
           "A100-80GB-PCIe": 1935e9}  # efficiencies are fractions of the card actually measured

# Architecture of the model being measured; load_model() sets them from config.json.
HIDDEN, INTER, VOCAB, N_HEADS, N_KV_HEADS, HEAD_DIM = 4096, 11008, 32000, 32, 32, 128


# ---------------------------------------------------------------------------
# bytes -- the same counts as pulp_layer_placement.weight_bytes / act_bytes (its test
# lifts these three functions out of this file and compares)
# ---------------------------------------------------------------------------
def kv_dim():
    return N_KV_HEADS * HEAD_DIM


def act_bytes_per_token(ctx):
    B = 2  # FP16
    kvd = kv_dim()
    q_out = N_HEADS * HEAD_DIM
    qkv = (HIDDEN + q_out + 2 * kvd) * B  # read x, write Q,K,V
    flash = (q_out + 2 * ctx * kvd + q_out) * B  # read Q + full-ctx K,V, write out
    oproj = (q_out + HIDDEN) * B
    gate = (HIDDEN + INTER) * B
    up = (HIDDEN + INTER) * B
    swiglu = (2 * INTER + INTER) * B
    down = (INTER + HIDDEN) * B
    misc = (6 * HIDDEN) * B
    return qkv + flash + oproj + gate + up + swiglu + down + misc


def weight_bytes_per_layer():
    kvd = kv_dim()
    q_out = N_HEADS * HEAD_DIM
    return (HIDDEN * q_out + 2 * HIDDEN * kvd + q_out * HIDDEN + 3 * HIDDEN * INTER) * 2


# ---------------------------------------------------------------------------
# model and timing helpers
# ---------------------------------------------------------------------------
def load_model(path):
    """One decoder layer + rotary of the model at path, random weights (timing only)."""
    global HIDDEN, INTER, VOCAB, N_HEADS, N_KV_HEADS, HEAD_DIM
    from transformers import AutoConfig
    import model_family

    cfg = AutoConfig.from_pretrained(os.path.expanduser(path))
    cfg._attn_implementation = "sdpa"  # as the benchmark
    HIDDEN, INTER, VOCAB = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    N_HEADS = cfg.num_attention_heads
    N_KV_HEADS = getattr(cfg, "num_key_value_heads", None) or N_HEADS
    HEAD_DIM = getattr(cfg, "head_dim", None) or HIDDEN // N_HEADS
    fam = model_family.resolve(cfg.model_type)
    layer = fam.decoder_layer(cfg, layer_idx=0).half().cuda().eval()
    rotary = fam.rotary_embedding(config=cfg, device=torch.device("cuda"))
    return layer, rotary


def layer_forward(layer, h, emb, cache, mask=None):
    """forward_through_layers in benchmark_pipeline_microbatching.py, one layer."""
    r = h
    x = layer.self_attn(hidden_states=layer.input_layernorm(h), position_embeddings=emb,
                        attention_mask=mask, past_key_values=cache, use_cache=True)[0]
    h = r + x
    return h + layer.mlp(layer.post_attention_layernorm(h))


def cache_at(layer, rotary, mb, ctx):
    """A DynamicCache holding ctx tokens for mb sequences (what decode step ctx-63 sees)."""
    from transformers import DynamicCache

    cache = DynamicCache()
    x = torch.randn(mb, ctx, HIDDEN, dtype=torch.float16, device="cuda")
    pos = torch.arange(ctx, device="cuda").unsqueeze(0).expand(mb, -1)
    with torch.no_grad():
        layer_forward(layer, x, rotary(x, pos), cache)
    return cache


def decode_input(rotary, mb, ctx):
    h = torch.randn(mb, 1, HIDDEN, dtype=torch.float16, device="cuda")
    pos = torch.full((mb, 1), ctx, dtype=torch.long, device="cuda")
    return h, rotary(h, pos), pos


def kernels(fn, copies=False):
    """(kernel count, summed kernel time in us) of one call of fn, from torch.profiler.
    copies=True also counts Memcpy/Memset: a contiguous clone() or copy_() on the device
    is one DtoD memcpy, not a kernel."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ks = [e for e in prof.events() if e.device_type == DeviceType.CUDA
          and (copies or not e.name.startswith(("Memcpy", "Memset")))]
    if not ks:
        seen = [f"{e.name} ({e.device_type})" for e in prof.events()][:20]
        raise RuntimeError(f"torch.profiler recorded no CUDA kernels; events seen: {seen}")
    return len(ks), sum(e.time_range.elapsed_us() for e in ks)


def wall_s(fn, reps=200, warmup=20):
    """Mean wall time of fn (fn must wait for its own work, as the transport does)."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def gemm_bytes_per_s(M, reps=50):
    """Weight streaming rate of one layer's decode GEMMs for M tokens: q,k,v,gate,up
    stacked in one matrix, then o and down. Big kernels, so launches do not show."""
    q, kv = N_HEADS * HEAD_DIM, kv_dim()
    wa = torch.randn(q + 2 * kv + 2 * INTER, HIDDEN, dtype=torch.float16, device="cuda")
    wo = torch.randn(HIDDEN, q, dtype=torch.float16, device="cuda")
    wd = torch.randn(HIDDEN, INTER, dtype=torch.float16, device="cuda")
    x = torch.randn(M, HIDDEN, dtype=torch.float16, device="cuda")
    y = torch.randn(M, q, dtype=torch.float16, device="cuda")
    z = torch.randn(M, INTER, dtype=torch.float16, device="cuda")

    def fn():
        F.linear(x, wa), F.linear(y, wo), F.linear(z, wd)

    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return weight_bytes_per_layer() / (start.elapsed_time(end) / reps * 1e-3)


def layer_kernel_us(layer, rotary, mb, ctx, reps=5):
    """(kernels, median kernel time in us) of one decode layer call at ctx cached tokens."""
    h, emb, _ = decode_input(rotary, mb, ctx)
    with torch.no_grad():
        layer_forward(layer, h, emb, cache_at(layer, rotary, mb, ctx))  # warm this shape
        runs = []
        for _ in range(reps):
            cache = cache_at(layer, rotary, mb, ctx)
            runs.append(kernels(lambda: layer_forward(layer, h, emb, cache)))
    return runs[0][0], statistics.median(us for _, us in runs)


def layer_dispatch_us(layer, rotary, mb, ctx, reps=30):
    """CPU time to queue one decode layer call (the GPU is idle at each start)."""
    h, emb, _ = decode_input(rotary, mb, ctx)
    cache = cache_at(layer, rotary, mb, ctx)
    out = []
    with torch.no_grad():
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            layer_forward(layer, h, emb, cache)
            out.append(time.perf_counter() - t0)
    return statistics.median(out) * 1e6


def device_copies(nbytes):
    """(D2H, H2D) bytes/s of one activation, as isend+FLUSH and RECV do them: a side
    stream, then a wait on that copy's event."""
    n = nbytes // 2
    src = torch.randn(n, dtype=torch.float16, device="cuda")
    dst = torch.empty(n, dtype=torch.float16, device="cuda")
    pinned = torch.empty(n, dtype=torch.float16).pin_memory()
    copy_stream, recv_stream = torch.cuda.Stream(), torch.cuda.Stream()
    ev = torch.cuda.Event()

    def d2h():
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            pinned.copy_(src, non_blocking=True)
        ev.record(copy_stream)
        ev.synchronize()

    def h2d():
        with torch.cuda.stream(recv_stream):
            dst.copy_(pinned, non_blocking=True)
        ev.record(recv_stream)
        ev.synchronize()

    return nbytes / wall_s(d2h), nbytes / wall_s(h2d)


def host_copies(nbytes):
    """(pinned -> SHM, SHM -> pinned) bytes/s: _flush_staging_to_shm (numpy slice
    assignment) and _queue_read_from_slot (torch copy_ from a numpy view of SHM)."""
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(create=True, size=nbytes)
    try:
        return _host_copy_rates(shm.buf, nbytes)  # its views of SHM are gone on return
    finally:
        shm.close()
        shm.unlink()


def _host_copy_rates(buf, nbytes):
    slot = np.ndarray((nbytes,), dtype=np.uint8, buffer=buf)
    pinned = torch.empty(nbytes // 2, dtype=torch.float16).pin_memory()
    raw = pinned.numpy().view(np.uint8)
    peer = torch.from_numpy(slot.view(np.float16))

    def send():
        slot[:nbytes] = raw[:nbytes]

    def recv():
        pinned.copy_(peer)

    return nbytes / wall_s(send), nbytes / wall_s(recv)


def token_staging_us(batch):
    """TokenExchange: D2H of next_tokens + stream sync, then H2D + stream sync."""
    dev_tok = torch.zeros((batch, 1), dtype=torch.long, device="cuda")
    host = torch.empty((batch, 1), dtype=torch.long).pin_memory()

    def exchange():
        host.copy_(dev_tok)
        torch.cuda.current_stream().synchronize()
        dev_tok.copy_(host)
        torch.cuda.current_stream().synchronize()

    return wall_s(exchange) * 1e6


def step_start_us(mb, seq):
    """Rank 0's work at a decode step start: position_ids built on the device (a pageable
    H2D that waits for itself), expand().contiguous(), the step event's first record."""
    def step():
        torch.tensor([[seq]], dtype=torch.long, device="cuda").expand(mb, -1).contiguous()
        torch.cuda.Event(enable_timing=True).record()

    return wall_s(step) * 1e6


def _gloo_worker(rank, world, port, batch, reps, q):
    """handshake + ACK (ranks 0 and 1, as FLUSH/RECV), next_tokens (last rank -> 0),
    barrier (all). Times are rank 0's. Second pass counts; the first is a warmup."""
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    hs, ack = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    tok = torch.zeros((batch, 1), dtype=torch.long)
    out = {}
    for _ in range(2):
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
        out["handshake_ack_us"] = (time.perf_counter() - t0) / reps * 1e6
        dist.barrier()
        t0 = time.perf_counter()
        for i in range(reps):
            if rank == world - 1:
                dist.send(tok, dst=0, tag=100_000 + i)
            elif rank == 0:
                dist.recv(tok, src=world - 1, tag=100_000 + i)
        out["tok_roundtrip_us"] = (time.perf_counter() - t0) / reps * 1e6
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(reps):
            dist.barrier()
        out["barrier_us"] = (time.perf_counter() - t0) / reps * 1e6
    if rank == 0:
        q.put(out)
    dist.destroy_process_group()


def gloo_probes(world, port, batch, reps=200):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_gloo_worker, args=(r, world, port, batch, reps, q))
             for r in range(world)]
    for p in procs:
        p.start()
    out = q.get(timeout=300)
    for p in procs:
        p.join(timeout=60)
    return out


# ---------------------------------------------------------------------------
# extra probes: things DEFAULT_CONSTANTS does not cover yet
# ---------------------------------------------------------------------------
def fit_latency(sizes, times_s):
    """Least-squares t = fixed + bytes / bw over the sizes: (fixed us, bytes/s)."""
    n = len(sizes)
    mx, my = sum(sizes) / n, sum(times_s) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(sizes, times_s)) / sum((x - mx) ** 2 for x in sizes)
    return (my - slope * mx) * 1e6, (1 / slope if slope > 0 else float("nan"))


def prefill_layer_us(layer, rotary, mb, seq, reps=5):
    """Kernel time of one prefill layer call: mb sequences of seq tokens, empty cache,
    the benchmark's causal mask (1, 1, seq, seq)."""
    from transformers import DynamicCache

    x = torch.randn(mb, seq, HIDDEN, dtype=torch.float16, device="cuda")
    pos = torch.arange(seq, device="cuda").unsqueeze(0)
    mask = torch.triu(torch.full((1, 1, seq, seq), torch.finfo(torch.float16).min,
                                 device="cuda"), diagonal=1).half()
    with torch.no_grad():
        emb = rotary(x, pos)
        layer_forward(layer, x, emb, DynamicCache(), mask)  # warm this shape
        runs = [kernels(lambda: layer_forward(layer, x, emb, DynamicCache(), mask))[1]
                for _ in range(reps)]
    return statistics.median(runs)


def layer_flops(mb, seq):
    q, kv = N_HEADS * HEAD_DIM, kv_dim()
    tokens = mb * seq
    return (2 * tokens * (HIDDEN * q + 2 * HIDDEN * kv + q * HIDDEN + 3 * HIDDEN * INTER)
            + 2 * mb * N_HEADS * seq * seq * HEAD_DIM * 2)


def endpoint_us(layer, mb, prefill_seq=None):
    """Kernel time (us) of each non-layer piece of one microbatch: embed gather (rank 0),
    clone (sending ranks), and on the last rank norm, lm_head, argmax + next_tokens write.
    prefill_seq=None is decode (1 token); else norm/embed/clone see mb x seq tokens."""
    t = prefill_seq or 1
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.float16, device="cuda")
    ids = torch.randint(0, VOCAB, (mb, t), device="cuda")
    hid = torch.randn(mb, t, HIDDEN, dtype=torch.float16, device="cuda")
    head = torch.randn(VOCAB, HIDDEN, dtype=torch.float16, device="cuda")
    tok = torch.zeros(mb, 1, dtype=torch.long, device="cuda")
    norm = layer.input_layernorm  # same RMSNorm class as the final norm
    out = {}
    with torch.no_grad():
        for name, fn in (
            ("embed", lambda: F.embedding(ids, table)),
            ("clone", lambda: hid.clone()),
            ("norm", lambda: norm(hid)),
            ("lm_head", lambda: F.linear(hid[:, -1:, :], head)),
            ("argmax", lambda: tok.copy_(torch.argmax(F.linear(hid[:, -1:, :], head), dim=-1))),
        ):
            fn()
            copies = name in ("clone", "argmax")  # clone and the next_tokens write are memcpys
            try:
                out[name] = statistics.median(kernels(fn, copies)[1] for _ in range(5))
            except RuntimeError as err:  # report it and keep going: nan in that column
                print(f"  [endpoint {name}, mb {mb}, tokens {t}] {err}", flush=True)
                out[name] = float("nan")
    out["argmax"] -= out["lm_head"]  # the line above also ran lm_head to get logits
    return out


def _import_transport():
    """The real transport engine. Its module imports helpers.py, which needs the
    `datasets` package for unrelated reasons; stub the one function it uses if absent."""
    try:
        import mig_transport_pipeline as mt
    except ImportError:
        import types

        stub = types.ModuleType("helpers")
        stub._compute_slot_size_mb = (
            lambda hidden_size=5120, max_mb_size=32, max_seq_len=64:
            max_mb_size * max_seq_len * hidden_size * 2 // (1024 * 1024) + 1)
        sys.modules["helpers"] = stub
        import mig_transport_pipeline as mt
    return mt


def _transport_worker(rank, port, hidden, mbs, tokens, reps, q):
    """Two ranks on this one GPU, using the real SHM transport exactly as run_pipeline:
    rank 0 isend -> flush -> wait (ACK), rank 1 irecv -> wait. Rank 0 times one
    microbatch handed over and acknowledged, for each microbatch size."""
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      MIG_SHM_PREFIX=f"mcprobe{port}", MIG_LOG_LEVEL="off")
    dist.init_process_group("gloo", rank=rank, world_size=2)
    torch.cuda.set_device(0)
    mt = _import_transport()
    out = {}
    for mb in mbs:
        mt.register_hooks(mb_size=mb, seq_len=tokens, hidden_size=hidden, num_slots=16)
        x = torch.randn(mb, tokens, hidden, dtype=torch.float16, device="cuda")
        buf = torch.empty_like(x)
        times = []
        for i in range(reps + 20):
            tag = 2000 + i
            if rank == 0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                h = dist.isend(x.clone(), dst=1, tag=tag)
                h.flush()
                h.wait()
                times.append(time.perf_counter() - t0)
            else:
                dist.irecv(buf, src=0, tag=tag).wait()
        out[mb] = statistics.median(times[20:]) * 1e6 if rank == 0 else None
        dist.barrier()
        mt.cleanup()
        dist.barrier()
    if rank == 0:
        q.put(out)
    dist.destroy_process_group()


def transport_handoff_us(port, mbs, tokens, reps=200):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_transport_worker, args=(r, port, HIDDEN, mbs, tokens, reps, q))
             for r in range(2)]
    for p in procs:
        p.start()
    out = q.get(timeout=600)
    for p in procs:
        p.join(timeout=60)
    return out


def detect_card(name):
    if "A100" in name:
        if torch.cuda.get_device_properties(0).total_memory < 45e9:
            return "A100-40GB"
        return "A100-80GB-PCIe" if "PCIe" in name else "A100-80GB"
    raise SystemExit(f"{name}: unknown card; set CONFIG['card'] and add its peak to CARD_BW")


# ---------------------------------------------------------------------------
# MIG on: CONFIG["mig"] = "slice" or "contention"
# ---------------------------------------------------------------------------
def slice_info(name):
    """(card, profile, GPCs, slice GB, card GB) from a MIG device name such as
    'NVIDIA A100-PCIE-40GB MIG 3g.20gb'. On a slice, total_memory is the slice's, so
    detect_card() would misread an 80GB card's 3g.40gb as a 40GB card; the name cannot."""
    m = re.search(r"MIG (\d)g\.(\d+)gb", name)
    if not m or "A100" not in name:
        raise SystemExit(f"{name}: not an A100 MIG slice; run with CUDA_VISIBLE_DEVICES=MIG-<uuid>")
    card_gb = 80 if "80GB" in name else 40
    card = "A100-40GB" if card_gb == 40 else (
        "A100-80GB-PCIe" if "PCIE" in name.upper() else "A100-80GB")
    return card, f"{m[1]}g.{m[2]}gb", int(m[1]), int(m[2]), card_gb


def run_slice(c, name):
    """Everything that may differ on a slice, against three peaks (see the docstring)."""
    card, profile, g, gb, card_gb = slice_info(name)
    peak = CARD_BW[c["card"] or card]
    shares = {"card": 1.0, "GPC": g / 7, "mem": gb / card_gb}
    layer, rotary = load_model(c["model_path"])
    print(f"{name}: {profile} of {card} (card peak {peak / 1e9:.0f} GB/s; shares: GPC {g}/7, "
          f"mem {gb}/{card_gb}); {c['model_path']}: hidden {HIDDEN}, vocab {VOCAB}")

    def effs(rate):  # one rate against the three peaks
        return "  ".join(f"vs {k} {rate / (peak * s):.3f}" for k, s in shares.items())

    # weight streaming
    print("\neff_weight (decode GEMMs, M tokens):")
    rates = {M: gemm_bytes_per_s(M) for M in c["mbs"]}
    for M, r in rates.items():
        print(f"  M={M:<3d} {r / 1e9:6.0f} GB/s   {effs(r)}")
    rate = statistics.mean(rates.values())
    print(f"  mean  {rate / 1e9:6.0f} GB/s   {effs(rate)}")

    # activation + KV traffic: as whole-GPU mode, with the weight time taken at this
    # slice's measured weight rate; an activation rate, then the three peaks
    kpl, _ = layer_kernel_us(layer, rotary, 1, c["seq_len"], reps=1)
    launch_us = layer_dispatch_us(layer, rotary, c["transport_mb"], c["seq_len"]) / kpl
    w_us = weight_bytes_per_layer() / rate * 1e6
    launch_term = kpl * launch_us
    # Two forms, as in whole-GPU mode. Kernel time is the profiler's summed kernel
    # durations (launch gaps excluded), so subtracting the CPU launch term as well
    # (the ILP form) often leaves nothing; the no-launch form is the usable one then.
    print(f"\neff_act: weight time {w_us:.0f} us, launch term {launch_term:.0f} us "
          f"({kpl} kernels x {launch_us:.1f} us)")
    print(f"  {'mb':>3} {'ctx':>4} {'kernel us':>10} {'ILP GB/s':>9} {'no-launch GB/s':>15}"
          f"   no-launch vs card / GPC / mem")
    sums = {"ilp": [0.0, 0.0], "nl": [0.0, 0.0]}  # bytes, seconds
    for mb in c["mbs"]:
        for ctx in c["ctxs"]:
            _, t = layer_kernel_us(layer, rotary, mb, ctx)
            b = mb * act_bytes_per_token(ctx)
            cols = []
            for key, left in (("ilp", (t - w_us - launch_term) * 1e-6), ("nl", (t - w_us) * 1e-6)):
                if left > 0:
                    sums[key][0] += b
                    sums[key][1] += left
                    cols.append(b / left)
                else:
                    cols.append(None)
            fmt = [f"{r / 1e9:.1f}" if r else "-" for r in cols]
            tail = effs(cols[1]) if cols[1] else "(no time left after the weights)"
            print(f"  {mb:3d} {ctx:4d} {t:10.0f} {fmt[0]:>9} {fmt[1]:>15}   {tail}")
    act_rate = sums["ilp"][0] / sums["ilp"][1] if sums["ilp"][1] else float("nan")
    act_rate_nl = sums["nl"][0] / sums["nl"][1] if sums["nl"][1] else float("nan")
    print(f"  time-equivalent ILP form  {act_rate / 1e9:.1f} GB/s   {effs(act_rate)}")
    print(f"  time-equivalent no-launch {act_rate_nl / 1e9:.1f} GB/s   {effs(act_rate_nl)}")

    # copies: this slice's copy engines
    dec = c["transport_mb"] * HIDDEN * 2
    pre = dec * c["seq_len"]
    d2h, h2d = device_copies(dec)
    d2h_p, h2d_p = device_copies(pre)
    print(f"\ncopies, one activation of mb {c['transport_mb']} (GB/s)   decode {dec // 1024} KiB"
          f"   prefill {pre // 1024} KiB")
    print(f"  D2H (pcie_bw)        {d2h / 1e9:8.2f} {d2h_p / 1e9:14.2f}")
    print(f"  H2D (recv_h2d_bw)    {h2d / 1e9:8.2f} {h2d_p / 1e9:14.2f}")

    # prefill: compute-heavier, so the GPC share may matter more than the memory share
    print(f"\nprefill, one layer, seq {c['seq_len']} (kernel time):")
    print(f"  {'mb':>3} {'us':>9} {'TFLOP/s':>8}   bytes rate vs card / GPC / mem")
    for mb in c["prefill_mbs"]:
        t = prefill_layer_us(layer, rotary, mb, c["seq_len"])
        r = (weight_bytes_per_layer() + mb * c["seq_len"] * act_bytes_per_token(c["seq_len"])) / (t * 1e-6)
        print(f"  {mb:3d} {t:9.0f} {layer_flops(mb, c['seq_len']) / (t * 1e-6) / 1e12:8.1f}   {effs(r)}")

    # endpoints: the lm_head is the big one on the last slice (set model_path to the model)
    print("\nendpoint kernels per microbatch (us):")
    names = ("embed", "clone", "norm", "lm_head", "argmax")
    print(f"  {'':12s} " + " ".join(f"{n:>8}" for n in names))
    for mb in c["copy_mbs"]:
        e = endpoint_us(layer, mb)
        print(f"  decode mb{mb:<3d} " + " ".join(f"{e[n]:8.1f}" for n in names))
    e = endpoint_us(layer, c["transport_mb"], c["seq_len"])
    print(f"  prefill mb{c['transport_mb']:<2d} " + " ".join(f"{e[n]:8.1f}" for n in names))

    def three(r):
        return ", ".join(f"vs_{k}={r / (peak * s):.3f}" for k, s in shares.items())

    print(f"\n# measured on {name}, {os.path.basename(os.path.expanduser(c['model_path']))}")
    print(f"""SLICE_CONSTANTS["{profile}"] = dict(
    weight_rate={rate:.3g},  # bytes/s; eff_weight {three(rate)}
    act_rate={act_rate:.3g},  # bytes/s, ILP form; eff_act {three(act_rate)}
    act_rate_no_launch={act_rate_nl:.3g},  # bytes/s; eff_act {three(act_rate_nl)}
    pcie_bw={d2h:.3g},
    recv_h2d_bw={h2d:.3g},
    launch_us={launch_us:.1f},
    kernels_per_layer={kpl},
)""")


def _copy_worker(uuid, nbytes, seconds, barrier, q):
    """One slice's D2H then H2D loop for `seconds`, started on the barrier so every slice
    copies at once; as device_copies(): a side stream, then a wait on the copy's event."""
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid  # before this process touches CUDA
    n = nbytes // 2
    src = torch.randn(n, dtype=torch.float16, device="cuda")
    dst = torch.empty(n, dtype=torch.float16, device="cuda")
    pinned = torch.empty(n, dtype=torch.float16).pin_memory()
    copy_stream, recv_stream = torch.cuda.Stream(), torch.cuda.Stream()
    ev = torch.cuda.Event()

    def d2h():
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            pinned.copy_(src, non_blocking=True)
        ev.record(copy_stream)
        ev.synchronize()

    def h2d():
        with torch.cuda.stream(recv_stream):
            dst.copy_(pinned, non_blocking=True)
        ev.record(recv_stream)
        ev.synchronize()

    out = {}
    for label, fn in (("d2h", d2h), ("h2d", h2d)):
        for _ in range(20):
            fn()
        barrier.wait()
        k, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn()
            k += 1
        out[label] = k * nbytes / (time.perf_counter() - t0)
    q.put((uuid, out))


def copy_rates(uuids, nbytes, seconds):
    """{uuid: {"d2h": bytes/s, "h2d": bytes/s}}, the given slices copying at the same time."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(len(uuids)), ctx.Queue()
    procs = [ctx.Process(target=_copy_worker, args=(u, nbytes, seconds, barrier, q)) for u in uuids]
    for p in procs:
        p.start()
    out = dict(q.get(timeout=300) for _ in uuids)
    for p in procs:
        p.join(timeout=60)
    return out


def run_contention(c):
    uuids = [u.strip() for u in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if u.strip()]
    if len(uuids) < 2 or not all(u.startswith("MIG-") for u in uuids):
        raise SystemExit("contention: set CUDA_VISIBLE_DEVICES to one card's MIG slices, "
                         "comma-separated, e.g. MIG-<a>,MIG-<b>")
    secs = c["contention_s"]
    print(f"{len(uuids)} slices; each copies {secs:g} s alone, then all together "
          f"(hidden {HIDDEN}, mb {c['transport_mb']})")
    for label, nb in (("decode", c["transport_mb"] * HIDDEN * 2),
                      ("prefill", c["transport_mb"] * c["seq_len"] * HIDDEN * 2)):
        solo = {u: copy_rates([u], nb, secs)[u] for u in uuids}
        both = copy_rates(uuids, nb, secs)
        print(f"\n{label} activation, {nb // 1024} KiB (GB/s):")
        print(f"  {'slice':40s} {'D2H alone':>10} {'together':>9} {'ratio':>6}"
              f" {'H2D alone':>10} {'together':>9} {'ratio':>6}")
        for u in uuids:
            s, b = solo[u], both[u]
            print(f"  {u:40s} {s['d2h'] / 1e9:10.2f} {b['d2h'] / 1e9:9.2f} {b['d2h'] / s['d2h']:6.2f}"
                  f" {s['h2d'] / 1e9:10.2f} {b['h2d'] / 1e9:9.2f} {b['h2d'] / s['h2d']:6.2f}")
        tot_s = sum(solo[u]["d2h"] for u in uuids)
        tot_b = sum(both[u]["d2h"] for u in uuids)
        print(f"  card total D2H: alone-sum {tot_s / 1e9:.2f}, together {tot_b / 1e9:.2f} GB/s")


# ---------------------------------------------------------------------------
def main():
    CONFIG = dict(
        # "off": whole GPU, MIG disabled (every DEFAULT_CONSTANTS entry);
        # "slice": one MIG slice (CUDA_VISIBLE_DEVICES=MIG-<uuid>);
        # "contention": one card's slices copying at once (CUDA_VISIBLE_DEVICES=MIG-<a>,MIG-<b>)
        mig="slice",
        contention_s=3.0,  # contention: seconds each slice copies, alone and then together
        model_path="lmsys/vicuna-7b-v1.5",  # HF id (only config.json is fetched) or a local model dir
        card=None,  # None: from the device; or "A100-40GB" / "A100-80GB"
        mbs=[1, 8, 32],  # microbatch sizes for the layer probes
        ctxs=[64, 192, 320, 448, 575],  # cached tokens: decode steps 1 .. 512 with seq 64
        transport_mb=8,  # activation of this microbatch for the copy probes (decode shape)
        seq_len=64,
        batch=32,  # next_tokens rows
        world=4,  # ranks for the gloo probes
        port=29900,
        copy_mbs=[2, 8, 32],  # microbatch sizes for the copy / handoff size sweep
        prefill_mbs=[1, 4, 16, 32],  # microbatch sizes for the prefill probe
    )
    c = CONFIG

    if c["mig"] == "contention":
        return run_contention(c)  # its worker processes open the slices themselves
    name = torch.cuda.get_device_name(0)
    if c["mig"] == "slice":
        return run_slice(c, name)
    if c["mig"] != "off":
        raise SystemExit(f"CONFIG['mig'] = {c['mig']!r}: use 'off', 'slice' or 'contention'")
    if "MIG" in name:
        raise SystemExit(f"{name} is a MIG slice: run on a whole GPU (MIG disabled), "
                         "e.g. CUDA_VISIBLE_DEVICES=<index>")
    card = c["card"] or detect_card(name)
    peak = CARD_BW[card]
    layer, rotary = load_model(c["model_path"])
    import transformers

    print(f"{name} as {card} (peak {peak / 1e9:.0f} GB/s); {c['model_path']}: hidden {HIDDEN}, "
          f"inter {INTER}, heads {N_HEADS}/{N_KV_HEADS} kv, head_dim {HEAD_DIM}; "
          f"torch {torch.__version__}, transformers {transformers.__version__}")

    # weight streaming
    rates = {M: gemm_bytes_per_s(M) for M in c["mbs"]}
    eff_weight = statistics.mean(rates.values()) / peak
    print("\neff_weight (decode GEMMs, M tokens): " + ", ".join(
        f"M={M}: {r / 1e9:.0f} GB/s = {r / peak:.3f}" for M, r in rates.items())
        + f"  -> mean {eff_weight:.3f}")

    # one layer: kernels, kernel time, CPU dispatch
    kpl, _ = layer_kernel_us(layer, rotary, 1, c["seq_len"], reps=1)
    dispatch = layer_dispatch_us(layer, rotary, c["transport_mb"], c["seq_len"])
    launch_us = dispatch / kpl
    with torch.no_grad():
        h, _, pos = decode_input(rotary, c["transport_mb"], c["seq_len"])
        rotary_kernels, _ = kernels(lambda: rotary(h, pos))
    print(f"kernels_per_layer {kpl}, rotary_kernels {rotary_kernels}, launch_us {launch_us:.1f} "
          f"(CPU {dispatch:.0f} us to queue one layer)")

    # activation + KV traffic, both ways of charging the launch term
    w_us = weight_bytes_per_layer() / (peak * eff_weight) * 1e6
    launch_term = kpl * launch_us
    print(f"\neff_act: one decode layer call; weight time {w_us:.0f} us, launch term "
          f"(kernels x launch_us) {launch_term:.0f} us")
    print(f"  {'mb':>3} {'ctx':>4} {'kernel us':>10} {'act ideal us':>13} {'ILP form':>9} {'no launch':>9}")
    ideal, ilp, nolaunch = [], [], []
    for mb in c["mbs"]:
        for ctx in c["ctxs"]:
            _, t = layer_kernel_us(layer, rotary, mb, ctx)
            a = mb * act_bytes_per_token(ctx) / peak * 1e6
            e1 = a / (t - w_us - launch_term) if t - w_us - launch_term > 0 else float("nan")
            e2 = a / (t - w_us) if t - w_us > 0 else float("nan")
            ideal.append(a), ilp.append(e1), nolaunch.append(e2)
            print(f"  {mb:3d} {ctx:4d} {t:10.0f} {a:13.1f} {e1:9.3f} {e2:9.3f}")

    def scalar(effs):  # time-equivalent: total ideal time / total measured time
        ok = [(a, e) for a, e in zip(ideal, effs) if e == e]
        return (sum(a for a, _ in ok) / sum(a / e for a, e in ok), len(ok)) if ok else (float("nan"), 0)

    eff_act, n_ok = scalar(ilp)
    eff_act_nl, n_nl = scalar(nolaunch)
    print(f"  ILP form ({n_ok}/{len(ilp)} samples usable): eff_act {eff_act:.3f}")
    print(f"  without the launch term in GPU time ({n_nl}/{len(ilp)}): eff_act {eff_act_nl:.3f}")

    # transport
    dec = c["transport_mb"] * HIDDEN * 2
    pre = dec * c["seq_len"]
    d2h, h2d = device_copies(dec)
    send, recv = host_copies(dec)
    d2h_p, h2d_p = device_copies(pre)
    send_p, recv_p = host_copies(pre)
    print(f"\ntransport, one activation of mb {c['transport_mb']} (GB/s)      decode {dec // 1024} KiB"
          f"   prefill {pre // 1024} KiB")
    for label, a, b in (("D2H (pcie_bw)", d2h, d2h_p), ("H2D (recv_h2d_bw)", h2d, h2d_p),
                        ("pinned -> SHM (memcpy_bw)", send, send_p),
                        ("SHM -> pinned (shm_to_pinned_bw)", recv, recv_p)):
        print(f"  {label:34s} {a / 1e9:8.2f} {b / 1e9:12.2f}")

    staging = token_staging_us(c["batch"])
    step = step_start_us(c["transport_mb"], c["seq_len"])
    g = gloo_probes(c["world"], c["port"], c["batch"])
    print(f"\ntoken_staging_us {staging:.1f}, step_start_us {step:.1f}, handshake_ack_us "
          f"{g['handshake_ack_us']:.1f}, tok_roundtrip_us {g['tok_roundtrip_us']:.1f}, "
          f"barrier_us {g['barrier_us']:.1f}")

    # printed before the extras, so a failing extra probe cannot lose it
    print("\n# measured on " + f"{name}, {os.path.basename(os.path.expanduser(c['model_path']))}"
          + f", mb {c['transport_mb']} transport")
    def num(x):  # a pasteable literal, also for a probe that came back NaN
        return f"{x:.3f}" if x == x else 'float("nan")'

    print(f"""DEFAULT_CONSTANTS = dict(
    eff_weight={eff_weight:.3f},
    eff_act={num(eff_act)},  # without the launch term in GPU time: {num(eff_act_nl)}
    launch_us={launch_us:.1f},
    kernels_per_layer={kpl},
    pcie_bw={d2h:.3g},
    recv_h2d_bw={h2d:.3g},
    shm_to_pinned_bw={recv:.3g},
    memcpy_bw={send:.3g},
    handshake_ack_us={g['handshake_ack_us']:.1f},
    tok_roundtrip_us={g['tok_roundtrip_us']:.1f},
    token_staging_us={staging:.1f},
    step_start_us={step:.1f},
    barrier_us={g['barrier_us']:.1f},
    rotary_kernels={rotary_kernels},
    timing_event_calls=2,  # from the benchmark code, not measured
    send_queue_calls=2,  # from the transport code, not measured
)""")

    # ---- extra measurements (not in DEFAULT_CONSTANTS yet) ----
    print("\n==== EXTRA: not in DEFAULT_CONSTANTS yet ====")

    # 1. copies at several sizes -> fixed latency + per-byte rate
    sizes = [mb * HIDDEN * 2 for mb in c["copy_mbs"]] + [c["transport_mb"] * c["seq_len"] * HIDDEN * 2,
                                                        32 * c["seq_len"] * HIDDEN * 2]
    rows = []
    for nb in sizes:
        a, b = device_copies(nb)
        s_, r_ = host_copies(nb)
        rows.append((nb, nb / a, nb / b, nb / s_, nb / r_))
    print("\ncopies vs size (us per copy):")
    print(f"  {'KiB':>8} {'D2H':>8} {'H2D':>8} {'pin>SHM':>8} {'SHM>pin':>8}")
    for nb, *ts in rows:
        print(f"  {nb // 1024:8d} " + " ".join(f"{t * 1e6:8.1f}" for t in ts))
    for i, label in enumerate(("D2H", "H2D", "pinned->SHM", "SHM->pinned")):
        fixed, bw = fit_latency([r[0] for r in rows], [r[i + 1] for r in rows])
        print(f"  fit {label:12s}: fixed {fixed:6.1f} us + bytes / {bw / 1e9:6.2f} GB/s")

    # 2. the real transport, one microbatch handed over and ACKed (2 ranks, this GPU)
    dec = transport_handoff_us(c["port"] + 1, c["copy_mbs"], 1)
    pre = transport_handoff_us(c["port"] + 2, [c["transport_mb"]], c["seq_len"])
    print("\nreal SHM transport, isend -> flush -> wait (ACK), median us per microbatch:")
    for mb, us in dec.items():
        print(f"  decode mb {mb:2d} ({mb * HIDDEN * 2 // 1024:5d} KiB): {us:8.1f}")
    for mb, us in pre.items():
        print(f"  prefill mb {mb:2d} ({mb * c['seq_len'] * HIDDEN * 2 // 1024:5d} KiB): {us:8.1f}")

    # 3. prefill layer
    print(f"\nprefill, one layer, seq {c['seq_len']} (kernel time):")
    print(f"  {'mb':>3} {'us':>9} {'bytes eff':>10} {'TFLOP/s':>8}")
    for mb in c["prefill_mbs"]:
        t = prefill_layer_us(layer, rotary, mb, c["seq_len"])
        tokens = mb * c["seq_len"]
        bw_eff = (weight_bytes_per_layer() + tokens * act_bytes_per_token(c["seq_len"])) / (t * 1e-6) / peak
        print(f"  {mb:3d} {t:9.0f} {bw_eff:10.3f} {layer_flops(mb, c['seq_len']) / (t * 1e-6) / 1e12:8.1f}")

    # 4. endpoints (embed, clone, norm, lm_head, argmax)
    print("\nendpoint kernels per microbatch (us):")
    names = ("embed", "clone", "norm", "lm_head", "argmax")
    print(f"  {'':12s} " + " ".join(f"{n:>8}" for n in names))
    for mb in c["copy_mbs"]:
        e = endpoint_us(layer, mb)
        print(f"  decode mb{mb:<3d} " + " ".join(f"{e[n]:8.1f}" for n in names))
    e = endpoint_us(layer, c["transport_mb"], c["seq_len"])
    print(f"  prefill mb{c['transport_mb']:<2d} " + " ".join(f"{e[n]:8.1f}" for n in names))

    # 5. GPU time of small kernels (the ILP charges launch_us each)
    with torch.no_grad():
        n_rot, us_rot = kernels(lambda: rotary(h, pos))
    print(f"\nsmall kernels: rotary {n_rot} kernels, {us_rot:.1f} us GPU total, "
          f"{us_rot / n_rot:.1f} us each (vs launch_us {launch_us:.1f} CPU)")


if __name__ == "__main__":
    main()
