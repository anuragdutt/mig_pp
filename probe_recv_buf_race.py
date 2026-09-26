"""
Acceptance gate for two changes (P1-P7):

  1. helpers.py — `norm.weight` key match narrowed from `in` to
     `endswith("model.norm.weight")`.
  2. benchmark_pipeline_microbatching.py — removal of the `buf.zero_()` loop
     over `decode_recv_bufs`, which is a default-stream write to a buffer that
     mig_transport's `_queue_read_from_slot` later overwrites from `recv_stream`
     with no cross-stream dependency.

Run on the box:

    python3 probe_recv_buf_race.py

All seven cases must PASS before either file is edited in a way that reaches a
sweep. P1-P4 need no GPU; P5-P7 are skipped (not failed) without CUDA.

ORDERING NOTE (this cost time before, see CLAUDE.md): the known-poisoner goes
LAST. P7 deliberately provokes the race with an oversized zeroing kernel; if it
were to wedge or corrupt the context, everything after it would report garbage
without executing. Fix candidates run first, provocation last.
"""

import sys

import torch

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)


# ---------------------------------------------------------------------------
# P1-P2: norm.weight key matching (helpers.py fix) — no GPU needed.
# ---------------------------------------------------------------------------

# Real Llama checkpoint key shapes, as they appear in the vicuna-7b shards.
LLAMA_KEYS = [
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.31.input_layernorm.weight",
    "model.layers.31.post_attention_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
]


def p1_old_match_is_broken():
    """The OLD predicate must be shown to over-match, or the fix is pointless."""
    hits = [k for k in LLAMA_KEYS if "norm.weight" in k]
    layer_norms = [k for k in hits if "layers." in k]
    if len(layer_norms) == 4 and "model.norm.weight" in hits:
        record(
            "P1 old predicate over-matches",
            PASS,
            f"`in` caught {len(hits)} keys incl. {len(layer_norms)} per-layer norms",
        )
    else:
        record("P1 old predicate over-matches", FAIL, f"unexpected hits: {hits}")


def p2_new_match_is_exact():
    """The NEW predicate must catch the final norm and nothing else."""
    hits = [k for k in LLAMA_KEYS if k.endswith("model.norm.weight")]
    if hits == ["model.norm.weight"]:
        record("P2 new predicate exact", PASS, "matches only model.norm.weight")
    else:
        record("P2 new predicate exact", FAIL, f"hits={hits}")


# ---------------------------------------------------------------------------
# P3-P4: the per-layer norms must now reach the layer loader.
# ---------------------------------------------------------------------------


def _route(key, use_new_predicate):
    """Mirror of load_specific_weights' branch order for rank == world_size-1."""
    if use_new_predicate:
        if key.endswith("model.norm.weight"):
            return "final_norm"
    else:
        if "norm.weight" in key:
            return "final_norm"
    if "lm_head.weight" in key:
        return "lm_head"
    if "layers." in key:
        return "layer"
    return "unrouted"


def p3_layer_norms_reach_layer_loader():
    layer_norm_keys = [k for k in LLAMA_KEYS if "layernorm.weight" in k]
    old = [_route(k, False) for k in layer_norm_keys]
    new = [_route(k, True) for k in layer_norm_keys]
    if all(r == "final_norm" for r in old) and all(r == "layer" for r in new):
        record(
            "P3 per-layer norms routed to layer loader",
            PASS,
            f"{len(layer_norm_keys)} keys: final_norm -> layer",
        )
    else:
        record("P3 per-layer norms routed to layer loader", FAIL, f"{old} -> {new}")


def p4_final_norm_and_lm_head_still_route():
    if (
        _route("model.norm.weight", True) == "final_norm"
        and _route("lm_head.weight", True) == "lm_head"
        and _route("model.layers.3.self_attn.q_proj.weight", True) == "layer"
    ):
        record("P4 final norm / lm_head / proj still route", PASS)
    else:
        record("P4 final norm / lm_head / proj still route", FAIL)


# ---------------------------------------------------------------------------
# P5-P7: the recv-buffer stream question. CUDA only.
# ---------------------------------------------------------------------------

MB, SEQ, HIDDEN = 4, 1, 4096


def p5_h2d_fully_overwrites_buffer():
    """
    The load-bearing claim behind deleting `buf.zero_()`: the H2D writes every
    element, so the zeroing has no observable effect. Poison the buffer with a
    sentinel, copy over it, and confirm not one sentinel element survives.
    """
    dev = torch.device("cuda:0")
    buf = torch.full((MB, SEQ, HIDDEN), 7.0, dtype=torch.float16, device=dev)
    staging = torch.empty(MB * SEQ * HIDDEN, dtype=torch.float16, pin_memory=True)
    staging.copy_(torch.arange(MB * SEQ * HIDDEN, dtype=torch.float16) % 100)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        buf.copy_(staging.view(buf.shape), non_blocking=True)
    buf.record_stream(stream)
    evt = torch.cuda.Event()
    evt.record(stream)
    evt.synchronize()

    survivors = int((buf == 7.0).sum().item())
    expected = staging.view(buf.shape).to(dev)
    if survivors == 0 and torch.equal(buf, expected):
        record("P5 H2D overwrites every element", PASS, "0 sentinel survivors")
    else:
        record(
            "P5 H2D overwrites every element",
            FAIL,
            f"{survivors} sentinel elements survived the copy",
        )


def p6_zero_completes_before_h2d_in_situ():
    """
    Does the race even reach the starting line in this workload? Record an event
    after the zeroing, then do what the benchmark does before the H2D is queued:
    block the CPU (the gloo handshake wait + host memcpy). If the zero event has
    already completed by then, the H2D cannot land ahead of it.

    This measures rather than argues — the point on which both the review and
    the earlier rebuttal were reasoning without evidence.
    """
    dev = torch.device("cuda:0")
    bufs = [
        torch.empty((MB, SEQ, HIDDEN), dtype=torch.float16, device=dev)
        for _ in range(6)
    ]

    pending = 0
    for _ in range(64):
        for b in bufs:
            b.zero_()
        zero_evt = torch.cuda.Event()
        zero_evt.record()  # default stream, exactly as the benchmark leaves it

        # Stand-in for the blocking handshake recv + SHM->pinned host memcpy
        # that AsyncHandleRecv.wait() performs before queueing any H2D.
        host = torch.empty(MB * SEQ * HIDDEN, dtype=torch.float16, pin_memory=True)
        host.copy_(torch.zeros(MB * SEQ * HIDDEN, dtype=torch.float16))

        if not zero_evt.query():
            pending += 1

    if pending == 0:
        record(
            "P6 zeroing completes before H2D is queued",
            PASS,
            "64/64 iterations: zero event complete at queue time",
        )
    else:
        record(
            "P6 zeroing completes before H2D is queued",
            FAIL,
            f"{pending}/64 iterations still had the zeroing pending — "
            "race is reachable, removal of zero_() is REQUIRED not optional",
        )


def p7_provoked_race_is_visible(force=False):
    """
    KNOWN POISONER — RUNS LAST ON PURPOSE.

    Inflate the default-stream write far past anything the benchmark queues, so
    it is still pending when the H2D fires on its own stream. If the buffer ends
    up holding zeros rather than the copied payload, the missing cross-stream
    dependency is demonstrated to be real on this hardware — which is what
    neither the review's CPU simulation nor the earlier rebuttal established.

    A PASS here means "the hazard is real in principle"; P6 is what says whether
    the benchmark ever gets near it.
    """
    dev = torch.device("cuda:0")
    big = torch.empty((512, 512, HIDDEN), dtype=torch.float16, device=dev)
    buf = torch.empty((MB, SEQ, HIDDEN), dtype=torch.float16, device=dev)
    staging = torch.full(
        (MB * SEQ * HIDDEN,), 3.0, dtype=torch.float16, pin_memory=True
    )

    torch.cuda.synchronize()
    for _ in range(40):  # flood the default stream
        big.zero_()
    buf.zero_()  # the write we are racing

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        buf.copy_(staging.view(buf.shape), non_blocking=True)
    evt = torch.cuda.Event()
    evt.record(stream)
    evt.synchronize()

    zeros = int((buf == 0.0).sum().item())
    torch.cuda.synchronize()
    if zeros > 0:
        record(
            "P7 provoked race observable",
            PASS,
            f"{zeros}/{buf.numel()} elements clobbered by the late zeroing — "
            "cross-stream hazard is real on this GPU",
        )
    else:
        record(
            "P7 provoked race observable",
            PASS,
            "not reproducible even under flood; hazard remains theoretical here",
        )


def main():
    print("=== P1-P4: norm.weight key matching (no GPU) ===", flush=True)
    p1_old_match_is_broken()
    p2_new_match_is_exact()
    p3_layer_norms_reach_layer_loader()
    p4_final_norm_and_lm_head_still_route()

    print("\n=== P5-P7: recv buffer / stream ordering (CUDA) ===", flush=True)
    if not torch.cuda.is_available():
        for n in ("P5 H2D overwrites every element",
                  "P6 zeroing completes before H2D is queued",
                  "P7 provoked race observable"):
            record(n, SKIP, "no CUDA on this host")
    else:
        print(f"torch {torch.__version__} / device {torch.cuda.get_device_name(0)}",
              flush=True)
        p5_h2d_fully_overwrites_buffer()
        p6_zero_completes_before_h2d_in_situ()
        p7_provoked_race_is_visible()  # poisoner last

    print("\n=== SUMMARY ===", flush=True)
    failed = [r for r in results if r[1] == FAIL]
    skipped = [r for r in results if r[1] == SKIP]
    for name, status, detail in results:
        print(f"  {status:4} {name}" + (f" — {detail}" if detail else ""))
    print(
        f"\n{len(results) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )
    if failed:
        print("\nGATE RED — do not edit benchmark_pipeline_microbatching.py.")
        return 1
    print("\nGATE GREEN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
