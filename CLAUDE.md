# mig_pp — working notes

Pipeline-parallel inference benchmark. Vicuna-7B split across MIG slices of one
A100-SXM4-40GB (3g.20gb + 2g.10gb + 2g.10gb), 3 ranks, gloo process group, with a
shared-memory transport engine layered underneath `dist.isend`/`dist.irecv`.

## Ground rules

**The GPU box is remote and Claude cannot reach it.** No SSH, no execution, no
reading its files. Write code locally; the user copies it over, runs it, and pastes
back logs. Never offer to run anything on the box, never ask "should I run it or
will you" — the user runs it, always.

Corollary: make anything the user runs self-contained and one-shot. Every round trip
costs a copy-paste, so front-load diagnostics and make one run answer as many
questions as possible.

**No local torch.** Neither system `python3` nor `.venv` has it. Nothing in
`tests/` can be executed on the Mac — it all runs on the box.

**Permission gate on benchmark edits.** A probe or test whose cases all pass is
required *before* modifying benchmark files. This was set after Claude patched
`benchmark_pipeline_microbatching.py` unasked; the patch was reverted and rebuilt
behind a gate.

**Staging/overhead measurements go to the logger only, never the results CSV.**
They are harness overhead, not a property of the configuration being swept.

**Stay on harness and experiment setup.** The user builds the predictive model. No
equation modelling unless explicitly asked.

## State as of 2026-09-24

Branch `test/40-gb-testing-latest`.

### Committed — gloo CUDA fix (`4d01c02`)

Symptom: all 10 sweep configs died identically at `dist.send(next_tokens)` with
`writev: Bad address`, then every later collective threw "Connection closed by peer".

Cause: **gloo is CPU-only.** It passes `tensor.data_ptr()` straight to `writev(2)`;
a CUDA device pointer there is EFAULT. `next_tokens` lives on `cuda:0`. Activations
dodge this because `dist.isend` is intercepted by the SHM engine and staged
device-to-host — but `patched_send`/`patched_recv` deliberately bypass that
machinery for small control tensors, so staging had to happen at the call site.

Probe evidence that settled it — device is the whole story, not dtype or size:

```
E1 (CPU):             data_ptr=0x597c64e65940  → PASS
E2 (CUDA):            data_ptr=0x302000000     → FAIL writev: Bad address
E5 (CUDA fp16, 1MB):  data_ptr=0x302000200     → FAIL (same)
```

Fix: `token_exchange.py` — a `TokenExchange` class holding one pinned host buffer,
allocated once, copying D2H before send and H2D after recv. Used by both
`probe_gloo_send.py` (the gate, T1–T7) and the benchmark, so the gate exercises the
shipped code rather than a copy of it.

Cost: ~0.0435ms / ~0.0480ms mean over 512 exchanges. Negligible against the ~745ms
ACK waits.

Gate-green via `probe_gloo_send.py`, and a sweep has since been run with it in
place — see Open items for the log analysis.

### Committed — async ACK fix (`912e999`)

Files: `async_handle_send.py`, `async_handle_recv.py`, `mig_transport_pipeline.py`.

Problem: the receiver's ACK send in `AsyncHandleRecv.wait()` is `_ORIGINAL_SEND`,
i.e. `dist.send`, which is **synchronous**. The sender only posted its matching
receive during its final drain, so the receiver blocked inside the ACK send until
the sender got around to draining — the receiver could not start computing until the
sender had finished all its microbatches.

User's CPU/Gloo experiment, 3 microbatches:

| Measurement | Current | ACK recv posted early |
|---|---|---|
| Downstream blocked sending ACK0 | 155.5 ms | 0.06 ms |
| Upstream blocked receiving ACK0 | 0.18 ms | 0.01 ms |
| Downstream MB0 starts | after upstream finishes all MBs | 155 ms earlier |

Fix as applied: `flush()` posts `_ORIGINAL_IRECV` for the ACK *before* sending the
handshake, storing both the buffer and the request on the handle; `wait()` drains
that stored request instead of calling `recv`. `wait()` also now checks the returned
slot index against the expected one and raises on mismatch.

Why it is safe — three rules, and they hold:

1. **Publish only after writing.** The handshake is sent after the data is in SHM.
2. **Acknowledge only after reading.** The receiver ACKs after its H2D lands; it then
   computes on its own GPU tensor, so SHM is dead to it.
3. **Reuse only after acknowledgment.** Posting `irecv` does *not* free the slot —
   it only means "ready to hear about it later." The slot is reclaimed in `wait()`,
   on request *completion*.

ACK tags are `tag + ACK_TAG_OFFSET` (`10_000_000`, `mig_transport_pipeline.py:90`),
so N outstanding receives cannot be confused with one another. The real hazard was
buffer lifetime — an `irecv` buffer GC'd while gloo still holds the pointer — which
is why the handle keeps a reference to both tensor and request.

**Measured on the GPU (sweep `transport_20260924_141712.log`).** `T26` ACK wait is
now 2.4-6.2ms on rank0, against ~745ms before the fix, and it sits well below the
downstream rank's 15.7-29.0ms compute (`B05`) — i.e. the sender no longer waits out
the receiver. Directly visible in the step trace too: at n=6, rank2's first
microbatch lands ~44ms before rank1 finishes its last, which the pre-fix blocking
ACK made impossible.

The original caveat still holds in the part that matters: the ~754ms of rank-1
compute was never an ACK cost and is not recovered. Only the blocking wait went.

### `tests/test_async_ack.py`

Covers exactly the hazards above: publish/reclaim ordering and idempotence, failed
ACK does not release the slot, wrong-slot ACK does not release the slot, receiver
ACK follows H2D completion, and a real two-process Gloo integration test for
receiver progress before sender drain plus tag reuse.

Run state unrecorded here. No torch locally, so on the box:

```bash
python3 -m unittest discover -s tests -p test_async_ack.py -v
```

This was the permission gate for the ACK change, which is now committed.

## Open items

1. **Re-run the sweep after the recv-buffer and norm-weight changes.** The
   12-config sweep in `transport_20260924_141712.log` was analysed and the
   pipeline verified correct (ordering, delivery, overlap — see below); the
   changes since are a dead-kernel removal and a fidelity fix, neither expected
   to move latency, but nothing has been run against them yet. Baseline to beat:
   `[13,10,9]` mb=24 at 40585ms.

   Verified from that sweep, so a future agent need not redo it: 20520 `T15`
   events with zero rank2-before-rank1 ordering violations; send count == recv
   count in every config; slot histogram exactly 6156/4104/4104/2052/2052/2052;
   `T27` slot spins 0 everywhere; 36/36 ranks reported latency. Transport
   overhead is ~7% of a step (69.2ms of stage compute inside a 74.5ms step at
   n=1). Microbatching *loses* here — n=1 40.6s, n=3 54.2s, n=6 76.7s — because
   forward-only decode re-reads the whole layer stack per microbatch, so
   splitting multiplies weight traffic. That is the workload, not a defect.
2. **Rank 1 reads stale `next_tokens`.** It propagates rank2→rank0 only
   (`benchmark_pipeline_microbatching.py:487`, `:592`) but line 547 reads it on every
   rank, so rank 1 sees stale zeros. Probably harmless since only rank 0 feeds embed.
   Flagged, awaiting a decision — not fixed.
3. **DCGM is unavailable.** Not a rename — `datacenter-gpu-manager` is absent from the
   box's apt repo entirely; `apt-cache search dcgm` returns only NSCQ libs. `pynvml` is
   already in the venv and is the proposed fallback. `dcgm_mem_monitor.py` (481 lines,
   unmodified) would need rewriting behind its existing interface:
   `setup_dcgm_group(mig_uuids, slice_gb)`, `start()`, `stop()`, `set_label()`,
   `clear()`, `save_csv(path)`, and `_samples` rows of
   `(ts, label, gpu_mb, gi0_mb, gi1_mb, gi2_mb)`.
4. **MIG on the Autoresearch H100 is blocked.** `CapEff: 0000000000000000` — the
   container has no capabilities, so `nvidia-smi -mig 1` cannot work. Ticket
   `tkt-bx3vt` filed. Log at `logs/h100_mig_enable_probe_2026-09-21.log`.

## Things that cost time before

**The box runs torch 2.14.0+cu130, not the `torch==2.13.0` pinned in
`requirements.txt:280`.** Reasoning from the stale pin produced a confidently wrong
conclusion: adversarial refuter agents claimed `ProcessGroupGloo` stages CUDA tensors
through pinned host buffers and voted the correct gloo diagnosis down 2/2. Claude
retracted a right answer because of it. The probe then proved the original diagnosis.
**Measurement beat both Claude's reasoning and the refuters'.** Check the box's actual
versions before reasoning from any pin.

**Probe ordering matters.** The first probe put the CUDA send second; its EFAULT tore
down the process group, so every later experiment reported "pre-barrier failed"
without executing — including the fix candidate, which therefore went untested. Put
fix candidates first and known-poisoners last, and say so in a comment.

**`dist.send` is synchronous.** Claude claimed the ACK send was "already async in the
direction that matters" — wrong, and it was the load-bearing argument. Also conflated
rank 0's ACK *receive* time (`[T11]`, 0.13ms) with rank 1's ACK *send* block, which
are different ranks and different directions.

**Recv buffers are not zeroed, and that is deliberate.** `prefill_recv_bufs` /
`decode_recv_bufs` are allocated with `torch.empty`, and the per-step
`buf.zero_()` loop over `decode_recv_bufs` was removed. Do not add either back.

Two independent reasons:

1. *Dead work.* The transport overwrites the whole buffer
   (`mig_transport_pipeline.py:575`, `tensor.copy_(staging.view(tensor.shape))`)
   and the only reader is after `handle.wait()`, so initial contents are never
   observed. `prefill_recv_bufs` never had the zeroing loop and has always run
   fine — same transport, same buffers, so the decode-side loop was provably
   doing nothing.
2. *Cross-stream write.* `zero_()` runs on the default stream; the transport's
   H2D runs on `recv_stream`, with no dependency between them. Independent
   streams are unordered, so a late zeroing could in principle land on top of a
   received activation. Silent: same kernels, same shapes, same timing — only
   the values would be wrong.

A code review flagged (2) as a live P1 bug and proposed per-receive readiness
events. `probe_recv_buf_race.py` measured it on the box instead:

| Case | Result | Reading |
|---|---|---|
| P6 | 64/64 zero event complete at H2D queue time | **Race unreachable at this workload's queue depths** |
| P7 | 16384/16384 elements clobbered under a deliberate flood | Hazard is genuine on this GPU, just never reached |

So the review was right that the dependency is missing and wrong that it was
firing; Claude was right that it was unreachable and wrong about why (it argued
the H2D would "queue behind" default-stream work — separate streams do not queue
behind each other). **Measurement beat both sides' reasoning, again.** Removal
was kept as cleanup, not as a correctness fix, and the sweep latencies from
before the change are therefore still valid.

This only holds while decode steps do not overlap. If successive steps are ever
allowed to run concurrently over the same buffers, the hazard becomes reachable
and the readiness event the review proposed is the right fix.

**`load_specific_weights` matched the final norm with `"norm.weight" in key`**,
which also matches `input_layernorm.weight` and `post_attention_layernorm.weight`.
On the last rank every decoder layernorm was copied into the final norm and then
`continue`d past the layer loader, so the per-layer norms were never loaded at
all. Fixed to `key.endswith("model.norm.weight")` (`helpers.py:81`). Output
fidelity only — shapes and kernels are unchanged, so it does not move latency.

**This is forward-only inference.** No backward pass. "1F1B" is the wrong vocabulary.

**`run_command` on Autoresearch uses `sh`, not `bash`** — redirections like `&>` fail
with `Syntax error: redirection unexpected`. Write a script file, then `bash file.sh`.

**`file_ticket` refs** only accept node/command/job/endpoint. A mission name in `refs`
is rejected as `bad ref`; put it in the body.

## Log tags

Transport logs are dense. Grep by tag rather than pasting whole files:

| Tag | Meaning |
|---|---|
| `T05` | isend posted |
| `T11` | ACK received, slot freed — remaining wait at drain, *not* downstream compute |
| `T14` | H2D queued |
| `T15` | H2D landed |
| `T16` | ACK sent |
| `T23`–`T27` | end-of-run summaries (D2H, H2D, handshake, ACK, slot spins) |
| `T28`/`T29` | verdict: no async sends / overlap working |

```bash
grep -E '\[T05\]|\[T11\]|\[T15\]|\[T16\]' transport-*.log | head -100
```

Note `T26`'s wording changed with the ACK fix: it used to be described as downstream
compute, which is no longer true now that receives are posted during `flush()`.

## Layout

| File | Role |
|---|---|
| `benchmark_pipeline_microbatching.py` | the sweep; gloo backend, 8-min timeout |
| `mig_transport_pipeline.py` | SHM engine; patches isend/irecv, leaves send/recv alone |
| `async_handle_send.py` / `async_handle_recv.py` | slot handles, handshake and ACK protocol |
| `token_exchange.py` | pinned CPU staging for `next_tokens` over gloo |
| `probe_gloo_send.py` | acceptance gate for the gloo fix (T1–T7) |
| `probe_recv_buf_race.py` | gate for the recv-buffer + norm fixes (P1–P7) |
| `tests/test_async_ack.py` | acceptance gate for the ACK fix — never run |
| `dcgm_mem_monitor.py` | memory sampling; DCGM unavailable, needs a pynvml rewrite |
