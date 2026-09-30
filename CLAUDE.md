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

**Local torch is CPU-only and partial.** `.venv` has torch 2.14.0 (CPU build) but
no transformers/numpy/datasets/pandas; system `python3` has none of them. The
torch-free suites run on the Mac (see the parallel-runner section). For torch-level
checks, install the pinned transformers into the scratchpad, not the venv:
`.venv/bin/python -m pip install --target <scratch>/pylib transformers==5.14.1
safetensors==0.8.0 numpy tqdm`, then `PYTHONPATH=<scratch>/pylib` (in zsh write
`${LIB}:tests`, not `$LIB:tests` — `:t` is a zsh modifier). Anything needing
CUDA, MIG or gloo across slices still only runs on the box.

**Permission gate on benchmark edits.** A probe or test whose cases all pass is
required *before* modifying benchmark files. This was set after Claude patched
`benchmark_pipeline_microbatching.py` unasked; the patch was reverted and rebuilt
behind a gate.

**Staging/overhead measurements go to the logger only, never the results CSV.**
They are harness overhead, not a property of the configuration being swept.

**Stay on harness and experiment setup.** The user builds the predictive model. No
equation modelling unless explicitly asked.

## State as of 2026-09-30 — 4 x A100-80GB, heavier models (uncommitted)

**The run box is `cc@mig` (repo `~/mig_pp`): 4 x A100-SXM4-80GB**, driver 560.35.05
(CUDA 12.6 max), MIG disabled as of 9/30. Its venv came with torch `2.14.1+cu130`
("driver too old (found version 12060)"); now `2.14.1+cu126` from the PyTorch cu126
index: 4 GPUs, fp16 matmul + SDPA ok. It has **transformers 5.18.0** (not the pinned
5.14.1): `tests/test_model_family_torch.py` passes against 5.18.0, including a Mistral
variant with explicit `head_dim` != hidden/heads (Mistral-Small-24B's 128 vs 160).
Enabling MIG there needs a reboot: `-mig 1` stays pending ("In use by another client")
and `nvidia-smi -r` is refused even with nothing in `fuser -v /dev/nvidia*` and
nvidia-persistenced stopped. MIG slices (and so their UUIDs) do not survive a reboot.
Hard budget: the run must finish in 2.5 days. (`pace@etracker2` was a mix-up: 2 x GTX
1080 Ti, no MIG; `/data` not writable there, hence `MODEL_ROOT = ~/models`.)

Layout 40/20/10/10 (3g.40gb + 2g.20gb + 1g.10gb + 1g.10gb: the 9/28 compute split,
twice the memory, ~1.3x the bandwidth, so latencies do not line up with 40GB data).
GPU time shrank to 2 days (48 h); the user comments out batch pairs by hand and
checks progress themselves (B64 first). One model per GPU, all 14 batch pairs (the user's call: the 16/32-microbatch pairs were
only expensive on the pre-ACK-fix harness): vicuna_13b `[22,11,6,6]` 33 splits,
llama_13b same, qwen_14b `[27,10,9,9]` 31, mistral_24b (Mistral-Small-24B-Base-2501)
`[22,10,7,5]` 29 = 462/462/434/406 configs. The user dropped nemo_12b for mistral_24b
and asked for wider small-slice caps than the April 80GB branches' `[24,10,5,5]` /
`[30,10,4,2]` (5GB-era 5s): the 10GB slices sit at their B64 capacity (13B 6, Qwen 9,
24B 7/5 -- Qwen's and 24B's last rank within ~300 MiB of the lm_head build peak, and
the smoke split does not exercise them), 0 OOM predicted; ranks 0-1 narrowed to fit.

Time. April 80GB data (branches `vic-13b-4mig-80gb`, `mistral-24b-4mig-80gb`; pre-ACK-fix
code): 13B `[24,10,5,5]` x 14 pairs = 34.8 h, 24B 49 splits = 102.6 h; wall - latency
per config ~80 s (13B), ~141 s (24B). New-harness estimate = 9/28's within-batch
microbatch profile x each model's April 2-microbatch latency + that overhead: per split
at 14 pairs 4124 s (13B), 4472 s (24B); x1.15-1.2 for more layers on slow slices ->
~43 h per lane (April pace: 59-73 h). The 3 high-microbatch pairs are ~1/3 of a split
on the new harness (46% in April). 14 pairs x the earlier 48/44/35 splits was ~52-63 h:
over budget. April's `hang` rows were not hangs: `JOIN_TIMEOUT_S = 1200` includes weight
load and April's (64,2) ran ~20 min; new-harness (64,2) estimate ~600-750 s. April
maxima give the 80GB slice sizes (40190/19965/9728 MiB): `memory_limits.LAYOUT_MIB`.
`oracle/80gb/20_10_5_5/` holds these 40/20/10/10 runs despite its name.

## State as of 2026-09-29 — 8-GPU parallel runner (committed `2236a5e`)

Goal: one box, 8 A100s, MIG on each; one **lane per GPU** running that GPU's model
list sequentially, all lanes at once, nothing shared between lanes. Vicuna-7B is
done (9/28); the other six models are on GPUs 0-5, Mistral-Nemo-12B on GPU 6, and a
full Vicuna-7B re-run (same 938 configs as 9/28) on GPU 7 to measure how much the new
box + eight lanes at once move latency (and to check the recovered 9/28 memory
mapping against NVML's per-UUID columns). `download_models.py` fetches all eight
(162.8 GB, one weight format each) into `MODEL_ROOT`; the two Llama-2 repos are gated.

Collisions that made 8 plain copies unsafe (all fixed, all default-preserving):

| Resource | Was | Now |
|---|---|---|
| Rendezvous port | `MASTER_PORT="29500"` hardcoded | `MASTER_PORT` constant; per job `base_port + gpu*10` |
| /dev/shm names | `mig_pipe_shm_{rank}_slot{N}`; startup *unlinks* any existing one | `$MIG_SHM_PREFIX` (`migpp_<run>_g<gpu>j<NN>`); legacy name when unset |
| Outputs | `benchmark.log`, CSVs, `logs/` in CWD | unchanged code; the manager runs each job with CWD = its job dir |
| Memory monitor | DCGM: `sudo pkill -f 'dcgmi dmon'`, one group name, GPU 0 hardcoded | `nvml_mem_monitor.py`, per MIG UUID, no sudo; planner refuses DCGM for >1 lane |
| Model | Llama classes + `.bin` only; other models lived on per-model branches | `model_family.resolve(model_type)` (llama, mistral, qwen2); safetensors + mmap'd `.bin`; `[B08]` coverage line |
| CPU | unbound | each lane `taskset` to its GPU's NUMA-local CPUs (lanes on a node share them) |

How a job gets its config: `parallel_plan.py` reads each model's `config.json`
(layers, hidden, type), writes `job.json`, the lane exports `MIG_EXP_CONFIG`, and the
benchmark's PARALLEL-RUN OVERRIDE block replaces its constants. Unset => standalone
run exactly as before (`tests/test_benchmark_wiring.py` execs the real config
section both ways).

Layer limits live in `layer_limits.py` (layout -> model -> per-rank caps, plus
`HEAD_ONLY_LAST_RANK`). `memory_limits.py` is the theory behind them: fp16 weights +
KV (x1.10) + context 250 MiB + prefill activations + the embed/lm_head fp32 build
peak. Calibrated on the 9/28 sweep it predicts 933/938 ok/OOM outcomes, and
reproduces `mig_pp/profiles/poc_layer_memory.csv` byte-for-byte. `[18,12,5,5]` is
exactly "rank0 = capacity at B64 - 1, ranks 1-3 = capacity at B32 - 1"; 7B models
share it, 13B uses the same rule (`[23,12,5,5]`), Qwen keeps a head-only last rank.

`generate_layer_splits` now delegates to `experiment_config` (torch-free), proven
equal to a frozen copy of the old body (67 splits x 14 = 938 = the 9/28 CSV) and,
with `min_last_rank_layers=0`, to the qwen-14b branch's generator. The qwen-7b branch
also allowed a head-only last rank (oracle has `(16,8,4,0)`) — the 9/24 generator
could not express either.

Gate status: torch-free suites green on Python 3.13 and 3.10, bash 3.2 and 5.3,
shellcheck clean; `tests/test_model_family_torch.py` (loader + stage maths vs HF,
3 families x 3 checkpoint formats) green on CPU against transformers 5.14.1.
**Box side not run yet:** `./run_parallel.sh smoke` is the acceptance gate — one tiny
config per job on every lane at once, then a PASS/FAIL verdict per job (state ok,
results rows ok, `[B08] missing=0` per rank, `[T01] shm_prefix=` = the job's prefix per
rank, memory trace within slice capacity, no /dev/shm leftovers). On the box,
`python3 -m unittest discover -s tests -p 'test_model_family_torch.py'` also runs
there on CPU in seconds.

Simplified on 2026-09-29 at the user's request ("get rid of over-engineering"):
the 2.5k-line preflight is gone (check + smoke cover it), and the planner, NVML
monitor and manager were cut to what is listed here.

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
3. **DCGM: availability varies by box; its rank mapping is wrong where it runs.**
   It was absent from one box's apt repo, but on the 9/26 box it ran — and fell
   back to `could not parse DCGM entity ids ... assuming ids 0..3` (see item 5).
   `nvml_mem_monitor.py` now implements the same interface by MIG UUID; parallel
   runs use it (`RUNNER mem_monitor="nvml"`). The standalone default is still DCGM.
4. **MIG on the Autoresearch H100 is blocked.** `CapEff: 0000000000000000` — the
   container has no capabilities, so `nvidia-smi -mig 1` cannot work. Ticket
   `tkt-bx3vt` filed. Log at `logs/h100_mig_enable_probe_2026-09-21.log`.

5. **9/28 memory columns are permuted** (latency and OOM status are fine — those come
   from the ranks). Recovered mapping, same for peak_*, avg_* and the trace:

   | column | actually holds |
   |---|---|
   | `*_rank0_20gb_mb` | rank 2 (5GB) |
   | `*_rank1_10gb_mb` | rank 3 (5GB, lm_head) |
   | `*_rank2_5gb_mb` | rank 1 (10GB) |
   | `*_rank3_5gb_mb` | rank 0 (20GB) |

   Three independent checks: column maxima sit at the true slices' capacities
   (4861/4863/9983/20085 MB); each rank's layer count correlates most with that
   column (r 0.51/0.51/0.63/0.84); with equal layers on ranks 2-3, col1-col0 = +203 MB
   (the 262 MB lm_head) in 184/209 rows. A pure reversal was considered first and is
   wrong for the two 5GB columns.
6. **Embed / lm_head are built fp32 on the GPU, then `.half()`** (`nn.Linear(...).to(device).half()`):
   6 bytes/param at the peak. Harmless for 32k vocabs (750 MiB) but it is what caps
   Qwen's last 5GB rank — 3118 MiB for Qwen2.5-7B (max 3 layers), 4455 MiB for
   Qwen2.5-14B (0 layers). Building them in fp16 directly would free ~2/3 of that.
   Not changed: it would move memory numbers against the 9/28 and oracle data.
   Awaiting a decision.
7. **Parallel-run interference is unmeasured.** Lanes are pinned to their GPU's NUMA
   node (lanes on one node share its cores); host memory bandwidth, PCIe switches
   (GPU pairs) and page cache are shared. Compare one model solo (`--gpus N`) against the same model
   in the full run before trusting cross-lane comparisons.
8. **Results CSV is written only at the end of a job's sweep** (pre-existing). A job
   killed at config 500/938 loses its rows; `benchmark.log` keeps per-config latency.
9. **13B at B32/B64 cannot fit on 20/10/5/5 at all** (26 GB weights + KV; the old 13B
   oracle runs only ever swept B8/B16), so `parallel_config.py` restricts the 13B
   models to B8/B16 and Qwen2.5-14B to B<=32. Put them back if OOM rows are wanted.

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
| `dcgm_mem_monitor.py` | DCGM memory sampling; GPU 0 only, not parallel-safe, mis-maps ranks (open item 5) |
| `nvml_mem_monitor.py` | same interface via NVML by MIG UUID; parallel-safe, no sudo |
| `run_parallel.sh` | bash manager: `discover check smoke start status stop` (+ `--detach`, `--resume`); `stop` only aborts early; one session per lane |
| `parallel_plan.py` | torch-free planner: validation, run dirs, report, smoke verdict |
| `parallel_config.py` | which models on which GPU (+ per-model batch overrides); the file you edit |
| `layer_limits.py` | per-layout, per-model layer caps (dictionary) + head-only-last-rank set |
| `memory_limits.py` | theoretical per-slice capacity per model/batch; `--validate` against a results CSV |
| `download_models.py` | fetches every MODELS entry into its configured path (`--check` = access/sizes/disk only) |
| `experiment_config.py` | job.json schema + the split enumeration (torch-free) |
| `model_family.py` | model_type -> classes, model path / weight-file discovery (torch-free at import) |
| `tests/test_run_parallel.py` | end-to-end: real planner + manager + stub benchmark on a fake 8-GPU box |
| `tests/test_model_family_torch.py` | loader + stage maths vs HF for llama/mistral/qwen2 (skips without transformers) |
