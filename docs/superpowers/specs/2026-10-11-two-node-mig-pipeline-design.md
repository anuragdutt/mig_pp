# Two-node MIG pipeline — design

Date: 2026-10-11. Status: approved section by section in conversation; this
spec awaits review. Not committed (the user does all git writes).

## Goal

Run the existing pipeline-parallel benchmark with one pipeline whose stages sit
on MIG slices of **two GPUs in two different nodes**. The experiment stays what
it is today — forward-only decode, a gloo process group, ranks respawned per
configuration, splits × batch pairs swept — with the slice placement fixed per
run.

## Decisions

| Question | Decision | Rejected |
|---|---|---|
| Shape of an experiment | Fixed placement per run: a per-rank list of (node, MIG UUID, slice GB) in pipeline order, contiguous or interleaved. Another placement is another run. | Placements swept by the harness (multiplies the config count); two independent pipelines (needs no harness change) |
| Cross-node transport | gloo TCP with pinned host staging | NCCL/RDMA (a second process group; NCCL on MIG unverified) |
| Launch | The same command on each node; the two parents meet on the lead's control port | Node A starting B over SSH |
| Split rule | Size-monotone over all rank pairs (§4) | No ordering; explicit split lists only |
| Architecture | Per-hop routing in the engine, one parent per node, a control channel between the parents | Relay processes (an extra hop inside the measured latency); torchrun persistent ranks (no per-config respawn, so not comparable with existing data) |

## Non-goals

- No placement sweep, no NCCL, no SSH orchestration.
- No change to single-node behaviour. Every change is a no-op when the two-node
  block is absent; tests and a single-node smoke prove it.
- No cost modelling. No harness-overhead numbers in the results CSV (logger only).

## Preconditions on the boxes

- Both nodes run the same checkout, in a separate directory such as
  `~/mig_pp_2node`, so single-node sweeps never import half-copied code (ranks
  re-import modules at every configuration). Same torch and transformers
  versions on both (enforced at connect). Model weights at the configured
  `path` on both.
- MIG slices exist on each node's GPU, and their UUIDs are in `PLACEMENT` (from
  `nvidia-smi -L` or `./run_parallel.sh discover` on that node).
- A private network between the nodes with **all TCP ports open both ways** on
  the chosen interface. gloo opens an ephemeral listening port per rank pair, on
  top of the master and control ports.

## 1. Surface

### Config: `two_node_config.py` (new, plain Python, read by the planner)

```python
import os

MODEL_ROOT = os.path.expanduser("~/models")

# addr:   this node's IP on the private network (gloo binds and connects on it)
# ifname: the NIC carrying addr (exported to every rank as GLOO_SOCKET_IFNAME)
# gpu:    physical GPU index holding this node's slices (NUMA pinning, inventory check)
NODES = {
    "A": dict(addr="10.0.0.1", ifname="eno1", gpu=0),
    "B": dict(addr="10.0.0.2", ifname="eno1", gpu=0),
}

# Pipeline order: (node, MIG UUID, slice GB). The node holding rank 0 is the lead.
PLACEMENT = [
    ("A", "MIG-...", 40),
    ("A", "MIG-...", 20),
    ("B", "MIG-...", 40),
    ("B", "MIG-...", 20),
]

# Run in this order. layer_limits: one cap per rank, required.
MODELS = {
    "vicuna_13b": dict(path=f"{MODEL_ROOT}/vicuna-13b-v1.5", layer_limits=[22, 11, 22, 11]),
}

SWEEP = dict(seq_len=64, max_new_tokens=512, max_runs=None, enforce_slice_ordering=True,
             batch_mb_pairs=[(8, 4), (8, 2)])

RUNNER = dict(
    base_port=29600,      # job k: master port base_port + 10k (single-node lanes use 29500-29570)
    control_port=29900,   # job k: control port control_port + k
    peer_timeout_s=1800,  # how long a parent waits for the other node to show up
    mem_monitor="nvml",   # "nvml" or "off"; DCGM is refused
    env={"MIG_LOG_LEVEL": "summary"},
)
```

Per-model overrides work as in `parallel_config.py`: `seq_len`,
`max_new_tokens`, `max_runs`, `batch_mb_pairs`, `splits`, `layer_limits`,
`min_last_rank_layers`. Unknown keys are errors.

The default ports sit outside the single-node lanes' range (29500–29570), so
both kinds of run can share a box.

### Commands

```
node A:  ./run_parallel.sh smoke -c two_node_config.py --node A
node B:  ./run_parallel.sh smoke -c two_node_config.py --node B
```

`check`, `start [--detach]`, `status`, `stop` and `--resume` work as today, each
with `--node X`. Start order does not matter: each parent waits up to
`peer_timeout_s` for the other.

Each node has its own run directory (`runs/<timestamp>[_smoke]_<node>`), NUMA
pinning (the local CPUs of `NODES[node]["gpu"]`) and `/dev/shm` prefix
(`migpp_<run>_<node>j<NN>`). With
`--node`, the runner uses its own lock file, so a two-node run and single-node
lanes on the same box do not refuse each other.

### Outputs

- The lead's job directory: `mig_benchmark_results.csv` and
  `mig_step_latencies.csv`, with today's columns for every rank plus one column,
  `placement` (e.g. `A40,A20,B40,B20`). Single-node CSVs are unchanged.
- Each node: its own `benchmark.log`, transport log and `mig_memory_trace.csv`
  (columns labelled with global rank numbers).
- A smoke verdict on each node. The lead's covers every rank.

## 2. Transport: per-hop routing

### Engine (`mig_transport_pipeline.py`)

- `register_hooks(..., rank_nodes=None)` passes `rank_nodes` (the node name of
  each rank) to `MIGPipelineTransport`. `None` means every rank is on one node:
  today's behaviour exactly.
- A peer is **remote** when `rank_nodes[peer] != rank_nodes[rank]`.
- Init attaches to the `/dev/shm` slots of same-node peers only. Today it
  attaches to every peer, and a remote one raises after 10 s.
- `isend(tensor, dst)`: slot and D2H queueing exactly as today
  (`_get_free_slot`, `_queue_copy_to_staging`, `T05`). Same node →
  `AsyncHandleSend` (unchanged). Remote → `AsyncHandleSendNet`.
- `irecv(tensor, src)`: same node → `AsyncHandleRecv` (unchanged). Remote →
  reserve a receive-staging slot (marked reserved until its H2D lands), post
  `_ORIGINAL_IRECV(recv_pinned_staging[rslot][:tensor.numel()], src, tag)`, and
  return `AsyncHandleRecvNet`. The sender sends exactly `tensor.numel()`
  elements, so message sizes always match.
- Remote hops require fp16 and raise `TypeError` otherwise. (The SHM path's
  non-fp16 fallback calls `_write_tensor_to_slot`
  (`mig_transport_pipeline.py:514`), which is defined nowhere. Pre-existing and
  unreachable for fp16 activations; noted, not changed.)

### Handles (`async_handle_net.py`, new)

Same interface as the SHM handles, so the benchmark's prefill and decode loops
(flush one microbatch behind, batched drain) do not change.

`AsyncHandleSendNet`

- `flush()`: wait on this slot's D2H event, then post
  `_ORIGINAL_ISEND(pinned_staging[slot][:numel], dst, tag)` and keep the buffer
  and request. The event wait counts toward `T23` (flush block), so the
  `T29`/`T30` overlap verdict keeps its meaning. Idempotent.
- `wait()`: `flush()` first if needed (with the `T06b` warning), wait for the
  gloo send to complete (`T33` time), then free the slot. If the wait raises,
  the slot stays taken.

`AsyncHandleRecvNet`

- `wait()`: wait for the gloo receive (`T34` time: upstream compute plus wire),
  queue the H2D on `recv_stream` (`record_stream` on the destination), sync that
  event (`T24`, H2D block), release the staging slot. Idempotent.

### Safety rules (the three ACK rules, cross-node form)

1. **Send only after writing.** The isend is posted only after the slot's D2H
   event has completed.
2. **Release receive staging only after reading.** The slot stays reserved from
   the irecv post until its H2D event completes.
3. **Reuse send staging only after completion.** The send slot is freed in
   `wait()`, after gloo reports the send complete, never in `flush()`.

Tags are unchanged (unique per step and microbatch), so many outstanding
receives from one source cannot be confused. Each handle keeps its buffer and
request alive until completion (the GC hazard noted for the ACK fix).

### Token exchange and rank environment

- `TokenExchange` is unchanged. When the last rank and rank 0 are on different
  nodes, every decode step makes one network round trip. That is real pipeline
  latency and belongs in the measurement.
- Every rank gets `MASTER_ADDR = node_addrs[rank_nodes[0]]` and
  `GLOO_SOCKET_IFNAME = <this node's ifname>`. Without the interface, gloo binds
  whatever the hostname resolves to — on Ubuntu often `127.0.1.1` — and remote
  connects fail. Single-node runs set neither: `127.0.0.1` as today.
- With the interface set, same-node handshakes and ACKs address the NIC's IP;
  Linux still delivers traffic to a local address through loopback.

### Logs (never the CSV)

| Tag | Level | Content |
|---|---|---|
| `T33` | WARNING (shown at summary level) | per rank: cross-node send completion wait, mean / total / count |
| `T34` | WARNING (shown at summary level) | per rank: cross-node receive wait (upstream compute + wire), mean / total / count |
| `T35` | INFO | per microbatch: cross-node isend posted (slot, tag, bytes) |
| `T36` | INFO | per microbatch: cross-node receive landed (tag, wait) |

`T22`–`T30` keep their meaning, with cross-node flushes and H2D blocks included
in `T23`/`T24`. `T33`/`T34` print only for ranks with cross-node traffic.

## 3. Orchestration

### Control channel (`two_node_control.py`, new, stdlib only)

- Plain TCP sockets (rather than `multiprocessing.connection`, for explicit
  accept and receive timeouts). The lead binds
  `node_addrs[lead]:<job's control port>` — the private address, not every
  interface — and accepts one peer. The other node retries connecting until
  `peer_timeout_s`.
- A message is a 4-byte big-endian length followed by a UTF-8 JSON object with a
  `type` field. Nothing is unpickled from the network.
- Torch-free, so the Mac tests and the stub benchmark drive the real module.
- Every failure surfaces as `PeerLost` or `Mismatch`. `PeerLost` means the
  channel is unusable: a frame may be half sent or half read, so close the
  socket and never reuse it. `Mismatch` means the nodes disagree. Send timeouts,
  broken pipes, unreadable frames and missing fields are included, so a caller
  catching those two catches everything.
- `result` stays one-way (non-lead → lead). If both sides send before either
  reads, they deadlock once a payload outgrows the socket buffers.

| type | Direction | Payload |
|---|---|---|
| `hello` | both | node, code hash, torch and transformers versions, job fingerprint, run id |
| `resume` | both, right after hello | "I already finished this job ok" (from `MIG_JOB_ALREADY_OK`); the job is skipped only if both say so, else both rerun it |
| `ready` | both, per config | config index, label (`s<split>_b<batch>_mb<mb>`) |
| `result` | non-lead → lead, per config | queue items, exit codes, hung flag, peak / avg MiB per local rank |
| `bye` | both, at the end, after writing outputs | ok: this node finished cleanly. The job is ok only if both are, so a failure on either node fails it on both |

- **Code hash**: sha256 over the files a run imports:
  `benchmark_pipeline_microbatching.py`, `mig_transport_pipeline.py`,
  `async_handle_send.py`, `async_handle_recv.py`, `async_handle_net.py`,
  `token_exchange.py`, `helpers.py`, `model_family.py`, `experiment_config.py`,
  `two_node_control.py`, `nvml_mem_monitor.py`. Not the whole repo: a node's own
  `parallel_config.py` must not abort the run.
- **Job fingerprint**: sha256 of the canonical JSON of everything both nodes must
  agree on — model key, type, layers, hidden size, heads, seq_len,
  max_new_tokens, max_runs, batch pairs, splits, ordering flag, min last-rank
  layers, rank_nodes, slice_gb, MIG UUIDs, node addresses, ports. Node-specific
  fields (node, ifname, shm prefix, run id, model path) are left out.

A `hello` mismatch logs both sides' values and exits non-zero on both nodes
before any rank starts. A match is logged by each parent as
`[B09] peer <node> matched: code <hash8> torch <ver> transformers <ver> job <hash8> run <run id>`,
which the smoke verdict's `peer` check reads.

Timeouts: connecting and accepting wait up to `peer_timeout_s`; `hello` must
follow within 60 s of connecting; every `ready` and `result` wait is bounded by
`JOIN_TIMEOUT + 300 s`.

### Per job (one model)

1. Connect, exchange `hello`, compare.
2. For each configuration `i` (the run plan's order is deterministic):
   1. Both send `ready i <label>` and wait for the other's. A mismatch aborts.
   2. Spawn only the local ranks (`rank_nodes[r] == this node`). The NVML
      monitor samples local UUIDs only.
   3. Join (`JOIN_TIMEOUT` 1200 s, as today) and drain the local queue.
   4. The non-lead sends `result`. The lead waits for it and merges.
   5. The lead builds the row with today's status precedence:
      `hang` > `OOM_rank<r>` > `crash` > `ok` > `timeout`.
3. At the end both send `bye`. The lead writes the CSVs; the non-lead writes its
   memory trace. Exit 0.

**Merge rules.** Queue keys already carry the global rank (`phase_rank<r>`,
`step_latencies_rank<r>`), so the merge is a dict union. On a key collision the
lead's value wins; only `oom` can collide, when ranks on both nodes OOM in the
same configuration. Exit codes and memory are per global rank. `hung` is true if
either node hung. `latency` comes from rank 0, which is always on the lead.

Ranks also put `weights_rank<r>` (loaded / expected / missing / format / files)
on the queue right after loading. The lead logs the remote ones in the B08
format with a `(node B)` suffix, so the existing verdict regex sees every rank.

Ranks on the non-lead node use only the shape of `input_ids`. Rank 0 embeds the
values, so the lead's input sample is the one that counts.

### Failures

| What happens | How it shows | Outcome |
|---|---|---|
| OOM on a rank of either node | the other ranks die on "Connection closed by peer" (as on one box today); `oom` arrives in the merged queue | row `OOM_rank<r>` |
| A rank hangs | both parents reach `JOIN_TIMEOUT` and terminate their local ranks | row `hang` |
| The peer parent is killed, or `stop` runs on the other node | control socket EOF without `bye`, noticed immediately | row `peer_lost`, the sweep stops, rows collected so far are written, exit non-zero |
| Network partition | a control wait exceeds `JOIN_TIMEOUT + 300 s` | same as the row above |
| The peer never starts | connect/accept exceeds `peer_timeout_s` | the job fails before any rank starts |

Each job has its own ports, so after a failure each node's next job can only meet
the other node's same job.

## 4. Sweep, job.json, planner, runner

### Split rule (`experiment_config.generate_layer_splits`)

New optional argument `rank_nodes=None` (`None` = one node). The recursion and
the output order stay the same. The check against the previous rank becomes a
check of candidate rank `j` against every earlier rank `i`:

- `gb[i] > gb[j]` → `l[i] > l[j]`; `gb[i] < gb[j]` → `l[i] < l[j]`.
- `gb[i] == gb[j]`, adjacent (`j == i + 1`), same node → `l[i] >= l[j]` (today's
  symmetry break).
- `gb[i] == gb[j]` otherwise → no constraint.

On a descending single-GPU layout (every layout that exists today) the
neighbour rules chain into these all-pairs rules, so the output is exactly
today's splits in today's order. Tests check this against the frozen copy for
every layout in `layer_limits.py`, and against a brute-force enumeration for
cross-node placements. `front_loaded_split` (used by smoke) is unchanged.

Every caller must pass `rank_nodes`. Without it, a two-node job silently gets the
neighbour rule. For 40 layers with limits `[22,11,22,11]` on A40,A20,B40,B20,
that is 87 splits against the cross-node rule's 989, with none in common. Both
nodes would compute the same wrong list, so the fingerprint cannot catch it.
`parallel_plan._finish_job` already passes `job.get("rank_nodes")` to the
enumeration and to `front_loaded_split`. That landed in stage 1 and does nothing
for single-node jobs, which carry no `rank_nodes`. The benchmark takes its
splits from job.json.

### Layer limits

Every `MODELS` entry carries `layer_limits=[...]`, one cap per rank. It is
required; there is no lookup in `layer_limits.py`, which stays untouched.
`min_last_rank_layers` comes from `layer_limits.HEAD_ONLY_LAST_RANK` as today
and can be overridden per model.

### job.json

An optional block, validated all-or-nothing by `experiment_config.validate_job`
(the schema stays 1):

```json
"rank_nodes":     ["A", "A", "B", "B"],
"node":           "A",
"node_addrs":     {"A": "10.0.0.1", "B": "10.0.0.2"},
"gloo_ifname":    "eno1",
"control_port":   29900,
"peer_timeout_s": 1800,
"placement":      "A40,A20,B40,B20"
```

Checks:

- `rank_nodes` has one entry per rank;
- `rank_nodes` spans exactly 2 nodes, because the control channel pairs the
  lead with one peer;
- each node name matches `[A-Za-z][A-Za-z_]*`. Names reach /dev/shm prefixes and
  placement labels, where a digit would make `A4`+40 read as `A`+440;
- `node` and every `rank_nodes` entry are keys of `node_addrs`;
- at least one rank is on `node`;
- `control_port` is in 1024–65535 and differs from `master_port`;
- `placement` matches `rank_nodes` and `slice_gb`.

### Benchmark (`benchmark_pipeline_microbatching.py`)

- New constants with single-node defaults: `RANK_NODES = None`,
  `THIS_NODE = None`, `MASTER_ADDR = "127.0.0.1"`, `GLOO_IFNAME = None`,
  `CONTROL_PORT = None`, `PEER_TIMEOUT_S = 1800`, `PLACEMENT = None`. The
  override block sets them from the two-node block when it is present.
- `run_pipeline`: takes `MASTER_ADDR` from the constant, sets
  `GLOO_SOCKET_IFNAME` when configured, calls
  `register_hooks(..., rank_nodes=RANK_NODES)`, and puts `weights_rank<r>` on the
  queue after loading.
- `main()`: spawns local ranks only, runs the monitor on local UUIDs with global
  labels, calls the control channel as in §3, merges rows, adds the `placement`
  column, writes no results CSV on the non-lead node, and handles `peer_lost`.

### NVML monitor (`nvml_mem_monitor.py`)

`setup_dcgm_group(mig_uuids, slice_gb, ranks=None)`: `ranks` labels the trace
columns with global rank numbers (`rank2_40gb_mb` on node B). `None` keeps
`0..n-1`, as today. If `ranks` does not line up with `mig_uuids` (and with
`slice_gb`, when given), the monitor prints one DISABLED line and turns itself
off instead of crashing `save_csv`.

### Planner (`parallel_plan.py`)

A config that defines `PLACEMENT` is the two-node form; defining `GPUS` as well
is an error. `check` and `plan` take `--node X`. `plan` requires it; without it,
`check` runs only the config-level checks. The checks:

- every `PLACEMENT` node is in `NODES`, and each `NODES` entry has `addr`,
  `ifname` and `gpu`;
- MIG UUIDs are unique;
- this node's UUIDs appear in `nvidia-smi -L` on `NODES[node]["gpu"]` with
  matching GB (the other node checks its own);
- `layer_limits` has one entry per rank and admits at least one split;
- `mem_monitor` is `nvml` or `off`;
- the ports of all jobs are distinct.

Output: one lane, `pipe_<node>`, with one job per `MODELS` entry in order, in
today's run-directory layout, so the runner, report and resume work unchanged.

Smoke verdict for two-node jobs:

| Check | Lead | Other node |
|---|---|---|
| exit | state ok, exit code 0 | same |
| results | all rows `ok`, `placement` column present | SKIP (the lead writes the CSV) |
| weights | B08 `missing=0` for every rank (remote ones logged by the lead) | its local ranks |
| memory | local ranks from the trace, remote ranks from the CSV peak columns; each `0 < peak <= slice size` | its local ranks from the trace |
| peer | a `[B09] peer ... matched` line in its `benchmark.log` | same |
| shm_leftover | none on this node | same |

### Runner (`run_parallel.sh`)

- `--node X` on any subcommand. It is forwarded to the planner (`check`,
  `smoke`, `start`) and selects a per-node lock,
  `/tmp/mig_pp_run_parallel.lock.node_X`, for `start`, `smoke`, `status` and
  `stop`. Per node, not one two-node lock, so the Mac tests can run both
  "nodes" at once. The lock is exported to the detached manager. The latest
  link is `runs/latest_X`.
- Nothing else changes. Lanes, sessions, detach, stop escalation, resume and
  `/dev/shm` cleanup are reused as they are.

## 5. Gates and tests

The CLAUDE.md permission gate: `benchmark_pipeline_microbatching.py` is not
modified until a probe passes on the box. This is the order the ACK fix
followed: the transport changes and their gate came first.

### Stage 1 — transport, control channel, probe

Files: `mig_transport_pipeline.py` (routing, a no-op with `rank_nodes=None`),
`async_handle_net.py`, `two_node_control.py`, `experiment_config.py` (split
rule, job.json block), `nvml_mem_monitor.py` (`ranks=`), `probe_two_node.py`,
tests.

On the Mac:

- **torch-free**: the split rule (frozen copy on every existing layout, brute
  force on cross-node placements); job.json two-node validation; the control
  channel (two parent processes on localhost: `hello` match and mismatch,
  lockstep, result shipping, EOF → `peer_lost`, timeouts); NVML `ranks=` labels.
- **CPU torch (`.venv`)**: the net handles with faked CUDA events over real
  two-process gloo on localhost, `rank_nodes = ["A", "B"]`. Checks: bit-exact
  payloads across many tags, the send slot freed only after send completion,
  receive staging held until its H2D, idempotent `flush`/`wait`, a failed wait
  keeps its slot. `tests/test_async_ack.py` stays green.

On the box: `MIG_NODE=A python3 probe_two_node.py` on A and
`MIG_NODE=B python3 probe_two_node.py` on B. The probe reads `NODES` and
`PLACEMENT` from `two_node_config.py`, uses its own fixed ports (29990 master,
29991 control, in a `CONFIG` dict at the top of `main()`), and prints PASS or
FAIL per case plus a VERDICT, one paste per node.

The cases run in this order: fix candidates first, the riskiest last, so an
early failure cannot mask later cases the way it did in the first gloo probe.

| Case | Checks |
|---|---|
| nic | Each parent checks that `NODES[node]["addr"]` is on `ifname` (`ip -4 -o addr show dev <ifname>`). gloo binds the address of `ifname` itself, so with the configured address on another NIC every case would pass over the wrong network. SKIP where there is no `ip`. |
| N1 | the control channel connects; code hash and torch / transformers versions match |
| N2 | gloo init across nodes with `MASTER_ADDR` and `GLOO_SOCKET_IFNAME`; all_gather of every rank's hostname; barrier |
| cuda_device | `torch.cuda.set_device` on the rank's MIG slice, so a bad UUID shows here and not as a gloo failure |
| N3 | raw gloo send / recv of pinned CPU tensors across nodes, decode- and prefill-sized, bit-exact |
| N4 | `TokenExchange` across nodes: from the last rank to rank 0 when they are on different nodes, otherwise from the first rank on the other node to rank 0 |
| N5 | the shipped engine on one cross-node hop: steps × microbatches, tags reused, a small slot pool (forces reuse), deterministic payloads compared bit-exact, zero slot spins |
| N6 | the shipped engine across the whole placement (SHM and network hops in one chain), bit-exact end to end |
| N7 | cross-node hop timing from N5 and N6 (information only; FAILs only if reading the counters raises) |
| exit | every rank process finished and exited 0 |
| exchange | the parents trade their rows so each prints the whole table; the lead sends first, so neither blocks the other |

Ranks report each case the moment it ends: one queue item per case, then a
finished marker. A rank that hangs or dies keeps the rows it already sent, and a
rank that ignores SIGTERM is killed. A failing case is reported before its
traceback is released, because freeing a pending gloo send can spin. That was
seen on macOS, with gloo's uv transport.

`TokenExchange` was N6 in the conversation. It moves to N4 because it is simpler
than the engine cases and should not be masked by them.

Payloads are generated on the CPU from a seed (step, microbatch, rank), moved to
the GPU, sent, and compared on arrival with `torch.equal`. Matching torch
versions on both nodes (N1) make the generated payloads identical.

### Stage 2 — only after probe VERDICT PASS on both nodes

Files: `benchmark_pipeline_microbatching.py`, `parallel_plan.py`,
`run_parallel.sh`, `two_node_config.py`, and `tests/fixtures/stub_benchmark.py`
(which speaks the control protocol when its job.json has `rank_nodes`).

On the Mac:

- `test_benchmark_wiring.py`: the config block with and without the two-node
  block;
- planner tests for the two-node form;
- an end-to-end `test_run_parallel.py` case with both "nodes" on the Mac (two
  run directories, `--node A` and `--node B`, the stub benchmark), where both
  verdicts PASS.

The existing suites stay green on Python 3.10 and 3.13 and on bash 3.2 and 5.

### Stage 3 — on the box

1. Single-node regression: `./run_parallel.sh smoke` with the existing
   `parallel_config.py` on one GPU → VERDICT PASS. This shows nothing changed by
   default.
2. Two-node smoke: `./run_parallel.sh smoke -c two_node_config.py --node A` on
   A, `--node B` on B → VERDICT PASS on both.
3. Full sweep: `./run_parallel.sh start -c two_node_config.py --node A --detach`
   on A, the same with `--node B` on B.

## Files

| File | Stage | Change |
|---|---|---|
| `mig_transport_pipeline.py` | 1 | `rank_nodes`; same-node peer attach; routing; receive-slot reservation; `T33`/`T34` summary |
| `async_handle_net.py` | 1 | new: `AsyncHandleSendNet`, `AsyncHandleRecvNet` |
| `two_node_control.py` | 1 | new: connect, hello, ready, result, bye; code hash; job fingerprint |
| `experiment_config.py` | 1 | all-pairs split rule with `rank_nodes`; job.json two-node block |
| `nvml_mem_monitor.py` | 1 | `ranks=` column labels |
| `probe_two_node.py` | 1 | new: the gate (N1–N7) |
| `tests/test_net_transport.py` | 1 | new |
| `tests/test_two_node_control.py` | 1 | new |
| `tests/test_experiment_config.py`, `tests/test_nvml_mem_monitor.py` | 1 | new cases |
| `benchmark_pipeline_microbatching.py` | 2 | constants and override; local spawn; control calls; merge; `placement` column; weights on the queue |
| `parallel_plan.py` | 1, 2 | stage 1: `_finish_job` passes `rank_nodes` to the split generator; stage 2: two-node config form, `--node`, its checks, node-aware verdict |
| `run_parallel.sh` | 2 | `--node` forwarding, two-node lock |
| `two_node_config.py` | 2 | new: the file you edit |
| `tests/fixtures/stub_benchmark.py` | 2 | control protocol when `rank_nodes` is set |
| `tests/test_benchmark_wiring.py`, `tests/test_parallel_plan.py`, `tests/test_run_parallel.py` | 2 | new cases |
| `CLAUDE.md` | 3 | state, layout table, new log tags — after the two-node smoke |

Unchanged: `async_handle_send.py`, `async_handle_recv.py`, `token_exchange.py`,
`helpers.py`, `model_family.py`, `layer_limits.py`, `memory_limits.py`,
`parallel_config.py`, `dcgm_mem_monitor.py`.

## Risks and open points

- **gloo send completion.** The design relies on a completed gloo send meaning
  gloo no longer reads the buffer. N5 tests exactly this under forced slot reuse.
  If it fails, the fallback is an explicit receiver ACK on the cross-node hop
  (rule 3 of the SHM protocol).
- **Resume across nodes.** Run `--resume` on both nodes. A job recorded `ok` on
  one node but not the other gets re-run alone, waits `peer_timeout_s`, and fails
  there. Rerun that model as a fresh run.
- **Shared NIC.** Cross-node traffic shares the interface with anything else on
  it (other jobs, storage). The harness does not control this; worth knowing
  when reading results.
- **Clock skew** between the nodes does not matter. No cross-node timestamps are
  compared: each rank times itself with CUDA events, and the reported latency is
  rank 0's.
- **MIG UUIDs do not survive a reboot.** Re-read them on the rebooted node and
  update `PLACEMENT`; the planner's inventory check catches stale ones.
- **`_write_tensor_to_slot`** in the SHM path's non-fp16 fallback is undefined.
  Pre-existing and unreachable for fp16 activations; not changed.
