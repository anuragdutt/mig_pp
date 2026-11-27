# Current tracks

What is in flight: what we are doing, where it stands, what comes next, and where the details
live. Replace a track's section when it changes; delete it when done. Results and details go in
the other notes (`learnings.md`, `problems.md`, `setup.md`, `mig_pp/milp/PROBES.md`,
`mig_pp/milp/SCORING.md`), not here.

Agreed sequence (2026-11-14, after outside review): 1 repeat runs -> 2 reconcile docs -> 3 probes
-> 4 freeze model + acceptance criteria -> 5 one test on an untouched model -> 6 change the model
only where a reproducible miss changes the pick.

Analysis scripts (Mac, not in the repo) live in the scratchpad:
/private/tmp/claude-501/-Users-limarkdcunha-Desktop-AP-mig-pp/c0ab995b-d33b-4792-9f5b-944608cd1f31/scratchpad

## NOW (2026-11-27, read this first in a new chat)

**Running:** P3 Run B step 1 on **mig5**, commit 2002413, started Thu 11/26 6:35 pm EST (box clock
23:35 UTC), ~37.1 h, ends ~Sat 11/28 7:40 am EST. Mistral-7B, main condition, 9 pairs x 136 splits
(`mistral7b_40gb_split_order.ALL`), 1 run each, sealed. The aborted Vicuna duplicate of 23:19 UTC is
not data (`TO-DO.md`).

**Done 2026-11-26/27:** Run A scored as pre-registered: PASS at both pairs (+0.57% / +0.52%, tau
0.87 / 0.73; `SCORING.md`). Probe v7.1 on mig5 under the main condition -> `probe_sets/mig5.json`,
step-3 gate Vicuna 36/36, Qwen 35/36 (`PROBES.md`). Boundary gap moved into
`mig_pp/milp/graph_placement.py` (identical to the sandbox; `tests/test_frozen_picks.py`).
**Mistral-7B frozen** (`SCORING.md`, "Pre-registered: Mistral-7B"): graph ILP picks (primary), GPipe
ILP + even / GPC- / memory-proportional baselines, acceptance >= 7/9 in the 2% group, median regret
<= 1%, worst <= 3%, seal note. **Uncommitted: the user commits after review.**

**Next on mig5:** step 1 ends ~Sat 11/28 7:40 am EST -> finish check (136 distinct `ok` per pair) ->
step 2 (rules in `TO-DO.md`: one-layer moves + rank 2/3 swap around the 3% group, graph pick +-1,
cap 25 per pair, plus the baselines missing from step 1) -> step 3 confirmation (3rd run for a pair if
median diff > 0.5% or any split > 2%). After Sat 11/28 11:45 pm: rank 0 = 25 + rank 2/3 swaps (~25 h).
**Then Qwen2.5-7B on mig5** (held-out main-condition run; not fully blind, its default-condition development check is scored; user 2026-11-27). Scripts for the finish check, noise
check and step-2 list: not written yet (`TO-DO.md`). Data counted under the rule: `runs.md`.

**Run condition (hard rule, user, 2026-11-26):** every run is `performance` + GPU locked +
`OMP_NUM_THREADS=1`; `schedutil`/default-thread runs only with the user's explicit OK after a written
justification (`mig_pp/milp/SCORING.md`, "Which condition counts").

**Working rules (user, 2026-11-26; memory has them too):** plan every run/command/code change with
the user and get an explicit OK first; one command per run, no chains/waiters unless asked; run
scripts never export -- export is a separate manual command; lease times are exact (buffer = copy +
one retry).

**Decisions made (2026-11-25/26):**
- Main condition for everything new (PCIe P3 and S runs): `performance` governor + GPU clock locked +
  1 thread per rank. Framed as a prospective protocol change; argument + sources: `justification.md`.
- P2 (mig4) outcome: drift, probe v7, (32,2) at max, Mistral (32,4) ok; the 7-pair Mistral pass never
  ran (waiter bug, ~12 GPU-h lost). Data `logs/mig4_p2_7_mistral_32_4/`. Mistral (`schedutil`) is
  sealed and now secondary.
- Run 17 analysed (blocks 1-5): `problems.md` "Run 17"; block-5 script
  `investigative_runs_results/scripts/run17_block5.py`.
- S runs (SXM4): big models only (Vicuna-13B, Llama-2-13B, Qwen2.5-14B, Mistral-Small-24B); B128
  skipped; all 14 pairs, 2-4 microbatch pairs first; S1 (Sun 11/29, 56 h) + S2 (Mon 12/14, 24 h) as
  one budget; experiment deadline 12/17. Search: model-independent coarse grid over the
  memory-feasible space + local refinement (multi-start within 3%, 12 one-layer moves) + confirmation
  repeats + the frozen ILP pick (review of 11/26). Old SXM4 oracle_v2 data: timing only.

**Open decisions (next chat):**
1. S grid: option A (3-layer grid, Qwen 4-layer, ~137 GPU-h phase 1) vs B (3-layer everywhere, ~162),
   plus the scaled PCIe-pattern split added per pair (user leaning to A + seeds; not confirmed).
   Draft builder: `wip/s1_build_lists.py` (drops the rank2>=rank3 ordering rule; per-pair memory caps).
2. Per-lane S1/S2 schedule with a hard refinement cutoff (not written yet).
3. P3 after Run B: Qwen2.5-7B (decided), 10/10/10/5 support; Llama-2 deprioritised; P3 checks for
   S (memory-edge configs, S1 script rehearsal, big-model download check).
4. `schedutil` block in S1 yes/no; exact S1 lease end.

**Work not in the repo's code yet (in `wip/`, see `wip/README.md`):** ILP kernel-boundary gap
(sandbox: 8/9 pairs in the 2% group at full clock, (32,16) the miss = thread pool), probe v7.1,
scoring scripts.

## 1. Ground truth from the runs (done)
- Every pair's near-best group and noise under default conditions (runs 2-14): `learnings.md`
  winners table, 16, 17. Used later to score the frozen ILP.

## 2. Graph ILP for mig1/mig2: connect it to the probe data
- Rule: constants from probes or theory, nothing fitted to benchmark results; only
  clock="schedutil" counts (`SCORING.md`).
- Part A done, not committed: GPU time composed from the probe's measured points (`layer_ops.py`);
  `probe_constants.py` writes each box's set to `probe_sets/<box>.json` (CONSTANTS["mig1"] / ["mig2"]).
- Part B done (not committed): CPU busy time from the step graph (harness code), governor threshold
  by step period and slow factor from the probe, thread-pool term, exact search for
  clock="schedutil"; probe sets as `probe_sets/*.json`. 33 + 4 tests pass, old sets bit-identical.
- Development check (`SCORING.md`): right at 8-16 microbatches; 2-4 microbatch pairs load rank 0
  too much (regret 3.2-4.7%). Diagnosed against the events runs: two probe-method gaps (launch
  timed cold, 15% high; GPU timed one slice at a time, 2-9% low). Probe v6 refuted both mechanisms
  (`PROBES.md`); the GPU gap is the idle time between kernels (spans vs summed durations), launch
  still ~10% high for an unknown reason. Probe sets now from v6. Next (proposed): probe v7.

## 3. Probes and step 3 (composed layer within 5% of the measured one at every point)
- Probe v3: step 3 fails (59 of 144); why, and the CPU findings: `PROBES.md`.
- Probe v5 (2026-11-16): step 3 at 142 of 144, median error -0.4% to -1.5% on every slice
  (`PROBES.md`). The 2 misses (-5.3 / -5.5%) are Qwen's bias GEMMs, which the probe does not
  measure; Vicuna and Mistral have no bias. Decided: step 3 passes for bias-free models; a
  GEMM-with-bias probe only when a bias model is scored. Probe v6 proposed (track 2).

## GPU chain (given 2026-11-15 midday; finished 2026-11-16 morning, both boxes)
- Probe v4 done on both boxes (not exported yet). Mistral-7B near sweep, all 9 pairs, 404 splits
  (run 16) started 16:41 UTC on both boxes, ~15 h. Run 15 (Vicuna near repeats at (32,16), (16,8),
  (16,4)) was skipped (an aborted attempt's `# run` line fooled the queue); queued after run 16.
  Commands: `setup.md`. Export: run 16 = second-to-last block, run 15 = last.
- **Run 16 is sealed: export it, but do not analyse it until the ILP is frozen** (untouched model).
- If the chain ends with GPUs free: a second Mistral pass (>= 3 repeats per split for scoring).

## Max-clock reruns (advisor request, 2026-11-16)
- Rerun the variance pairs at max clock, two conditions (user's call): default vs max = CPU
  `performance` + GPU locked at max (log only if MIG refuses the lock). Temperature not
  controlled, only logged. Vicuna only. Both boxes run the same schedule (user's call): problem
  pairs (8,4), (16,8), (16,4), (8,2), NEAR_BEST + BY_PAIR splits, blocks default/max/max/default
  (2 runs per condition per box); then the 5 stable pairs at max, NEAR_BEST, once per box. ~27 h
  per box. Purpose: max-clock data at all 9 pairs to check the ILP against (validation only).
- Run 17 = `clock_study.sh` (commit bd39771), started 2026-11-16 22:07 UTC on both boxes. LOST: the
  machines were taken down before it finished; no data. GPU lock works under MIG (seen at start).
- Rerun 2026-11-24 on mig3 (one box, 2 runs per condition): blocks 1-4 done, all ok, exported to
  `logs/mig3_run17a/`, results in `problems.md` "Run 17". Confirms run 6: noise comes from the CPU
  governor (default up to 24%, max <= 1.3%); `schedutil` costs 2-8% on the best split.
- Block 5 (stable pairs at max, once): (32,16), (32,8), (16,2), (32,4) all ok, stopped 04:55 UTC
  11/25 during (32,2); full export `logs/mig3_run17c/` (supersedes 17a). (32,2) at max -> P2.
  Block 5 not analysed yet.

## 4-5. Freeze and test (after 2-3)
- Freeze formulas, constants and acceptance criteria in `SCORING.md` first. Proposed: median regret
  <= 1%, worst <= 3% vs the median of >= 3 default repeats; pick inside the pair's near-best group
  at >= 80% of pairs.
- Then one test on an untouched model (Llama-2-7B or Mistral-7B; Qwen is no longer untouched).

## Parked
- From `docs/archive/open-questions.md` (2026-11-12/14), still open: (2) did nsys cause the serial
  staircase in the (32,16) timeline (rerun with kernel tracing off, NVTX only); (5) the KV-cache copy
  (`CatArrayBatchedCopy`, ~20% of rank 0's GPU time under nsys): its untraced share and growth with
  microbatch size (run 4's NCU reports, not analysed). The other 8 are answered there.

- Batch 64 (needs the 80GB box); cross-node; the 80GB paired diagnostic (`problems.md`, "Still open").
