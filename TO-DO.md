# TO-DO (P3 Run B and after)

Deferred items, agreed with the user on 2026-11-26. Current state stays in `current.md` (NOW).

## Runbook: mig5 from Sat 11/28 (presets in `clock_study.sh`, one command each, export by hand)

Export names the Mac scripts expect: `~/Desktop/AP/logs/mig5_p3_runB_step1`, `_step2`, `_step3`,
`_step3b`, `_post`, `mig5_p3_vicuna_topup` (step names `vicuna7b_vtop_`). Each export packs the last `# run` block + the 3 clock logs.

| # | On the Mac first | On mig5 (after `git pull`, check `git log --oneline -1`) |
|---|---|---|
| 1 | export step 1; `runb_finish_check.py` ALL OK; `runb_lists.py` step=2; commit `mistral7b_step2_split_order.py` | `INV_HOST=mig5 ./clock_study.sh runB_step2 --dry-run`, then without `--dry-run` |
| 2 | export step 2; finish check (module `mistral7b_step2_split_order`); `runb_lists.py` step=3; commit `mistral7b_step3_split_order.py` | `... ./clock_study.sh runB_step3` |
| 3 | export step 3; finish check; `runb_noise_check.py`; if it wrote `mistral7b_step3b_split_order.py`, commit it | `... ./clock_study.sh runB_step3_rerun` (only if written) |
| 4 | - | `... ./clock_study.sh vicuna_topup` (~45 min) |
| 5 | - | `... ./clock_study.sh runB_post` (~24 h; 88 splits x 9 pairs) |

Steps 4-5 need no new files. Then score Mistral (steps 1-3 only), then Qwen.
`runb_finish_check.py` CONFIG per step (module / prefix): step 1 `mistral7b_40gb_split_order` /
`mistral7b_clock_`; step 2 `mistral7b_step2_split_order` / `mistral7b_s2_`; step 3
`mistral7b_step3_split_order` / `mistral7b_s3_`; rerun `mistral7b_step3b_split_order` / `mistral7b_s3b_`;
post `mistral7b_post_split_order` / `mistral7b_post_`; Vicuna `vicuna7b_topup_split_order` /
`vicuna7b_vtop_` with `passes=2` (all others 1). `runb_noise_check.py` takes each earlier export with
its prefix (step 1 `mistral7b_clock_`, step 2 `mistral7b_s2_`). Both refuse a wrong or incomplete
export (wrong phase, a folder given twice, repeated oracle, missing pair or split) instead of
reporting OK. `runb_lists.py` checks each export's phase and one oracle per pair, and for step 3 a
measurement of every frozen pick and every baseline that fits (else "NO LIST WRITTEN"); it does not
count splits, so run the finish check to ALL OK first.

## Before Run B step 1 ends (~Sat 11/28 7:41 am EST: started Thu 11/26 6:35 pm EST + 37.1 h)

- **Finish check for Run B** (review point 3, 2026-11-26): per step `mistral7b_clock_blk01_*`, count the
  distinct splits with status `ok` in the results CSV (expect 136 per pair, 1,224 in all) and list
  every other row by status (OOM, crash, timeout; `benchmark_pipeline_microbatching.py` ~line 1081).
  `run_parallel.sh` counts processes, not configurations, so its "ok" is not enough. Prints counts
  only, never latencies (Mistral is sealed until the picks are frozen).
- **Noise script** (point D): per-pair aggregates only (median / p90 / worst repeat difference,
  # splits > 1% / > 2%); 3rd run for a pair if median diff > 0.5% or any split > 2%. No split names,
  no latencies.
- **Freeze:** picks written to `SCORING.md` "Pre-registered: Mistral-7B" with the seal note
  (2026-11-26); acceptance threshold approved 2026-11-27 (below). Left: the user's commit.

## Decisions still open for Run B

- **Step 3 (confirmation) set, fixed 2026-11-27:** a 2nd run of every split within 3% of the pair's
  best after steps 1-2 (single-run values), plus the frozen graph ILP pick, the GPipe ILP pick and the
  three rule baselines for that pair, whatever their distance. Repeat difference of a split =
  100 x |t2 - t1| / ((t1 + t2) / 2) (order-free). If a pair's repeat differences have median > 0.5%
  or any split > 2%, that pair's **same confirmation set** is run a 3rd time (not its 136+ list).
  Score = median over each split's runs.
- **The post-deadline pass (rank 0 = 25, ranks 2/3 swapped) does not change the primary score:**
  the primary result uses steps 1-3 only. The pass is reported separately as a coverage check; if it
  finds a split faster than steps 1-3's best, regret against it is reported as a sensitivity result.
- Acceptance threshold: proposed by Claude, approved by the user 2026-11-27 (>= 7/9 in the 2% group, median regret <= 1%, worst <= 3%;
  `SCORING.md`).
- **Step 2 rules (set 2026-11-27, before any Run B data is opened):** one round, one command. Per pair,
  the candidates not yet measured are: (a) every one-layer move and the rank 2 <-> rank 3 swap of each
  split within 3% of the pair's best in step 1; (b) every one-layer move of the frozen graph ILP pick
  (k = 1); (c) the baselines below. All must fit in memory. Each unique candidate counts once.
  Cap 25 splits per pair for (a) + (b), sorted by (fastest step-1 latency among the candidate's
  parents, then the candidate split as a tuple) and taken from the top: deterministic, so the run
  list is reproducible. (c) is always added. A script
  builds the list from step 1 and prints only split lists, never latencies. Budget: <= 25 + 2 splits
  x 9 pairs x ~110 s = ~7 h at most; expected ~3-5 h.
- **Step 2 also measures the baselines missing from step 1** (frozen in `SCORING.md`, 2026-11-26):
  [8, 8, 8, 8] at all 9 pairs; GPipe ILP picks [16, 9, 3, 4] at (8,2) (16,2) (32,4) (32,2) and
  [18, 10, 1, 3] at (16,4). All fit in memory.
- Run order: not needed (the run will not be stopped early); same order at every pair.

## After Sat 11/28 11:45 pm EST (mig5, while S1 runs)

- **Qwen2.5-7B runs on mig5 after Mistral is complete** (user, 2026-11-27): a held-out
  main-condition run, not a fully blind test (its development check under default conditions is
  already scored in `SCORING.md`); say so in the paper. Needs its own split list (it may put 0 layers on the last rank), probe-gated picks
  (the bias GEMM question: v7.1's 3-D GEMMs are in the mig5 probe set), frozen before its data.

- Rank 0 = 25 (12 splits per pair) and ranks 2 / 3 swapped (78 per pair): ~25 h.

## Before S1 (Sun 11/29 2:45 am EST)

- **`layer_ops.py` KV-append fallback** (review 2026-11-27): a KV-head count the probe set has no fit
  for (e.g. 16) ends in a bare `KeyError: 'kv_append_fixed_us'`; make it a clear error naming the
  missing head count, with a test. The edit was blocked while the freeze was staged; frozen picks do
  not depend on it (Mistral's 8 KV heads are covered). Matters for the S models on SXM4.

## After Run B completes (mig5, user's call 2026-11-26)

- **Vicuna-7B baseline top-up** (user, 2026-11-27; after the Mistral runs, before Qwen): only the
  baselines / ILP picks never measured under the main condition. mig5 = mig3's card, so run 17's data
  stands: no anchors, no repeats. One command, preset `vicuna_topup` (`vicuna7b_topup_split_order.py`),
  2 passes everywhere (simpler than matching (32,8) and (16,2)'s single runs: +4 runs, ~8 min).
  11 splits, 22 runs as planned + 4, ~45 min:

  | Pair | Runs | Measure |
  |---|---|---|
  | (8,4) | 2 | [8,8,8,8], [14,9,5,4], GPipe [20,10,1,1] |
  | (16,8) | 2 | [14,9,5,4] |
  | (32,16) | 2 | [14,9,5,4] |
  | (8,2) | 2 | [8,8,8,8], [14,9,5,4] |
  | (16,4) | 2 | [14,9,5,4], GPipe [16,9,3,4] |
  | (32,8) | 1 | [14,9,5,4], GPipe [18,10,1,3] |
  | (16,2) | 1 | [8,8,8,8], [14,9,5,4] |
  | (32,4), (32,2) | - | nothing missing |

  All graph ILP picks (frozen setup: mig5 set, live gap, `clock="fast"`) are already measured.
  **Even [8,8,8,8] does not fit for Vicuna-7B at 6 pairs** (memory model): batch 32 -> ranks 2 and 3
  out of memory (6 layers fit on each 5GB slice); (16,8), (16,4) -> rank 3 only (7 fit: it also holds
  the output head and its logits). Report as "does not fit"; not run. Edge cases by one layer: rank 2
  at (16,8) / (16,4) and rank 3 at (16,2) fit exactly 8.

- **Mistral-7B validation layer** (step-3 gate): add `~/models/Mistral-7B-Instruct-v0.3` (local
  path; the HF repo is gated) to `validate_models` (`mig_pp/milp/measure_common.py` ~line 1170),
  then a short validation-only MIG probe on mig5 under the main condition. One layer, random
  weights: a check, not benchmark data. It comes after the freeze, so it can only report whether
  the frozen ILP composes Mistral's layer within 5%; it cannot change the picks.

## Box housekeeping

- 2026-11-26 23:19 EST: `--probes --dry-run` on mig5 at commit 2f814f3 (before `--probes` existed)
  started Run A's config for real. Its `# run` line in `investigation_runs.txt` and
  `runs/20261126_231944` are an aborted duplicate: not data. Before any run on a box, check
  `git log --oneline -1` there shows the commit the command needs.
