# Scoring the ILP against measured sweeps

How we check `pulp_layer_placement.py` against oracle CSVs. **The CSVs are verification
data only. Nothing in the model is fitted to them.** Constants come from
`measure_constants.py`; the formulas come from the harness code and NVIDIA's MIG docs.

## Which condition counts (2026-11-26, hard rule)

**Every run from 2026-11-26 on uses the main condition: `performance` CPU governor + GPU SM clock
locked at max + `OMP_NUM_THREADS=1` per rank. Score and pick against main-condition runs (`clock="fast"`,
no pool term).** A `schedutil` or default-thread run needs the user's explicit OK after a written
justification (question it answers, GPU-h, why main-condition data cannot answer it). Why the
condition changed, and how the paper reports it: `justification.md`. Data from before the change
(oracle_v2, runs 2-14, run 16, P2's Mistral (32,4), run 17's default blocks) keep their original
conditions and are reported as separate, secondary results; they are not rerun.

### Superseded: which clock setting counted before (2026-11-14)

**Score and pick with `clock="schedutil"` only.** The runs we use (oracle_v2 and anything
compared with it) run with the default CPU thread pool: the harness never sets
`OMP_NUM_THREADS`. On mig1/mig2 they run under the `schedutil` governor, the boxes' default
(not recorded in the data, so almost certain rather than measured). The 80GB box's (cc@mig)
governor during oracle_v2 is unknown (problems.md, "Run 6"). These conditions are what the ILP
must predict. `clock="fast"` (every CPU core at full clock) and the clean
runs it matches (run 10: `performance` + `OMP_NUM_THREADS=1`) are a diagnostic only: they
check the GPU side of the model with the CPU effects removed. Do not report `clock="fast"`
results as the ILP's accuracy.

## Development check: probe sets, nothing fitted (2026-11-16)

`graph_placement` with `hardware_constants.CONSTANTS["mig1"]` / `["mig2"]` (probe_sets/*.json from
probe v5 + v3's CPU probes; written by `probe_constants.py`) and `clock="schedutil"`: GPU time
composed from the probe's measured points (step 3: 142/144 within 5%), CPU clock from the
governor probe and the CPU busy time read off the harness code, thread pool from the CPU probe,
split picked by exact search (~13 s for the 9 pairs). Measured: Vicuna-7B single-node default runs
2, 5, 9, 11-14, both boxes pooled. A development check, not the frozen test.

| Pair | Pick (mig1 set) | Measured best | Regret | In the 2% group |
|---|---|---|---|---|
| (16,2) | [17,9,4,2] (mig2 set: [16,9,4,3], 1.2%) | [17,9,4,2] | 0.0% | yes |
| (32,4) | [17,9,4,2] | [17,9,4,2] | 0.0% | yes |
| (32,2) | [16,9,4,3] | [16,9,4,3] | 0.0% | yes |
| (16,4) | [19,9,3,1] | [18,9,4,1] | 1.2% | yes |
| (32,8) | [19,9,3,1] | [19,10,2,1] | 3.2% | no |
| (8,2) | [17,10,4,1] | [18,9,4,1] | 4.6% | no |
| (8,4) | [29,1,1,1] | [26,4,1,1] | 3.9% | no |
| (16,8) | [29,1,1,1] | [22,8,1,1] | 4.7% | no |
| (32,16) | [23,7,1,1] | [20,8,3,1] | 4.7% | no |

Median regret 3.2%, worst 4.7%, 4 of 9 picks in the group: below the proposed acceptance criteria.
Right at 8-16 microbatches; at 2 microbatches it loads rank 0 too much (as on the 80GB box, "Open
points").

**Diagnosis against the events runs** (runs 3 and 11, 195 split runs, Vicuna, default conditions;
scratchpad `diag_clock.py`, `layer_gpu.py`, `step_parts.py`, `whatif.py`). Validation only:
- Clock levels: the model matches the measured level (launch ms per layer) for rank 0 in 191/195
  split runs, ranks 1-3 in 151-158/195. CPU busy shares agree on ranks 0-1; ranks 2-3 run ~8-10
  points busier than modelled (CPU work outside the model).
- Latency error grows with rank 0's layers at (8,4) / (16,8): -1% at [24,6,1,1], -14% at [29,1,1,1].
  In the step: rank 0's own part is 3-6 ms short in every split, and the tail after rank 0's last
  publish is short with 1-2 layers on rank 1 but up to 11 ms long with 4-8 slowed layers there.
- Two input gaps, both from how the probes measure:
  - Launch: the harness launches at 0.67 ms per layer (full clock) and 1.50 (slowed); the model
    uses 0.77 and 1.73 (40 x the probe's 19.3 us, x2.25). The slow factor matches; the base is 15%
    high because the CPU probe times one layer call after a synchronize, cold, while the harness
    launches a stack of layers back to back.
  - GPU: the harness's layer time is 2-9% above the composed one on every rank (rank 0 at (8,4):
    0.885 vs 0.838 ms per layer); the probes ran one slice at a time, the pipeline runs all four.
  - Kernel-level launch / GPU interleaving was checked and is small (~0.3-0.4 ms per microbatch).
- What-ifs with the harness's own values (diagnostic, not adopted): launch 16.8 us and GPU x1.05
  give median regret 1.2%, worst 3.2%, 5/9 in the group; (8,4) then picks its measured winner.
  Still off: (32,8) 3.2%, (32,16) 2.4%, (8,2) / (16,2) 2.2%.

**With probe v6 sets** (launch 18.5 us, concurrency factors ~1; runs 2, 5, 9, 11-15): Vicuna
median regret 2.9-3.1%, worst 4.8%, 4/9 in the group ((8,2) fixed, (16,8) 2.9%, (16,4) 3.1%);
Qwen2.5-7B (run 3's 15 splits per pair, 2 runs each; a second development check) median 1.5%,
worst 2.6%. Diagnostic what-ifs on Vicuna (not adopted): + a GPU gap of 1.2 us per kernel gives
median 1.1%, worst 4.2%, 5/9; + that gap and launch at the harness's 16.8 us gives median 0.8%,
worst 3.2%, 7/9, with (8,4) picking its measured winner. Next: probe v7 measures both (`PROBES.md`).

## What is measured

For every (batch, microbatch) pair in the CSV, the ILP predicts the latency of every measured split:

| Metric | Meaning | Good |
|---|---|---|
| **Median error** | (predicted − measured) / measured, over all rows. + = model too slow | near 0% |
| **p10 / p90** | spread of that error | narrow |
| **Error by N** | median error per number of microbatches N = batch / mb | flat across N |
| **Kendall tau** | does the model order the splits like the measurements? 1 = same order, 0 = random | near 1 |
| **Pick rank** | the split the model thinks is fastest (among measured ones): where it really placed | 1 |
| **Pick penalty** | how much slower that pick really ran than the true best measured split | near 0% |
| **Free pick** | `optimize_placement` over *every* split that fits in memory, flagged if it was measured | — |

Pick rank and pick penalty are what matter for choosing a split. Median error is how far
off the absolute times are.

## How to run

The script below is self-contained. It needs PuLP **3.x** (`pulp<4`: PuLP 4.0 removed
`PULP_CBC_CMD`). Run it from `mig_pp/milp/`. The ILP's own example config in `main()` is
not used: the config here comes from the CSV.

```bash
cd mig_pp/milp
python3 score.py ../../oracle_v2/80gb/40_20_10_10/vicuna/vicuna_13b.csv vicuna_13b
python3 score.py ../../oracle_v2/80gb/40_20_10_10/llama/llama_13b.csv llama_13b
python3 score.py ../../oracle_v2/80gb/40_20_10_10/qwen/qwen_14b.csv qwen_14b
python3 score.py ../../oracle_v2/80gb/40_20_10_10/mistral/mistral_24b.csv mistral_24b
```

Arguments: CSV path, then the model key in `memory_limits.REFERENCE_SHAPES`. The card is
hardcoded to `A100-80GB` (the CSV cannot tell); the slice layout comes from the CSV header.

```python
# score.py -- ILP vs a measured sweep CSV. CSV is for checking only, never fitting.
import statistics, sys, warnings
warnings.simplefilter("ignore")
import pulp_layer_placement as plp
import validate as V

CSV, MODEL = sys.argv[1], sys.argv[2]
rows, gb = V.load_rows(CSV)
cfg = dict(model=MODEL, slice_gb=gb, card="A100-80GB", seq_len=64, max_new_tokens=512,
           samples=8, min_last_rank_layers=1, constants=plp.DEFAULT_CONSTANTS)

pairs = {}
for r in rows:
    pairs.setdefault((r["batch"], r["mb"]), []).append(r)
print(f"{CSV}: {len(rows)} ok rows, {len(pairs)} pairs, layout {gb}")

errs, by_n, taus, ranks, pen, free_info = [], {}, [], [], [], []
for (b, mb), grp in sorted(pairs.items()):
    steps = plp.prepare(cfg, b, mb, 8)
    meas = {}  # split -> (fastest measured ms, predicted ms)
    for r in grp:
        p = plp.evaluate(list(r["split"]), steps, cfg)
        e = (p - r["meas"]) / r["meas"] * 100
        errs.append(e)
        by_n.setdefault(b // mb, []).append(e)
        if r["split"] not in meas or r["meas"] < meas[r["split"]][0]:
            meas[r["split"]] = (r["meas"], p)
    order = sorted(meas, key=lambda s: meas[s][0])  # by measured latency
    pick = min(meas, key=lambda s: meas[s][1])  # model's best measured split
    taus.append(V.kendall_tau([meas[s][1] for s in order], [meas[s][0] for s in order]))
    ranks.append((order.index(pick) + 1, len(order)))
    pen.append((meas[pick][0] - meas[order[0]][0]) / meas[order[0]][0] * 100)
    free, _ = plp.optimize_placement(cfg, b, mb)
    free_info.append((b, mb, free, tuple(free) in meas, order[0]))

s = sorted(errs)
print(f"error% median {statistics.median(errs):+.1f}  p10 {s[len(s)//10]:+.1f}  "
      f"p90 {s[9*len(s)//10]:+.1f}  mean|err| {statistics.mean(abs(e) for e in errs):.1f}")
print("by N: " + "  ".join(f"N={n}:{statistics.median(v):+.0f}%" for n, v in sorted(by_n.items())))
print(f"Kendall tau median {statistics.median(taus):+.2f} (min {min(taus):+.2f}, max {max(taus):+.2f})")
print(f"pick rank median {statistics.median(r for r, _ in ranks):.0f} of "
      f"~{statistics.median(n for _, n in ranks):.0f}, penalty median {statistics.median(pen):+.1f}%, "
      f"worst {max(pen):+.1f}%")
for b, mb, free, measured, best in free_info:
    print(f"  B{b:<2d} mb{mb:<2d} measured best {list(best)}  ILP free pick {free}"
          f"{' (measured)' if measured else ''}")
```

`validate.py` covers the same ground in its own format. This script adds pick penalty,
error by N, and the summary lines used in the tables below.

## Results, 2026-10-17 (40/20/10/10 on A100-80GB, `oracle_v2/80gb/40_20_10_10/`)

Constants: `DEFAULT_CONSTANTS` from `measure_constants.py` on a whole A100 80GB PCIe (job
2302084). No MIG measurements yet.

### Before the fixes

| Model | Median error | Kendall tau | Pick rank | Penalty median / worst |
|---|---|---|---|---|
| vicuna_13b | +56% | 0.48 | 32/46 | +8.1% / — |
| llama_13b | +56% | 0.48 | 32/46 | +8.1% / +9.3% |
| qwen_14b | +40% | 0.16 | 24/41 | +10.1% / +29.3% |
| mistral_24b | +36% | 0.29 | 34/41 | +10.8% / +15.0% |

The error grew with N (up to +68–100% at N=32): a per-microbatch cost was counted too many times.

### Fixes (from code logic and NVIDIA docs, not from the data)

1. **Kernel launch time counted twice.** It was in a layer's GPU time *and* in the CPU
   launch edge `Lc` (F ≥ A + Lc). Now only in `Lc`. Applies to the layers and to the
   per-microbatch extra kernels (rotary, embed, clone, norm/lm_head/argmax).
2. **Slice bandwidth follows memory slices, not GPCs.** NVIDIA MIG docs: each memory slice
   is 1/8 of capacity *and* bandwidth; 3g.40gb has 4/8 of memory but 3/7 of SMs. Bandwidth =
   card peak × memory slices / 8 (`MEMORY_SLICES`), was × GPCs / 7.

### After the fixes

| Model | Median error | p10…p90 | Kendall tau | Pick rank | Penalty median / worst |
|---|---|---|---|---|---|
| vicuna_13b | +8.4% | −10…+16% | 0.67 | 4/46 | +1.5% / +4.3% |
| llama_13b | +8.5% | −10…+16% | 0.66 | 4/46 | +1.1% / +4.5% |
| qwen_14b | +6.6% | −14…+12% | 0.79 | 5/41 | +2.0% / +2.9% |
| mistral_24b | +1.2% | −6…+10% | 0.66 | 2/41 | +1.4% / +3.4% |

Error by N after the fixes:

| Model | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|
| vicuna_13b / llama_13b | +10% | +9% | +5% | +8% | +4% |
| qwen_14b | −2% | 0% | +9% | +10% | +10% |
| mistral_24b | 0% | +1% | +1% | +8% | +9% |

In plain words: among measured splits, the model's favourite is within 1–2% of the real best
(worst 4.5%) on all four models.

## Open points

- **~10% too slow at N ≥ 16 on qwen_14b / mistral_24b.** Something per microbatch is still too
  big. Suspects: `timing_event_calls` / `send_queue_calls` (never measured), and copy speeds
  measured on the whole GPU (small slices have fewer copy engines). Fix by measuring on
  MIG slices, not by tuning to the CSV.
- **Free picks are mostly unmeasured.** At N=2–4 the ILP picks extreme splits like
  [45,1,1,1] or [36,2,1,1]. For qwen_14b it puts 14–16 layers on rank 1, but the sweep stopped
  at 10. Only running a few of these will tell whether the model is right there.
- **Sweep ranges** (layers per rank 0/1/2/3): vicuna/llama_13b 17–23 / 7–11 / 3–6 / 1–6;
  qwen_14b 20–28 / 8–10 / 5–9 / 1–9; mistral_24b 18–23 / 7–10 / 4–7 / 1–5.

## Sandbox: boundary gap from probe v7, scored at full clock vs run 17 (2026-11-25)

Not in the repo's ILP yet (sandbox copy of `graph_placement.terms()`: GPU time per layer +
kernels x the slice's measured gap per kernel, `boundary_gap` from probe v7, interpolated in
log(mb); the same per kernel outside the layers). Probe set: mig4 v7 (`probe_constants.py`; v3's
`probe_cpu_interactions.json` copied in, as before). `clock="fast"`, scored against run 17's
`performance` + GPU-locked data on mig3 (problem pairs: mean of blocks 2 + 3; stable pairs: block 5,
one run). Scored at full clock at the user's request (2026-11-25); the main-score decision is open.

| | No gap | Gap, GPU side (queued) | Gap, live |
|---|---|---|---|
| Median regret | 0.85% | 0.00% | 0.00% |
| Worst regret | 3.51% | 3.51% | 3.51% |
| Pick in the 2% group | 6/8 | 7/8 | 7/8 |
| Median prediction error | -6.5% | -5.5% | -5.0% |
| Median Kendall tau | 0.69 | 0.76 | 0.78 |

Live gap, per pair: winner at (8,2), (16,4), (32,8), (16,2), (32,4); (16,8) [23,6,2,1] (0.04%, a
tie); (8,4) [27,3,1,1] +1.6% (in the group; still rank-0 heavy); (32,16) [22,7,2,1] vs measured
[21,7,3,1] +3.5% under every variant, tau 0.15: the open pair.

- (32,2), scored later the same day against P2's (32,2) at max on mig4 (53/53 ok, one run; probes and
  benchmark from the same box): pick [16, 9, 4, 3] = measured best (0.00%), free pick the same, tau
  0.94 with the live gap (0.92 without). With it, 8 of 9 pairs in the 2% group; (32,16) is the only miss.
- The absolute level (predictions 4-9% fast) is not a target (user, 2026-11-25): picks and ranking are.
- Caveats: probes from mig4, benchmark from mig3; block 5 is one run per split.

## Pre-registered: (32,16) and (16,8) at max + 1 thread (written 2026-11-25, before the run)

Purpose: check the condition planned for S runs (`performance` + GPU locked + `OMP_NUM_THREADS=1`,
the thread setting torchrun uses for multi-process jobs) and the ILP under it, before S1.
(32,16) is the pair where the receive copy pools at default threads (mb 16 x 4096 = 65,536 >
32,768); (16,8) is the control (same 2 microbatches, 32,768: never pooled).

Model: sandbox `graph_placement` (not the repo's), probe set mig4 (probe v7), live boundary gap,
no pool term (1 thread), `clock="fast"`, exact search. Picks, frozen before the data:

| Pair | Frozen pick | Predicted | In the measured list |
|---|---|---|---|
| (32,16) | [23, 6, 2, 1] | 43.50 s | NEAR_BEST + BY_PAIR (62 splits; includes run 10's 13) |
| (16,8) | [23, 6, 2, 1] | 36.10 s | NEAR_BEST + BY_PAIR (48) |

List for (32,16) changed from NEAR_BEST (58) to NEAR_BEST + BY_PAIR (62) before any data (2026-11-25),
so run 10's 13 one-thread splits are measured again on the same box. Run: `clock_study.sh` with
VALIDATE="32,16;16,8", VALIDATE_PASSES=2 (forward, reversed), THREADS=1, max (~5.7 h).

Added before any data (2026-11-25, after a review): (32,16) once more with torch's default threads
(`CONTRAST="32,16"`, step condition `max`), between the two 1-thread passes, so the thread effect
is measured on one box with drift balanced. Not a pick test. Run total ~7.4 h.

Pass: pick within 2% of the measured best (median of the passes); also report regret and
Kendall tau over the list. Context: at default threads (run 17 max) [23, 6, 2, 1] was +12.2% at
(32,16); run 10 (1 thread) did not measure it. Below the threshold, 1 thread vs default threads
differ by a uniform 0.6-1.1% (box offset) with the same winners (run 10 vs run 17 max, 8 pairs).

### Result (scored 2026-11-26, P3 Run A on mig5): PASS at both pairs

Data `../logs/mig5_p3_runA/` (`clock_study.sh`, commit 2f814f3; steps `clock_blk01_max-t1_*` and
`clock_blk03_max-t1_*`, the two 1-thread passes, forward then reversed; `clock_blk02_max_b32_mb16`,
the default-thread contrast, is not part of the pick test). Harness: 62 / 48 / 62 / 62 / 48 rows, all
`ok`; governor and GPU clock restored at the end. Measured = median of the two 1-thread passes.

| Pair | Measured best | Frozen pick [23, 6, 2, 1] | Regret | Rank | Kendall tau (list) | Verdict |
|---|---|---|---|---|---|---|
| (32,16) | [21, 8, 2, 1], 47.64 s | 47.91 s | +0.57% | 4 / 62 | +0.87 | **PASS** |
| (16,8) | [22, 7, 2, 1], 39.96 s | 40.17 s | +0.52% | 2 / 48 | +0.73 | **PASS** |

- 2% groups: (32,16) 13 splits, (16,8) 10 splits; the pick is in both.
- Repeat difference pass 1 vs pass 2: median 0.16% / 0.21%, worst 1.35% / 1.09%.
- Reproduction: the sandbox model (setup above) gives [23, 6, 2, 1] at 43.50 s / 36.10 s, equal to
  the pre-registered values; tau uses its predictions for every split in the list.
- Predicted level ~9% below measured (43.50 vs 47.91 s): not a target (picks and ranking are).

## Pre-registered: Mistral-7B, blind test #1 (written 2026-11-26, before any Run B data is opened)

Data to score against: P3 Run B on mig5 (steps `mistral7b_clock_blk01_max-t1_*`, then step 2 and the
confirmation repeats), main condition (`performance` + GPU locked at 1410 MHz + `OMP_NUM_THREADS=1`),
20/10/5/5. Run B step 1 started Thu 11/26 6:35 pm EST; nothing of it has been read.

Model (frozen): `mig_pp/milp/graph_placement.py` with the kernel-boundary gap (moved from the sandbox
2026-11-26; identical to the sandbox Run A passed with: 448 predictions equal, exact-search picks
equal at 9/9 pairs; pinned by `tests/test_frozen_picks.py`). Commit: the one that adds this section.
Probe set `probe_sets/mig5.json` (probe v7.1, same card and condition as Run B), `gap_mode="live"`, no
pool term (the set has none), `clock="fast"`, exact search (`optimize_placement`), model `mistral_7b`,
seq_len 64, max_new_tokens 512, samples 8, min_last_rank_layers 1.

| Pair | Frozen pick | Predicted | In Run B step 1 (136 splits) |
|---|---|---|---|
| (8,4) | [27, 3, 1, 1] | 31.00 s | yes |
| (16,8) | [27, 3, 1, 1] | 31.81 s | yes |
| (32,16) | [26, 4, 1, 1] | 34.69 s | yes |
| (8,2) | [19, 9, 3, 1] | 49.89 s | yes |
| (16,4) | [19, 9, 3, 1] | 50.40 s | yes |
| (32,8) | [20, 8, 3, 1] | 51.56 s | yes |
| (16,2) | [16, 9, 4, 3] | 77.34 s | yes |
| (32,4) | [17, 9, 4, 2] | 78.53 s | yes |
| (32,2) | [16, 9, 4, 3] | 130.27 s | yes |

Per pair: pass = pick within 2% of the best measured split (median of its runs). Report regret,
rank, Kendall tau over the measured list, and n of 9 picks in the 2% group.

**Acceptance (proposed by Claude, approved by the user 2026-11-27, before any Run B data is opened):** the blind test
passes when all three hold: **at least 7 of 9 picks in the 2% group**, **median regret <= 1%**,
**worst regret <= 3%**. Kendall tau and the baselines' regret are reported as context, with no
threshold. Basis: development model at full clock 8/9 in the group, median 0-0.85% (sandbox section
above); Run A +0.52% / +0.57%.

**What is scored:** Run B steps 1-3 only (`TO-DO.md` holds the step 2 and step 3 rules, fixed
2026-11-27). Measured value = median over a split's runs; the best measured split and the 2% group
come from those medians. Step 3 repeats every split within 3% of the best plus the frozen picks and
the baselines. The later pass on mig5 (rank 0 = 25, ranks 2/3 swapped) does not change this score; if
it finds a faster split, regret against it is reported separately as a sensitivity result.

**Seal note:** oracle_v2's Mistral-7B best splits (`schedutil`, cross-node, not this setup) were seen
by the user and Claude on Thu 11/26 5:35 pm EST, before this freeze. The picks above come
mechanically from the frozen model and probe set; no setting was chosen with Mistral in mind (the
setup is Run A's pre-registered one with the probe set swapped to the same-card mig5 set).

### Baselines for the Mistral-7B blind test (frozen 2026-11-26, before any Run B data is opened)

Primary model: the graph ILP above. Baselines, scored the same way (regret vs the best measured split):

| Pair | Even | GPC-proportional 3:2:1:1 | Memory-proportional 4:2:1:1 | GPipe ILP pick (predicted) |
|---|---|---|---|---|
| (8,4) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [20, 10, 1, 1] (28.65 s) |
| (16,8) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [20, 10, 1, 1] (29.82 s) |
| (32,16) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [20, 10, 1, 1] (32.81 s) |
| (8,2) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [16, 9, 3, 4] (44.35 s) |
| (16,4) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [18, 10, 1, 3] (45.26 s) |
| (32,8) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [18, 10, 3, 1] (47.21 s) |
| (16,2) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [16, 9, 3, 4] (71.45 s) |
| (32,4) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [16, 9, 3, 4] (72.91 s) |
| (32,2) | [8, 8, 8, 8] | [14, 9, 5, 4] | [16, 8, 4, 4] | [16, 9, 3, 4] (125.65 s) |

- GPipe ILP: `mig_pp/milp/gpipe_placement.py`, the same inputs as the graph ILP (its stage times are the
  graph ILP's step terms, gap included: mig5 set, live gap, `clock="fast"`); only the step law differs
  (fill + (N-1) x slowest stage). Exact search; equals brute force at (32,16) and (32,2).
- Even, GPC- and memory-proportional are rules, not models. A profile-based stage balancer is not a
  separate baseline: the GPipe law with probed stage times is that approach in exact form.
- In Run B step 1 already: GPC- and memory-proportional, GPipe at (8,4) (16,8) (32,16) (32,8). Added to
  step 2 (all fit in memory): [8, 8, 8, 8] at all 9 pairs; GPipe [16, 9, 3, 4] at (8,2) (16,2) (32,4)
  (32,2) and [18, 10, 1, 3] at (16,4).
- **Primary model rule (user, 2026-11-26):** the graph ILP stays primary; the GPipe ILP is a
  comparison and would only become primary if it scores better than the graph ILP. If that switch is
  made on Run B data, the paper must say the primary was chosen after scoring the blind test (Mistral
  then is no longer a blind test for the chosen model; the held-out main-condition Qwen2.5-7B run
  on mig5 would be its first unseen test, and that run is not fully blind either: Qwen's
  default-condition development check is already scored above).
