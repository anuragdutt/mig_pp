# Runs under the main condition

Which data counts under the hard rule of 2026-11-26: `performance` governor + GPU SM clock locked at
max + `OMP_NUM_THREADS=1` per rank (`mig_pp/milp/SCORING.md`, "Which condition counts"). Update
this table when a run is exported. Data folders are under `../logs/` (sibling of this repo).

## Vicuna-7B, PCIe 20/10/5/5

| Pair | Source | Splits | Runs per split | Counts because | Data |
|---|---|---|---|---|---|
| (8,4) | run 17 max blocks (mig3) | 48 | 2 | mb <= 8: decode receive never pools, so default threads ~ 1 thread | `mig3_run17c/` |
| (16,4) | run 17 max blocks (mig3) | 51 | 2 | same | `mig3_run17c/` |
| (8,2) | run 17 max blocks (mig3) | 38 | 2 | same | `mig3_run17c/` |
| (16,8) | run 17 max (mig3) + P3 Run A (mig5) | 48 | 4 (2 + 2) | same; Run A is exactly the rule | `mig3_run17c/` + Run A (not exported yet) |
| (32,8) | run 17 block 5 at max (mig3) | 35 | 1 | mb <= 8 | `mig3_run17c/` |
| (16,2) | run 17 block 5 at max (mig3) | 48 | 1 | mb <= 8 | `mig3_run17c/` |
| (32,4) | run 17 block 5 at max (mig3) | 48 | 1 | mb <= 8 | `mig3_run17c/` |
| (32,2) | P2, run 17 leftover at max (mig4) | 53 | 1 | mb <= 8 (different card: mig4) | `mig4_p2_5_clock/` |
| (32,16) | P3 Run A only (mig5, started 2026-11-26) | 62 | 2 | over the pool threshold (16 x 4096 = 65,536 > 32,768): only 1-thread data counts | Run A (not exported yet) |

## Everything else

| Model / layout / card | Data under the rule |
|---|---|
| Mistral-7B (PCIe) | None. Run 16 and P2's (32,4) are `schedutil`: secondary, sealed |
| Qwen2.5-7B, Llama-2-7B (PCIe) | None |
| 10/10/10/5 layout (PCIe) | None |
| SXM4, all models | None |

Not counted: run 10 (`performance` + 1 thread, but GPU not locked), run 8 (`schedutil`), oracle_v2,
runs 2-14, run 17's default blocks, and run 17's (32,16) at max (default threads, over the threshold).

## Notes

- Run 17's max blocks count at the 8 pairs only through the threshold argument (`justification.md`
  sections 2 and 6: per-pair medians +0.4 to +1.1% default vs 1 thread, winners equal or within
  0.1%, cross-host). Prefill pools at every pair; that is covered only by those small whole-run
  differences.
- Check after Run A: (16,8) with 1 thread (Run A) vs run 17's (16,8) at max with default threads.
  Same card (mig5 = mig3's card), so it tests the argument directly. If they agree within max-clock
  repeat noise (~0.2-1%), the 8 pairs stand without reruns.
- Runs per split (protocol of 2026-11-27, `TO-DO.md` / `SCORING.md`): 2 runs for the near-best group
  (within 3% of the best) plus the frozen picks and baselines; a 3rd run of that set only for a pair
  whose repeat differences exceed median 0.5% or any split 2%. Replaces the earlier ">= 3 runs".
  Vicuna is the development model: a top-up for it is decided after the (16,8) check (open).
