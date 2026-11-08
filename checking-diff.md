# Checking the PCIe vs SXM4 difference

## Why we are doing this

All existing one-GPU 80GB data was measured on **A100-SXM4-80GB** cards (box `cc@mig`). The new
4-GPU machine has **A100 80GB PCIe** cards. We want to extend that data with new runs (extreme
splits for the bigger models, the wider layer limits for the 7B models).

The two cards are not automatically interchangeable:

| A100 80GB | SXM4 (existing data) | PCIe (new box) |
|---|---|---|
| Memory bandwidth | 2039 GB/s | ~1935 GB/s |
| Power limit | 400 W or more | ~300 W |

These specs are from memory of NVIDIA's datasheets. Confirm them on the box with:

```bash
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm,clocks.max.mem --format=csv
```

We can't assume our workload stays clear of these limits. The slice probes show MIG slices reaching
82–91% of their memory-bandwidth share, so a ~5% lower peak could show up in latency.

The risk is not a uniform slowdown, which would leave the ranking of splits intact. The risk is a
difference that varies **by split**. Every ILP score compares a split with the best split in the
same table. If base splits come from SXM4 and new splits from PCIe, part of any gap would be the
card rather than the split.

So before extending the data, we re-run a small set of already-measured configs on PCIe and compare
them with the SXM4 numbers.

## What to run

Run on 40/20/10/10 (`3g.40gb` + `2g.20gb` + `1g.10gb` + `1g.10gb`), with the same harness as the
existing data, 4 lanes at once as the old runs did, `MIG_LOG_LEVEL=summary` and `skip_done=False`.

Each split runs at these 4 (batch, microbatch) pairs, which cover both sides of the 64 KiB hand-off
jump and both low and high microbatch counts:

**(16, 4), (16, 8), (32, 16), (64, 2)**

| GPU | Model | Splits | One pass |
|---|---|---|---|
| 0 | Vicuna-13B | [23,11,4,2], [20,11,5,4], [23,9,4,4], [23,8,6,3], [21,7,6,6], [17,11,6,6] | ~1.6 h |
| 1 | Qwen-14B | [28,10,7,3], [27,10,7,4], [26,9,8,5], [28,8,6,6], [20,10,9,9], [26,10,6,6], [25,10,7,6] | ~2.1 h |
| 2 | Mistral-7B | [16,10,4,2], [15,10,4,3], [17,11,2,2], [20,7,4,1], [19,5,4,4], [20,8,3,1], [12,11,5,4] | ~1.2 h |
| 3 | Qwen-7B | [14,9,4,1], [11,10,5,2], [18,8,1,1], [20,5,2,1], [9,7,6,6], [20,6,1,1], [9,8,7,4] | ~1.3 h |

The splits span best to worst in the existing data, plus the rank-0-heaviest and rank-0-lightest
splits. Times are estimates from the old latencies plus per-config overhead.

**Run the whole set twice** (about 4.5 h in total). The second pass measures the new box's own
run-to-run noise.

SXM4 reference data: `logs/oracle/intra_gpu/80gb/40_20_10_10/*/*/gpu*/*_<model>/mig_benchmark_results.csv`.
The bigger models are also in `oracle_v2/intra_gpu/80gb/40_20_10_10/`.

## Pass rules (fixed before seeing the results)

Judge the bigger models and the smaller models separately. For each config:

ratio = PCIe latency ÷ SXM4 latency

| Check | Rule | Why |
|---|---|---|
| Noise | The two PCIe passes agree within ~0.5% (median) | Otherwise the comparison below means nothing |
| Time shift | The model's median ratio is within ±3% | Absolute times are comparable |
| Ranking | After dividing by that median, at least 90% of configs are within ±1% and none is beyond ±2% | The gaps that matter are small: best vs runner-up is 0–1.5% |
| Order | Splits are in the same order within each pair (Kendall tau ≥ 0.9) | Same best split |

## What each result means

| Result | Decision |
|---|---|
| All checks pass | Extend the data; treat the cards as equivalent |
| Only the time shift fails | Extend for choosing splits; label the card and don't compare absolute times across cards |
| Ranking or order fails | Keep PCIe data separate and run complete sets on it |

## Where it goes in the schedule

This runs first, before any new data is collected, because the result decides how the new runs are
used and stored.
