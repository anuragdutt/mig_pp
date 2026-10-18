# Scoring the ILP against measured sweeps

How we check `pulp_layer_placement.py` against oracle CSVs. **The CSVs are verification
data only. Nothing in the model is fitted to them.** Constants come from
`measure_constants.py`; the formulas come from the harness code and NVIDIA's MIG docs.

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
