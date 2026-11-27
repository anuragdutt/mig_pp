# Probes: hardware measurements for the ILP

The ILP's constants come from these probes or from theory, never from benchmark results
(`SCORING.md`). Probes run on mig1 and mig2 (A100-PCIE-40GB, MIG 20/10/5/5, `schedutil`, default
thread pool) and are exported to `~/Desktop/AP/logs/mig{1,2}_probes*` (commands in `setup.md`).

| Script | Measures |
|---|---|
| `probe_cpu_interactions.py` | receive copiers next to a process launching GPU work; governor duty cycles |
| `measure_constants_cpu.py` | launch time per kernel, host copies, gloo messages, token staging, step start, receive copy with idle gaps, pooled copy with 4 processes |
| `measure_constants_mig.py` | GPU primitives per MIG slice (steady state, CUDA graphs) + validation layers of Vicuna-7B and Qwen2.5-7B |

**Step 3, the gate before any probe set goes into the ILP:** one decoder layer's GPU time,
composed from the primitives (`layer_ops.py`), must be within 5% of the measured validation layer
at every point (`check_layers.py`). The measured layers are a check only, never an input; a miss is
fixed in the op list or the probe, never by adjusting numbers.

## Probe runs

| Run | Date | Commit | Folder | Notes |
|---|---|---|---|---|
| 1 | 2026-11-14 | before `0136bcf` | `mig{1,2}_probes` | single calls after idle gaps; the boxes disagreed 8-44% on the primitives while the real layers agreed 0-1% |
| v2 | 2026-11-15 | `0136bcf` | `mig{1,2}_probes_v2` | steady state, per head shape; no raw points, repeat pass, windows or gap sweep (`5006cd9` was not pushed yet). Superseded by v3 |
| v3 | 2026-11-15 | `5006cd9` | `mig{1,2}_probes_v3` | complete: raw points, repeat pass, windows, receive copies with gaps, interference sweep |

## Probe v3: GPU

**Step 3 fails: 59 of 144 points within 5%** (probe run 1: 40, v2: 57). The composed layer is
mostly too low: Vicuna decode -4 to -10%, prefill -5 to -20%, worst -21%.

- **Repeatable.** Repeat pass vs first pass: median difference 0.2-0.4% (host copies up to ~20%).
  mig1 vs mig2: median within 1%, single points up to ~10% apart. So the misses are the method.
- **A straight line per primitive (fixed + size / rate) cannot follow the points:**
  - GEMM: achieved bandwidth swings +-11% by shape and row count (cuBLAS picks different kernels);
    not smooth in either.
  - Attention: time depends on microbatch size and context separately (mb 1: 137-476 GB/s; mb 32:
    339-593 GB/s); the line misses by up to 28%.
  - KV append: sizes under ~20 MB run above the slice's peak bandwidth (L2 hits); large sizes run
    at a steady ~685 GB/s on 3g.20gb, 88% of peak. The line, pulled up by the small ones, is 17%
    too fast at real sizes.
  - Elementwise and device copy: small ops take 1.2-2.3 us, close to their lines (~1% of a layer).
- **Using the measured points directly** (GEMM by shape and rows, attention and KV append on the
  microbatch x context grid; scratchpad `lookup_test.py`): decode 70 of 108 points within 5%,
  median -3.7%, worst -11.6%.
- **What is left depends on microbatch size only, not context** (3g.20gb Vicuna: ~32 / 70 / 130 us
  at mb 1 / 8 / 32), so it sits in the small kernels (norm, rotary, broadcast ops), not attention.
  The validation layers record only their total, so the kernel can't be named yet.
- **Prefill:** no prefill-attention primitive (charged by its KV bytes); -5 to -18%.
- **Grid gaps:** Qwen's 512 x 3584 and 3584 x 18944 GEMMs are not in the GEMM grid; the harness's
  microbatch sizes 2, 4 and 16 are not in the row / microbatch grids (1, 8, 32).

## Probe v3: CPU

- **Launch time per kernel:** 19.3-19.8 us in all 3 probe runs, both boxes.
- **Governor:** a mostly idle process runs at 0.39-0.47 of full speed (launches ~2.3x as long).
  Full speed from 40% busy at a 40 ms period and 30% at 120 ms: mig1 in all 3 runs; mig2 moves by
  ~10 points and flips at low busy fractions. "4 at once" is random between runs (idle processes
  sometimes stay fast).
- **Pooled receive copy (mb 16 = 65,536 values) depends on what else runs:**

  | Condition | Per copy |
  |---|---|
  | one process, back to back | 27 us |
  | one process, 10-50 ms idle between copies | 0.25-0.36 ms |
  | one copier next to a process launching back to back | **6.6 ms** |
  | 3 copiers at once | ~35 ms |
  | 4 processes at once, every 20 ms | 46-47 ms (only 21-22 of 50 copies/s) |

  The harness measured ~6.5 ms per receive (`problems.md`, "Run 8"): the one-copier case, as in a
  pipeline where receives on different ranks rarely coincide. One thread per copy: 25-38 us.
- **Launches next to receive copiers:** x1.46-1.51 next to 3 pooled copiers launching back to back,
  x1.08-1.10 next to one; x0.67-0.71 when the launcher idles 25 ms between bursts (the copiers keep
  the clock up). One-thread copies and copies of 32,768 values: no effect.
- **gloo messages** (handshake, token round trip, barrier) vary 30-45% between runs, favouring
  either box: noise, not a box difference.

## Probe v4 (written 2026-11-15, MIG probe only; rerun pending)

`measure_common.py` extended; the CPU probes are unchanged, so v4 reuses v3's CPU files:
1. Validation layers save each kernel's name and time (`kernels` per point), to name the op the
   composition misses.
2. The layers' small ops by type, as generic ops (`op_*`): RMSNorm's steps, residual add, SiLU,
   gate x up, rotary's broadcast multiply, half negate and cat; rows 1-2048, widths 4096 / 14336.
3. GEMM rows 1, 2, 4, 8, 16, 32 and 16 shapes (adds Qwen's k/v and down projections, 14336-wide
   MLPs, a 13B down projection, 32768- and 152064-row lm_heads); prefill GEMM rows 64-2048.
4. Attention and KV append at microbatch sizes 1, 2, 4, 8, 16, 32.
5. Prefill attention (`attention_prefill_h*`): masked SDPA with K/V repeated, per query-head count.
6. A fit with no growth with size (tiny ops) gives fixed time only, not NaN.

About 4x the points of v3.

### Probe v4 results (2026-11-15, `mig{1,2}_probes_v4`, v3's CPU files copied in)

`layer_ops.layer_us` now composes from the measured points when a slice has them (GEMM by shape and
rows, attention and KV append on the microbatch x context grid, prefill attention on microbatch x
length, small ops by type and size), and from the lines otherwise. Small ops carry their type
(`_op_tag`). The attention output's `.contiguous()` is free (no kernel in the per-kernel lists), so
prefill is 42 kernels for Vicuna and 44 for Qwen, as measured.

**Step 3: 109 of 144 points within 5%** (v3: 59). Median error by slice, both boxes: 3g.20gb -4.3 to
-5.0%, 2g.10gb -2.6 to -3.1%, 1g.5gb -2.3 to -2.7%; worst: prefill mb 32, -10 to -13%. Still below
the bar, and the bias is larger on the 20GB slice.

Per op, decode (composed vs the layer's own kernels, matched in order; scratchpad `per_op.py`):
- GEMM: measured / composed 1.01-1.02.
- Small ops: 1.2-2x in the layer, ~+1 us each, ~30 us per layer. The probe times them back to back
  in a CUDA graph; the harness and the validation layers launch them one by one (eager).
- Attention: ~1.15x, +16-31 us per layer. The layer attends to ctx + 1 keys: at L = 65 it takes
  1.4-1.7x the probe's L = 64 (one more 64-key block); at L = 321 / 576, 1.0-1.17x.
- KV append: 1.03-1.12x.

Prefill: the GEMM shapes not in the prefill sweep (the down projections, Qwen's k / v) are scaled
from another shape and miss by up to 2x; the per-op order of prefill does not match the trace's
(the empty cache's two small kernels run after the cats), so it is compared in total only.

## Probe v5 (written 2026-11-15, MIG probe only; to run 2026-11-16 morning)

1. Every GPU primitive timed the way the harness and the validation layers run: launched eagerly
   one by one, kernel time from the profiler (`eager_us`), instead of CUDA graph replays. Points
   carry method "kernel".
2. Small ops measured inside generic layer-order chains, per kernel (`chain_us`): RMSNorm's 8 steps,
   rotary's 5 (cos multiply, half negate, cat, sin multiply, add), SiLU x up + residual add.
3. Decode attention at L = 64k and 64k + 1 for k = 1-9, and 1088 / 1089 (the KV append keeps 64,
   320, 576, 1088: a copy has no block steps).
4. Prefill GEMM for all 13 non-lm_head shapes, rows 64-2048.

About 870 timed points per pass and slice; expect ~1 h on each box.

### Probe v5 results (2026-11-16, `mig{1,2}_probes_v5`, v3's CPU files copied in)

Complete on both boxes: every sweep has its full point count in both passes, all GPU points timed
as "kernel", every validation point grouped per kernel, SM clock 1395-1410 MHz.

**Step 3: 142 of 144 points within 5%** (v4: 109, v3: 59). Decode 106/108, prefill 36/36. Median
error per slice, model and box: -0.4% to -1.5% (it was -2.3% on 5GB to -5.0% on 20GB with v4), so
the remaining bias is small and about the same on every slice.

- Per op, decode (`per_op.py`): GEMM 1.01, attention 0.99 (block edges + eager timing fixed it; v4:
  ~1.15), KV append 1.01; small ops 1.0-1.7x their chain values, ~1 us each.
- The 2 misses: Qwen decode mb 32 ctx 64 on 3g.20gb, -5.3% (mig1) / -5.5% (mig2). Largest gap: the
  q projection, 54 us in the layer vs 46 composed. Qwen's q / k / v projections have a bias, and
  with a bias cuBLAS runs a different GEMM kernel (its name carries "relu"); the probe measures
  GEMMs without bias. Vicuna and Mistral have no bias: Vicuna is 12/12 on every slice.
- **Decision (2026-11-16): step 3 passes for bias-free models** (Vicuna-7B, Llama, Mistral-7B: the
  models scored). Known limit: models with q / k / v biases (Qwen2) compose ~5% low at large
  microbatches on 3g.20gb until the probe measures GEMMs with a bias (a small probe addition).
- Repeatability: repeat vs first pass median 0.2-0.4%, 95th percentile 5-11% (eager timing of
  single small points varies more than graph timing did). mig2 vs mig1: median -1.3% to +0.7%,
  single points up to ~18% apart on 3g.20gb; the composed layers agree between boxes.

## Probe v6 (written 2026-11-16; CPU + MIG probes; to run)

From the development check's diagnosis against the events runs (`SCORING.md`): two places where
the probes did not measure the way the pipeline runs. Method fixes only, nothing fitted:
1. **Launch** (`measure_constants_cpu.py`, `layer_stack_dispatch_us`): CPU time per decode layer
   call in a stack of 16 calls back to back, each call timed, the first of each stack left out, as
   `forward_through_layers` launches them. Before: one call right after a synchronize, cold, 19.3 us
   per kernel, ~15% above the harness's own 0.67 ms per layer. The old number is kept as
   `launch_us_cold`.
2. **Concurrency** (`measure_constants_mig.py` + `measure_common.concurrency`): per slice, a decode
   GEMM (11008 x 4096, M 8), decode attention (h32, mb 8, L 321) and an RMSNorm chain, timed alone,
   with a decode-like load (`gpu_load`) on every other slice of the card, and alone again; factor =
   loaded / alone per kind. The ILP's composed GPU time takes them per op (`layer_us(loaded=True)`:
   GEMM and KV append x gemm, attention x attention, small ops and copies x small); step 3 keeps
   composing alone, as the validation layers are measured.

The repeat pass is off in v6 (`CONFIG["repeat"]`; v3-v5 showed 0.2-0.4%). Expect ~5-10 min (CPU
probe) + ~40 min (MIG probe) per box. On the Mac, v3's `probe_cpu_interactions.json` is copied in.

### Probe v6 results (2026-11-16, `mig{1,2}_probes_v6`, v3's `probe_cpu_interactions.json` copied in)

Complete on both boxes. Both of v6's hypotheses were **refuted**:
- **Launch:** timed as a back-to-back stack, 18.47 / 18.50 us per kernel (spread 0.5%), against
  18.9-19.0 us cold. The cold start was ~2.5%, not the ~15% gap; the events runs time launch the
  same way (perf_counter at each layer boundary, `trace_bench.py`) and see 0.67 ms per layer, so
  ~10% is still unexplained. Grad mode is the same (`no_grad` in both).
- **Concurrency:** with a decode-like load on the other 3 slices, each slice's speed stays the same:
  GEMM x1.00, small ops x1.00-1.01, attention x1.00-1.03, SM clock 1410 MHz. Power and clocks
  shared between slices are not what makes the harness's layers 2-9% slower.
- Step 3 on v6's primitives: 142/144 again (v6's GPU numbers agree with v5's).

**The GPU gap is a definition mismatch:** the events runs' `layer_ms` is a CUDA-event span around
each layer, which includes the GPU's idle gap between consecutive kernels; the composition sums
profiler kernel durations, which leave it out. Rank 0 at (8,4): 0.885 vs 0.838 ms per layer, ~1.2
us per kernel boundary. A fixed gap per kernel weighs most on the fastest slice (+5.7% on 20GB,
+1.8% on 5GB), which pushes layers off rank 0.

### Next: probe v7 (proposed)
1. **Kernel-boundary gap:** per slice, the GPU span (CUDA events) of a long eager sequence of mixed
   generic kernels (GEMMs, the small-op chains, attention) against the sum of their profiler
   durations; gap per kernel = the difference / kernels. The ILP adds it per kernel.
2. **Launch:** time the harness's own `forward_through_layers` on a stack of distinct layer objects,
   with the CPU frequency recorded during the measurement, to find the remaining ~10%.


## Probe v7 (written 2026-11-25; CPU + MIG probes; to run in P2)

The three items above, as method additions; nothing fitted, nothing the ILP reads yet (the probe
sets carry the new numbers only when the folder has them, so v6 sets rebuild byte-identical):
1. **Kernel-boundary gap** (`measure_common.boundary_gap`, MIG probe, per slice): a stack of 8
   generic decode layers (`gap_layer`: Llama-7B shapes, the harness's op order, 40 kernels, two
   weight sets in turn), mb 1 / 8 / 32, ctx 320, 5 windows. `kernel_us` = profiler sum;
   `span_queued_us` = CUDA-event span with every launch queued behind a sleep kernel (GPU-side gaps
   only; `queued_ok` says the CPU finished first); `span_live_us` = launched live from an idle GPU.
   Gap per kernel = (span - kernel_us) / kernels, both ways.
2. **Launch as the harness issues it** (`measure_common.harness_launch_us`, CPU probe, last step):
   8 distinct layers (own weights, own cache slots), mb 8, ctx 64; three methods alternating per
   window: the benchmark's own `forward_through_layers` (one call / layers), the same layers one by
   one (`per_layer_distinct`), and v6's one layer object called again (`per_layer_same_v6`). The
   clock of the CPU each window ran on is sampled every 20 ms by a child process.
3. **GEMM with a bias** (`primitives`, MIG probe): `F.linear(x, w, b)` for Qwen2.5-7B's q (3584 x
   3584) and k / v (512 x 3584) projections, decode M 1-32 and prefill M 64-2048, as
   `gemm_bias_decode` / `gemm_bias_prefill`.

Concurrency is off from v7 (`CONFIG["concurrency"]`; v6: x1.00-1.03).

Gate (Mac, 2026-11-25): v6 folders rebuild `probe_sets/mig{1,2}.json` byte-identical; `layer_us`
identical with v7 keys present (162 points, 3 models x 6 mb x 3 ctx x 3 slices); CPU smoke of
`gap_layer` (mb 1 / 8 / 32 shapes) and of `harness_launch_us` end to end on a tiny Llama, both
through the real `forward_through_layers` and the fallback; `test_pulp_layer_placement` 34/34
(needs PuLP < 3: 4.x changed `LpVariable`), `test_layer_ops_torch` 4/4. Not testable on the Mac:
the CUDA timing itself (`torch.cuda._sleep`, events, profiler) and the CPU clock sampler (Linux
`/sys`); the box run is their first test.

Ran on mig4 in P2 (2026-11-25, `schedutil`, GPU unlocked): ~1 min CPU probe + ~6 min MIG probe;
data `../logs/mig4_p2_5_clock/probes_v7/`. Two defects found there, fixed in v7.1.

## Probe v7.1 (2026-11-26; in the code from 2026-11-26, was `wip/measure_common.sandbox.patch`)

Changes to `measure_common.py`, nothing the ILP reads changes (it takes kernels per layer from
`layer_ops` and `launch_us` from the CPU probe):
1. **Kernels per layer counted under `no_grad`**, as every timed call: v7 counted 35, not 40, so its
   `launch_harness` us-per-kernel was ~14% high (printed only).
2. **CPU clock read in-process** (`cpu_khz`, `scaling_cur_freq` of the CPU the process ran on, before
   and after every stack); v7's sampler child recorded nothing on the box. Under the main condition
   this records that the `performance` governor held during the probe.
3. **3-D-input GEMMs** (`gemm3d_decode`, `gemm_bias3d_decode`): `gemm_bias_shapes` with an (mb, 1, K)
   input, as a decoder layer's `nn.Linear` gets it, without and with the bias (Qwen's q projection:
   bias or input shape?). Adds two MIG-probe groups (~1-3 min).

Gate (Mac, 2026-11-26): smoke test 18/18 (scratchpad `smoke_v71.py`: count under `no_grad`, clock
per window and per method, `freq_samples`, no-clock fallback, both 3-D groups fitted with (M,1,N)
outputs; CUDA calls stubbed, `layer_forward` fallback path); `test_layer_ops_torch` 4/4. Not
testable on the Mac: CUDA timing and `/sys` on the box. First box run: P3 on mig5, under the main
condition (`CLOCK_MODE=probes ./clock_study.sh`).

### Probe v7.1 results (2026-11-26, mig5, `../logs/mig5_probes_v71/`, `clock_study.sh --probes`, commit c39ae5e)

First box run of v7.1, under the main condition: governor `performance`, GPU locked at 1410 MHz,
`OMP_NUM_THREADS=1` (constants_cpu.json host block). Both probes ok.
- **CPU clock held:** 3.517 GHz median in all 300 in-process samples (v7.1 item 2 works on the box).
- **Kernels per decode layer 40** (v7 counted 35): v7.1 item 1 confirmed. Launch 18.85 us per kernel;
  as the harness issues it 21.0-21.3 us per kernel.
- **Kernel-boundary gap, queued, us per kernel (mb 1 / 8 / 32):** 20GB 0.78 / 0.88 / 0.75, 10GB
  0.61 / 0.67 / 0.76, 5GB 0.74 / 0.70 / 0.79.
- Probe set `probe_sets/mig5.json` (`probe_constants.py`): no `probe_cpu_interactions.json`, so no
  clock or pool model, `clock="fast"` only (the ILP refuses `schedutil` with it). mig1 / mig2 rebuilt
  alongside, byte-identical.
- **Step 3 gate** (`check_layers.py`): Vicuna-7B 36/36 within 5% (median -1.0% to -1.4% per slice);
  Qwen2.5-7B 35/36, the miss is 20GB decode mb32 ctx64 at -5.3%, the same point that misses on mig1
  and mig2. Mistral-7B has no validation layer yet (`TO-DO.md`).
