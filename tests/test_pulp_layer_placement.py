"""Tests for mig_pp/milp/graph_placement.py and validate.py. Torch-free; the MILP
tests need PuLP and skip without it. Run locally:
    python3 -m unittest discover -s tests -p 'test_pulp_layer_placement.py' -v
With PuLP from a scratch dir (not the venv):
    PYTHONPATH=<scratch>/pylib .venv/bin/python -m unittest discover -s tests \
        -p 'test_pulp_layer_placement.py' -v
"""

import ast
import csv
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MILP = ROOT / "mig_pp" / "milp"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MILP))

try:
    import pulp  # noqa: F401  (only to know whether the MILP tests can run)

    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False

import memory_limits as ml  # noqa: E402
import graph_placement as plp  # noqa: E402

SHARED = ("HS", "TOK", "BARRIER", "STEP")


def cfg(model="vicuna_7b", gb=(20, 10, 5, 5), card="A100-40GB", T=16, **kw):
    return dict(dict(model=model, slice_gb=list(gb), card=card, seq_len=64, max_new_tokens=T,
                     samples=8, min_last_rank_layers=1, constants=plp.DEFAULT_CONSTANTS), **kw)


def zero_values(R):
    """Every DAG term at 0 for R ranks."""
    v = dict.fromkeys(SHARED, 0.0)
    for s in range(R):
        v[f"Cg{s}"] = v[f"Lc{s}"] = 0.0
        if s < R - 1:
            v[f"D{s}"] = v[f"M{s}"] = 0.0
        if s > 0:
            v[f"RECV{s}"] = 0.0
    return v


def end(R, N, values, sink="token"):
    edges, order = plp.step_graph(R, N, sink)
    return plp.longest_path(edges, order, values)


def simulate_step(R, N, v, sink="token"):
    """One step written the way run_pipeline executes it (each rank's CPU program in
    order, a clock per CUDA stream) instead of as the edge list: an independent oracle."""
    upstream = None  # handshake delivery times of rank s-1
    for s in range(R):
        last = s == R - 1
        cpu = v["STEP"] if s == 0 else 0.0
        gpu = copy = 0.0  # default stream, copy_stream
        landed, published = [0.0] * N, [0.0] * N
        for k in range(N):
            if s > 0:  # RECV(k): wait for the handshake, memcpy, H2D, ACK
                cpu = max(cpu, upstream[k]) + v[f"RECV{s}"]
            start = cpu
            gpu = max(max(start, gpu) + v[f"Cg{s}"], start + v[f"Lc{s}"])
            cpu = start + v[f"Lc{s}"]  # LAUNCH(k) done; isend queued D2H(k)
            if not last:
                copy = max(gpu, copy) + v[f"D{s}"]
                landed[k] = copy
                if k >= 1:  # FLUSH(k-1): event wait, memcpy, handshake
                    cpu = max(cpu, landed[k - 1]) + v[f"M{s}"] + v["HS"]
                    published[k - 1] = cpu
        if not last:
            cpu = max(cpu, landed[N - 1]) + v[f"M{s}"] + v["HS"]  # FLUSH(N-1)
            published[N - 1] = cpu
            upstream = published
        elif sink == "token":  # the token D2H waits for the GPU
            return max(cpu, gpu) + v["TOK"]
        else:  # final step: straight to the barrier
            return cpu + v["BARRIER"]


class Bytes(unittest.TestCase):
    """measure_constants.py defines EFF_WEIGHT / EFF_ACT against its own byte counts, so
    the model must count the same way. It imports torch: its functions are lifted out of
    the source and run here."""

    def test_same_bytes_as_measure_constants(self):
        tree = ast.parse((MILP / "measure_constants.py").read_text())
        wanted = {"kv_dim", "act_bytes_per_token", "weight_bytes_per_layer"}
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        self.assertEqual({f.name for f in funcs}, wanted)
        for key, s in ml.REFERENCE_SHAPES.items():
            ns = dict(HIDDEN=s["H"], INTER=s["I"], N_HEADS=s["heads"], N_KV_HEADS=s["kv"],
                      HEAD_DIM=s["hd"])
            exec(compile(ast.Module(body=funcs, type_ignores=[]), "measure_constants.py", "exec"), ns)
            with self.subTest(model=key):
                self.assertEqual(plp.weight_bytes(s), ns["weight_bytes_per_layer"]())
                for ctx in (64, 320, 575):
                    self.assertEqual(plp.act_bytes(s, ctx), ns["act_bytes_per_token"](ctx))

    def test_weights_match_memory_limits(self):
        for key, s in ml.REFERENCE_SHAPES.items():  # which also counts the norms and Qwen's bias
            extra = 2 * s["H"] + ((s["heads"] + 2 * s["kv"]) * s["hd"] if s["bias"] else 0)
            self.assertAlmostEqual(ml.MemoryModel(s).w * ml.MIB, plp.weight_bytes(s) + 2 * extra, places=3)


class StepGraph(unittest.TestCase):
    def test_two_microbatches_closed_form(self):
        # One-behind publish with free transport: C0 + SUM max(C_s, C_s+1) + C_last.
        rng = random.Random(1)
        for R in (2, 3, 4, 5):
            for _ in range(20):
                C = [rng.uniform(0.5, 5) for _ in range(R)]
                v = zero_values(R)
                for s in range(R):
                    v[f"Cg{s}"] = C[s]
                want = C[0] + sum(max(C[s], C[s + 1]) for s in range(R - 1)) + C[-1]
                self.assertAlmostEqual(end(R, 2, v), want, places=9)
        v = zero_values(4)
        for s, c in enumerate([3.0, 1.0, 3.0, 1.0]):
            v[f"Cg{s}"] = c
        self.assertEqual(end(4, 2, v), 13.0)  # GPipe would say 8 + 3 = 11

    def test_one_microbatch_is_a_serial_chain(self):
        rng = random.Random(2)
        for R in (2, 3, 4):
            v = {t: rng.uniform(0.01, 0.1) for t in SHARED}
            chain = v["STEP"] + v["TOK"]
            for s in range(R):
                v[f"Cg{s}"] = cg = rng.uniform(1, 5)
                v[f"Lc{s}"] = 0.1 * cg  # GPU-bound
                chain += cg
                if s < R - 1:
                    v[f"D{s}"], v[f"M{s}"] = rng.uniform(0, .1), rng.uniform(0, .1)
                    chain += v[f"D{s}"] + v[f"M{s}"] + v["HS"]
                if s > 0:
                    v[f"RECV{s}"] = rng.uniform(0, .1)
                    chain += v[f"RECV{s}"]
            self.assertAlmostEqual(end(R, 1, v), chain, places=9)

    def test_graph_equals_the_harness_program(self):
        # Large CPU, copy and memcpy costs too, so every edge gets to bind somewhere.
        rng = random.Random(4)
        for _ in range(400):
            R, N = rng.randint(2, 5), rng.randint(1, 9)
            v = {k: rng.choice([0.0, rng.uniform(0, 3)]) for k in zero_values(R)}
            for s in range(R):
                v[f"Cg{s}"], v[f"Lc{s}"] = rng.uniform(0.1, 5), rng.uniform(0, 6)
            for sink in ("token", "last"):
                self.assertAlmostEqual(end(R, N, v, sink), simulate_step(R, N, v, sink), places=9)

    def test_last_step_ends_at_the_barrier(self):
        # The last rank's own queued GPU work is outside rank 0's window.
        v = zero_values(4)
        for s, c in enumerate([2.0, 2.0, 2.0, 1.0]):
            v[f"Cg{s}"] = c
        a = end(4, 4, v, "last")
        v["Cg3"] = 50.0
        self.assertEqual(end(4, 4, v, "last"), a)
        self.assertGreater(end(4, 4, v, "token"), 200.0)


class Latency(unittest.TestCase):
    def test_step_groups_cover_every_decode_step(self):
        for T in (1, 2, 3, 512):
            for samples in (None, 1, 8, 1000):
                g = plp.step_groups(64, T, samples)
                with self.subTest(T=T, samples=samples):
                    self.assertTrue(g[0][2] and g[-1][3] == "last")
                    self.assertEqual(sum(w for w, _, _, _ in g[1:]), T)
                    self.assertEqual(g[-1][1], 64 + T - 1)  # step t sees seq + t - 1 cached tokens

    def test_sampled_decode_close_to_every_step(self):
        for model, gb, card, split, b, mb in [
            ("vicuna_7b", (20, 10, 5, 5), "A100-40GB", [14, 10, 5, 3], 32, 8),
            ("mistral_24b", (40, 20, 10, 10), "A100-80GB", [19, 10, 6, 5], 64, 2),
        ]:
            c = cfg(model, gb, card, T=512)
            exact = plp.predict_ms(c, split, b, mb)
            sampled = plp.evaluate(split, plp.prepare(c, b, mb, 8), c)
            self.assertLess(abs(sampled - exact) / exact, 1e-4)

    def test_layers_scale_gpu_and_cpu_terms_only(self):
        shape = ml.REFERENCE_SHAPES["vicuna_7b"]
        bw = plp.slice_peak_memory_bandwidths([20, 10, 5, 5], "A100-40GB")
        tm = plp.terms(shape, bw, plp.DEFAULT_CONSTANTS, 4, 64.0, False, 64)
        for name, (per_layer, _) in tm.items():
            self.assertEqual(per_layer != 0, name[:2] in ("Cg", "Lc"), name)

    def test_prefill_moves_the_whole_prompt(self):
        shape = ml.REFERENCE_SHAPES["vicuna_7b"]
        bw = plp.slice_peak_memory_bandwidths([20, 10, 5, 5], "A100-40GB")
        pre = plp.terms(shape, bw, plp.DEFAULT_CONSTANTS, 4, 64, True, 64)
        dec = plp.terms(shape, bw, plp.DEFAULT_CONSTANTS, 4, 64, False, 64)
        self.assertAlmostEqual(pre["D0"][1], 64 * dec["D0"][1])
        self.assertEqual(pre["STEP"][1], 0.0)
        self.assertGreater(dec["STEP"][1], 0.0)


class Space(unittest.TestCase):
    def test_validate_layout(self):
        bw = plp.slice_peak_memory_bandwidths([40, 20, 10, 10], "A100-80GB")
        self.assertAlmostEqual(bw[0], 2039e9 * 4 / 8)
        self.assertAlmostEqual(bw[3], 2039e9 / 8)
        with self.assertRaises(KeyError):
            plp.slice_peak_memory_bandwidths([20, 10, 5, 5], "A100-80GB")  # no 5GB slice on the 80GB card

    def test_bounds_are_memory_limits(self):
        for gb, card in (((20, 10, 5, 5), "A100-40GB"), ((40, 20, 10, 10), "A100-80GB")):
            for model in ("vicuna_7b", "qwen_7b", "mistral_24b"):
                s = ml.REFERENCE_SHAPES[model]
                for b, mb in ((8, 4), (64, 2), (64, 32)):
                    try:
                        lower, upper = plp.layer_bounds(cfg(model, gb, card, T=512), b, mb)
                    except ValueError:
                        continue
                    mm = ml.MemoryModel(s, 64, 512)
                    want = [min(s["L"], mm.max_layers_that_can_fit(r, 4, b, mb, cap))
                            for r, cap in enumerate(ml.slice_mibs(list(gb)))]
                    self.assertEqual(upper, want)
                    self.assertEqual(lower, [1, 1, 1, 1])

    def test_nothing_fits(self):
        with self.assertRaises(ValueError):
            plp.layer_bounds(cfg("mistral_24b", (20, 10, 5, 5), T=512), 64, 2)

    def test_all_splits(self):
        got = plp.all_splits(14, [1, 1, 1, 0], [10, 6, 5, 3])
        want = [[a, b, c, d] for a in range(1, 11) for b in range(1, 7)
                for c in range(1, 6) for d in range(0, 4) if a + b + c + d == 14]
        self.assertEqual(got, want)


@unittest.skipUnless(HAVE_PULP, "PuLP not installed")
class Milp(unittest.TestCase):
    CASES = [
        ("vicuna_7b", (20, 10, 5, 5), "A100-40GB", 32, 8),
        ("vicuna_7b", (20, 10, 5, 5), "A100-40GB", 64, 2),
        ("qwen_7b", (20, 10, 5, 5), "A100-40GB", 16, 8),
        ("mistral_24b", (40, 20, 10, 10), "A100-80GB", 64, 2),
        ("vicuna_13b", (40, 20, 10, 10), "A100-80GB", 16, 2),
        ("vicuna_7b", (20, 10, 10), "A100-40GB", 32, 16),  # three ranks
        ("mistral_7b", (40, 20, 10, 10), "A100-80GB", 8, 2),
    ]

    def test_matches_brute_force(self):
        for model, gb, card, b, mb in self.CASES:
            c = cfg(model, gb, card)
            split, obj = plp.optimize_placement(c, b, mb)
            best, best_ms, tried = plp.brute_force(c, b, mb)
            with self.subTest(model=model, gb=gb, b=b, mb=mb):
                self.assertGreater(tried, 1)
                self.assertLess(abs(obj - best_ms) / best_ms, 1e-7)  # objective, not split: ties
                self.assertAlmostEqual(plp.evaluate(split, plp.prepare(c, b, mb, 8), c), obj,
                                       delta=1e-6 * obj)

    def test_head_only_last_rank(self):
        c = cfg("qwen_7b", min_last_rank_layers=0)
        split, obj = plp.optimize_placement(c, 8, 4)
        best, best_ms, _ = plp.brute_force(c, 8, 4)
        self.assertLess(abs(obj - best_ms) / best_ms, 1e-7)
        self.assertEqual(sum(split), 28)


class Validate(unittest.TestCase):
    def test_csv_formats_and_model_names(self):
        import validate

        with tempfile.TemporaryDirectory() as d:
            oracle = os.path.join(d, "vicuna_7B.csv")
            with open(oracle, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["20gb", "10gb", "5gb", "5gb", "batch_size", "microbatch_size",
                            "num_microbatches", "peak_rank0_40gb_mb", "peak_rank1_20gb_mb",
                            "peak_rank2_10gb_mb", "peak_rank3_10gb_mb", "total_latency_ms", "status"])
                w.writerow([18, 12, 1, 1, 8, 4, 2, 0, 0, 0, 0, 1000.0, "ok"])
                w.writerow([17, 12, 2, 1, 8, 4, 2, 0, 0, 0, 0, "", "OOM_rank3"])
            rows, gb = validate.load_rows(oracle)
            self.assertEqual(gb, [40, 20, 10, 10])  # the peak columns win over the split header
            self.assertEqual([r["split"] for r in rows], [(18, 12, 1, 1)])
            self.assertEqual(validate.model_from_path(oracle), "vicuna_7b")
            job = os.path.join(d, "gpu2", "01_qwen_7b")
            os.makedirs(job)
            self.assertEqual(validate.model_from_path(os.path.join(job, "results.csv")), "qwen_7b")

            bench = os.path.join(d, "mig_benchmark_results.csv")
            with open(bench, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["split", "batch_size", "microbatch_size", "peak_rank0_20gb_mb",
                            "peak_rank1_10gb_mb", "total_latency_ms", "status"])
                w.writerow(["[18, 12, 1, 1]", 8, 4, 0, 0, 900.0, "ok"])
            rows, gb = validate.load_rows(bench)
            self.assertEqual((gb, rows[0]["split"]), ([20, 10], (18, 12, 1, 1)))

    def test_kendall_tau(self):
        import validate

        self.assertEqual(validate.kendall_tau([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertEqual(validate.kendall_tau([1, 2, 3], [30, 20, 10]), -1.0)


class SliceConstants(unittest.TestCase):
    """A constants set with `slices` (hardware_constants "a100_40gb_pcie_mig") sets each rank's
    GPU time from its own slice's measured rates."""

    def test_decode_layer_time_per_slice(self):
        mig = plp.hardware_constants.CONSTANTS["a100_40gb_pcie_mig"]
        shape = ml.REFERENCE_SHAPES["vicuna_7b"]
        steps = plp.prepare(cfg(constants=mig), 16, 8, 8)
        _, step_terms, _ = steps[1]  # a decode group
        ctx = 64 + (1 + 15 // 8) / 2 - 1  # the first decode range's midpoint context (T = 16)
        for rank, gb in enumerate((20, 10, 5, 5)):
            s = mig["slices"][gb]
            want = (plp.weight_bytes(shape) / s["weight_rate"] + s["layer_fixed_us"] * 1e-6
                    + 8 * plp.act_bytes(shape, ctx) / s["act_rate"]) * 1e3
            self.assertAlmostEqual(step_terms[f"Cg{rank}"][0], want, places=9)
        # 1g.5gb: about 3.3 ms per layer at mb 8 (the probe measured 2.67-3.82 ms over ctx 64-575)
        self.assertTrue(2.6 < step_terms["Cg3"][0] < 3.9)

    def test_without_slices_nothing_changes(self):
        whole = plp.DEFAULT_CONSTANTS
        a = plp.prepare(cfg(constants=whole), 16, 8, 8)
        b = plp.prepare(cfg(constants=dict(whole, slices={})), 16, 8, 8)
        self.assertEqual([t for _, t, _ in a], [t for _, t, _ in b])


class Clock(unittest.TestCase):
    """CONFIG clock="schedutil": slow ranks' launches take clock_slow_factor x as long."""

    MIG = plp.hardware_constants.CONSTANTS["a100_40gb_pcie_mig"]
    CASES = [(8, 4), (16, 8), (32, 8), (16, 2)]

    def test_matches_brute_force_and_binds(self):
        c = cfg(constants=self.MIG, clock="schedutil")
        some_slow = False
        for b, mb in self.CASES:
            split, obj = plp.optimize_placement(c, b, mb)
            best, best_ms, _ = plp.brute_force(c, b, mb)
            steps = plp.prepare(c, b, mb, 8)
            lower, upper = plp.layer_bounds(c, b, mb)
            some_slow |= any(plp.slow_ranks(s, steps, c) for s in plp.all_splits(32, lower, upper))
            with self.subTest(b=b, mb=mb):
                self.assertLess(abs(obj - best_ms) / best_ms, 1e-6)
                self.assertAlmostEqual(plp.evaluate(split, steps, c), obj, delta=1e-6 * obj)
        self.assertTrue(some_slow)  # the clock constraint is not vacuous in these cases

    def test_thresholds_at_the_extremes(self):
        steps = plp.prepare(cfg(), 16, 4, 8)
        split = [18, 9, 3, 2]
        fast = plp.evaluate(split, steps, cfg(constants=self.MIG))
        never = dict(self.MIG, clock_busy_threshold=[0.0] * 4)
        always = dict(self.MIG, clock_busy_threshold=[1.01] * 4)  # a rank is never busy the whole step
        self.assertEqual(plp.slow_ranks(split, steps, cfg(constants=never, clock="schedutil")), set())
        self.assertEqual(plp.evaluate(split, steps, cfg(constants=never, clock="schedutil")), fast)
        self.assertEqual(plp.slow_ranks(split, steps, cfg(constants=always, clock="schedutil")), {0, 1, 2, 3})
        all_slow = sum(w * plp.longest_path(e, o, plp.term_ms_for(split, t, {0, 1, 2, 3}, 2.26))
                       for w, t, (e, o) in steps) + self.MIG["barrier_us"] * 1e-3
        self.assertAlmostEqual(plp.evaluate(split, steps, cfg(constants=always, clock="schedutil")), all_slow)
        self.assertGreater(all_slow, fast)


def fake_primitives(scale):
    """A made-up slice holding every primitive the composition reads (probe v2 key names);
    GPU rates x scale (the copies to and from the host are not scaled)."""
    p = dict(gemm_decode_fixed_us=8.0, gemm_decode_rate=6e11, gemm_prefill_fixed_us=12.0, gemm_prefill_rate=6e13,
             attention_mha_h32_kv32_d128_fixed_us=10.0, attention_mha_h32_kv32_d128_rate=3e11,
             attention_gqa_h28_kv4_d128_fixed_us=10.0, attention_gqa_h28_kv4_d128_rate=3e11,
             kv_append_kv32_fixed_us=6.0, kv_append_kv32_rate=4e11, kv_append_kv4_fixed_us=6.0, kv_append_kv4_rate=4e11,
             elementwise_fixed_us=3.0, elementwise_rate=5e11, device_copy_fixed_us=3.0, device_copy_rate=5e11,
             d2h_fixed_us=15.0, d2h_rate=1.3e9, h2d_fixed_us=12.0, h2d_rate=2.0e9)
    return {k: v * scale if k.endswith("_rate") and not k.startswith(("d2h", "h2d")) else v for k, v in p.items()}


# CPU numbers from any measured set; slices hold primitives, as probe_constants.py prints them
PROBED = dict(plp.DEFAULT_CONSTANTS, slices={20: fake_primitives(1.0), 10: fake_primitives(0.5), 5: fake_primitives(0.25)})


class ComposedFromPrimitives(unittest.TestCase):
    """Slices holding probe primitives: every GPU time is composed from them (layer_ops), and the
    launches are counted from the same kernel lists."""

    def test_terms_per_rank(self):
        shape = ml.REFERENCE_SHAPES["vicuna_7b"]
        steps = plp.prepare(cfg(constants=PROBED), 16, 8, 8)
        launch_ms = PROBED["launch_us"] * 1e-3
        for (_, t, _), seq, ctx in ((steps[0], 64, 64), (steps[1], None, 64 + (1 + 15 // 8) / 2 - 1)):
            for rank, gb in enumerate((20, 10, 5, 5)):
                p = PROBED["slices"][gb]
                layer = plp.layer_ops.layer_us(plp.layer_ops.layer_ops_from_shape(shape, 8, ctx, seq), p)
                extras = plp.layer_ops.extras_from_shape(shape, 8, ctx, seq)
                roles = ["rotary"] + ["embed"] * (rank == 0) + (["clone"] if rank < 3 else ["head"])
                fixed = [plp.layer_ops.layer_us(extras[role], p) for role in roles]
                with self.subTest(prefill=seq, rank=rank):
                    self.assertAlmostEqual(t[f"Cg{rank}"][0], layer[0] * 1e-3, places=12)
                    self.assertAlmostEqual(t[f"Cg{rank}"][1], sum(f[0] for f in fixed) * 1e-3, places=12)
                    self.assertEqual(round(t[f"Lc{rank}"][0] / launch_ms), 42 if seq else 40)
                    calls = sum(f[2] for f in fixed) + 2 + 2 * (rank < 3)  # + timing events, send queue calls
                    self.assertAlmostEqual(t[f"Lc{rank}"][1], calls * launch_ms, places=12)
        decode, act = steps[1][1], 8 * 4096 * 2
        p20, p10 = PROBED["slices"][20], PROBED["slices"][10]
        self.assertAlmostEqual(decode["D0"][1], (p20["d2h_fixed_us"] + act / p20["d2h_rate"] * 1e6) * 1e-3)
        recv_s = (act / PROBED["shm_to_pinned_bw"] + (p10["h2d_fixed_us"] + act / p10["h2d_rate"] * 1e6) * 1e-6
                  + PROBED["launch_us"] * 1e-6 + PROBED["handshake_ack_us"] * 0.5e-6)
        self.assertAlmostEqual(decode["RECV1"][1], recv_s * 1e3)

    def test_loaded_slowdown_in_the_ilp_only(self):
        """A slice's concurrency factors (probe v6) scale the ILP's composed GPU time per op kind;
        layer_us composes alone unless asked (step 3)."""
        conc = dict(factors=dict(gemm=1.10, attention=1.20, small=1.30))
        loaded = dict(PROBED, slices={gb: dict(p, concurrency=conc) for gb, p in PROBED["slices"].items()})
        shape, p = ml.REFERENCE_SHAPES["vicuna_7b"], PROBED["slices"][20]
        ops = plp.layer_ops.layer_ops_from_shape(shape, 8, 64)
        alone = plp.layer_ops.layer_us(ops, p)[1]
        want = sum(us * {"gemm": 1.10, "kv_append": 1.10, "attention": 1.20}.get(kind, 1.30) for kind, us in alone.items())
        self.assertAlmostEqual(plp.layer_ops.layer_us(ops, dict(p, concurrency=conc))[0], sum(alone.values()))
        self.assertAlmostEqual(plp.layer_ops.layer_us(ops, dict(p, concurrency=conc), loaded=True)[0], want)
        t_alone = plp.prepare(cfg(constants=PROBED), 16, 8, 8)[1][1]
        t_loaded = plp.prepare(cfg(constants=loaded), 16, 8, 8)[1][1]
        self.assertGreater(t_loaded["Cg0"][0], 1.09 * t_alone["Cg0"][0])
        self.assertEqual(t_loaded["Lc0"], t_alone["Lc0"])  # launches are CPU work: unchanged

    def test_gemms_are_weight_streaming(self):
        """With every primitive but the decode GEMM free, a decode layer is its weights over the GEMM
        rate: the same bytes as weight_bytes()."""
        prims = {k: 0.0 if k.endswith("_fixed_us") else 1e30 for k in fake_primitives(1.0)}
        prims["gemm_decode_rate"] = 6e11
        for model in ("vicuna_7b", "qwen_7b"):
            shape = ml.REFERENCE_SHAPES[model]
            us = plp.layer_ops.layer_us(plp.layer_ops.layer_ops_from_shape(shape, 4, 300), prims)[0]
            self.assertAlmostEqual(us * 1e-6, plp.weight_bytes(shape) / 6e11, delta=1e-12)

    @unittest.skipUnless(HAVE_PULP, "needs PuLP")
    def test_milp_matches_brute_force(self):
        for model, b, mb in (("vicuna_7b", 32, 8), ("vicuna_7b", 8, 2), ("qwen_7b", 16, 8)):
            c = cfg(model, constants=PROBED)
            split, obj = plp.optimize_placement(c, b, mb)
            best, best_ms, tried = plp.brute_force(c, b, mb)
            with self.subTest(model=model, b=b, mb=mb):
                self.assertGreater(tried, 1)
                self.assertLess(abs(obj - best_ms) / best_ms, 1e-7)

    def test_no_clock_model_yet(self):
        with self.assertRaises(ValueError):
            plp.prepare(cfg(constants=PROBED, clock="schedutil"), 16, 8, 8)


def simulate_busy(R, N, v, sink, step_ms, tok_half_ms):
    """Each rank's CPU busy time, tallied while walking its program in order (as simulate_step
    does): RECV, launches and FLUSH's spin + memcpy + handshake are busy, the waits for upstream
    handshakes sleep; STEP on every rank; the last rank's token D2H spins on its GPU."""
    upstream, busy = None, []
    for s in range(R):
        last = s == R - 1
        cpu = v["STEP"] if s == 0 else 0.0
        b = v["STEP"] if s == 0 else step_ms
        gpu = copy = 0.0
        landed, published = [0.0] * N, [0.0] * N
        for k in range(N):
            if s > 0:
                cpu = max(cpu, upstream[k]) + v[f"RECV{s}"]  # the wait before it sleeps
                b += v[f"RECV{s}"]
            start = cpu
            gpu = max(max(start, gpu) + v[f"Cg{s}"], start + v[f"Lc{s}"])
            cpu = start + v[f"Lc{s}"]
            b += v[f"Lc{s}"]
            if not last:
                copy = max(gpu, copy) + v[f"D{s}"]
                landed[k] = copy
                if k >= 1:
                    b += max(0.0, landed[k - 1] - cpu) + v[f"M{s}"] + v["HS"]  # spin on D2H(k-1)
                    cpu = max(cpu, landed[k - 1]) + v[f"M{s}"] + v["HS"]
                    published[k - 1] = cpu
        if not last:
            b += max(0.0, landed[N - 1] - cpu) + v[f"M{s}"] + v["HS"]
            cpu = max(cpu, landed[N - 1]) + v[f"M{s}"] + v["HS"]
            published[N - 1] = cpu
            upstream = published
            if s == 0 and sink == "token":
                b += tok_half_ms
        elif sink == "token":
            b += max(0.0, gpu - cpu) + tok_half_ms
        busy.append(b)
    return busy


CLOCKED = dict(PROBED, clock_slow_factor=2.3, clock_threshold_by_period={40: 0.4, 120: 0.3})


class ProbeClock(unittest.TestCase):
    """Probe sets with clock="schedutil": CPU busy time from the step graph, the governor probe's
    threshold, exact search over the splits."""

    def test_cpu_busy_matches_program_order(self):
        rng = random.Random(7)
        constants = dict(step_start_us=80.0, token_staging_us=56.0)
        for _ in range(300):
            R, N, sink = rng.choice([2, 3, 4]), rng.choice([1, 2, 3, 5, 8]), rng.choice(["token", "last"])
            v = zero_values(R)
            for key in v:
                v[key] = rng.choice([0.0, rng.uniform(0.05, 1.0), rng.uniform(1.0, 30.0)])
            edges, order = plp.step_graph(R, N, sink)
            times = plp.event_times(edges, order, v)
            got = plp.cpu_busy_ms(times, v, R, N, constants, sink == "token")
            want = simulate_busy(R, N, v, sink, 0.08, 0.028 if sink == "token" else 0.0)
            with self.subTest(R=R, N=N, sink=sink):
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, places=9)

    def test_threshold_between_periods(self):
        self.assertEqual(plp.busy_threshold(CLOCKED, 20.0), 0.4)
        self.assertAlmostEqual(plp.busy_threshold(CLOCKED, 80.0), 0.35)
        self.assertEqual(plp.busy_threshold(CLOCKED, 200.0), 0.3)

    def test_slow_set_is_consistent(self):
        """Thresholds 0 / above 1 slow nobody / everybody; in between, every rank left at full
        clock is busy enough in the schedule of the returned slow set."""
        split = [20, 8, 3, 1]
        for b, mb in ((16, 8), (32, 4)):
            steps = plp.prepare(cfg(constants=CLOCKED, clock="schedutil"), b, mb, 8)
            never = dict(CLOCKED, clock_threshold_by_period={40: 0.0, 120: 0.0})
            always = dict(CLOCKED, clock_threshold_by_period={40: 1.01, 120: 1.01})
            self.assertEqual(plp.slow_ranks(split, steps, cfg(constants=never, clock="schedutil")), set())
            self.assertEqual(plp.slow_ranks(split, steps, cfg(constants=always, clock="schedutil")), {0, 1, 2, 3})
            c = cfg(constants=CLOCKED, clock="schedutil")
            slow = plp.slow_ranks(split, steps, c)
            busy, total, n_steps = [0.0] * 4, 0.0, 0.0
            for weight, step_terms, (edges, order) in steps:
                term_ms = plp.term_ms_for(split, step_terms, slow, CLOCKED["clock_slow_factor"])
                times = plp.event_times(edges, order, term_ms)
                total += weight * times["END"]
                n_steps += weight
                for r, x in enumerate(plp.cpu_busy_ms(times, term_ms, 4, plp.num_microbatches_of(order), CLOCKED,
                                                      plp._ends_with_token(edges))):
                    busy[r] += weight * x
            threshold = plp.busy_threshold(CLOCKED, total / n_steps)
            with self.subTest(b=b, mb=mb, slow=slow):
                for r in set(range(4)) - slow:
                    self.assertGreaterEqual(busy[r], threshold * total)

    def test_exact_search_picks_the_fastest_split(self):
        c = cfg(constants=CLOCKED, clock="schedutil")
        split, obj = plp.optimize_placement(c, 16, 8)
        steps = plp.prepare(c, 16, 8, 8)
        lower, upper = plp.layer_bounds(c, 16, 8)
        everything = [plp.evaluate(s, steps, c) for s in plp.all_splits(32, lower, upper)]
        self.assertAlmostEqual(obj, min(everything), places=9)
        self.assertAlmostEqual(plp.evaluate(split, steps, c), obj, places=9)

    def test_pool_term_only_above_the_grain(self):
        pooled = dict(PROBED, recv_copy_pool_us=6600.0, pool_launch_slowdown=1.1, pool_min_values=32768)
        for mb, expect_pool in ((8, False), (16, True)):  # 8 x 4096 = 32768 (not above), 16 x 4096 above
            plain = plp.prepare(cfg(constants=PROBED), 32, mb, 8)[1][1]
            with_pool = plp.prepare(cfg(constants=pooled), 32, mb, 8)[1][1]
            with self.subTest(mb=mb):
                if expect_pool:
                    self.assertAlmostEqual(with_pool["Lc0"][0], 1.1 * plain["Lc0"][0])
                    self.assertGreater(with_pool["RECV1"][1], plain["RECV1"][1] + 6.6 - 0.01)
                else:
                    self.assertEqual(with_pool, plain)


class ProbeConstants(unittest.TestCase):
    """probe_constants.py: a box's probe folder -> the block pasted into hardware_constants.py."""

    def test_clock_and_pool_from_the_cpu_probe(self):
        import probe_constants as pc
        inter = dict(governor=dict(speed={"40": {"0.1": 0.44, "0.2": 0.43, "0.3": 0.42, "0.4": 1.0, "0.5": 0.98},
                                          "120": {"0.1": 0.45, "0.2": 1.02, "0.3": 0.99, "0.4": 1.0}}),
                     interference=[dict(copier=[65536, 1, 25, None], dispatch_gap_ms=0, slowdown=1.09,
                                        copiers=[[dict(median=6600.0)], [dict(median=6500.0)]])])
        factor, thresholds = pc.clock_from_probe(inter)
        self.assertAlmostEqual(factor, 1 / 0.435, places=6)  # median of the 4 slow cells
        self.assertEqual(thresholds, {40: 0.4, 120: 0.2})
        self.assertEqual(pc.pool_from_probe(inter), (6550.0, 1.09))
        self.assertEqual(pc.pool_from_probe(dict(interference={"old": {}})), (None, None))

    def test_folder_to_probe_set(self):
        """A box's probe folder -> a set with every fit and the first pass's points (time + shape),
        written as JSON and read back with int keys, as hardware_constants.py loads it."""
        import json
        import probe_constants as pc
        cpu = {k: dict(median=float(PROBED[k]), spread=0.01, windows=[float(PROBED[k])])
               for k in pc.CPU_KEYS if k not in ("kernels_per_layer", "rotary_kernels")}
        cpu.update(kernels_per_layer=40, rotary_kernels=9, host=dict(cpu="test", governor="schedutil"))
        point = dict(units=33554432, us=59.4, method="kernel", residual=0.01, N=4096, K=4096, M=1, replays_us=[59.4])
        mig = [dict(fake_primitives(scale), gb=gb, profile=f"{g}g.{gb}gb", gemm_decode_points=[point],
                    repeat=fake_primitives(scale), validation={})
               for gb, g, scale in ((20, 3, 1.0), (10, 2, 0.5), (5, 1, 0.25))]
        inter = dict(governor=dict(speed={"40": {"0.2": 0.43, "0.5": 1.0}, "120": {"0.2": 1.0, "0.5": 1.0}}),
                     interference=[dict(copier=[65536, 1, 25, None], dispatch_gap_ms=0, slowdown=1.1,
                                        copiers=[[dict(median=6600.0)]])])
        with tempfile.TemporaryDirectory() as d:
            for name, obj in (("constants_cpu", cpu), ("constants_mig", mig), ("probe_cpu_interactions", inter)):
                with open(f"{d}/{name}.json", "w") as f:
                    json.dump(obj, f)
            consts = pc.from_probes(d)
            pc.save(consts, f"{d}/box.json")
            back = pc.load(f"{d}/box.json")
        trimmed = dict(units=33554432, us=59.4, N=4096, K=4096, M=1)  # raw extras dropped
        self.assertEqual(consts["slices"][10], dict(fake_primitives(0.5), gemm_decode_points=[trimmed]))
        self.assertEqual((consts["kernels_per_layer"], consts["launch_us"]), (40, PROBED["launch_us"]))
        self.assertEqual(consts["clock_threshold_by_period"], {40: 0.5, 120: 0.2})
        self.assertEqual((consts["recv_copy_pool_us"], consts["pool_launch_slowdown"]), (6600.0, 1.1))
        self.assertEqual(back, consts)  # the JSON round trip, keys and all


if __name__ == "__main__":
    unittest.main()
