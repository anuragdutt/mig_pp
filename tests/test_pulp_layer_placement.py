"""Tests for mig_pp/milp/pulp_layer_placement.py and validate.py. Torch-free; the MILP
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
import pulp_layer_placement as plp  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
