"""Torch-free tests for layer_limits.py and memory_limits.py.

Run locally:
    python3 -m unittest discover -s tests -p 'test_memory_limits.py' -v
"""

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiment_config as ec  # noqa: E402
import layer_limits as ll  # noqa: E402
import memory_limits as ml  # noqa: E402
import parallel_plan as pp  # noqa: E402

LAYOUT = [20, 10, 5, 5]
LAYOUT_80 = [40, 20, 10, 10]
RESULTS_928 = ROOT / "mig_benchmark_results_9_28.csv"
PROFILE = ROOT / "mig_pp" / "profiles" / "poc_layer_memory.csv"
# The April 80GB runs sit under oracle/80gb/20_10_5_5/ although their slices
# were 40/20/10/10 (their columns say so: peak_rank0_40gb_mb, ...).
ORACLE_80 = ROOT / "oracle" / "80gb" / "20_10_5_5"


def oracle_splits(path):
    # The header repeats a slice size, which DictReader would collapse: read
    # the split columns positionally.
    with open(path) as f:
        return {tuple(int(x) for x in r[:4]) for r in list(csv.reader(f))[1:] if r}


class LayerLimitsFile(unittest.TestCase):
    def test_entries_well_formed(self):
        for layout, models in ll.LAYER_LIMITS.items():
            n = len(layout.split("_"))
            for key, limits in models.items():
                with self.subTest(layout=layout, model=key):
                    self.assertEqual(len(limits), n)
                    self.assertTrue(all(isinstance(x, int) and x >= 0 for x in limits))

    def test_every_entry_enumerates_splits(self):
        # A limit vector with no valid split would make `check` fail for
        # that model; catch it here instead.
        for layout, models in ll.LAYER_LIMITS.items():
            gb = [int(x) for x in layout.split("_")]
            for key in models:
                limits, min_last = ll.limits_for(key, gb)
                layers = ml.REFERENCE_SHAPES[key]["L"]
                with self.subTest(model=key):
                    self.assertTrue(
                        ec.generate_layer_splits(layers, limits, gb, True, min_last)
                    )

    def test_head_only_models_exist(self):
        every = {k for m in ll.LAYER_LIMITS.values() for k in m}
        self.assertLessEqual(ll.HEAD_ONLY_LAST_RANK, every)

    def test_lookup_errors_name_the_problem(self):
        with self.assertRaises(KeyError) as cm:
            ll.limits_for("vicuna_13b", [20, 20, 20])
        self.assertIn("20_20_20", str(cm.exception))
        with self.assertRaises(KeyError) as cm:
            ll.limits_for("vicuna_7b", LAYOUT_80)  # known layout, model not in it
        self.assertIn("vicuna_7b", str(cm.exception))
        with self.assertRaises(KeyError) as cm:
            ll.limits_for("gemma_7b", LAYOUT)
        self.assertIn("gemma_7b", str(cm.exception))

    def test_head_only_flag(self):
        self.assertEqual(ll.limits_for("qwen_14b", LAYOUT), ([30, 15, 7, 0], 0))
        self.assertEqual(ll.limits_for("llama_7b", LAYOUT), ([18, 12, 5, 5], 1))

    def test_7b_class_shares_the_vicuna_split_set(self):
        s = {
            k: ec.generate_layer_splits(32, ll.limits_for(k, LAYOUT)[0], LAYOUT)
            for k in ("vicuna_7b", "llama_7b", "mistral_7b")
        }
        self.assertEqual(s["vicuna_7b"], s["llama_7b"])
        self.assertEqual(s["vicuna_7b"], s["mistral_7b"])
        self.assertEqual(len(s["vicuna_7b"]), 67)


class Layout80GB(unittest.TestCase):
    """40/20/10/10 on A100-80GB, sized to a ~2.5-day sweep."""

    def splits(self, key):
        limits, min_last = ll.limits_for(key, LAYOUT_80)
        return ec.generate_layer_splits(
            ml.REFERENCE_SHAPES[key]["L"], limits, LAYOUT_80, True, min_last
        )

    def test_split_counts_fit_the_budget(self):
        # x 14 batch pairs: ~43 h per lane estimated against a 60 h budget.
        # Changing a vector changes the run time: re-check the budget.
        counts = {k: len(self.splits(k)) for k in ll.LAYER_LIMITS["40_20_10_10"]}
        self.assertEqual(counts, {"vicuna_13b": 33, "llama_13b": 33, "qwen_14b": 31, "mistral_24b": 29})

    def test_13b_models_share_a_split_set(self):
        self.assertEqual(self.splits("llama_13b"), self.splits("vicuna_13b"))

    def test_every_slice_holds_layers(self):
        # Small slices up to their B64 capacity; none head-only on this layout.
        for key in ll.LAYER_LIMITS["40_20_10_10"]:
            self.assertTrue(all(min(s) >= 1 for s in self.splits(key)), key)

    def test_april_13b_splits_are_a_subset_not_the_sweep(self):
        # The April [24, 10, 5, 5] sweep reached rank 0 up to 24; the new vector
        # trades that end for more layers on the 10GB slices. Overlap keeps a
        # cross-check against the April rows.
        path = ORACLE_80 / "vicuna_13B.csv"
        if not path.exists():
            self.skipTest("oracle CSV not present")
        ours, april = {tuple(s) for s in self.splits("vicuna_13b")}, oracle_splits(path)
        self.assertEqual(len(april), 22)
        self.assertTrue(ours & april)
        self.assertGreater(max(s[3] for s in ours), max(s[3] for s in april))

    def test_slice_capacities(self):
        self.assertEqual(ml.slice_mibs(LAYOUT_80), [40192, 19968, 9728, 9728])
        self.assertEqual(ml.slice_mibs(LAYOUT), [20096, 9984, 4864, 4864])

    def test_every_config_fits(self):
        # The April 13B sweep had no OOM at any of the 14 pairs; the model agrees
        # and extends that to the two GQA models.
        pairs = pp.SWEEP_DEFAULTS["batch_mb_pairs"]
        for key in ll.LAYER_LIMITS["40_20_10_10"]:
            mm = ml.MemoryModel(ml.REFERENCE_SHAPES[key])
            for s in self.splits(key):
                for b, mb in pairs:
                    self.assertEqual(mm.oom_ranks(s, b, mb, LAYOUT_80), [], (key, s, b, mb))


class MemoryModelShapes(unittest.TestCase):
    def test_weights_match_repo_profile(self):
        # mig_pp/profiles/poc_layer_memory.csv was measured on Vicuna-7B.
        if not PROFILE.exists():
            self.skipTest("profile CSV not present")
        with open(PROFILE) as f:
            rows = list(csv.DictReader(f))
        block = next(int(r["param_bytes"]) for r in rows if r["component"] == "decoder_block")
        embed_head = next(int(r["param_bytes"]) for r in rows if r["layer_id"] == "-1")
        mm = ml.MemoryModel(ml.REFERENCE_SHAPES["vicuna_7b"])
        self.assertEqual(round(mm.w * ml.MIB), block)
        self.assertEqual(round(2 * mm.emb * ml.MIB), embed_head)
        # kv_bytes_peak there is one sequence at 64 + 512 tokens.
        kv = next(int(r["kv_bytes_peak"]) for r in rows if r["component"] == "decoder_block")
        self.assertEqual(round(mm.kv_seq * ml.MIB), kv)

    def test_shape_from_config(self):
        qwen = ml.shape_from_config({
            "model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584,
            "intermediate_size": 18944, "num_attention_heads": 28,
            "num_key_value_heads": 4, "vocab_size": 152064,
        })
        self.assertEqual(qwen, ml.REFERENCE_SHAPES["qwen_7b"])
        llama = ml.shape_from_config({
            "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
            "intermediate_size": 11008, "num_attention_heads": 32, "vocab_size": 32000,
        })
        self.assertEqual(llama, ml.REFERENCE_SHAPES["llama_7b"])
        # Mistral-Small-24B: explicit head_dim that is not hidden / heads.
        m24 = ml.shape_from_config({
            "model_type": "mistral", "num_hidden_layers": 40, "hidden_size": 5120,
            "intermediate_size": 32768, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 131072,
        })
        self.assertEqual(m24["hd"], 128)


class Calibration(unittest.TestCase):
    def test_predicts_928_outcomes(self):
        if not RESULTS_928.exists():
            self.skipTest("9/28 results CSV not present")
        right, n = ml.validate(str(RESULTS_928), "vicuna_7b", LAYOUT)
        self.assertEqual(n, 938)
        self.assertGreaterEqual(right, 930)

    def test_rule_reproduces_the_vicuna_vector(self):
        mm = ml.MemoryModel(ml.REFERENCE_SHAPES["vicuna_7b"])
        caps = {
            b: [mm.max_layers(r, 4, b, b // 2, ml.slice_mib_for(LAYOUT[r])) for r in range(4)]
            for b in (32, 64)
        }
        self.assertEqual([caps[64][0] - 1] + [c - 1 for c in caps[32][1:]], [18, 12, 5, 5])

    def test_qwen_last_rank_capped_by_lm_head_build(self):
        for key, cap in (("qwen_7b", 3), ("qwen_14b", 0)):
            mm = ml.MemoryModel(ml.REFERENCE_SHAPES[key])
            for b in (8, 64):
                self.assertEqual(mm.max_layers(3, 4, b, 2, ml.slice_mib_for(5)), cap, (key, b))

    def test_13b_cannot_fit_at_batch_32(self):
        mm = ml.MemoryModel(ml.REFERENCE_SHAPES["vicuna_13b"])
        caps = [mm.max_layers(r, 4, 32, 2, ml.slice_mib_for(LAYOUT[r])) for r in range(4)]
        self.assertLess(sum(caps), 40)
        caps16 = [mm.max_layers(r, 4, 16, 2, ml.slice_mib_for(LAYOUT[r])) for r in range(4)]
        self.assertGreaterEqual(sum(caps16), 40)


if __name__ == "__main__":
    unittest.main()
