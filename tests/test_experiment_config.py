"""Torch-free tests for experiment_config.py and model_family.py.

Run locally (no torch needed):
    python3 -m unittest discover -s tests -p 'test_experiment_config.py' -v

Gate for the parallel-run changes to the benchmark: the benchmark's
generate_layer_splits() now delegates to experiment_config, so the
equivalence test below compares against a frozen copy of the original.
"""

import itertools
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiment_config as ec  # noqa: E402
import model_family as mf  # noqa: E402

# Verbatim copy of benchmark_pipeline_microbatching.generate_layer_splits at
# cf16b9e, before it delegated to experiment_config. It reads module globals,
# so each call execs it with those globals injected. Do not edit — it is the
# reference the new implementation must reproduce exactly (same splits, same
# order: smoke runs take the last one).
_ORIGINAL_SRC = '''
def generate_layer_splits():
    valid_splits = []
    n = WORLD_SIZE

    def _ok(prev_idx, prev_layers, layers):
        """Ordering constraint between rank prev_idx and the next rank."""
        if not ENFORCE_SLICE_ORDERING:
            return True
        if SLICE_GB[prev_idx] == SLICE_GB[prev_idx + 1]:
            return prev_layers >= layers
        return prev_layers > layers

    def _recurse(rank, assigned, remaining):
        # Last rank takes whatever is left — no need to enumerate it.
        if rank == n - 1:
            if not (1 <= remaining <= LAYER_LIMITS[rank]):
                return
            if assigned and not _ok(rank - 1, assigned[-1], remaining):
                return
            valid_splits.append(assigned + [remaining])
            return

        # Leave at least one layer for each rank still to come.
        max_here = min(LAYER_LIMITS[rank], remaining - (n - rank - 1))
        for count in range(1, max_here + 1):
            if assigned and not _ok(rank - 1, assigned[-1], count):
                continue
            _recurse(rank + 1, assigned + [count], remaining - count)

    _recurse(0, [], TOTAL_LAYERS)
    return valid_splits
'''


# Verbatim copy of generate_layer_splits on origin/qwen-14b-4mig (7acbbbd):
# 4 ranks only, last rank may hold 0 layers. min_last_rank_layers=0 must
# reproduce it exactly.
_QWEN14B_SRC = '''
def generate_layer_splits():
    valid_splits = []
    for l0 in range(1, LAYER_LIMITS[0] + 1):
        for l1 in range(1, LAYER_LIMITS[1] + 1):
            for l2 in range(1, LAYER_LIMITS[2] + 1):
                l3 = TOTAL_LAYERS - (l0 + l1 + l2)
                # Enforce that bigger MIG instances always get more layers than smaller ones.
                # 20GB (l0) > 10GB (l1) > 5GB (l2) >= 5GB (l3).
                # The last two are both 5GB so they're allowed to be equal.
                if 0 <= l3 <= LAYER_LIMITS[3] and l0 > l1 > l2 >= l3:
                    valid_splits.append([l0, l1, l2, l3])
    return valid_splits
'''


def qwen14b_splits(total, limits):
    g = {"TOTAL_LAYERS": total, "LAYER_LIMITS": list(limits)}
    exec(_QWEN14B_SRC, g)
    return g["generate_layer_splits"]()


def original_splits(total, limits, slice_gb, ordering):
    g = {
        "WORLD_SIZE": len(limits),
        "TOTAL_LAYERS": total,
        "LAYER_LIMITS": list(limits),
        "SLICE_GB": list(slice_gb),
        "ENFORCE_SLICE_ORDERING": ordering,
    }
    exec(_ORIGINAL_SRC, g)
    return g["generate_layer_splits"]()


def valid_job(**over):
    job = {
        "schema": 1,
        "model_path": "/models/vicuna-7b-v1.5",
        "model_type": "llama",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "seq_len": 64,
        "max_new_tokens": 512,
        "max_runs": None,
        "batch_mb_pairs": [[8, 4], [64, 2]],
        "mig_uuids": [f"MIG-{i:08x}-0000-0000-0000-000000000000" for i in range(4)],
        "slice_gb": [20, 10, 5, 5],
        "layer_limits": [18, 12, 5, 5],
        "enforce_slice_ordering": True,
        "splits": None,
        "master_port": 29530,
        "shm_prefix": "migpp_20260929_161500_g3j0",
        "mem_monitor": "nvml",
    }
    job.update(over)
    return job


class SplitEquivalence(unittest.TestCase):
    # Real topologies from this repo's branches, plus edge cases.
    REAL = [
        (32, [18, 12, 5, 5], [20, 10, 5, 5]),  # vicuna-7b, this branch
        (32, [22, 14, 7, 7], [20, 10, 5, 5]),
        (32, [18, 9, 5, 4], [20, 10, 5, 5]),  # llama-7b / mistral-7b
        (28, [16, 8, 4, 2], [20, 10, 5, 5]),  # qwen2.5-7b
        (40, [22, 10, 5, 5], [20, 10, 5, 5]),  # llama-2-13b
        (48, [30, 15, 7, 0], [20, 10, 5, 5]),  # qwen2.5-14b (limit 0 => none)
        (40, [30, 10, 4, 2], [20, 10, 5, 5]),  # mistral-24b on 80GB
        (32, [20, 14, 14], [20, 10, 10]),  # 3-slice layout
        (32, [32], [40]),  # one slice
        (5, [1, 1, 1], [10, 10, 10]),  # infeasible
    ]

    def test_real_topologies_match_original(self):
        for total, limits, gb in self.REAL:
            for ordering in (True, False):
                with self.subTest(total=total, limits=limits, ordering=ordering):
                    self.assertEqual(
                        ec.generate_layer_splits(total, limits, gb, ordering),
                        original_splits(total, limits, gb, ordering),
                    )

    def test_exhaustive_small_space_matches_original(self):
        gb_layouts = [[20, 10, 5, 5], [10, 10, 10], [20, 10], [5, 5, 5, 5]]
        for gb in gb_layouts:
            for limits in itertools.product(range(0, 6), repeat=len(gb)):
                for total in range(1, 12):
                    for ordering in (True, False):
                        self.assertEqual(
                            ec.generate_layer_splits(total, limits, gb, ordering),
                            original_splits(total, limits, gb, ordering),
                            msg=f"total={total} limits={limits} gb={gb} ord={ordering}",
                        )

    def test_first_config_of_928_sweep(self):
        # mig_benchmark_results_9_28.csv row 1 is split [12, 10, 5, 5]; the
        # memory-trace label s18_12_1_1 is the last split.
        s = ec.generate_layer_splits(32, [18, 12, 5, 5], [20, 10, 5, 5])
        self.assertEqual(s[0], [12, 10, 5, 5])
        self.assertEqual(s[-1], [18, 12, 1, 1])
        self.assertEqual(
            ec.front_loaded_split(32, [18, 12, 5, 5], [20, 10, 5, 5]), [18, 12, 1, 1]
        )

    def test_front_loaded_split_none_when_infeasible(self):
        self.assertIsNone(ec.front_loaded_split(48, [30, 15, 7, 0], [20, 10, 5, 5]))

    def test_min_last_zero_matches_qwen14b_branch(self):
        gb = [20, 10, 5, 5]
        cases = [(48, [30, 15, 7, 0])]
        cases += [
            (t, lim)
            for lim in itertools.product(range(1, 7), range(1, 6), range(1, 5), range(0, 4))
            for t in range(4, 16)
        ]
        for total, limits in cases:
            self.assertEqual(
                ec.generate_layer_splits(total, limits, gb, True, min_last_rank_layers=0),
                qwen14b_splits(total, limits),
                msg=f"total={total} limits={limits}",
            )

    def test_min_last_zero_covers_qwen14b_oracle(self):
        # Every split the Qwen2.5-14B oracle measured must be reachable.
        oracle = ROOT / "oracle" / "40gb" / "20_10_5_5" / "qwen_14B.csv"
        if not oracle.exists():
            self.skipTest("oracle CSV not present")
        import csv

        # The header repeats "5gb", which DictReader would collapse — read
        # the split columns positionally.
        with open(oracle) as f:
            rows = list(csv.reader(f))[1:]
        seen = {tuple(int(x) for x in r[:4]) for r in rows if r}
        ours = {
            tuple(s)
            for s in ec.generate_layer_splits(
                48, [30, 15, 7, 0], [20, 10, 5, 5], True, min_last_rank_layers=0
            )
        }
        self.assertTrue(seen, "oracle has no rows")
        self.assertLessEqual(seen, ours)

    def test_min_last_default_is_original(self):
        self.assertEqual(
            ec.generate_layer_splits(32, [18, 12, 5, 5], [20, 10, 5, 5]),
            ec.generate_layer_splits(
                32, [18, 12, 5, 5], [20, 10, 5, 5], True, min_last_rank_layers=1
            ),
        )


class JobValidation(unittest.TestCase):
    def test_valid_job_passes(self):
        ec.validate_job(valid_job())

    def test_reports_every_problem(self):
        job = valid_job(num_layers=True, master_port=80, shm_prefix="bad/prefix")
        del job["seq_len"]
        with self.assertRaises(ec.JobConfigError) as cm:
            ec.validate_job(job)
        msg = str(cm.exception)
        for fragment in ("num_layers", "master_port", "shm_prefix", "missing 'seq_len'"):
            self.assertIn(fragment, msg)

    def test_bool_is_not_an_int(self):
        with self.assertRaises(ec.JobConfigError):
            ec.validate_job(valid_job(max_new_tokens=True))

    def test_length_mismatch(self):
        with self.assertRaises(ec.JobConfigError) as cm:
            ec.validate_job(valid_job(layer_limits=[18, 12, 5]))
        self.assertIn("layer_limits", str(cm.exception))

    def test_duplicate_uuid(self):
        u = "MIG-aaaaaaaa-0000-0000-0000-000000000000"
        with self.assertRaises(ec.JobConfigError) as cm:
            ec.validate_job(valid_job(mig_uuids=[u, u, u + "1", u + "2"]))
        self.assertIn("twice", str(cm.exception))

    def test_indivisible_batch(self):
        with self.assertRaises(ec.JobConfigError):
            ec.validate_job(valid_job(batch_mb_pairs=[[8, 3]]))

    def test_explicit_splits_checked(self):
        ec.validate_job(valid_job(splits=[[18, 12, 1, 1]]))
        ec.validate_job(valid_job(splits=[[18, 12, 2, 0]]))  # head-only last rank
        with self.assertRaises(ec.JobConfigError):
            ec.validate_job(valid_job(splits=[[18, 12, 1]]))
        with self.assertRaises(ec.JobConfigError):
            ec.validate_job(valid_job(splits=[[18, 12, 1, 2]]))  # sums to 33

    def test_mem_monitor_values(self):
        for m in ec.MEM_MONITORS:
            ec.validate_job(valid_job(mem_monitor=m))
        with self.assertRaises(ec.JobConfigError):
            ec.validate_job(valid_job(mem_monitor="dcgmi"))


class LoadJob(unittest.TestCase):
    def test_unset_means_standalone(self):
        self.assertIsNone(ec.load_job({}))
        self.assertIsNone(ec.load_job({ec.JOB_ENV_VAR: "  "}))

    def test_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "job.json")
            with open(p, "w") as f:
                json.dump(valid_job(), f)
            self.assertEqual(ec.load_job({ec.JOB_ENV_VAR: p})["master_port"], 29530)
            with open(p, "w") as f:
                json.dump(valid_job(master_port=1), f)
            with self.assertRaises(ec.JobConfigError):
                ec.load_job({ec.JOB_ENV_VAR: p})

    def test_missing_file_is_loud(self):
        with self.assertRaises(OSError):
            ec.load_job({ec.JOB_ENV_VAR: "/nonexistent/job.json"})

    def test_count_configs(self):
        job = valid_job()
        n_splits = len(ec.generate_layer_splits(32, [18, 12, 5, 5], [20, 10, 5, 5]))
        self.assertEqual(ec.count_configs(job), n_splits * 2)
        self.assertEqual(ec.count_configs(valid_job(max_runs=3)), 3)
        self.assertEqual(ec.count_configs(valid_job(splits=[[18, 12, 1, 1]])), 2)


def _write_safetensors(path, names):
    header = {n: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for n in names}
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")


class ModelFamily(unittest.TestCase):
    def test_supported_types(self):
        self.assertEqual(mf.supported_model_types(), ["llama", "mistral", "qwen2"])

    def test_unknown_type_is_clear(self):
        with self.assertRaises(mf.UnsupportedModelError) as cm:
            mf.family_spec("gemma2")
        self.assertIn("llama, mistral, qwen2", str(cm.exception))

    def test_spec_names_are_the_branch_classes(self):
        self.assertEqual(
            mf.family_spec("qwen2")[1:],
            ("Qwen2DecoderLayer", "Qwen2RMSNorm", "Qwen2RotaryEmbedding"),
        )
        self.assertEqual(mf.family_spec("llama")[1], "LlamaDecoderLayer")

    def test_resolve_model_path_dir_and_hub_cache(self):
        with tempfile.TemporaryDirectory() as d:
            local = os.path.join(d, "vicuna")
            os.makedirs(local)
            self.assertEqual(mf.resolve_model_path(local), local)

            repo = os.path.join(d, "hub", "models--lmsys--vicuna-7b-v1.5")
            snap = os.path.join(repo, "snapshots", "abc123")
            os.makedirs(snap)
            os.makedirs(os.path.join(repo, "refs"))
            with open(os.path.join(repo, "refs", "main"), "w") as f:
                f.write("abc123\n")
            self.assertEqual(
                mf.resolve_model_path(
                    "lmsys/vicuna-7b-v1.5", hub_cache=os.path.join(d, "hub")
                ),
                snap,
            )
            with self.assertRaises(mf.ModelPathError):
                mf.resolve_model_path("lmsys/absent", hub_cache=os.path.join(d, "hub"))
            with self.assertRaises(mf.ModelPathError):
                mf.resolve_model_path(os.path.join(d, "nope"))

    def test_read_model_config(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(
                    {
                        "model_type": "qwen2",
                        "num_hidden_layers": 28,
                        "hidden_size": 3584,
                        "num_attention_heads": 28,
                        "vocab_size": 152064,
                    },
                    f,
                )
            cfg = mf.read_model_config(d)
            self.assertEqual(
                (cfg["model_type"], cfg["num_layers"], cfg["hidden_size"]),
                ("qwen2", 28, 3584),
            )
            self.assertFalse(cfg["tie_word_embeddings"])

    def test_find_weights_prefers_safetensors_and_checks_shards(self):
        with tempfile.TemporaryDirectory() as d:
            # .bin-only (Vicuna v1.5 layout)
            with open(os.path.join(d, "pytorch_model.bin.index.json"), "w") as f:
                json.dump(
                    {
                        "weight_map": {
                            "model.layers.0.mlp.up_proj.weight": "pytorch_model-00001-of-00002.bin",
                            "lm_head.weight": "pytorch_model-00002-of-00002.bin",
                        }
                    },
                    f,
                )
            for s in ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"):
                open(os.path.join(d, s), "wb").close()
            wf = mf.find_weight_files(d)
            self.assertEqual((wf.fmt, len(wf.files)), ("bin", 2))

            # Both formats present: safetensors wins.
            _write_safetensors(
                os.path.join(d, "model.safetensors"),
                ["model.layers.0.mlp.up_proj.weight", "model.norm.weight"],
            )
            wf = mf.find_weight_files(d)
            self.assertEqual(wf.fmt, "safetensors")

    def test_missing_shard_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors"}}, f)
            with self.assertRaises(mf.WeightFilesError) as cm:
                mf.find_weight_files(d)
            self.assertIn("model-00001-of-00002.safetensors", str(cm.exception))

    def test_partial_safetensors_falls_back_to_complete_bin(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": {"a": "model-00001-of-00001.safetensors"}}, f)
            with open(os.path.join(d, "pytorch_model.bin.index.json"), "w") as f:
                json.dump({"weight_map": {"a": "pytorch_model-00001-of-00001.bin"}}, f)
            open(os.path.join(d, "pytorch_model-00001-of-00001.bin"), "wb").close()
            wf = mf.find_weight_files(d)
            self.assertEqual(wf.fmt, "bin")
            self.assertEqual(len(wf.skipped), 1)
            self.assertIn("model-00001-of-00001.safetensors", wf.skipped[0])

    def test_no_weights_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            # Mistral's native file is not an HF checkpoint.
            open(os.path.join(d, "consolidated.safetensors"), "wb").close()
            with self.assertRaises(mf.WeightFilesError):
                mf.find_weight_files(d)


if __name__ == "__main__":
    unittest.main()
