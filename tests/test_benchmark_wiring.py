"""Torch-free checks on the parallel-run wiring in the benchmark and transport.

Run locally (no torch needed):
    python3 -m unittest discover -s tests -p 'test_benchmark_wiring.py' -v

The benchmark and transport import torch, so they cannot be imported here.
Instead this executes the benchmark's real module-level CONFIGURATION section
(sliced out of the source) with stub monitor modules, and inspects the source
of both files with ast. It proves the two properties the gate needs:

  1. MIG_EXP_CONFIG unset => every constant is exactly what the section alone
     would produce with the override block deleted (standalone unchanged).
  2. MIG_EXP_CONFIG set   => every job.json key lands in its constant, the SHM
     prefix is exported for the transport, and the monitor is chosen by it.
"""

import ast
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiment_config  # noqa: E402

BENCH = (ROOT / "benchmark_pipeline_microbatching.py").read_text()
TRANSPORT = (ROOT / "mig_transport_pipeline.py").read_text()

CONFIG_START = "# --- CONFIGURATION ---"
CONFIG_END = "# Dist message tag bases"
OVERRIDE_START = "# --- PARALLEL-RUN OVERRIDE ---"
OVERRIDE_END = "# The DCGM monitor pkills"

CONSTANTS = [
    "MODEL_NAME", "TOTAL_LAYERS", "HIDDEN_SIZE", "HEADS", "SEQ_LEN",
    "MAX_NEW_TOKENS", "MAX_RUNS", "BATCH_MB_PAIRS", "MIG_UUIDS", "SLICE_GB",
    "LAYER_LIMITS", "ENFORCE_SLICE_ORDERING", "MIN_LAST_RANK_LAYERS", "SPLITS",
    "MASTER_PORT", "MEM_MONITOR", "WORLD_SIZE",
]


def config_section(drop_override=False):
    start = BENCH.index(CONFIG_START)
    end = BENCH.index(CONFIG_END, start)
    src = BENCH[start:end]
    if drop_override:
        a = src.index(OVERRIDE_START)
        b = src.index(OVERRIDE_END, a)
        src = src[:a] + src[b:]
    return src


def fake_monitor(name):
    m = types.ModuleType(name)
    m.disabled = False

    def disable():
        m.disabled = True

    m.disable = disable
    return m


def exec_config(env, drop_override=False):
    dcgm, nvml = fake_monitor("dcgm_mem_monitor"), fake_monitor("nvml_mem_monitor")
    g = {"os": os, "experiment_config": experiment_config, "__name__": "bench_cfg"}
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(
        sys.modules, {"dcgm_mem_monitor": dcgm, "nvml_mem_monitor": nvml}
    ):
        exec(compile(config_section(drop_override), "bench_config", "exec"), g)
        g["_env_after"] = dict(os.environ)
    g["_fakes"] = (dcgm, nvml)
    return g


def qwen_job(tmp, **over):
    job = {
        "schema": 1,
        "model_path": "/data/models/Qwen2.5-7B",
        "model_type": "qwen2",
        "num_layers": 28,
        "hidden_size": 3584,
        "num_heads": 28,
        "seq_len": 64,
        "max_new_tokens": 8,
        "max_runs": 1,
        "batch_mb_pairs": [[8, 4]],
        "mig_uuids": [f"MIG-{i}0000000-0000-0000-0000-000000000000" for i in range(4)],
        "slice_gb": [20, 10, 5, 5],
        "layer_limits": [16, 8, 4, 2],
        "enforce_slice_ordering": True,
        "min_last_rank_layers": 1,
        "splits": [[16, 8, 3, 1]],
        "master_port": 29530,
        "shm_prefix": "migpp_20260929_161500_g3j01",
        "mem_monitor": "nvml",
        "job_id": "gpu3/01_qwen_7b",
    }
    job.update(over)
    p = os.path.join(tmp, "job.json")
    with open(p, "w") as f:
        json.dump(job, f)
    return job, p


class StandaloneUnchanged(unittest.TestCase):
    def test_override_block_is_a_noop_without_job(self):
        with_block = exec_config({})
        without_block = exec_config({}, drop_override=True)
        for name in CONSTANTS:
            self.assertEqual(with_block[name], without_block[name], name)
        self.assertNotIn(experiment_config.SHM_PREFIX_ENV_VAR, with_block["_env_after"])

    def test_standalone_defaults_match_previous_hardcoding(self):
        # Before this change run_pipeline set MASTER_PORT="29500" and the
        # module imported dcgm_mem_monitor unconditionally.
        g = exec_config({})
        self.assertEqual(g["MASTER_PORT"], 29500)
        self.assertEqual(g["MEM_MONITOR"], "dcgm")
        self.assertIs(g["monitor"], g["_fakes"][0])
        self.assertIsNone(g["SPLITS"])
        self.assertEqual(g["MIN_LAST_RANK_LAYERS"], 1)


class JobOverride(unittest.TestCase):
    def test_every_job_key_lands(self):
        with tempfile.TemporaryDirectory() as d:
            job, p = qwen_job(d)
            g = exec_config({experiment_config.JOB_ENV_VAR: p})
        expect = {
            "MODEL_NAME": job["model_path"],
            "TOTAL_LAYERS": 28,
            "HIDDEN_SIZE": 3584,
            "HEADS": 28,
            "SEQ_LEN": 64,
            "MAX_NEW_TOKENS": 8,
            "MAX_RUNS": 1,
            "BATCH_MB_PAIRS": [(8, 4)],
            "MIG_UUIDS": job["mig_uuids"],
            "SLICE_GB": [20, 10, 5, 5],
            "LAYER_LIMITS": [16, 8, 4, 2],
            "ENFORCE_SLICE_ORDERING": True,
            "MIN_LAST_RANK_LAYERS": 1,
            "SPLITS": [[16, 8, 3, 1]],
            "MASTER_PORT": 29530,
            "MEM_MONITOR": "nvml",
            "WORLD_SIZE": 4,
        }
        for name, value in expect.items():
            self.assertEqual(g[name], value, name)
        # BATCH_MB_PAIRS entries are tuples, as the hardcoded list's are.
        self.assertIsInstance(g["BATCH_MB_PAIRS"][0], tuple)
        self.assertEqual(
            g["_env_after"][experiment_config.SHM_PREFIX_ENV_VAR], job["shm_prefix"]
        )
        self.assertIs(g["monitor"], g["_fakes"][1])
        self.assertFalse(g["_fakes"][1].disabled)

    def test_monitor_off_disables_nvml(self):
        with tempfile.TemporaryDirectory() as d:
            _, p = qwen_job(d, mem_monitor="off")
            g = exec_config({experiment_config.JOB_ENV_VAR: p})
        self.assertIs(g["monitor"], g["_fakes"][1])
        self.assertTrue(g["_fakes"][1].disabled)

    def test_monitor_dcgm_selectable(self):
        with tempfile.TemporaryDirectory() as d:
            _, p = qwen_job(d, mem_monitor="dcgm")
            g = exec_config({experiment_config.JOB_ENV_VAR: p})
        self.assertIs(g["monitor"], g["_fakes"][0])

    def test_invalid_job_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            _, p = qwen_job(d, master_port=80)
            with self.assertRaises(experiment_config.JobConfigError):
                exec_config({experiment_config.JOB_ENV_VAR: p})

    def test_override_covers_every_required_job_key(self):
        # A job key the benchmark never reads would be silently ignored.
        block = config_section()
        block = block[block.index(OVERRIDE_START) : block.index(OVERRIDE_END)]
        metadata_only = {"schema", "model_type"}  # validated, not a constant
        for key in (
            "model_path", "num_layers", "hidden_size", "num_heads", "seq_len",
            "max_new_tokens", "max_runs", "batch_mb_pairs", "mig_uuids",
            "slice_gb", "layer_limits", "enforce_slice_ordering",
            "min_last_rank_layers", "splits", "master_port", "mem_monitor",
            "shm_prefix",
        ):
            if key in metadata_only:
                continue
            self.assertIn(f'"{key}"', block, key)


class SourceChecks(unittest.TestCase):
    def test_no_llama_classes_left_in_benchmark(self):
        tree = ast.parse(BENCH)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for cls in ("LlamaConfig", "LlamaDecoderLayer", "LlamaRMSNorm", "LlamaRotaryEmbedding"):
            self.assertNotIn(cls, names)

    def test_master_port_not_hardcoded(self):
        self.assertNotIn('os.environ["MASTER_PORT"] = "29500"', BENCH)
        self.assertIn('os.environ["MASTER_PORT"] = str(MASTER_PORT)', BENCH)

    def test_split_generator_delegates(self):
        tree = ast.parse(BENCH)
        fn = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "generate_layer_splits"
        )
        src = ast.get_source_segment(BENCH, fn)
        self.assertIn("experiment_config.generate_layer_splits(", src)
        self.assertNotIn("_recurse", src)

    def test_transport_shm_names_all_prefixed(self):
        # Every SharedMemory(name=...) and every unlink-by-name must go
        # through _slot_name; a literal "mig_pipe_shm_" f-string anywhere
        # would reintroduce the cross-run collision.
        self.assertNotIn('f"mig_pipe_shm_', TRANSPORT)
        tree = ast.parse(TRANSPORT)
        # String literals only (comments mention the name too).
        literals = [
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value.startswith("mig_pipe_shm")
        ]
        self.assertEqual(literals, ["mig_pipe_shm"])  # the legacy default only
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_slot_name"
        ]
        # create, peer attach, cleanup
        self.assertGreaterEqual(len(calls), 3)

    def test_transport_prefix_default_is_legacy(self):
        tree = ast.parse(TRANSPORT)
        g = {"os": os}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_LEGACY_SHM_PREFIX" for t in node.targets
            ):
                exec(compile(ast.Module([node], []), "t", "exec"), g)
            if isinstance(node, ast.FunctionDef) and node.name in ("_shm_prefix", "_slot_name"):
                exec(compile(ast.Module([node], []), "t", "exec"), g)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(g["_slot_name"](g["_shm_prefix"](), 2, 7), "mig_pipe_shm_2_slot7")
        with mock.patch.dict(os.environ, {"MIG_SHM_PREFIX": "migpp_x_g1j01"}):
            self.assertEqual(g["_slot_name"](g["_shm_prefix"](), 0, 0), "migpp_x_g1j01_0_slot0")


if __name__ == "__main__":
    unittest.main()
