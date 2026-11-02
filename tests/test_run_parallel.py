"""End-to-end tests for run_parallel.sh with the real planner and a stub benchmark.

Run locally (stdlib only, no torch, no GPU):
    python3 -m unittest discover -s tests -p 'test_run_parallel.py' -v

Each test builds a fake 8-GPU box in a temp dir: an `nvidia-smi -L` fixture,
model directories whose config.json / safetensors headers match the real
shapes, and a copy of parallel_config.py pointed at them, with
tests/fixtures/stub_benchmark.py standing in for the benchmark. What is under
test is the real run_parallel.sh + parallel_plan.py: process isolation,
concurrency, stop/kill, resume, lock, verdict.
"""

import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_parallel.sh"
STUB = ROOT / "tests" / "fixtures" / "stub_benchmark.py"
sys.path.insert(0, str(ROOT))

import memory_limits as ml  # noqa: E402

MODEL_TYPES = {
    "llama_7b": "llama", "mistral_7b": "mistral", "qwen_7b": "qwen2",
    "vicuna_13b": "llama", "llama_13b": "llama", "qwen_14b": "qwen2",
    "vicuna_7b": "llama",
}


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); it is dead for our purposes.
    try:
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    except OSError:
        return True
    return bool(st.strip()) and not st.strip().startswith("Z")


class FakeBox:
    def __init__(self, root: Path):
        self.root = root
        self.shm = root / "shm"
        self.shm.mkdir()
        (root / "models").mkdir()
        for key, mt in MODEL_TYPES.items():
            a = ml.REFERENCE_SHAPES[key]
            d = root / "models" / key
            d.mkdir()
            (d / "config.json").write_text(json.dumps({
                "model_type": mt, "num_hidden_layers": a["L"], "hidden_size": a["H"],
                "intermediate_size": a["I"], "num_attention_heads": a["heads"],
                "num_key_value_heads": a["kv"], "vocab_size": a["V"],
            }))
            names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
            names += [f"model.layers.{i}.mlp.up_proj.weight" for i in range(a["L"])]
            hdr = json.dumps({n: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for n in names}).encode()
            (d / "model.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0\0")
        lines = []
        for g in range(8):
            lines.append(f"GPU {g}: NVIDIA A100-SXM4-40GB (UUID: GPU-{uuid.UUID(int=g)})")
            for dev, prof in enumerate(["3g.20gb", "2g.10gb", "1g.5gb", "1g.5gb"]):
                lines.append(f"  MIG {prof:<11s} Device  {dev}: (UUID: {self.uuid(g, dev)})")
        (root / "smi_l.txt").write_text("\n".join(lines) + "\n")
        (root / "smi_query.txt").write_text(
            "\n".join(f"{g}, 00000000:{0x10 + g:02X}:00.0" for g in range(8)) + "\n")

    @staticmethod
    def uuid(g, r):
        return f"MIG-{uuid.UUID(int=1000 + g * 10 + r)}"

    def config(self, lanes, stubs=None):
        """lanes: {gpu: [model keys]}; stubs: {model key: stub dict}."""
        stubs = stubs or {}
        models = ",\n".join(
            f'    "{k}": dict(path="{self.root / "models" / k}", stub={stubs.get(k, {"sleep": 1})!r})'
            for k in MODEL_TYPES
        )
        gpus = ",\n".join(
            f"    dict(gpu={g}, slice_gb=[20, 10, 5, 5], "
            f"mig_uuids={[self.uuid(g, r) for r in range(4)]!r}, models={ms!r})"
            for g, ms in lanes.items()
        )
        text = (
            f"MODELS = {{\n{models},\n}}\nGPUS = [\n{gpus},\n]\n"
            f"SWEEP = dict(batch_mb_pairs=[(8, 4), (16, 2)])\n"
            f"RUNNER = dict()\n"
        )
        p = self.root / f"config_{len(list(self.root.glob('config_*.py')))}.py"
        p.write_text(text)
        return p

    def env(self):
        e = dict(os.environ)
        e.update({
            "MIG_PARALLEL_SMI_L": str(self.root / "smi_l.txt"),
            "MIG_PARALLEL_SMI_QUERY": str(self.root / "smi_query.txt"),
            "MIG_PARALLEL_BENCH": str(STUB),
            "MIG_SHM_DIR": str(self.shm),
            "MIG_PARALLEL_POLL": "1",
            "MIG_PARALLEL_LOCK": str(self.root / "lock"),
            "MIG_PARALLEL_RUNS_DIR": str(self.root / "runs"),
            "MIG_PARALLEL_STOP_GRACE": "3",
            "PYTHON": sys.executable,
        })
        return e

    def run(self, *args, timeout=60, check=False):
        p = subprocess.run(["bash", str(SCRIPT), *map(str, args)], env=self.env(),
                           capture_output=True, text=True, timeout=timeout)
        if check and p.returncode != 0:
            raise AssertionError(f"exit {p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
        return p

    def latest(self):
        return Path(os.path.realpath(self.root / "runs" / "latest"))


def job_dirs(run_dir: Path):
    return sorted(p.parent for p in run_dir.glob("gpu*/*/job.json"))


def read(p: Path) -> str:
    return p.read_text().strip() if p.exists() else ""


class RunParallel(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="runpar_"))
        self.box = FakeBox(self.tmp)

    def tearDown(self):
        # Never leave stub processes behind, even when a test fails.
        for rec in self.tmp.glob("runs/*/gpu*/*/stub_record.json"):
            try:
                r = json.loads(rec.read_text())
                for pid in [r["pid"], *r.get("children", [])]:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except (OSError, ValueError, KeyError):
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def records(self, run_dir):
        return {str(j.relative_to(run_dir)): json.loads((j / "stub_record.json").read_text())
                for j in job_dirs(run_dir) if (j / "stub_record.json").exists()}

    def test_lanes_parallel_jobs_sequential_and_isolated(self):
        cfg = self.box.config({0: ["llama_7b", "qwen_7b"], 1: ["mistral_7b"], 2: ["vicuna_13b", "qwen_14b"]},
                              stubs={k: {"sleep": 2} for k in MODEL_TYPES})
        p = self.box.run("start", "-c", cfg, check=True, timeout=90)
        run_dir = self.box.latest()
        recs = self.records(run_dir)
        self.assertEqual(len(recs), 5, p.stdout)
        # Distinct cwd / prefix per job; one process group per LANE (a lane's
        # jobs run inside its session), matching the lane's pgid file.
        for field in ("cwd", "prefix"):
            self.assertEqual(len({r[field] for r in recs.values()}), 5, field)
        self.assertEqual(len({r["pgid"] for r in recs.values()}), 3)
        for jid, r in recs.items():
            lane = jid.split("/")[0]
            self.assertEqual(str(r["pgid"]), read(run_dir / "lanes" / lane / "pgid"), jid)
        self.assertEqual({r["port"] for r in recs.values()}, {29500, 29510, 29520})
        for jid, r in recs.items():
            self.assertEqual(os.path.realpath(r["cwd"]), os.path.realpath(run_dir / jid))
            self.assertTrue(r["prefix"].startswith("migpp_"))
            self.assertEqual(r["env"]["PYTHONUNBUFFERED"], "1")
        # Lanes overlap in time; jobs within a lane do not.
        first = [recs["gpu0/01_llama_7b"], recs["gpu1/01_mistral_7b"], recs["gpu2/01_vicuna_13b"]]
        self.assertLess(max(r["started"] for r in first), min(r["finished"] for r in first))
        self.assertGreaterEqual(recs["gpu0/02_qwen_7b"]["started"], recs["gpu0/01_llama_7b"]["finished"])
        self.assertGreaterEqual(recs["gpu2/02_qwen_14b"]["started"], recs["gpu2/01_vicuna_13b"]["finished"])
        for j in job_dirs(run_dir):
            self.assertEqual(read(j / "state"), "ok")
        self.assertEqual(list(self.box.shm.iterdir()), [])
        self.assertFalse((self.tmp / "lock").exists(), "lock not released")
        # Each model keeps its own results CSV in its own job dir.
        for j in job_dirs(run_dir):
            self.assertTrue((j / "mig_benchmark_results.csv").exists(), j)

    def test_failed_job_does_not_stop_its_lane_or_others(self):
        cfg = self.box.config({0: ["llama_7b", "qwen_7b"], 1: ["mistral_7b"]},
                              stubs={"llama_7b": {"sleep": 1, "rc": 3}})
        p = self.box.run("start", "-c", cfg, timeout=60)
        self.assertEqual(p.returncode, 1, "a failed job must make start exit 1")
        run_dir = self.box.latest()
        self.assertEqual(read(run_dir / "gpu0/01_llama_7b/state"), "failed")
        self.assertEqual(read(run_dir / "gpu0/01_llama_7b/exit_code"), "3")
        self.assertEqual(read(run_dir / "gpu0/02_qwen_7b/state"), "ok")
        self.assertEqual(read(run_dir / "gpu1/01_mistral_7b/state"), "ok")

    def test_stop_kills_every_process_and_cleans_shm(self):
        cfg = self.box.config({0: ["llama_7b", "qwen_7b"], 1: ["mistral_7b"], 3: ["llama_13b"]},
                              stubs={k: {"sleep": 60, "leak_shm": True} for k in MODEL_TYPES})
        self.box.run("start", "-c", cfg, "--detach", check=True, timeout=30)
        run_dir = self.box.latest()
        deadline = time.time() + 20
        while len(self.records(run_dir)) < 3 and time.time() < deadline:
            time.sleep(0.2)
        recs = self.records(run_dir)
        self.assertEqual(len(recs), 3)
        self.assertTrue(any(self.box.shm.iterdir()), "stubs should be holding segments")
        status = self.box.run("status", check=True)
        self.assertIn("running", status.stdout)
        # A second start while this one runs must be refused.
        again = self.box.run("start", "-c", cfg, timeout=30)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("another run is active", again.stderr)

        self.box.run("stop", check=True, timeout=60)
        pids = [p for r in recs.values() for p in [r["pid"], *r["children"]]]
        time.sleep(0.5)
        self.assertEqual([p for p in pids if pid_alive(p)], [], "processes survived stop")
        self.assertFalse(pid_alive(int(read(run_dir / "manager.pid"))), "manager survived stop")
        self.assertEqual(list(self.box.shm.iterdir()), [], "segments left after stop")
        states = {str(j.relative_to(run_dir)): read(j / "state") for j in job_dirs(run_dir)}
        self.assertEqual(states["gpu0/01_llama_7b"], "killed")
        self.assertEqual(states["gpu0/02_qwen_7b"], "pending", "a stopped lane must not start its next job")
        self.assertFalse((self.tmp / "lock").exists())

        # --resume re-runs what did not finish ok, and skips what did.
        (run_dir / "gpu1/01_mistral_7b/state").write_text("ok\n")
        for j in job_dirs(run_dir):  # make the re-run quick
            job = json.loads((j / "job.json").read_text())
            job["stub_sleep"] = 1
            job["stub_leak_shm"] = False
            (j / "job.json").write_text(json.dumps(job))
        r = self.box.run("start", "--resume", run_dir, check=True, timeout=90)
        self.assertIn("resuming", r.stdout)
        for jid in ("gpu0/01_llama_7b", "gpu0/02_qwen_7b", "gpu3/01_llama_13b"):
            self.assertEqual(read(run_dir / jid / "state"), "ok", jid)
        def attempts(jid):
            return read(run_dir / jid / "stdout.log").count("===== attempt")
        self.assertEqual(attempts("gpu1/01_mistral_7b"), 1, "an ok job must be skipped on resume")
        self.assertEqual(attempts("gpu0/01_llama_7b"), 2, "a killed job must be re-run")
        self.assertEqual(attempts("gpu0/02_qwen_7b"), 1, "a never-started job runs once")

    def test_stop_escalates_to_kill(self):
        cfg = self.box.config({0: ["llama_7b"]}, stubs={"llama_7b": {"sleep": 60, "ignore_term": True}})
        self.box.run("start", "-c", cfg, "--detach", check=True, timeout=30)
        run_dir = self.box.latest()
        deadline = time.time() + 20
        while not self.records(run_dir) and time.time() < deadline:
            time.sleep(0.2)
        rec = self.records(run_dir)["gpu0/01_llama_7b"]
        t0 = time.time()
        self.box.run("stop", check=True, timeout=60)
        self.assertGreaterEqual(time.time() - t0, 2.5, "should have waited the grace period")
        time.sleep(0.5)
        self.assertFalse(pid_alive(rec["pid"]), "TERM-ignoring benchmark survived")

    def test_smoke_verdict_fails_when_prefix_not_honored(self):
        # A transport that ignored MIG_SHM_PREFIX logs another prefix in [T01].
        cfg = self.box.config({0: ["llama_7b"], 1: ["mistral_7b"]},
                              stubs={"mistral_7b": {"sleep": 2, "wrong_prefix": True}, "llama_7b": {"sleep": 2}})
        p = self.box.run("smoke", "-c", cfg, timeout=60)
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertRegex(p.stdout, r"FAIL\s+gpu1/01_mistral_7b")
        self.assertIn("VERDICT: FAIL", p.stdout)
        ok = self.box.config({0: ["llama_7b"], 1: ["mistral_7b"]}, stubs={k: {"sleep": 2} for k in MODEL_TYPES})
        p = self.box.run("smoke", "-c", ok, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("VERDICT: PASS", p.stdout)

    def test_planning_error_launches_nothing(self):
        # Same MIG UUID on two GPUs: the planner must refuse.
        cfg = self.box.config({0: ["llama_7b"], 1: ["mistral_7b"]})
        text = cfg.read_text().replace(self.box.uuid(1, 0), self.box.uuid(0, 0))
        cfg.write_text(text)
        p = self.box.run("start", "-c", cfg, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("planning failed", p.stderr)
        self.assertFalse((self.tmp / "lock").exists())

    def test_shellcheck_clean(self):
        sc = shutil.which("shellcheck")
        if not sc:
            self.skipTest("shellcheck not installed")
        p = subprocess.run([sc, "-s", "bash", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout)


if __name__ == "__main__":
    unittest.main()
