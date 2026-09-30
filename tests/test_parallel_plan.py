"""Unit tests for parallel_plan.py. Stdlib only: no torch, no GPU.

    python3 -m unittest discover -s tests -p 'test_parallel_plan.py' -v

FakeBox (tests/test_run_parallel.py) provides model dirs and nvidia-smi
fixtures for an 8-GPU A100 box with every GPU split 20/10/5/5. The manager
itself (run_parallel.sh) is tested end to end in test_run_parallel.py.
"""

import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_config as ec  # noqa: E402
import parallel_plan as pp  # noqa: E402
from test_run_parallel import FakeBox  # noqa: E402

# Every shape nvidia-smi -L prints: a 40GB GPU listed out of rank order, an
# 80GB GPU with a +me profile, an old-driver MIG UUID, a GPU without MIG.
SMI_L = """\
GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-00000000-0000-0000-0000-000000000000)
  MIG 1g.5gb      Device  0: (UUID: MIG-a0000000-0000-0000-0000-000000000000)
  MIG 3g.20gb     Device  1: (UUID: MIG-a1000000-0000-0000-0000-000000000000)
  MIG 1g.5gb      Device  2: (UUID: MIG-a2000000-0000-0000-0000-000000000000)
  MIG 2g.10gb     Device  3: (UUID: MIG-a3000000-0000-0000-0000-000000000000)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-00000000-0000-0000-0000-000000000001)
  MIG 3g.40gb     Device  0: (UUID: MIG-b0000000-0000-0000-0000-000000000000)
  MIG 1g.10gb+me  Device  1: (UUID: MIG-b1000000-0000-0000-0000-000000000000)
GPU 2: A100-SXM4-40GB (UUID: GPU-00000000-0000-0000-0000-000000000002)
  MIG 3g.20gb Device 0: (UUID: MIG-GPU-00000000-0000-0000-0000-000000000002/1/0)
GPU 3: NVIDIA A100-SXM4-40GB (UUID: GPU-00000000-0000-0000-0000-000000000003)
"""
GPU0_UUIDS = [f"MIG-a{i}000000-0000-0000-0000-000000000000" for i in range(4)]


def cli(*args):
    """pp.main(args) -> (exit code, stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = pp.main([str(a) for a in args])
    return rc, out.getvalue()


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])


class Discover(unittest.TestCase):
    def test_parse_and_paste_ready_block(self):
        gpus = pp.parse_nvidia_smi_l(SMI_L)
        self.assertEqual([g["index"] for g in gpus], [0, 1, 2, 3])
        self.assertEqual([m["uuid"] for m in gpus[0]["migs"]], GPU0_UUIDS)  # listed order kept
        self.assertEqual([(m["profile"], m["gb"]) for m in gpus[1]["migs"]],
                         [("3g.40gb", 40), ("1g.10gb+me", 10)])
        self.assertEqual(gpus[2]["migs"][0]["uuid"], "MIG-GPU-00000000-0000-0000-0000-000000000002/1/0")
        self.assertEqual(gpus[3]["migs"], [])

        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.write(SMI_L)
            f.flush()
            rc, out = cli("discover", "--smi-l", f.name)
        ns = {}
        exec(out, ns)  # paste-ready: it must be valid Python
        g0 = ns["GPUS"][0]
        self.assertEqual((rc, g0["gpu"], g0["models"], g0["slice_gb"]), (0, 0, [], [20, 10, 5, 5]))
        u = GPU0_UUIDS
        self.assertEqual(g0["mig_uuids"], [u[1], u[3], u[0], u[2]])  # the 5GB tie keeps listed order
        self.assertEqual(ns["GPUS"][1]["slice_gb"], [40, 10])
        self.assertEqual([g["gpu"] for g in ns["GPUS"]], [0, 1, 2])  # GPU 3 is only a comment


class _Box(unittest.TestCase):
    """A FakeBox, with the planner's env overrides pointed at it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.box = FakeBox(self.tmp)
        # An empty sysfs, so the real /sys of the machine running the tests cannot leak in.
        env = mock.patch.dict(os.environ, {**self.box.env(),
                                           "MIG_PARALLEL_SYSFS": str(self.tmp / "sysfs")})
        env.start()
        self.addCleanup(env.stop)

    def plan(self, lanes, edit=lambda text: text, **kw):
        cfg = self.box.config(lanes)
        cfg.write_text(edit(cfg.read_text()))
        return pp.build_plan(str(cfg), **kw)

    def problems(self, lanes, edit=lambda text: text, **kw):
        with self.assertRaises(pp.PlanError) as cm:
            self.plan(lanes, edit, **kw)
        return str(cm.exception)

    @staticmethod
    def jobs(plan):
        return {j["model_key"]: j for lane in plan["lanes"] for j in lane["jobs"]}


class Planning(_Box):
    def test_inventory_mismatch_names_gpu_rank_uuid(self):
        u = [FakeBox.uuid(0, r) for r in range(4)]
        swapped = [u[1], u[0], u[2], u[3]]  # the 10GB slice listed as rank 0
        msg = self.problems({0: ["llama_7b"]}, lambda t: t.replace(repr(u), repr(swapped)))
        self.assertIn(f"gpu 0 rank 0: {u[1]} expected 20gb on GPU 0, got 2g.10gb on GPU 0", msg)
        self.assertIn(f"gpu 0 rank 1: {u[0]} expected 10gb on GPU 0, got 3g.20gb on GPU 0", msg)

    def test_duplicate_uuid_across_gpus(self):
        dup = FakeBox.uuid(0, 0)
        msg = self.problems({0: ["llama_7b"], 1: ["mistral_7b"]},
                            lambda t: t.replace(FakeBox.uuid(1, 0), dup))
        self.assertIn(f"MIG UUID {dup} appears twice: gpu 0 rank 0 and gpu 1 rank 0", msg)

    def test_placeholder_uuids(self):
        # gpu 6 has models=[] and placeholders too: a spare GPU is not checked at all.
        msg = self.problems({0: ["llama_7b"], 6: []}, lambda t: t
                            .replace(FakeBox.uuid(0, 2), "MIG-PASTE-GPU0-RANK2")
                            .replace(FakeBox.uuid(6, 0), "MIG-PASTE-GPU6-RANK0"))
        self.assertEqual(msg, "1 problem(s):\n  - gpu 0: mig_uuids still hold template placeholders; "
                              "run ./run_parallel.sh discover and paste its block")

    def test_dcgm_only_for_one_lane_on_gpu0(self):
        dcgm = lambda t: t.replace("RUNNER = dict()", 'RUNNER = dict(mem_monitor="dcgm")')  # noqa: E731
        msg = self.problems({0: ["llama_7b"], 1: ["mistral_7b"]}, dcgm)
        self.assertIn("'dcgm' needs exactly one lane, on gpu 0", msg)
        one_lane = self.jobs(self.plan({0: ["llama_7b"]}, dcgm))
        self.assertEqual(one_lane["llama_7b"]["job"]["mem_monitor"], "dcgm")

    def test_limits_come_from_layer_limits_py(self):
        jobs = self.jobs(self.plan({0: ["qwen_7b"], 1: ["vicuna_13b"]}))
        qwen, vicuna = jobs["qwen_7b"]["job"], jobs["vicuna_13b"]["job"]
        self.assertEqual((qwen["layer_limits"], qwen["min_last_rank_layers"]), ([18, 12, 5, 3], 0))
        self.assertEqual((vicuna["layer_limits"], vicuna["min_last_rank_layers"]), ([23, 12, 5, 5], 1))
        self.assertEqual(qwen["splits"],
                         ec.generate_layer_splits(28, [18, 12, 5, 3], [20, 10, 5, 5], True, 0))
        self.assertEqual(jobs["qwen_7b"]["n_configs"], len(qwen["splits"]) * 2)  # FakeBox sweeps 2 pairs

    def test_explicit_layer_limits_override_wins(self):
        plan = self.plan({0: ["llama_7b"]}, lambda t: t.replace(
            "models=['llama_7b']", "models=[dict(model='llama_7b', layer_limits=[18, 9, 5, 4])]"))
        job = self.jobs(plan)["llama_7b"]["job"]
        self.assertEqual((job["layer_limits"], job["min_last_rank_layers"]), ([18, 9, 5, 4], 1))
        self.assertEqual(job["splits"], ec.generate_layer_splits(32, [18, 9, 5, 4], [20, 10, 5, 5]))

    def test_model_missing_from_layer_limits_py(self):
        path = self.tmp / "models" / "llama_7b"
        msg = self.problems({0: ["extra_7b"]}, lambda t: t.replace(
            "MODELS = {\n", f'MODELS = {{\n    "extra_7b": dict(path="{path}"),\n'))
        self.assertIn("gpu0/01_extra_7b: layer_limits.py has no entry for model 'extra_7b' "
                      "on layout '20_10_5_5'", msg)

    def test_smoke_overrides(self):
        plan = self.plan({0: ["llama_7b"], 1: ["qwen_14b"]}, smoke=True)
        self.assertTrue(plan["run_id"].endswith("_smoke"))
        for j in self.jobs(plan).values():
            job = j["job"]
            got = (job["max_new_tokens"], job["max_runs"], job["batch_mb_pairs"], j["n_configs"])
            self.assertEqual(got, (8, 1, [[8, 4]], 1))
            front = ec.front_loaded_split(job["num_layers"], job["layer_limits"], job["slice_gb"],
                                          True, job["min_last_rank_layers"])
            self.assertEqual(job["splits"], [front])
            self.assertIn("_smoke_", job["shm_prefix"])

    def test_run_dir_layout_and_job_env_sources_in_bash(self):
        cpulist = self.tmp / "sysfs/bus/pci/devices/0000:10:00.0/local_cpulist"  # FakeBox's gpu 0
        cpulist.parent.mkdir(parents=True)
        cpulist.write_text("0-23,48-71\n")
        odd_env = 'RUNNER = dict(env=dict(ODD="it\'s a b"))'  # quoting must survive job.env
        plan = self.plan({0: ["llama_7b", "mistral_7b"], 3: ["qwen_7b"]},
                         lambda t: t.replace("RUNNER = dict()", odd_env))
        run = Path(pp.write_run_dir(plan))
        self.assertEqual((run / "lanes.txt").read_text().splitlines(), ["gpu0", "gpu3"])
        self.assertEqual((run / "lanes/gpu0/jobs.txt").read_text().splitlines(),
                         [str(run / "gpu0/01_llama_7b"), str(run / "gpu0/02_mistral_7b")])
        self.assertEqual((run / "lanes/gpu0/cpus").read_text(), "0-23,48-71\n")
        self.assertEqual((run / "lanes/gpu3/cpus").read_text(), "")  # no sysfs entry: unbound
        self.assertTrue(any("gpu 3" in w and "unbound" in w for w in plan["warnings"]), plan["warnings"])
        self.assertEqual((run / "lanes/gpu3/gpu").read_text(), "3\n")
        j = json.loads((run / "plan.json").read_text())["lanes"][0]["jobs"][0]
        self.assertEqual(set(j), {"job_id", "job_dir", "model_key", "shm_prefix", "n_configs"})
        out = subprocess.run(
            ["bash", "-c", '. "$1/job.env" && printf "%s\\n" "$MIG_SHM_PREFIX" "$MIG_EXP_CONFIG" "$ODD"',
             "_", j["job_dir"]], capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertEqual(out, [j["shm_prefix"], f"{j['job_dir']}/job.json", "it's a b"])
        self.assertEqual((Path(j["job_dir"]) / "state").read_text(), "pending\n")
        ec.validate_job(json.loads((Path(j["job_dir"]) / "job.json").read_text()))


class RunDir(_Box):
    def finished_smoke_run(self):
        """A 4-job smoke run dir whose jobs all wrote clean output, in the real formats."""
        plan = self.plan({0: ["llama_7b", "mistral_7b"], 1: ["qwen_7b", "vicuna_13b"]}, smoke=True)
        run = pp.write_run_dir(plan)
        for j in self.jobs(plan).values():
            d, job, ranks = Path(j["job_dir"]), j["job"], range(len(j["job"]["mig_uuids"]))
            (d / "state").write_text("ok\n")
            (d / "exit_code").write_text("0\n")
            (d / "benchmark.log").write_text("[1/1] Split [18, 12, 1, 1] | Batch: 8\n" + "".join(
                f"[B08][Rank {r}] weights: loaded=9/9 missing=0 format=safetensors files=1\n"
                for r in ranks))
            (d / "logs").mkdir()
            (d / "logs/transport_1.log").write_text("".join(
                f"12:00:00.000 [INFO] [T01][rank{r}] init transport: world_size=4 num_slots=6 "
                f"shm_prefix={j['shm_prefix']}\n" for r in ranks))
            write_csv(d / "mig_benchmark_results.csv", ["split", "status"], [["[18, 12, 1, 1]", "ok"]])
            cols = [f"rank{r}_{gb}gb_mb" for r, gb in enumerate(job["slice_gb"])]
            write_csv(d / "mig_memory_trace.csv", ["timestamp", "label", "gpu_mb"] + cols,
                      [["12:00:00.000", "s", 1] + [gb * 512 for gb in job["slice_gb"]]])
        return run, self.jobs(plan)

    def test_verdict_passes_clean_jobs(self):
        run, _ = self.finished_smoke_run()
        rc, out = cli("report", run, "--verdict")
        self.assertEqual(rc, 0, out)
        self.assertRegex(out, r"gpu0\s+0\s+29500\s+01_llama_7b\s+ok\s+1/1\s+1/0/0")  # the status row
        self.assertEqual(out.count("\nPASS  "), 6 * 4, out)  # six checks, four jobs
        self.assertTrue(out.rstrip().endswith("VERDICT: PASS"), out)

    def test_verdict_fails_each_broken_job(self):
        run, jobs = self.finished_smoke_run()
        d = {k: Path(j["job_dir"]) for k, j in jobs.items()}
        log = d["llama_7b"] / "benchmark.log"
        log.write_text(log.read_text().replace("[B08][Rank 2]", "[B99][Rank 2]"))
        t01 = d["mistral_7b"] / "logs/transport_1.log"
        t01.write_text(t01.read_text().replace(jobs["mistral_7b"]["shm_prefix"], "mig_pipe_shm"))
        (self.box.shm / f"{jobs['qwen_7b']['shm_prefix']}_0_slot0").write_bytes(b"\0")
        write_csv(d["vicuna_13b"] / "mig_benchmark_results.csv", ["split", "status"],
                  [["[23, 12, 3, 2]", "ok"], ["[23, 12, 3, 2]", "OOM_rank1"]])
        rc, out = cli("report", run, "--verdict")
        self.assertEqual(rc, 1, out)
        self.assertIn("VERDICT: FAIL (4)", out)
        for key, check in (("llama_7b", "weights"), ("mistral_7b", "shm_prefix"),
                           ("qwen_7b", "shm_leftover"), ("vicuna_13b", "results")):
            self.assertIn(f"FAIL  {jobs[key]['job_id']}  {check}:", out)


if __name__ == "__main__":
    unittest.main()
