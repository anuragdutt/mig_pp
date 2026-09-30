#!/usr/bin/env python3
"""
Planner behind run_parallel.sh: parallel_config.py -> a run directory.

One lane per GPU: a lane runs its models one after another (they share the
GPU's MIG slices) and lanes run side by side. Each (GPU, model) job gets its
own directory, port and /dev/shm prefix, a job.json the benchmark reads
through MIG_EXP_CONFIG, and a job.env the lane sources. Stdlib only.

  discover [--smi-l FILE]                  paste-ready GPUS block from nvidia-smi -L
  check [-c CONFIG]                        validate; one row per job with its sweep size
  plan [-c CONFIG] [--smoke] [--gpus 0,3]  check, then write the run dir (last line: its path)
  report RUN_DIR [--verdict]               status table; --verdict adds the smoke checks
"""

import argparse
import csv
import glob
import json
import os
import re
import runpy
import shlex
import subprocess
import sys
import time
from collections import Counter

import experiment_config as ec
import layer_limits
import model_family as mf

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(REPO, "parallel_config.py")

# Missing SWEEP / RUNNER keys take these. batch_mb_pairs is the benchmark's own list.
SWEEP_DEFAULTS = dict(
    seq_len=64, max_new_tokens=512, max_runs=None, enforce_slice_ordering=True,
    batch_mb_pairs=[[8, 4], [8, 2], [16, 8], [16, 4], [16, 2], [32, 16], [32, 8],
                    [32, 4], [32, 2], [64, 32], [64, 16], [64, 8], [64, 4], [64, 2]],
)
RUNNER_DEFAULTS = dict(base_port=29500, mem_monitor="nvml", env={})
# What a MODELS entry or a GPU's dict(model=...) may set. SWEEP < MODELS < GPU.
OVERRIDE_KEYS = set(SWEEP_DEFAULTS) | {"splits", "layer_limits", "min_last_rank_layers", "stub"}
GPU_KEYS = {"gpu", "slice_gb", "mig_uuids", "models"}
# config.json fields copied into every job.json.
MODEL_FIELDS = ("model_type", "num_layers", "hidden_size", "num_heads", "tie_word_embeddings")
# plan.json carries only these per job; the full settings are in each job.json.
PLAN_JOB_KEYS = ("job_id", "job_dir", "model_key", "shm_prefix", "n_configs")


class PlanError(Exception):
    """The config cannot be run. The message lists every problem found."""


def _fail(problems):
    raise PlanError(f"{len(problems)} problem(s):\n  - " + "\n  - ".join(problems))


def _unknown(d, allowed, where):
    return [f"{where}: unknown key '{k}'" for k in d if k not in allowed]


def _read(path):
    """A file's text, stripped; '' if it cannot be read (e.g. not written yet)."""
    try:
        with open(path, errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def _write(path, text):
    with open(path, "w") as f:
        f.write(text + "\n" if text else "")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _csv(path):
    """(header, rows as dicts) of a CSV, or None if there is no such file."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames or [], list(reader)
    except FileNotFoundError:
        return None


def _table(header, rows):
    cells = [[str(c) for c in row] for row in [header, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in cells]


# --- nvidia-smi and sysfs -----------------------------------------------------
def _smi(args, path=None):
    """nvidia-smi output, or the saved copy at path; None if nvidia-smi is unavailable."""
    if path:
        with open(path) as f:
            return f.read()
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                              check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None


_GPU_LINE = re.compile(r"^GPU (\d+): (.+?) \(UUID:")
# Old drivers print MIG UUIDs as MIG-GPU-<uuid>/<gi>/<ci>; [^\s)]+ takes both forms.
_MIG_LINE = re.compile(r"^\s+MIG (\S+)\s+Device\s+\d+: \(UUID: (MIG-[^\s)]+)\)")


def parse_nvidia_smi_l(text):
    """`nvidia-smi -L` -> [{"index", "name", "migs": [{"uuid", "profile", "gb"}]}], as listed."""
    gpus = []
    for line in text.splitlines():
        m = _GPU_LINE.match(line)
        if m:
            gpus.append({"index": int(m.group(1)), "name": m.group(2), "migs": []})
        elif gpus and (m := _MIG_LINE.match(line)):
            gb = re.search(r"\.(\d+)gb", m.group(1))  # "3g.20gb" -> 20, "1g.10gb+me" -> 10
            gpus[-1]["migs"].append({"uuid": m.group(2), "profile": m.group(1),
                                     "gb": int(gb.group(1)) if gb else None})
    return gpus


def _discover_block(gpus):
    lines = ["GPUS = ["]
    for g in gpus:
        if not g["migs"]:
            lines.append(f"    # gpu {g['index']}: {g['name']}, no MIG devices")
            continue
        # Rank 0 = the largest slice. sorted() is stable, so ties keep nvidia-smi order.
        migs = sorted(g["migs"], key=lambda m: -(m["gb"] or 0))
        lines.append(f"    dict(gpu={g['index']}, slice_gb={[m['gb'] for m in migs]}, models=[],")
        lines.append(f"         mig_uuids={json.dumps([m['uuid'] for m in migs])}),"
                     f"  # {g['name']}: {' '.join(m['profile'] for m in migs)}")
    return "\n".join(lines + ["]"])


def _numa_cpus(gpu_indices, warnings):
    """gpu -> its NUMA-local CPU list for taskset, so a lane's host-side staging stays
    on its GPU's socket. Lanes on one node share the list; '' where unreadable."""
    query = _smi(["--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                 os.environ.get("MIG_PARALLEL_SMI_QUERY")) or ""
    bus = dict(line.replace(" ", "").split(",", 1) for line in query.splitlines() if "," in line)
    sysfs, cpus = os.environ.get("MIG_PARALLEL_SYSFS", "/sys"), {}
    for g in gpu_indices:
        domain, _, rest = bus.get(str(g), "").partition(":")
        bdf = f"{domain[-4:]}:{rest}".lower()  # nvidia-smi "00000000:10:1C.0" -> "0000:10:1c.0"
        path = os.path.join(sysfs, "bus/pci/devices", bdf, "local_cpulist")
        cpus[g] = _read(path) if rest else ""
    unbound = [str(g) for g in gpu_indices if not cpus[g]]
    if unbound:
        warnings.append(f"no NUMA-local CPU list for gpu {', '.join(unbound)}; they run unbound")
    return cpus


# --- planning -----------------------------------------------------------------
def _check_gpus(gpu_cfg, models, problems, warnings):
    """Validate the GPU entries that have models. -> [(gpu, slice_gb, uuids, [(key, overrides)])]"""
    smi_l = _smi(["-L"], os.environ.get("MIG_PARALLEL_SMI_L"))
    if smi_l is None:
        warnings.append("nvidia-smi not found: MIG UUIDs not cross-checked against the inventory")
    inventory = {m["uuid"]: (g["index"], m)
                 for g in parse_nvidia_smi_l(smi_l or "") for m in g["migs"]}
    lanes, seen_gpu, seen_uuid = [], [], {}
    for g in gpu_cfg:
        if g.get("models") == []:
            continue  # a spare GPU: no lane, nothing to check
        idx = g.get("gpu")
        where = f"gpu {idx}"
        bad = _unknown(g, GPU_KEYS, where)
        bad += [f"{where}: missing '{k}'" for k in sorted(GPU_KEYS - set(g))]
        if not isinstance(idx, int) or idx in seen_gpu:
            bad.append(f"{where}: gpu must be an int, each GPU listed once")
        if not bad and len(g["mig_uuids"]) != len(g["slice_gb"]):
            bad.append(f"{where}: {len(g['mig_uuids'])} mig_uuids for {len(g['slice_gb'])} slices")
        seen_gpu.append(idx)
        problems += bad
        if bad:
            continue
        slice_gb, uuids = g["slice_gb"], g["mig_uuids"]
        for r, u in enumerate(uuids):
            if u in seen_uuid:
                problems.append(f"MIG UUID {u} appears twice: {seen_uuid[u]} and {where} rank {r}")
            seen_uuid.setdefault(u, f"gpu {idx} rank {r}")
        if any("PASTE" in str(u) for u in uuids):
            problems.append(f"{where}: mig_uuids still hold template placeholders; "
                            "run ./run_parallel.sh discover and paste its block")
        elif smi_l is not None:
            for r, (u, want) in enumerate(zip(uuids, slice_gb)):
                on_gpu, mig = inventory.get(u, (None, None))
                if on_gpu != idx or mig["gb"] != want:
                    got = f"{mig['profile']} on GPU {on_gpu}" if mig else "not in nvidia-smi -L"
                    problems.append(f"{where} rank {r}: {u} expected {want}gb on GPU {idx}, "
                                    f"got {got}")
        jobs = []
        for ref in g["models"]:
            over = dict(ref) if isinstance(ref, dict) else {"model": ref}
            key = over.pop("model", None)
            if key not in models:
                problems.append(f"{where}: model {key!r} is not a MODELS key")
            else:
                problems += _unknown(over, OVERRIDE_KEYS, f"{where} {key}")
                jobs.append((key, over))
        lanes.append((idx, slice_gb, uuids, jobs))
    return lanes


def _finish_job(job, s, key, smoke):
    """Add layer_limits, min_last_rank_layers and splits; validate. -> (job or None, problems)"""
    if "layer_limits" in s:  # an explicit override wins over layer_limits.py
        limits, table_min = s["layer_limits"], 1
    else:
        try:
            limits, table_min = layer_limits.limits_for(key, job["slice_gb"])
        except KeyError as e:
            return None, [e.args[0]]  # str(KeyError) would wrap the message in quotes
    job.update(layer_limits=limits, min_last_rank_layers=s.get("min_last_rank_layers", table_min))
    job = json.loads(json.dumps(job))  # validate exactly what the benchmark will read
    try:
        ec.validate_job(job)
    except ec.JobConfigError as e:
        return None, [line.strip()[2:] for line in str(e).splitlines()[1:]]  # drop "  - "
    args = (job["num_layers"], job["layer_limits"], job["slice_gb"],
            job["enforce_slice_ordering"], job["min_last_rank_layers"])
    if smoke:  # the split least likely to OOM a small slice
        front = ec.front_loaded_split(*args)
        job["splits"] = [front] if front else []
    elif not job["splits"]:
        job["splits"] = ec.generate_layer_splits(*args)
    if not job["splits"]:
        return None, [f"no split fits layer_limits {job['layer_limits']} with "
                      f"min_last_rank_layers {job['min_last_rank_layers']}"]
    return job, []


def build_plan(config_path, smoke=False, gpus=None):
    """Validate the config and lay out every job for write_run_dir(). Raises PlanError
    listing every problem, having written nothing. gpus: keep only these GPU indices."""
    config_path = os.path.abspath(config_path)
    try:
        ns = runpy.run_path(config_path)
    except Exception as e:  # the config is user code: report it, do not crash
        _fail([f"cannot load {config_path}: {type(e).__name__}: {e}"])
    # Only these four names are read; helpers such as LAYOUT or _gpu() are ignored.
    models, gpu_cfg = ns.get("MODELS"), ns.get("GPUS")
    sweep, runner = ns.get("SWEEP", {}), ns.get("RUNNER", {})
    problems, warnings = [], []
    if not (isinstance(models, dict) and models):
        problems.append("MODELS must be a non-empty dict of key: dict(path=...)")
    if not (isinstance(gpu_cfg, list) and all(isinstance(g, dict) for g in gpu_cfg)):
        problems.append("GPUS must be a list of dict(gpu=, slice_gb=, mig_uuids=, models=)")
    problems += [f"{name} must be a dict"
                 for name, d in (("SWEEP", sweep), ("RUNNER", runner)) if not isinstance(d, dict)]
    if problems:
        _fail(problems)  # nothing else can be checked without these

    problems += _unknown(sweep, SWEEP_DEFAULTS, "SWEEP")
    problems += _unknown(runner, RUNNER_DEFAULTS, "RUNNER")
    sweep, runner = {**SWEEP_DEFAULTS, **sweep}, {**RUNNER_DEFAULTS, **runner}
    if runner["mem_monitor"] not in ec.MEM_MONITORS:
        problems.append(f"RUNNER.mem_monitor must be one of {', '.join(ec.MEM_MONITORS)}, "
                        f"got {runner['mem_monitor']!r}")
        runner["mem_monitor"] = "nvml"  # reported once here, not again for every job
    good = set()  # MODELS keys whose entry is usable
    for key, entry in models.items():
        if isinstance(entry, dict) and entry.get("path"):
            good.add(key)
            problems += _unknown(entry, OVERRIDE_KEYS | {"path"}, f"MODELS[{key!r}]")
        else:
            problems.append(f"MODELS[{key!r}] needs path=<model dir, or cached HF repo id>")
    if gpus is not None:
        known = [g.get("gpu") for g in gpu_cfg]
        problems += [f"--gpus {i}: no such gpu in GPUS" for i in gpus if i not in known]
        gpu_cfg = [g for g in gpu_cfg if g.get("gpu") in gpus]
    lanes = _check_gpus(gpu_cfg, models, problems, warnings)
    if runner["mem_monitor"] == "dcgm" and [lane[0] for lane in lanes] != [0]:
        problems.append("RUNNER.mem_monitor 'dcgm' needs exactly one lane, on gpu 0: "
                        "dcgm_mem_monitor.py hardcodes GPU 0 and pkills every `dcgmi dmon` "
                        "on the box; use 'nvml'")

    used = list(dict.fromkeys(key for lane in lanes for key, _ in lane[3]))
    unused = [str(k) for k in models if k not in used]
    if unused:
        warnings.append(f"MODELS not run on any GPU: {', '.join(unused)}")
    info = {}  # model key -> (model dir, config.json fields)
    for key in (k for k in used if k in good):
        try:
            model_dir = mf.resolve_model_path(str(models[key]["path"]))
            mc = mf.read_model_config(model_dir)
            mf.family_spec(mc["model_type"])
            mf.find_weight_files(model_dir)
            info[key] = (model_dir, mc)
        except (OSError, ValueError) as e:  # model_family's errors subclass these, as does bad JSON
            problems.append(f"model {key}: {e}")

    runs_dir = os.path.abspath(os.environ.get("MIG_PARALLEL_RUNS_DIR") or f"{REPO}/runs")
    run_id = base = time.strftime("%Y%m%d_%H%M%S") + ("_smoke" if smoke else "")
    n = 1
    while os.path.exists(os.path.join(runs_dir, run_id)):
        n += 1
        run_id = f"{base}_{n}"
    run_dir, safe_id = os.path.join(runs_dir, run_id), re.sub(r"[^A-Za-z0-9_]", "_", run_id)
    cpus = _numa_cpus([lane[0] for lane in lanes], warnings)
    plan_lanes = []
    for idx, slice_gb, uuids, refs in lanes:
        lane_id, port, jobs = f"gpu{idx}", runner["base_port"] + 10 * idx, []
        for nn, (key, over) in enumerate(refs, 1):
            if key not in info:
                continue  # the model's problem is already reported
            job_id = f"{lane_id}/{nn:02d}_{key}"
            prefix = f"migpp_{safe_id}_g{idx}j{nn:02d}"  # per job: no two jobs share /dev/shm slots
            s = {**sweep, **models[key], **over}  # SWEEP < MODELS[key] < this GPU's dict
            model_dir, mc = info[key]
            job = dict(
                {k: s[k] for k in SWEEP_DEFAULTS}, **{k: mc[k] for k in MODEL_FIELDS},
                schema=ec.JOB_SCHEMA_VERSION, model_path=model_dir, mig_uuids=uuids,
                slice_gb=slice_gb, splits=s.get("splits"), master_port=port, shm_prefix=prefix,
                mem_monitor=runner["mem_monitor"], run_id=run_id, lane=lane_id, gpu_index=idx,
                job_id=job_id, model_key=key, smoke=smoke,
            )
            job.update({f"stub_{k}": v for k, v in s.get("stub", {}).items()})  # for the test stub
            if smoke:  # one tiny configuration per job
                job.update(max_new_tokens=8, batch_mb_pairs=[[8, 4]], max_runs=1)
            job, errors = _finish_job(job, s, key, smoke)
            problems += [f"{job_id}: {e}" for e in errors]
            if job:
                total = len(job["splits"]) * len(job["batch_mb_pairs"])
                jobs.append({"job_id": job_id, "job_dir": os.path.join(run_dir, job_id),
                             "model_key": key, "shm_prefix": prefix, "job": job,
                             "n_configs": min(total, job["max_runs"] or total)})
        plan_lanes.append({"lane": lane_id, "gpu": idx, "port": port, "cpus": cpus[idx],
                           "slice_gb": slice_gb, "jobs": jobs})
    if problems:
        _fail(problems)
    env = [(str(k), str(v)) for k, v in runner["env"].items()]
    return {"run_id": run_id, "smoke": smoke, "config": config_path, "lanes": plan_lanes,
            "run_dir": run_dir, "env": env, "warnings": warnings}


def write_run_dir(plan):
    """Write the run directory build_plan() laid out (run_parallel.sh reads it). -> its path"""
    run_dir = plan["run_dir"]
    os.makedirs(run_dir)  # never exist_ok: two runs must not share a directory
    public = {k: plan[k] for k in ("run_id", "smoke", "config")}
    public["lanes"] = [dict(lane, jobs=[{k: j[k] for k in PLAN_JOB_KEYS} for j in lane["jobs"]])
                       for lane in plan["lanes"]]
    _write(os.path.join(run_dir, "plan.json"), json.dumps(public, indent=2))
    _write(os.path.join(run_dir, "lanes.txt"), "\n".join(lane["lane"] for lane in plan["lanes"]))
    for lane in plan["lanes"]:
        d = os.path.join(run_dir, "lanes", lane["lane"])
        os.makedirs(d)
        _write(os.path.join(d, "jobs.txt"), "\n".join(j["job_dir"] for j in lane["jobs"]))
        _write(os.path.join(d, "cpus"), lane["cpus"])
        _write(os.path.join(d, "gpu"), str(lane["gpu"]))
        for j in lane["jobs"]:
            os.makedirs(j["job_dir"])
            job_json = os.path.join(j["job_dir"], "job.json")
            _write(job_json, json.dumps(j["job"], indent=2))
            # Unbuffered so stdout.log is live; no tqdm bars filling the logs.
            env = [("MIG_EXP_CONFIG", job_json), ("MIG_SHM_PREFIX", j["shm_prefix"]),
                   ("PYTHONUNBUFFERED", "1"), ("TQDM_DISABLE", "1"), *plan["env"]]
            _write(os.path.join(j["job_dir"], "job.env"),
                   "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env))
            _write(os.path.join(j["job_dir"], "shm_prefix"), j["shm_prefix"])
            _write(os.path.join(j["job_dir"], "state"), "pending")
    return run_dir


def _summary(plan):
    rows = [(lane["lane"], lane["gpu"], lane["port"], j["model_key"], j["job"]["model_type"],
             j["job"]["num_layers"], len(j["job"]["splits"]), j["n_configs"])
            for lane in plan["lanes"] for j in lane["jobs"]]
    out = _table(("LANE", "GPU", "PORT", "MODEL", "TYPE", "LAYERS", "SPLITS", "CONFIGS"), rows)
    out.append(f"total {len(rows)} jobs, {sum(r[-1] for r in rows)} configs")
    return out + [f"WARNING: {w}" for w in plan["warnings"]]


# --- report / verdict -------------------------------------------------------
# Job files are found as RUN_DIR/<job_id>, not via plan.json's absolute job_dir,
# so a run dir copied off the box still reports.
_PROGRESS = re.compile(r"\[(\d+)/(\d+)\] Split")
_B08 = re.compile(r"\[B08\]\[Rank (\d+)\] weights: loaded=\d+/\d+ missing=(\d+)")
_T01 = re.compile(r"\[T01\]\[rank(\d+)\] .*shm_prefix=(\S+)")


def _report(plan, run_dir):
    lines = [f"run {plan['run_id']} (smoke: {'yes' if plan['smoke'] else 'no'})"]
    rows, states = [], Counter()
    for lane in plan["lanes"]:
        pgid, lane_dead = _read(os.path.join(run_dir, "lanes", lane["lane"], "pgid")), False
        if pgid.isdigit():
            try:
                os.killpg(int(pgid), 0)
            except ProcessLookupError:
                lane_dead = True
            except OSError:  # EPERM: the group exists
                pass
        for j in lane["jobs"]:
            d = os.path.join(run_dir, j["job_id"])
            state = _read(os.path.join(d, "state")) or "-"
            if state == "running" and lane_dead:
                state = "stale"  # the lane died without recording an outcome
            states[state] += 1
            done = _PROGRESS.findall(_read(os.path.join(d, "benchmark.log")))
            table = _csv(os.path.join(d, "mig_benchmark_results.csv"))
            counts = "-"
            if table is not None:
                st = [r.get("status") or "" for r in table[1]]
                ok, oom = st.count("ok"), sum(s.startswith("OOM_") for s in st)
                counts = f"{ok}/{oom}/{len(st) - ok - oom}"
            rows.append((lane["lane"], lane["gpu"], lane["port"], j["job_id"].split("/", 1)[1],
                         state, "/".join(done[-1]) if done else "-", counts))
    lines += _table(("LANE", "GPU", "PORT", "JOB", "STATE", "PROGRESS", "OK/OOM/OTHER"), rows)
    return lines + ["states: " + ", ".join(f"{n} {s}" for s, n in states.items())]


def _job_checks(d, prefix, shm_dir):
    """[(ok, check, detail)] for one job of a finished smoke run; ok None means SKIP."""
    job, out = _load_json(os.path.join(d, "job.json")), []
    world = len(job["mig_uuids"])
    state, code = (_read(os.path.join(d, name)) or "-" for name in ("state", "exit_code"))
    out.append((state == "ok" and code == "0", "exit", f"state {state}, exit_code {code}"))
    rows = (_csv(os.path.join(d, "mig_benchmark_results.csv")) or ([], []))[1]
    not_ok = [r.get("status") for r in rows if r.get("status") != "ok"]
    out.append((bool(rows) and not not_ok, "results", f"{len(rows)} rows, not ok: {not_ok or '-'}"))
    # Every rank loaded every tensor; a rank's worst B08 line counts.
    missing = {}
    for r, z in _B08.findall(_read(os.path.join(d, "benchmark.log"))):
        missing[int(r)] = max(missing.get(int(r), 0), int(z))
    bad = [r for r in range(world) if missing.get(r) != 0]
    out.append((not bad, "weights", f"ranks without a missing=0 B08 line: {bad or '-'}"))
    # T01 prints the prefix the transport really used: proof MIG_SHM_PREFIX got through.
    level = re.search(r"^export MIG_LOG_LEVEL=(.*)$", _read(os.path.join(d, "job.env")), re.M)
    level = shlex.split(level.group(1))[0].strip().lower() if level else ""
    if level in ("off", "summary"):
        out.append((None, "shm_prefix", f"MIG_LOG_LEVEL={level} does not log T01"))
    else:
        seen = {(int(r), p) for path in glob.glob(os.path.join(d, "logs", "transport_*.log"))
                for r, p in _T01.findall(_read(path))}
        bad = [r for r in range(world) if (r, prefix) not in seen]
        shows = f"; T01 shows {sorted({p for _, p in seen})}" if bad else ""
        out.append((not bad, "shm_prefix", f"ranks without T01 for {prefix}: {bad or '-'}{shows}"))
    if job.get("mem_monitor") != "nvml":
        out.append((None, "memory", f"mem_monitor is {job.get('mem_monitor')}"))
    else:
        rows, bad = (_csv(os.path.join(d, "mig_memory_trace.csv")) or ([], []))[1], []
        for r, gb in enumerate(job["slice_gb"]):
            col = f"rank{r}_{gb}gb_mb"
            peak = max([float(row[col]) for row in rows if row.get(col)] or [0.0])
            # 0: the slice was never read. Above the slice's size: some other device was.
            if not 0 < peak <= gb * 1024 * 1.02:
                bad.append(f"{col} max {peak:g}")
        ok = bool(rows) and not bad
        out.append((ok, "memory", f"{len(rows)} samples, out of range: {bad or '-'}"))
    left = glob.glob(os.path.join(shm_dir, prefix + "_*"))
    out.append((not left, "shm_leftover", f"{len(left)} {prefix}_* file(s) in {shm_dir}"))
    return out


def _verdict(plan, run_dir):
    """The smoke checks, one line each, then the verdict. -> (lines, number of FAILs)"""
    shm_dir, lines = os.environ.get("MIG_SHM_DIR", "/dev/shm"), []
    for lane in plan["lanes"]:
        for j in lane["jobs"]:
            d = os.path.join(run_dir, j["job_id"])
            for ok, name, detail in _job_checks(d, j["shm_prefix"], shm_dir):
                status = "SKIP" if ok is None else "PASS" if ok else "FAIL"
                lines.append(f"{status}  {j['job_id']}  {name}: {detail}")
    fails = sum(line.startswith("FAIL") for line in lines)
    return lines + [f"VERDICT: FAIL ({fails})" if fails else "VERDICT: PASS"], fails


# --- CLI ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="parallel_plan.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("discover", help="paste-ready GPUS block from nvidia-smi -L")
    p.add_argument("--smi-l", metavar="FILE", help="saved `nvidia-smi -L` output to read instead")
    p = sub.add_parser("check", help="validate the config and show every job's sweep")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    p = sub.add_parser("plan", help="check, then write a run directory; the last line is its path")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    p.add_argument("--smoke", action="store_true", help="one tiny config per job")
    p.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")], metavar="0,3",
                   help="only these GPU indices")
    p = sub.add_parser("report", help="status of every job in a run directory")
    p.add_argument("run_dir")
    p.add_argument("--verdict", action="store_true", help="add the smoke checks; exit 1 on a FAIL")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "discover":
            text = _smi(["-L"], args.smi_l or os.environ.get("MIG_PARALLEL_SMI_L"))
            if text is None:
                _fail(["nvidia-smi not found; pass --smi-l FILE with saved `nvidia-smi -L` output"])
            print(_discover_block(parse_nvidia_smi_l(text)))
        elif args.cmd == "check":
            print("\n".join(_summary(build_plan(args.config)) + ["check: OK"]))
        elif args.cmd == "plan":
            plan = build_plan(args.config, smoke=args.smoke, gpus=args.gpus)
            run_dir = write_run_dir(plan)
            print("\n".join(_summary(plan) + [run_dir]))  # run_parallel.sh takes the last line
        else:
            plan = _load_json(os.path.join(args.run_dir, "plan.json"))
            lines, fails = _report(plan, args.run_dir), 0
            if args.verdict:
                verdict, fails = _verdict(plan, args.run_dir)
                lines += verdict
            print("\n".join(lines))
            return 1 if fails else 0
    except (PlanError, OSError) as e:
        print(f"parallel_plan.py {args.cmd}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
