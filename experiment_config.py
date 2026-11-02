"""
Per-job configuration for parallel runs. Torch-free on purpose.

benchmark_pipeline_microbatching.py hardcodes one run's setup (MODEL_NAME,
MIG_UUIDS, LAYER_LIMITS, ...). To run several of those side by side — one
per physical GPU — parallel_plan.py writes a job.json per (GPU, model) job
and run_parallel.sh points MIG_EXP_CONFIG at it. The benchmark's config
block then replaces its constants from that file.

MIG_EXP_CONFIG unset => load_job() returns None and the benchmark runs on its
own constants exactly as before. Nothing here changes a standalone run.

No torch/transformers imports: the planner, run_parallel.sh and tests/ all
import this on machines without torch.
"""

import json
import os
import re
from typing import Dict, List, Optional, Sequence

# Env var the lane runner sets to this job's job.json.
JOB_ENV_VAR = "MIG_EXP_CONFIG"

# Env var the SHM transport reads its segment-name prefix from. Every job gets
# its own prefix so parallel runs cannot attach to, or unlink, each other's
# slots. Unset => the transport's legacy "mig_pipe_shm" prefix.
SHM_PREFIX_ENV_VAR = "MIG_SHM_PREFIX"

JOB_SCHEMA_VERSION = 1

# dcgm: dcgm_mem_monitor.py (GPU 0 only, and not safe to run twice at once —
#       it pkills every `dcgmi dmon` on the box and shares one group name).
# nvml: nvml_mem_monitor.py (per-MIG-UUID reads, no sudo, parallel-safe).
# off:  no sampling; memory columns are 0.
MEM_MONITORS = ("dcgm", "nvml", "off")

_SHM_PREFIX_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")


class JobConfigError(ValueError):
    """job.json is missing a field or holds a value the benchmark cannot run."""


def _is_int(v) -> bool:
    # bool is an int subclass; JSON true/false must not pass as a layer count.
    return isinstance(v, int) and not isinstance(v, bool)


def load_job(env: Optional[Dict[str, str]] = None) -> Optional[dict]:
    """
    Read and validate the job.json named by MIG_EXP_CONFIG.

    Returns None when the variable is unset or empty — the standalone case.
    Raises JobConfigError (or OSError / json.JSONDecodeError) when it is set
    but unusable: a parallel job must fail loudly rather than fall back to
    the hardcoded Vicuna constants and silently run the wrong experiment.
    """
    env = os.environ if env is None else env
    path = (env.get(JOB_ENV_VAR) or "").strip()
    if not path:
        return None
    with open(path, "r") as f:
        job = json.load(f)
    validate_job(job)
    return job


def validate_job(job: dict) -> None:
    """Raise JobConfigError listing every problem, not just the first."""
    problems: List[str] = []

    def need(key, check, what):
        if key not in job:
            problems.append(f"missing '{key}'")
            return False
        if not check(job[key]):
            problems.append(f"'{key}' must be {what}, got {job[key]!r}")
            return False
        return True

    def pos_int(v):
        return _is_int(v) and v > 0

    need("schema", lambda v: v == JOB_SCHEMA_VERSION, f"{JOB_SCHEMA_VERSION}")
    need("model_path", lambda v: isinstance(v, str) and v, "a non-empty string")
    need("model_type", lambda v: isinstance(v, str) and v, "a non-empty string")
    need("num_layers", pos_int, "a positive int")
    need("hidden_size", pos_int, "a positive int")
    need("num_heads", pos_int, "a positive int")
    need("seq_len", pos_int, "a positive int")
    need("max_new_tokens", pos_int, "a positive int")
    need("max_runs", lambda v: v is None or pos_int(v), "null or a positive int")
    need(
        "enforce_slice_ordering", lambda v: isinstance(v, bool), "true or false"
    )
    need(
        "master_port",
        lambda v: _is_int(v) and 1024 <= v <= 65535,
        "an int in 1024..65535",
    )
    need(
        "shm_prefix",
        lambda v: isinstance(v, str) and bool(_SHM_PREFIX_RE.match(v)),
        "1-64 chars of [A-Za-z0-9_]",
    )
    need("mem_monitor", lambda v: v in MEM_MONITORS, f"one of {MEM_MONITORS}")
    if "min_last_rank_layers" in job:
        need("min_last_rank_layers", lambda v: _is_int(v) and v in (0, 1), "0 or 1")

    pairs_ok = need(
        "batch_mb_pairs",
        lambda v: isinstance(v, list)
        and len(v) > 0
        and all(
            isinstance(p, (list, tuple))
            and len(p) == 2
            and pos_int(p[0])
            and pos_int(p[1])
            for p in v
        ),
        "a non-empty list of [batch, microbatch] positive-int pairs",
    )
    if pairs_ok:
        for b, mb in job["batch_mb_pairs"]:
            if b % mb != 0:
                problems.append(
                    f"batch_mb_pairs: batch {b} not divisible by microbatch {mb}"
                )

    uuids_ok = need(
        "mig_uuids",
        lambda v: isinstance(v, list)
        and len(v) > 0
        and all(isinstance(u, str) and u.startswith("MIG-") for u in v),
        "a non-empty list of 'MIG-...' strings",
    )
    if uuids_ok and len(set(job["mig_uuids"])) != len(job["mig_uuids"]):
        problems.append("mig_uuids: the same MIG UUID appears twice")

    world = len(job["mig_uuids"]) if uuids_ok else None

    for key, lo in (("slice_gb", 1), ("layer_limits", 0)):
        if need(
            key,
            lambda v, lo=lo: isinstance(v, list)
            and all(_is_int(x) and x >= lo for x in v),
            f"a list of ints >= {lo}",
        ):
            if world is not None and len(job[key]) != world:
                problems.append(
                    f"'{key}' has {len(job[key])} entries but mig_uuids has {world}"
                )

    splits = job.get("splits")
    if "splits" in job and splits is not None:
        if not isinstance(splits, list) or not splits:
            problems.append("'splits' must be null or a non-empty list")
        else:
            for s in splits:
                # 0 is allowed: a rank with no decoder layers still runs its
                # embed / norm+lm_head and forwards activations.
                if not (isinstance(s, list) and all(_is_int(x) and x >= 0 for x in s)):
                    problems.append(f"splits: {s!r} is not a list of ints >= 0")
                    continue
                if world is not None and len(s) != world:
                    problems.append(f"splits: {s} has {len(s)} ranks, want {world}")
                if _is_int(job.get("num_layers")) and sum(s) != job["num_layers"]:
                    problems.append(
                        f"splits: {s} sums to {sum(s)}, model has "
                        f"{job['num_layers']} layers"
                    )

    if problems:
        raise JobConfigError("invalid job config:\n  - " + "\n  - ".join(problems))


def generate_layer_splits(
    total_layers: int,
    layer_limits: Sequence[int],
    slice_gb: Sequence[int],
    enforce_ordering: bool = True,
    min_last_rank_layers: int = 1,
) -> List[List[int]]:
    """
    Every way to distribute total_layers across the slices, subject to the
    per-slice capacity (layer_limits) and the symmetry-breaking ordering rule.

    Same algorithm, same output order as the benchmark's original
    generate_layer_splits(), which now delegates here so the planner can
    count and validate a sweep without importing torch.
    tests/test_experiment_config.py checks equivalence against a frozen copy
    of the original.

    Ordering rule (enforce_ordering=True): between neighbouring ranks, equal
    slices need prev >= next (permuting layers between identical slices only
    duplicates configurations), a bigger slice needs prev > next.

    min_last_rank_layers=0 lets the last rank hold no decoder layers — only
    the final norm + lm_head. That is how the qwen-14b-4mig branch swept
    Qwen2.5-14B (LAYER_LIMITS [30, 15, 7, 0]: the 5GB slice is nearly full
    with a 152k-vocab lm_head alone). With the default of 1 the output is
    exactly the original's; with 0 it is exactly that branch's.
    """
    valid_splits: List[List[int]] = []
    n = len(layer_limits)

    def _ok(prev_idx, prev_layers, layers):
        if not enforce_ordering:
            return True
        if slice_gb[prev_idx] == slice_gb[prev_idx + 1]:
            return prev_layers >= layers
        return prev_layers > layers

    def _recurse(rank, assigned, remaining):
        # Last rank takes whatever is left — no need to enumerate it.
        if rank == n - 1:
            if not (min_last_rank_layers <= remaining <= layer_limits[rank]):
                return
            if assigned and not _ok(rank - 1, assigned[-1], remaining):
                return
            valid_splits.append(assigned + [remaining])
            return

        # Leave at least one layer for each middle rank still to come, and
        # min_last_rank_layers for the last one.
        max_here = min(
            layer_limits[rank], remaining - (n - rank - 2) - min_last_rank_layers
        )
        for count in range(1, max_here + 1):
            if assigned and not _ok(rank - 1, assigned[-1], count):
                continue
            _recurse(rank + 1, assigned + [count], remaining - count)

    _recurse(0, [], total_layers)
    return valid_splits


def job_splits(job: dict) -> List[List[int]]:
    """The splits a job will sweep: its explicit 'splits', else the enumeration."""
    if job.get("splits"):
        return [list(s) for s in job["splits"]]
    return generate_layer_splits(
        job["num_layers"],
        job["layer_limits"],
        job["slice_gb"],
        job["enforce_slice_ordering"],
        job.get("min_last_rank_layers", 1),
    )


def count_configs(job: dict) -> int:
    """Configurations the benchmark will run for this job (after MAX_RUNS)."""
    total = len(job_splits(job)) * len(job["batch_mb_pairs"])
    if job.get("max_runs") is not None:
        total = min(total, job["max_runs"])
    return total


def front_loaded_split(
    total_layers: int,
    layer_limits: Sequence[int],
    slice_gb: Sequence[int],
    enforce_ordering: bool = True,
    min_last_rank_layers: int = 1,
) -> Optional[List[int]]:
    """
    The last split in enumeration order: the most layers on rank 0, then rank
    1, and so on — the fewest on the small slices. Used by smoke runs, which
    want the configuration least likely to OOM a 5GB slice. None if the
    limits admit no split at all.
    """
    splits = generate_layer_splits(
        total_layers, layer_limits, slice_gb, enforce_ordering, min_last_rank_layers
    )
    return splits[-1] if splits else None
