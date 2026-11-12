"""
Standalone measurements for the investigative runs. Nothing else should be running on GPU 0.

    python3 ncu_investigate.py

Two parts, written to investigation_ncu/:

  layer_scale.csv  The same layer alone on the 3g, 2g and 1g slice, at each microbatch size and
                   context length. Timed with torch.cuda.Event, no ncu.
  ncu_<config>_r<rank>.ncu-rep  Nsight Compute report per rank, for the GUI (open two to compare)
  ncu_ranks.csv    Each rank of each split alone on that rank's slice, under ncu (its first layer, one
                   decode step; the rank's other layers run unprofiled to keep the context real): SM and
                   DRAM utilization, busy SMs, DRAM bytes, L2 hit rate. Decode step only, covering
                   the layer stack. No transport and no other rank, because ncu serializes kernels.

With INV_SPLITS / INV_PAIRS set (see parallel_config.py) only those configs are profiled, and
INV_NCU_SCALE=0 skips layer_scale.csv; investigate.sh's per-pair step uses both. INV_NCU_RANKS=0
profiles rank 0 only. Each attempt's full output is saved as ncu_<config>_r<rank>[.sudo].out.txt.

The script re-runs itself as the worker, selected by INV_MODE.
"""

import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

CONFIG = {
    "model": "vicuna_13b",
    "seq_len": 64,
    "splits": [[23, 11, 4, 2], [20, 11, 5, 4], [21, 7, 6, 6], [23, 9, 4, 4], [17, 11, 6, 6]],
    "pairs": [(32, 16), (64, 2)],
    "gpu": 0,
    "ncu_context": 320,  # middle of a run: 64 prompt + half of 512 generated
    "ncu_steps": 1,
    "metrics": [
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__cycles_active.avg",
        "sm__cycles_elapsed.avg",
        "dram__bytes.sum",
        "lts__t_sector_hit_rate.pct",
    ],
    "scale_layer": 20,
    "scale_ranks": [0, 1, 2],  # slices 3g, 2g, 1g
    "scale_mbs": [2, 16],
    "scale_contexts": [64, 320, 576],
    "scale_reps": 25,
    "scale_warmup": 5,
    # Saved in each .ncu-rep for the Nsight Compute GUI; more sections = more replay passes.
    "gui_sections": ["SpeedOfLight", "SpeedOfLight_RooflineChart", "MemoryWorkloadAnalysis",
                     "Occupancy", "WarpStateStats", "LaunchStats"],
    "out_dir": "investigation_ncu",
}

RANGE_NAME = "INV_STACK"


# ---------------------------------------------------------------------------
# WORKER: runs on one slice, chosen by CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------------------------


def build(model_path, indices):
    import torch
    import torch.nn as nn
    from transformers import AutoConfig

    import model_family
    from helpers import load_specific_weights

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "sdpa"
    family = model_family.resolve(config.model_type)
    layers = nn.ModuleList(
        [family.decoder_layer(config, layer_idx=i).half().to(device) for i in indices])
    # rank 1 of 4: this process only owns decoder layers, not the embedding or lm_head.
    load_specific_weights(1, 4, model_path, layers, indices, {})
    layers.eval()
    return config, layers, family.rotary_embedding(config=config, device=device), device


def fill_cache(forward, layers, config, rope, device, mb, context):
    """A KV cache holding `context` tokens, as a real run has at that point."""
    import torch
    from transformers import DynamicCache

    cache = DynamicCache()
    hidden = torch.randn(mb, context, config.hidden_size, dtype=torch.float16, device=device)
    positions = torch.arange(0, context, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.full((1, 1, context, context), torch.finfo(torch.float16).min, device=device)
    mask = torch.triu(mask, diagonal=1).half()
    forward(layers, hidden, rope(hidden, positions), mask, cache).sum().item()
    return cache


def decode_input(config, rope, device, mb, position):
    import torch

    hidden = torch.randn(mb, 1, config.hidden_size, dtype=torch.float16, device=device)
    positions = torch.tensor([[position]], dtype=torch.long, device=device).expand(mb, -1).contiguous()
    return hidden, rope(hidden, positions)


def worker_rank(job):
    """Decode through the rank's layers; only the first layer sits in the NVTX range ncu profiles.
    Layers on one rank have the same shapes, context and slice, so one layer stands for all of them,
    at a fraction of ncu's replay time."""
    import torch

    from benchmark_pipeline_microbatching import forward_through_layers as forward

    config, layers, rope, device = build(job["model_path"], job["indices"])
    print("SM_COUNT", torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
    with torch.no_grad():
        cache = fill_cache(forward, layers, config, rope, device, job["mb"], job["context"])
        for step in range(job["steps"]):
            hidden, position_embeddings = decode_input(config, rope, device, job["mb"],
                                                       job["context"] + step)
            torch.cuda.nvtx.range_push(RANGE_NAME)
            out = forward(layers[:1], hidden, position_embeddings, None, cache)
            torch.cuda.nvtx.range_pop()
            out = forward(layers[1:], out, position_embeddings, None, cache)
            out.sum().item()


def worker_scale(job):
    """One layer at each (mb, context); events are read after the loop finishes."""
    import torch

    from benchmark_pipeline_microbatching import forward_through_layers as forward

    config, layers, rope, device = build(job["model_path"], [job["layer"]])
    with torch.no_grad():
        for mb in job["mbs"]:
            for context in job["contexts"]:
                cache = fill_cache(forward, layers, config, rope, device, mb, context)
                inputs = [decode_input(config, rope, device, mb, context + i)
                          for i in range(job["reps"])]
                starts = [torch.cuda.Event(enable_timing=True) for _ in inputs]
                ends = [torch.cuda.Event(enable_timing=True) for _ in inputs]
                for i, (hidden, position_embeddings) in enumerate(inputs):
                    starts[i].record()
                    forward(layers, hidden, position_embeddings, None, cache)
                    ends[i].record()
                ends[-1].synchronize()  # one event, after the loop; no device-wide sync
                times = [s.elapsed_time(e) for s, e in zip(starts, ends)][job["warmup"]:]
                print("RESULT", json.dumps({
                    "mb": mb, "context": context, "reps": len(times),
                    "median_ms": statistics.median(times), "min_ms": min(times),
                    "max_ms": max(times)}), flush=True)


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------


def run(cmd, env=None, timeout=3600):
    full = dict(os.environ, **(env or {}))
    try:
        r = subprocess.run(cmd, env=full, timeout=timeout, text=True, capture_output=True)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except OSError as e:
        return 127, f"{type(e).__name__}: {e}"


def worker_env(uuid, mode, job):
    return {"CUDA_VISIBLE_DEVICES": uuid, "INV_MODE": mode, "INV_JOB": json.dumps(job),
            "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))}


def parse_ncu_csv(path, metrics):
    """Per-kernel rows of an ncu --csv log: [{metric: value}]; the units row is dropped."""
    if not os.path.exists(path):
        return []
    with open(path, errors="ignore") as f:
        rows = list(csv.reader(line for line in f if line.startswith('"')))
    head = next((r for r in rows if r and r[0] == "ID"), None)
    if head is None or any(m not in head for m in metrics):
        return []
    out = []
    for row in rows:
        if row is head or len(row) < len(head):
            continue
        try:
            out.append({m: float(row[head.index(m)].replace(",", "")) for m in metrics})
        except ValueError:
            continue  # units row
    return out


def aggregate(rows, sm_count):
    time_key = "gpu__time_duration.sum"
    total = sum(r[time_key] for r in rows)

    def weighted(metric):
        return sum(r[metric] * r[time_key] for r in rows) / total if total else ""

    elapsed = sum(r["sm__cycles_elapsed.avg"] for r in rows)
    active = sum(r["sm__cycles_active.avg"] for r in rows)
    busy_fraction = active / elapsed if elapsed else 0.0
    return {
        "n_kernels": len(rows),
        "kernel_time_s": total,
        "sm_util_pct": weighted("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "dram_util_pct": weighted("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "l2_hit_pct": weighted("lts__t_sector_hit_rate.pct"),
        "dram_bytes": sum(r["dram__bytes.sum"] for r in rows),
        "sm_count": sm_count,
        "busy_sms": busy_fraction * sm_count if sm_count else "",
    }


def ncu_attempt(ncu, cmd, env, report, metrics):
    """(exit code, output, rows). Profiles into report.ncu-rep (opens in the Nsight Compute GUI),
    then reads the per-kernel metrics back from it. Starts from no report; a run that exits
    non-zero gives no rows."""
    path = report + ".ncu-rep"
    if os.path.exists(path):
        os.remove(path)
    rc, out = run(cmd, env=env)
    with open(report + (".sudo" if cmd[0] == "sudo" else "") + ".out.txt", "w") as f:
        f.write(f"$ {' '.join(cmd)}\nexit {rc}\n\n{out}")  # full output, for diagnosing failures
    if rc != 0 or not os.path.exists(path):
        return rc, out, []
    # --page raw: one row per kernel launch and one column per metric.
    rc_csv, text = run([ncu, "--import", path, "--csv", "--page", "raw", "--print-units", "base"])
    with open(report + ".csv", "w") as f:
        f.write(text)
    return rc, out, parse_ncu_csv(report + ".csv", metrics) if rc_csv == 0 else []


def profile_rank(ncu, job, uuid, report, metrics):
    """(aggregate dict or None, note). Hardware counters may need admin rights: tries plain, then sudo."""
    env = worker_env(uuid, "rank", job)
    # Match ncu_attempt's import path even when ncu defaults to compressed .ncu-repz reports.
    # --clock-control none: MIG does not allow locking clocks, and unlocked clocks match the real runs.
    args = [ncu, "-o", report + ".ncu-rep", "--clock-control", "none", "--target-processes", "all",
            "--nvtx", "--nvtx-include", RANGE_NAME + "/", "--metrics", ",".join(metrics)]
    for section in CONFIG["gui_sections"]:
        args += ["--section", section]
    args += [sys.executable, os.path.abspath(__file__)]
    rc, out, rows = ncu_attempt(ncu, args, env, report, metrics)
    note = "ok"
    if not rows:
        sudo = ["sudo", "-n", "env"] + [f"{k}={v}" for k, v in env.items()]
        rc2, out2, rows = ncu_attempt(ncu, sudo + args, None, report, metrics)
        out = out + "\n" + out2
        if rows:
            note = "needed sudo"
        else:
            # The error, not the progress bars: ncu's ==ERROR== lines and Python tracebacks first.
            lines = [l.strip() for l in out.splitlines() if l.strip() and "shards:" not in l]
            errors = [l for l in lines if "==ERROR==" in l or "Error" in l or "Traceback" in l]
            note = f"FAILED rc={rc}/{rc2}: " + " | ".join((errors or lines)[-6:])[:600] + \
                   f" (full output: {os.path.basename(report)}.out.txt)"
    if not rows:
        return None, note
    m = re.search(r"SM_COUNT (\d+)", out)
    return aggregate(rows, int(m.group(1)) if m else 0), note


def run_scale(model_path, uuids, slice_gb, out_dir):
    """True when every slice produced results."""
    ok = True
    path = os.path.join(out_dir, "layer_scale.csv")
    job = {"model_path": model_path, "layer": CONFIG["scale_layer"], "mbs": CONFIG["scale_mbs"],
           "contexts": CONFIG["scale_contexts"], "reps": CONFIG["scale_reps"],
           "warmup": CONFIG["scale_warmup"]}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice_gb", "layer", "mb", "context", "reps", "median_ms", "min_ms", "max_ms"])
        for rank in CONFIG["scale_ranks"]:
            print(f"layer timing on {slice_gb[rank]} GB slice", flush=True)
            rc, out = run([sys.executable, os.path.abspath(__file__)],
                          env=worker_env(uuids[rank], "scale", job))
            found = [json.loads(line[7:]) for line in out.splitlines() if line.startswith("RESULT ")]
            if rc != 0 or not found:
                print(f"  FAILED rc={rc}: {out.strip()[-600:]}", flush=True)
                ok = False
            for r in found:
                w.writerow([slice_gb[rank], CONFIG["scale_layer"], r["mb"], r["context"], r["reps"],
                            round(r["median_ms"], 4), round(r["min_ms"], 4), round(r["max_ms"], 4)])
                print(f"  mb {r['mb']:>2} context {r['context']:>3}: {r['median_ms']:.3f} ms", flush=True)
            f.flush()
    print(f"wrote {path}", flush=True)
    return ok


def run_ncu(model_path, uuids, slice_gb, out_dir):
    """True when every rank of every config was profiled."""
    ncu = shutil.which("ncu")
    if ncu is None:
        print("ncu not found: no hardware counters", flush=True)
        return False
    path = os.path.join(out_dir, "ncu_ranks.csv")
    columns = ["n_kernels", "kernel_time_s", "sm_util_pct", "dram_util_pct", "l2_hit_pct",
               "dram_bytes", "sm_count", "busy_sms"]
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:  # appended: one call per pair adds that pair's rows
        w = csv.writer(f)
        if new_file:
            w.writerow(["split", "batch", "mb", "rank", "slice_gb", "layers", "profiled_layer", "context"]
                   + columns + ["note"])
        for split in CONFIG["splits"]:
            for batch, mb in CONFIG["pairs"]:
                for rank, count in enumerate(split):
                    if CONFIG["ranks"] is not None and rank not in CONFIG["ranks"]:
                        continue
                    first = sum(split[:rank])
                    job = {"model_path": model_path, "indices": list(range(first, first + count)),
                           "mb": mb, "context": CONFIG["ncu_context"], "steps": CONFIG["ncu_steps"]}
                    name = "_".join(map(str, split)) + f"_b{batch}_mb{mb}_r{rank}"
                    print(f"ncu {name}", flush=True)
                    result, note = profile_rank(ncu, job, uuids[rank],
                                                os.path.join(out_dir, f"ncu_{name}"),
                                                CONFIG["metrics"])
                    print(f"  {result if result else ''} ({note})", flush=True)
                    values = [result[c] for c in columns] if result else [""] * len(columns)
                    w.writerow(["_".join(map(str, split)), batch, mb, rank, slice_gb[rank], count, first,
                                CONFIG["ncu_context"]] + values + [note])
                    f.flush()
                    if result is None:
                        print("ncu failed; stopping here so one problem is not repeated 40 times. "
                              f"Paste the note above. Log: {name}", flush=True)
                        return False
    print(f"wrote {path}", flush=True)
    return True


def main():
    mode = os.environ.get("INV_MODE")
    if mode == "rank":
        worker_rank(json.loads(os.environ["INV_JOB"]))
        return
    if mode == "scale":
        worker_scale(json.loads(os.environ["INV_JOB"]))
        return

    import parallel_config

    # Model, splits and pairs follow the box (parallel_config.INV_SETUP).
    inv = parallel_config.INV
    CONFIG.update(model=inv["model"], splits=inv["core"], pairs=inv["pairs"],
                  scale_mbs=sorted({mb for _, mb in inv["pairs"]}))
    # investigate.sh's per-pair step profiles that pair's best and worst split only.
    if parallel_config.INV_SPLITS and parallel_config.INV_PAIRS:
        CONFIG.update(splits=parallel_config.INV_SPLITS, pairs=parallel_config.INV_PAIRS)
    do_scale = os.environ.get("INV_NCU_SCALE", "1") != "0"
    # INV_NCU_RANKS=0 profiles only rank 0: a quick check that ncu works at all.
    ranks = os.environ.get("INV_NCU_RANKS", "").strip()
    CONFIG["ranks"] = [int(r) for r in ranks.split(",")] if ranks else None
    print(f"setup {parallel_config.INV_SETUP}: {inv['model']}, splits {inv['core']}, pairs {inv['pairs']}",
          flush=True)
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    entry = next(g for g in parallel_config.GPUS if g["gpu"] == CONFIG["gpu"])
    model_path = parallel_config.MODELS[CONFIG["model"]]["path"]
    scale_ok = run_scale(model_path, entry["mig_uuids"], entry["slice_gb"], CONFIG["out_dir"]) if do_scale else True
    ncu_ok = run_ncu(model_path, entry["mig_uuids"], entry["slice_gb"], CONFIG["out_dir"])
    if not (scale_ok and ncu_ok):
        print(f"FAILED: layer timing {'ok' if scale_ok else 'incomplete'}, "
              f"ncu {'ok' if ncu_ok else 'incomplete'}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
