"""
Turns one box's exported probe folder into a constants set for the ILPs, named by the box, and
writes it to probe_sets/<box>.json, which hardware_constants.py loads as CONSTANTS["<box>"]. Edit
CONFIG in main(), then

    python3 mig_pp/milp/probe_constants.py

Reads from each folder (as exported from the box):
    constants_cpu.json   measure_constants_cpu.py: launch time per kernel, host copies, gloo
                         messages, token staging, step start (median of the probe's windows)
    constants_mig.json   measure_constants_mig.py: the generic GPU primitives of each MIG slice
                         (first pass; the repeat pass and the raw points stay in the JSON as checks)
    probe_cpu_interactions.json  probe_cpu_interactions.py: the governor's duty-cycle curve (slow
                         factor, busy threshold per step period) and the pooled receive copy

graph_placement composes every GPU time from these primitives and their measured points (layer_ops)
and takes the CPU clock and the thread pool from the last file. Torch-free.
"""

import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_limits  # noqa: E402
import layer_ops  # noqa: E402

CPU_KEYS = ("launch_us", "kernels_per_layer", "rotary_kernels", "memcpy_bw", "shm_to_pinned_bw",
            "handshake_ack_us", "tok_roundtrip_us", "token_staging_us", "step_start_us", "barrier_us")
NOT_MEASURED = dict(timing_event_calls=2, send_queue_calls=2,  # CPU-only CUDA calls, read off the harness code
                    pool_min_values=32768)  # PyTorch's GRAIN_SIZE: a host copy of more values runs on its thread pool
CLOCK_FAST_SPEED = 0.7  # a duty-cycle cell ran at full clock if >= 0.7 x the 100%-busy reference (slow level ~0.43)


def median(v):
    """A probe number: {median, spread, windows} (probe v2) or a plain number (probe run 1)."""
    return v["median"] if isinstance(v, dict) else v


def clock_from_probe(inter):
    """(slow factor, {step period ms: busy threshold}) from the governor duty-cycle probe: the slow
    level is the median speed of the cells below CLOCK_FAST_SPEED (factor = 1 / level); per period,
    the threshold is the lowest busy fraction from which every higher one ran at full clock."""
    slow, thresholds = [], {}
    for period, row in inter["governor"]["speed"].items():
        cells = sorted((float(f), s) for f, s in row.items())
        edge = None
        for f, s in reversed(cells):
            if s < CLOCK_FAST_SPEED:
                break
            edge = f
        thresholds[int(period)] = edge if edge is not None else 1.0  # never fast: always below
        slow += [s for _, s in cells if s < CLOCK_FAST_SPEED]
    return (1 / statistics.median(slow) if slow else 1.0), thresholds


def pool_from_probe(inter):
    """(us per pooled receive copy, launch slowdown next to it) from the interference sweep: one pooled
    copier (65,536 values, default threads) next to a process launching back to back -- in a pipeline
    the receives of different ranks rarely overlap and the other ranks keep launching. (None, None)
    for probe files without that row (before probe v3)."""
    rows = inter.get("interference")
    for r in rows if isinstance(rows, list) else []:
        if r["copier"] == [65536, 1, 25, None] and r["dispatch_gap_ms"] == 0:
            worst = [max(c, key=lambda q: q["median"]) for c in r["copiers"]]
            return statistics.median(w["median"] for w in worst), r["slowdown"]
    return None, None


def from_probes(folder):
    """The constants set of one box: CPU numbers + slices {GB: primitive fits} + the CPU clock and the
    thread pool when the folder has probe_cpu_interactions.json."""
    folder = os.path.expanduser(folder)
    cpu = json.load(open(f"{folder}/constants_cpu.json"))
    mig = json.load(open(f"{folder}/constants_mig.json"))
    out = {k: median(cpu[k]) for k in CPU_KEYS}
    out.update(NOT_MEASURED)
    if os.path.exists(f"{folder}/probe_cpu_interactions.json"):
        inter = json.load(open(f"{folder}/probe_cpu_interactions.json"))
        out["clock_slow_factor"], out["clock_threshold_by_period"] = clock_from_probe(inter)
        pool_us, slowdown = pool_from_probe(inter)
        if pool_us is not None:
            out.update(recv_copy_pool_us=pool_us, pool_launch_slowdown=slowdown)
    if "launch_harness" in cpu:  # launch as the harness issues a stack, with the CPU clock (probe v7 on)
        h = cpu["launch_harness"]
        out["launch_harness"] = dict(h["methods"], kernels_per_layer=h["kernels_per_layer"], layers=h["layers"])
    out["slices"] = {s["gb"]: slice_set(s) for s in sorted(mig, key=lambda s: -s["gb"])}
    out["source"] = folder
    return out


POINT_DROP = ("replays_us", "residual", "method")  # kept in the probe's own JSON, not needed to compose


def slice_set(s):
    """One slice's primitives: every fit, every first-pass measured point (time + shape), and the
    slowdown with the card's other slices busy when the probe measured it."""
    out = {k: v for k, v in s.items() if k.endswith(("_fixed_us", "_rate"))}
    if "concurrency" in s:  # slowdown with the card's other slices busy (probe v6 on)
        out["concurrency"] = s["concurrency"]
    if "boundary_gap" in s:  # the GPU's idle gap between kernels (probe v7 on); raw windows stay in the probe JSON
        out["boundary_gap"] = {mb: {k: v for k, v in r.items() if k != "windows"} for mb, r in s["boundary_gap"]["by_mb"].items()}
    for k, pts in s.items():
        if k.endswith("_points"):
            out[k] = [{kk: vv for kk, vv in p.items() if kk not in POINT_DROP} for p in pts]
    return out


def save(consts, path):
    """Writes a set as JSON (slices keyed by GB as text; load() turns them back)."""
    with open(path, "w") as f:
        json.dump(consts, f, indent=1)


def load(path):
    """A set written by save(), with int keys again (slices by GB, thresholds by period ms)."""
    with open(path) as f:
        s = json.load(f)
    s["slices"] = {int(gb): prims for gb, prims in s["slices"].items()}
    if "clock_threshold_by_period" in s:
        s["clock_threshold_by_period"] = {int(p): v for p, v in s["clock_threshold_by_period"].items()}
    return s


def main():
    CONFIG = dict(
        boxes={  # set name -> exported probe folder (constants_cpu.json, constants_mig.json, probe_cpu_interactions.json)
            "mig1": "~/Desktop/AP/logs/mig1_probes_v6",
            "mig2": "~/Desktop/AP/logs/mig2_probes_v6",
        },
        out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_sets"),
        check_model="vicuna_7b",  # the probe counted kernels on this model's decode layer (mb 1, ctx 64)
    )
    c = CONFIG
    shape = memory_limits.REFERENCE_SHAPES[c["check_model"]]
    ops_layer = len(layer_ops.layer_ops_from_shape(shape, 1, 64))
    ops_rotary = len(layer_ops.extras_from_shape(shape, 1, 64)["rotary"])
    os.makedirs(c["out_dir"], exist_ok=True)

    for name, folder in c["boxes"].items():
        consts = from_probes(folder)
        host = json.load(open(os.path.join(os.path.expanduser(folder), "constants_cpu.json")))["host"]
        nan = [f"{gb}:{k}" for gb, p in consts["slices"].items() for k, v in p.items()
               if isinstance(v, float) and math.isnan(v)]
        same = (consts["kernels_per_layer"], consts["rotary_kernels"]) == (ops_layer, ops_rotary)
        path = os.path.join(c["out_dir"], f"{name}.json")
        save(consts, path)
        print(f"\n{name}: {folder}; {host.get('cpu')}, governor {host.get('governor')}, "
              f"OMP_NUM_THREADS {host.get('omp_num_threads') or 'unset'}, transformers {host.get('transformers', '?')}")
        print(f"  slices (GB): {list(consts['slices'])}; points per slice: "
              f"{[sum(len(v) for k, v in p.items() if k.endswith('_points')) for p in consts['slices'].values()]}")
        print(f"  kernels per decode layer / rotary: probe {consts['kernels_per_layer']} / {consts['rotary_kernels']}, "
              f"op lists {ops_layer} / {ops_rotary} ({'match' if same else 'MISMATCH: check layer_ops'})")
        print(f"  launch {consts['launch_us']:.2f} us per kernel; clock: slow factor {consts.get('clock_slow_factor', '-')}, thresholds {consts.get('clock_threshold_by_period', '-')}; "
              f"pool: {consts.get('recv_copy_pool_us', '-')} us, launches x{consts.get('pool_launch_slowdown', '-')}")
        conc = {gb: p["concurrency"]["factors"] for gb, p in consts["slices"].items() if "concurrency" in p}
        print(f"  slowdown with the other slices busy: " + ("; ".join(
            f"{gb}GB " + ", ".join(f"{k} x{v:.3f}" for k, v in f.items()) for gb, f in conc.items()) or "not measured (before probe v6)"))
        gaps = {gb: {mb: round(r["gap_queued_us_per_kernel"], 2) for mb, r in p["boundary_gap"].items()}
                for gb, p in consts["slices"].items() if "boundary_gap" in p}
        print("  kernel-boundary gap, us per kernel (queued): " + (str(gaps) if gaps else "not measured (before probe v7)"))
        if "launch_harness" in consts:
            print("  launch as the harness issues it: " + ", ".join(
                f"{k} {v['us_per_kernel']:.2f} us/kernel" for k, v in consts["launch_harness"].items() if isinstance(v, dict)))
        if nan:
            print(f"  NOT FITTED (nan): {', '.join(nan)}")
        print(f"  wrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
