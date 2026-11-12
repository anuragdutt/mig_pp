"""
Joins the investigative runs into a few CSVs. Torch-free; edit CONFIG in main() and run, with
the setup the runs came from (on the Mac it defaults to 80gb):

    INV_SETUP=40gb python3 analyze_investigation.py

Outputs in CONFIG["out_dir"]:
  overhead.csv    latency of each config: control vs layer-events vs DCGM run
  ranks.csv       one row per config and rank: where the rank's time goes, bottleneck share,
                  memory, transport, CPU, DCGM and ncu values
  layers.csv      one row per config, rank, layer and phase: median layer time and FLOPs
  neighbors.csv   each neighbor split against its base split
  copies.csv      median transport copy timings per config and rank (verbose run)
  power.csv       power, clocks and throttling per config and run
  bottleneck.csv  per config and sampled decode step: slowest rank's stack time
  token_wait.csv  rank 0's wait for the next token each step (events run)
  extreme.csv     rank-0-heavier splits against the winner [23,11,4,2], from the same run
"""

import ast
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

import memory_limits
import parallel_config  # INV: model, slices and reference splits of the setup (INV_SETUP)

JOB_GLOB = "gpu0/*"
TAGS_SUMMARY = re.compile(r"\[(T2[3-6])\]\[rank(\d)\].*?mean=([\d.]+)ms(?: total=([\d.]+)ms)? over (\d+)")
SECTION = re.compile(r"\[T22\]\[rank(\d)\].*split=(\[[^\]]*\]) mb=(\d+) latency=(\d+)ms")
LINE_TIME = re.compile(r"^(\d\d:\d\d:\d\d\.\d+)")
DETAIL = re.compile(r"\[(T11|T14|T15)\]\[rank(\d)\].*?(?:received in|queued in|blocked=)\s*([\d.]+)\s*(us|µs|ms|s)\b")


def key_of(split_text):
    return "_".join(str(x) for x in ast.literal_eval(split_text))


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def job_dirs(run_dir):
    if not run_dir:
        return []
    return sorted(d for d in glob.glob(os.path.join(run_dir, JOB_GLOB)) if os.path.isdir(d))


def results(run_dir):
    """(split, batch, mb) -> results row, status ok only."""
    out = {}
    for d in job_dirs(run_dir):
        for r in read_csv(os.path.join(d, "mig_benchmark_results.csv")):
            if r["status"] == "ok":
                out[(key_of(r["split"]), int(r["batch_size"]), int(r["microbatch_size"]))] = r
    return out


def clock_s(text):
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def in_window(t, window):
    """t and window are seconds of the day; a start below 0 means inference began before midnight."""
    start, end = window
    return start <= t <= end or (start < 0 and t >= start + 86400)


def windows(run_dir, res):
    """config -> (start, end) of the timed inference, in seconds of the day.

    Rank 0's T22 line is written right after inference ends, for every config. It has no batch
    size, so the row is the one with this split and microbatch and the closest latency."""
    out = {}
    for d in job_dirs(run_dir):
        for path in glob.glob(os.path.join(d, "logs", "transport_*.log")):
            with open(path, errors="ignore") as f:
                for line in f:
                    section, stamp = SECTION.search(line), LINE_TIME.match(line)
                    if not section or not stamp or section[1] != "0":
                        continue
                    split, mb, latency = key_of(section[2]), int(section[3]), float(section[4])
                    same = [k for k in res if k[0] == split and k[2] == mb]
                    if same:
                        key = min(same, key=lambda k: abs(float(res[k]["total_latency_ms"]) - latency))
                        end = clock_s(stamp[1])
                        out[key] = (end - latency / 1000.0, end)
    return out


def transport_summaries(run_dir, batches):
    """(split, mb, rank) -> {tag: (mean_ms, total_ms)}; only used to join on configs of one batch."""
    out = defaultdict(dict)
    current = {}
    for d in job_dirs(run_dir):
        for path in glob.glob(os.path.join(d, "logs", "transport_*.log")):
            with open(path, errors="ignore") as f:
                for line in f:
                    m = SECTION.search(line)
                    if m:
                        current[int(m[1])] = (key_of(m[2]), int(m[3]), int(m[1]))
                        continue
                    m = TAGS_SUMMARY.search(line)
                    if m and int(m[2]) in current:
                        total = float(m[4]) if m[4] else float(m[3]) * int(m[5])  # logged total is exact
                        out[current[int(m[2])]][m[1]] = (float(m[3]), total)
    return out


def copy_details(run_dir):
    """Verbose run. (split, mb, rank) -> {tag: [ms, ...]}. Detail lines come before their T22 line."""
    pending = defaultdict(lambda: defaultdict(list))
    out = {}
    for d in job_dirs(run_dir):
        for path in glob.glob(os.path.join(d, "logs", "transport_*.log")):
            with open(path, errors="ignore") as f:
                for line in f:
                    m = DETAIL.search(line)
                    if m:
                        scale = {"us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1000.0}[m[4]]
                        pending[int(m[2])][m[1]].append(float(m[3]) * scale)
                        continue
                    m = SECTION.search(line)
                    if m:
                        rank = int(m[1])
                        out[(key_of(m[2]), int(m[3]), rank)] = pending.pop(rank, {})
    return out


def layer_rows(run_dir):
    out = []
    for d in job_dirs(run_dir):
        for path in sorted(glob.glob(os.path.join(d, "layer_latency_rank*.csv"))):
            out += read_csv(path)
    return out


def cpu_rows(run_dir):
    out = {}
    for d in job_dirs(run_dir):
        for path in sorted(glob.glob(os.path.join(d, "cpu_time_rank*.csv"))):
            for r in read_csv(path):
                out[(r["split"], int(r["batch"]), int(r["mb"]), int(r["rank"]))] = r
    return out


def layer_flops(shape, mb, tokens, context):
    """Multiplies + adds of one decoder layer: tokens = new tokens, context = tokens attended over."""
    H, I, heads, kv, hd = shape["H"], shape["I"], shape["heads"], shape["kv"], shape["hd"]
    rows = mb * tokens

    def linear(in_size, out_size):
        return 2 * rows * in_size * out_size

    flops = linear(H, heads * hd) + 2 * linear(H, kv * hd)  # q, k, v
    flops += 2 * 2 * mb * tokens * context * heads * hd  # q.k scores, then scores.v
    flops += linear(heads * hd, H)  # o_proj
    flops += 2 * linear(H, I) + linear(I, H)  # MLP gate, up, down
    flops += 2 * 3 * rows * H  # two RMSNorms: square, sum, scale
    return flops


def stats_by_layer(rows):
    """((split,batch,mb,rank,layer,phase)) -> [ms] and sampled-context list."""
    out = defaultdict(list)
    for r in rows:
        phase = "prefill" if int(r["step"]) == 0 else "decode"
        out[(r["split"], int(r["batch"]), int(r["mb"]), int(r["rank"]), int(r["layer"]), phase)].append(float(r["layer_ms"]))
    return out


def bottleneck(rows):
    """Per config: stack time per (step, mb_idx, rank), for the decode steps only."""
    stacks = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if int(r["step"]) == 0:
            continue
        cfg = (r["split"], int(r["batch"]), int(r["mb"]))
        stacks[cfg][(int(r["step"]), int(r["mb_idx"]), int(r["rank"]))] += float(r["layer_ms"])
    return stacks


def main():
    CONFIG = {
        "model": parallel_config.INV["model"],
        "seq_len": 64,
        "slice_gb": parallel_config.INV["slice_gb"],
        # Each is a run directory under runs/ (or logs/). "" = skipped.
        "control": "",
        "events": "",
        "dcgm": "",
        "neighbors32": "",
        "neighbors64": "",
        "transport": "",
        "extreme": "",
        "ncu_dir": "investigation_ncu",
        "samples_dir": "investigation_samples",
        "sample_tags": {"control": "core_control", "events": "core_events", "dcgm": "core_dcgm"},
        "out_dir": "analysis/investigation",
    }
    # Run directories that investigate.sh recorded, for any CONFIG entry left empty above. The file
    # is append-only, so a step that was rerun uses its latest run (its samples are the latest too).
    if os.path.exists("investigation_runs.txt"):
        latest = {}
        with open("investigation_runs.txt") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    latest[parts[0]] = parts[1]
        for step, run_dir in latest.items():
            if CONFIG.get(step) == "":
                CONFIG[step] = run_dir
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    out = CONFIG["out_dir"]
    shape = memory_limits.REFERENCE_SHAPES[CONFIG["model"]]

    control = results(CONFIG["control"])
    events = results(CONFIG["events"])
    dcgm_res = results(CONFIG["dcgm"])

    rows = []
    for key in sorted(control):
        c = float(control[key]["total_latency_ms"])
        row = list(key) + [round(c, 1)]
        for other in (events, dcgm_res):
            if key in other:
                v = float(other[key]["total_latency_ms"])
                row += [round(v, 1), round((v / c - 1) * 100, 3)]
            else:
                row += ["", ""]
        rows.append(row)
    write_csv(os.path.join(out, "overhead.csv"),
              ["split", "batch", "mb", "control_ms", "events_ms", "events_pct", "dcgm_ms", "dcgm_pct"], rows)

    ev_rows = layer_rows(CONFIG["events"])
    by_layer = stats_by_layer(ev_rows)
    layer_out = []
    decode_steps = sorted({int(r["step"]) for r in ev_rows if int(r["step"]) > 0})
    mid_context = CONFIG["seq_len"] + (st.mean(decode_steps) if decode_steps else 0)
    for (split, batch, mb, rank, layer, phase), v in sorted(by_layer.items()):
        tokens, context = (CONFIG["seq_len"], CONFIG["seq_len"]) if phase == "prefill" else (1, mid_context)
        layer_out.append([split, batch, mb, rank, layer, phase, len(v), round(st.median(v), 4),
                          round(min(v), 4), round(max(v), 4), int(layer_flops(shape, mb, tokens, context))])
    write_csv(os.path.join(out, "layers.csv"),
              ["split", "batch", "mb", "rank", "layer", "phase", "samples", "median_ms", "min_ms", "max_ms", "flops"],
              layer_out)

    stacks = bottleneck(ev_rows)
    bottleneck_rows, share = [], defaultdict(lambda: defaultdict(int))
    for cfg, table in stacks.items():
        points = defaultdict(dict)
        for (step, mb_idx, rank), ms in table.items():
            points[(step, mb_idx)][rank] = ms
        for (step, mb_idx), ranks in sorted(points.items()):
            slowest = max(ranks, key=ranks.get)
            share[cfg][slowest] += 1
            bottleneck_rows.append(list(cfg) + [step, mb_idx, slowest, round(ranks[slowest], 3)]
                                   + [round(ranks.get(r, 0.0), 3) for r in range(4)])
    write_csv(os.path.join(out, "bottleneck.csv"),
              ["split", "batch", "mb", "step", "mb_idx", "slowest_rank", "slowest_ms",
               "rank0_ms", "rank1_ms", "rank2_ms", "rank3_ms"], bottleneck_rows)

    tsum = transport_summaries(CONFIG["control"], None)
    cpu = cpu_rows(CONFIG["events"])
    ncu = {(r["split"], int(r["batch"]), int(r["mb"]), int(r["rank"])): r
           for r in read_csv(os.path.join(CONFIG["ncu_dir"], "ncu_ranks.csv"))}
    dcgm_samples = read_csv(os.path.join(CONFIG["samples_dir"], CONFIG["sample_tags"]["dcgm"] + "_dcgm.csv"))
    dcgm_win = windows(CONFIG["dcgm"], dcgm_res)
    rank_rows = []
    for key in sorted(control):
        split, batch, mb = key
        r = control[key]
        layers_by_rank = [int(x) for x in split.split("_")]
        total_points = sum(share[key].values()) if key in share else 0
        win = dcgm_win.get(key)
        for rank, n_layers in enumerate(layers_by_rank):
            decode = float(r[f"decode_ms_rank{rank}"])
            compute_mean = float(r[f"decode_compute_mean_ms_rank{rank}"])
            max_new = int(r["max_new_tokens"])
            n_mb = batch // mb
            compute_total = compute_mean * max_new * n_mb
            ts = tsum.get((split, mb, rank), {})
            # Per config in the results CSV. The memory trace file keeps only a job's last config.
            peak = float(r[f"peak_rank{rank}_{CONFIG['slice_gb'][rank]}gb_mb"])
            mean_mem = float(r[f"avg_rank{rank}_{CONFIG['slice_gb'][rank]}gb_mb"])
            cap = memory_limits.slice_mibs(CONFIG["slice_gb"])[rank]
            n = ncu.get((split, batch, mb, rank), {})
            c = cpu.get((split, batch, mb, rank), {})
            sm, dram = [], []
            if win:
                for s in dcgm_samples:
                    if int(s["rank"]) == rank and in_window(clock_s(s["timestamp"]), win):
                        if s["sm_active"] != "":
                            sm.append(float(s["sm_active"]))
                        if s["dram_active"] != "":
                            dram.append(float(s["dram_active"]))
            sm_count = n.get("sm_count", "")
            busy = round(st.mean(sm) * float(sm_count), 2) if sm and sm_count not in ("", None) else ""
            rank_rows.append([
                split, batch, mb, rank, CONFIG["slice_gb"][rank], n_layers, round(float(r["total_latency_ms"]), 1),
                round(decode, 1), round(compute_total, 1), round(100 * compute_total / decode, 2),
                round(decode - compute_total, 1), r[f"barrier_idle_ms_rank{rank}"],
                round(100 * share[key][rank] / total_points, 2) if total_points else "",
                peak, round(100 * peak / cap, 2), round(mean_mem, 1),
                *[round(ts[t][1] / 1000.0, 3) if t in ts else "" for t in ("T23", "T24", "T25", "T26")],
                c.get("cpu_user_s", ""), c.get("cpu_sys_s", ""), c.get("torch_threads", ""), c.get("affinity_cpus", ""),
                round(st.mean(sm), 4) if sm else "", round(st.mean(dram), 4) if dram else "", busy,
                n.get("sm_util_pct", ""), n.get("dram_util_pct", ""), n.get("busy_sms", ""), n.get("sm_count", ""),
                n.get("dram_bytes", ""), n.get("l2_hit_pct", ""),
            ])
    write_csv(os.path.join(out, "ranks.csv"),
              ["split", "batch", "mb", "rank", "slice_gb", "layers", "total_ms", "decode_ms", "compute_ms",
               "compute_pct_of_decode", "not_compute_ms", "barrier_idle_ms", "bottleneck_share_pct",
               "peak_mem_mib", "peak_mem_pct_of_slice", "mean_mem_mib", "t23_s", "t24_s", "t25_s", "t26_s",
               "cpu_user_s", "cpu_sys_s", "torch_threads", "affinity_cpus", "dcgm_sm_active", "dcgm_dram_active",
               "dcgm_busy_sms", "ncu_sm_util_pct", "ncu_dram_util_pct", "ncu_busy_sms", "sm_count", "ncu_dram_bytes",
               "ncu_l2_hit_pct"], rank_rows)

    neighbors = []
    bases = {pair: "_".join(map(str, split)) for pair, split in parallel_config.INV["neighbor_bases"].items()}
    for run_key in ("neighbors32", "neighbors64"):
        for (split, batch, mb), r in sorted(results(CONFIG[run_key]).items()):
            base = bases.get((batch, mb))
            if base is None or (base, batch, mb) not in control:
                continue
            b, n = [int(x) for x in base.split("_")], [int(x) for x in split.split("_")]
            base_ms = float(control[(base, batch, mb)]["total_latency_ms"])
            ms = float(r["total_latency_ms"])
            change = [n[i] - b[i] for i in range(4)]
            neighbors.append([split, batch, mb, base, str(change), round(base_ms, 1), round(ms, 1),
                              round((ms / base_ms - 1) * 100, 3)])
    write_csv(os.path.join(out, "neighbors.csv"),
              ["split", "batch", "mb", "base_split", "layer_change", "base_ms", "split_ms", "vs_base_pct"], neighbors)

    copies = []
    for (split, mb, rank), tags in sorted(copy_details(CONFIG["transport"]).items()):
        row = [split, mb, rank]
        for tag in ("T11", "T14", "T15"):
            v = tags.get(tag, [])
            row += [len(v), round(st.median(v), 4) if v else "", round(max(v), 4) if v else ""]
        copies.append(row)
    write_csv(os.path.join(out, "copies.csv"),
              ["split", "mb", "rank", "ack_n", "ack_wait_med_ms", "ack_wait_max_ms", "h2d_queue_n", "h2d_queue_med_ms",
               "h2d_queue_max_ms", "h2d_block_n", "h2d_block_med_ms", "h2d_block_max_ms"], copies)

    waits = defaultdict(list)
    for d in job_dirs(CONFIG["events"]):
        for r in read_csv(os.path.join(d, "token_wait_rank0.csv")):
            waits[(r["split"], int(r["batch"]), int(r["mb"]))].append(float(r["wait_ms"]))
    token_rows = []
    for key, v in sorted(waits.items()):
        total_ms = float(events[key]["total_latency_ms"]) if key in events else None
        v = sorted(v)
        token_rows.append(list(key) + [len(v), round(st.median(v), 3), round(v[int(0.9 * (len(v) - 1))], 3),
                                       round(sum(v) / 1000.0, 3),
                                       round(100 * sum(v) / total_ms, 2) if total_ms else ""])
    write_csv(os.path.join(out, "token_wait.csv"),
              ["split", "batch", "mb", "steps", "median_ms", "p90_ms", "total_s", "pct_of_latency"], token_rows)

    extreme_rows = []
    extreme = results(CONFIG["extreme"])
    for (split, batch, mb), r in sorted(extreme.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        base = extreme.get(("23_11_4_2", batch, mb))
        if base is None:
            continue
        ms, base_ms = float(r["total_latency_ms"]), float(base["total_latency_ms"])
        extreme_rows.append([split, batch, mb, int(split.split("_")[0]), round(ms, 1), round(base_ms, 1),
                             round((ms / base_ms - 1) * 100, 3)])
    write_csv(os.path.join(out, "extreme.csv"),
              ["split", "batch", "mb", "rank0_layers", "split_ms", "winner_ms", "vs_winner_pct"], extreme_rows)

    power_rows = []
    for name, tag in CONFIG["sample_tags"].items():
        samples = read_csv(os.path.join(CONFIG["samples_dir"], tag + "_power.csv"))
        res = {"control": control, "events": events, "dcgm": dcgm_res}[name]
        for key, win in sorted(windows(CONFIG[name], res).items()):
            def f(row, col):
                return float(row[col])
            inside = [s for s in samples if s["timestamp"] and in_window(clock_s(s["timestamp"].split(" ")[-1]), win)]
            if not inside:
                continue
            power = [f(s, "power.draw") for s in inside]
            sm = [f(s, "clocks.sm") for s in inside]
            memc = [f(s, "clocks.mem") for s in inside]
            throttled = sum(1 for s in inside if int(s["clocks_throttle_reasons.active"].strip(), 16) & ~0x1)
            power_rows.append([name] + list(key) + [len(inside), round(st.mean(power), 1), round(max(power), 1),
                                                    round(st.mean(sm), 1), min(sm), round(st.mean(memc), 1),
                                                    round(100 * throttled / len(inside), 2)])
    write_csv(os.path.join(out, "power.csv"),
              ["run", "split", "batch", "mb", "samples", "power_mean_w", "power_max_w", "sm_clock_mean_mhz",
               "sm_clock_min_mhz", "mem_clock_mean_mhz", "pct_samples_throttled"], power_rows)


if __name__ == "__main__":
    main()
