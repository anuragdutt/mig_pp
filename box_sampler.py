"""
Samples GPU 0 while a run is going. Start it just before the run, stop it with Ctrl-C (or SIGTERM).

    python3 box_sampler.py core_control          # power, clocks, temperature, throttle reasons
    python3 box_sampler.py core_dcgm dcgm        # the same, plus DCGM SM and DRAM activity per slice

Writes to investigation_samples/: <tag>_power.csv and, with dcgm, <tag>_dcgm.csv.

DCGM rows are matched to ranks by reading each slice's GPU-instance id from NVML and looking it up
in `dcgmi discovery -c`, so the two identical 1g slices cannot be swapped.
"""

import csv
import os
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime

import parallel_config

CONFIG = {
    "gpu": 0,
    "power_interval_s": 1,
    "dcgm_interval_ms": 500,
    # 1002 SM active, 1003 SM occupancy, 1005 DRAM active
    "dcgm_fields": "1002,1003,1005",
    "group_name": "investigate_group",
    "out_dir": "investigation_samples",
}


def gpu_entry(gpu):
    return next(g for g in parallel_config.GPUS if g["gpu"] == gpu)


def entities_by_rank(gpu, uuids):
    """[DCGM entity id of rank 0, of rank 1, ...]"""
    import pynvml

    pynvml.nvmlInit()
    gi_ids = []
    for uuid in uuids:
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        except (TypeError, ctypes_error()):
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
        gi_ids.append(pynvml.nvmlDeviceGetGpuInstanceId(handle))
    pynvml.nvmlShutdown()

    out = subprocess.run(["sudo", "-n", "dcgmi", "discovery", "-c"], text=True, capture_output=True)
    print(out.stdout + out.stderr)
    entity_of_gi = {}
    for m in re.finditer(r"I (\d+)/(\d+)\s*\|\s*GPU Instance \(EntityID: (\d+)\)", out.stdout):
        if int(m.group(1)) == gpu:
            entity_of_gi[int(m.group(2))] = int(m.group(3))
    print(f"GPU-instance ids by rank: {gi_ids}; DCGM entity by GPU instance: {entity_of_gi}")
    return [entity_of_gi[g] for g in gi_ids]


def ctypes_error():
    import ctypes

    return ctypes.ArgumentError


def stamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return ""


def run_dcgm(path, entities, stop):
    rank_of = {e: r for r, e in enumerate(entities)}
    made = subprocess.run(["sudo", "-n", "dcgmi", "group", "-c", CONFIG["group_name"], "-a",
                           ",".join(f"i:{e}" for e in entities)], text=True, capture_output=True)
    print(made.stdout + made.stderr)
    m = re.search(r"group ID of (\d+)", made.stdout)
    if not m:
        print("DCGM group could not be created; no DCGM samples")
        return
    group = m.group(1)
    proc = subprocess.Popen(["sudo", "-n", "dcgmi", "dmon", "-e", CONFIG["dcgm_fields"], "-g", group,
                             "-d", str(CONFIG["dcgm_interval_ms"])],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "rank", "sm_active", "sm_occupancy", "dram_active"])
            for line in proc.stdout:
                if stop.is_set():
                    break
                cols = line.split()
                # e.g. "I 3   0.912  0.400  0.610"; the first column's name varies by version
                if len(cols) < 5 or not cols[0][0].isalpha() or not cols[1].isdigit():
                    continue
                rank = rank_of.get(int(cols[1]))
                if rank is not None:
                    w.writerow([stamp(), rank] + [parse_float(c) for c in cols[-3:]])
                    f.flush()
    finally:
        proc.terminate()
        subprocess.run(["sudo", "-n", "pkill", "-f", f"dcgmi dmon -e {CONFIG['dcgm_fields']}"],
                       capture_output=True)
        subprocess.run(["sudo", "-n", "dcgmi", "group", "-d", group], capture_output=True)


def _stop(signum, frame):
    raise KeyboardInterrupt


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    # Explicit, because a process started in the background by a script inherits SIGINT as
    # ignored; investigate.sh stops this with SIGTERM, a terminal with Ctrl-C.
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    tag = sys.argv[1]
    use_dcgm = len(sys.argv) > 2 and sys.argv[2] == "dcgm"
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    power_path = os.path.join(CONFIG["out_dir"], f"{tag}_power.csv")
    dcgm_path = os.path.join(CONFIG["out_dir"], f"{tag}_dcgm.csv")

    query = "timestamp,index,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active"
    power_file = open(power_path, "w")
    power_file.write(query + "\n")
    power_file.flush()
    power = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
         "-i", str(CONFIG["gpu"]), "-l", str(CONFIG["power_interval_s"])],
        stdout=power_file, stderr=subprocess.STDOUT)

    stop = threading.Event()
    thread = None
    try:
        if use_dcgm:
            uuids = gpu_entry(CONFIG["gpu"])["mig_uuids"]
            thread = threading.Thread(
                target=run_dcgm, args=(dcgm_path, entities_by_rank(CONFIG["gpu"], uuids), stop),
                daemon=True)
            thread.start()

        print(f"sampling GPU {CONFIG['gpu']} -> {power_path}" + (f" and {dcgm_path}" if use_dcgm else ""))
        print("Ctrl-C to stop")
        power.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        power.terminate()
        power_file.close()
        if thread is not None:
            thread.join(timeout=5)
    print("stopped")


if __name__ == "__main__":
    main()
