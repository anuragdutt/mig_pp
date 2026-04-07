import subprocess
import threading
import re
import csv
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import time

DCGM_GROUP_NAME = "mig-bench"
SAMPLE_INTERVAL_MS = 500
FIELD_FBUSED = "252"  # DCGM's internal code for "Framebuffer Memory Used"
NUM_MIG_INSTANCES = 4

# DCGM GPU-I entity IDs (from: sudo dcgmi dmon -e 252)
# These are NOT nvidia-smi GI IDs — they're DCGM's own numbering
# Mapping confirmed by idle memory footprint
DCGM_ID_TO_RANK = {
    3: 0,  # GPU-I 3 → 37MB idle → 3g.40gb (Rank 0)
    2: 1,  # GPU-I 2 → 25MB idle → 2g.20gb (Rank 1)
    0: 2,  # GPU-I 0 → 12MB idle → 1g.10gb (Rank 2)
    1: 3,  # GPU-I 1 → 12MB idle → 1g.10gb (Rank 3)
}

# Change the sample format to hold all 4 MIG instances
# Samples: (timestamp, label, gpu_mb, gi0_mb, gi1_mb, gi2_mb, gi3_mb)
_samples: List[Tuple[str, str, int, int, int, int, int]] = []

# Maps DCGM GPU-I entity index → rank
# Populated by setup_dcgm_group()
# _gi_index_to_rank: Dict[int, int] = {}

_label = "idle"
_stop = threading.Event()
_proc = None


# Simple terminal command running function
def _run(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


# Simple parsing function
def _parse_dmon_line(line: str) -> Optional[Tuple[str, int]]:
    """
    Parse dcgmi dmon lines like:
      GPU-I 0   4096.0
      GPU 0     38000.0
    """
    line = line.strip()
    m = re.match(r"(GPU(?:-I)?)\s+(\d+)\s+([\d.]+)", line)
    if not m:
        return None
    entity = f"{m.group(1)} {m.group(2)}"
    value_mb = int(float(m.group(3)))
    return entity, value_mb


def _get_group_id() -> int:
    out = _run("sudo dcgmi group -l")
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if DCGM_GROUP_NAME in line:
            for j in range(max(0, i - 3), i):
                m = re.search(r"->\s*(\d+)", lines[j])
                if m:
                    return int(m.group(1))
    raise RuntimeError(f"DCGM group '{DCGM_GROUP_NAME}' not found.")


def setup_dcgm_group():
    """Create DCGM group with GPU 0 + all 4 MIG instances."""

    # Delete existing group if present
    out = _run("sudo dcgmi group -l")
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if DCGM_GROUP_NAME in line:
            for j in range(max(0, i - 3), i):
                m = re.search(r"->\s*(\d+)", lines[j])
                if m:
                    gid = m.group(1)
                    _run(f"sudo dcgmi group -d {gid}")
                    print(f"[MemMonitor] Removed old group {gid}")
            break

    # Create group with GPU 0
    out = _run(f"sudo dcgmi group -c {DCGM_GROUP_NAME} -a 0")
    print(f"[MemMonitor] {out.strip()}")

    # Parse the new group ID
    m = re.search(r"group ID of (\d+)", out)
    if not m:
        raise RuntimeError(f"Failed to create DCGM group. Output: {out}")
    group_id = m.group(1)

    # Add each GPU instance individually
    for dcgm_id in sorted(DCGM_ID_TO_RANK.keys()):
        out = _run(f"sudo dcgmi group -g {group_id} -a i:{dcgm_id}")
        print(f"[MemMonitor] Added i:{dcgm_id} → {out.strip()}")

    print(f"[MemMonitor] DCGM ID → Rank: {DCGM_ID_TO_RANK}")


def _sample_loop(group_id: int):
    global _proc
    cmd = f"sudo dcgmi dmon -e {FIELD_FBUSED} -g {group_id} -d {SAMPLE_INTERVAL_MS}"

    _proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    sweep = {}
    for line in _proc.stdout:
        if _stop.is_set():
            break

        parsed = _parse_dmon_line(line)
        if not parsed:
            continue

        entity, value = parsed
        sweep[entity] = value

        gi_keys = sorted(k for k in sweep if k.startswith("GPU-I"))
        if "GPU 0" in sweep and len(gi_keys) >= NUM_MIG_INSTANCES:
            rank_mem = [0] * NUM_MIG_INSTANCES
            for gi_key in gi_keys:
                # Extract DCGM ID from "GPU-I 3" → 3
                dcgm_id = int(gi_key.split()[1])
                rank = DCGM_ID_TO_RANK.get(dcgm_id)
                if rank is not None and rank < NUM_MIG_INSTANCES:
                    rank_mem[rank] = int(sweep[gi_key])

            _samples.append(
                (
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    _label,
                    int(sweep["GPU 0"]),
                    rank_mem[0],
                    rank_mem[1],
                    rank_mem[2],
                    rank_mem[3],
                )
            )
            sweep = {}

    try:
        _proc.terminate()
        _proc.wait(timeout=5)
    except Exception:
        pass


# --- PUBLIC API ---


def start():
    # Clean up any orphaned dcgmi from previous runs
    subprocess.run("sudo pkill -f 'dcgmi dmon'", shell=True, capture_output=True)
    time.sleep(0.5)

    group_id = _get_group_id()
    _stop.clear()
    threading.Thread(target=_sample_loop, args=(group_id,), daemon=True).start()
    print(
        f"[MemMonitor] Sampling FB memory via DCGM group {group_id} (target=5GB MIG)."
    )


def stop():
    _stop.set()
    if _proc:
        try:
            _proc.terminate()
        except Exception:
            pass

    # Kill the real dcgmi process that sudo spawned
    subprocess.run("sudo pkill -f 'dcgmi dmon'", shell=True, capture_output=True)
    print("[MemMonitor] Stopped.")


def set_label(phase: str):
    global _label
    _label = phase


def clear():
    _samples.clear()


def save_csv(path="mig_memory.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "label",
                "gpu_mb",
                "rank0_40gb_mb",
                "rank1_20gb_mb",
                "rank2_10gb_mb",
                "rank3_10gb_mb",
            ]
        )
        w.writerows(_samples)
    print(f"[MemMonitor] Saved {len(_samples)} samples → {path}")
