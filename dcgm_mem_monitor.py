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

# Change the sample format to hold all 4 MIG instances
# Samples: (timestamp, label, gpu_mb, gi0_mb, gi1_mb, gi2_mb, gi3_mb)
_samples: List[Tuple[str, str, int, int, int, int, int]] = []

# Maps DCGM GPU-I entity index → rank
# Populated by setup_dcgm_group()
_gi_index_to_rank: Dict[int, int] = {}

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
    """
    Auto-discover all MIG GPU Instance IDs via nvidia-smi,
    map them to ranks by matching UUIDs, and create a DCGM group
    with the physical GPU + all 4 GPU instances.
    """
    global _gi_index_to_rank

    # --- Step 1: Discover GI IDs and their UUIDs ---
    # nvidia-smi mig -lgi outputs lines like:
    # +-------+-------+-------+------+------+------+-------+
    # | GPU   | Name  |  ...  |  GI  | ...  | ... | UUID  |
    # Format varies, so we parse nvidia-smi -L which is simpler

    # Get MIG device list: "MIG 3g.20gb Device 0: ... (UUID: MIG-xxxxx)"
    out = _run("nvidia-smi -L")

    # Also get GI ID mapping from nvidia-smi
    gi_out = _run("sudo nvidia-smi mig -lgi")

    # Parse GI IDs from nvidia-smi mig -lgi
    # Lines look like:
    # +----+----------------------+-------+---------+...
    # | 0  | 0  |  3g.20gb  | ...
    # We need GI ID (second column) and profile name
    gi_ids = []
    for line in gi_out.splitlines():
        # Match lines with GI data: "| GPU | GI ID | CI ID | ..."
        m = re.match(r"\s*\|\s*\d+\s*\|\s*(\d+)\s*\|", line)
        if m:
            gi_ids.append(int(m.group(1)))

    if len(gi_ids) < NUM_MIG_INSTANCES:
        print(
            f"[MemMonitor] WARNING: Found only {len(gi_ids)} GPU instances, "
            f"expected {NUM_MIG_INSTANCES}. GI IDs: {gi_ids}"
        )
        print(f"[MemMonitor] Raw output:\n{gi_out}")

    print(f"[MemMonitor] Discovered GI IDs: {gi_ids}")

    # --- Step 2: Delete existing group if present ---
    out = _run("sudo dcgmi group -l")
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if DCGM_GROUP_NAME in line:
            for j in range(max(0, i - 3), i):
                m = re.search(r"->\s*(\d+)", lines[j])
                if m:
                    gid = m.group(1)
                    _run(f"sudo dcgmi group -d {gid}")
                    print(f"[MemMonitor] Successfully removed group {gid}")
            break

    # --- Step 3: Create group with GPU 0 + all GPU instances ---
    # DCGM entity format: "0" for GPU 0, "i:<N>" for GPU Instance N
    entity_parts = ["0"] + [f"i:{gi}" for gi in gi_ids]
    entity_str = ",".join(entity_parts)

    out = _run(f"sudo dcgmi group -c {DCGM_GROUP_NAME} -a {entity_str}")
    print(f"[MemMonitor] {out.strip()}")

    # --- Step 4: Build DCGM GPU-I index → rank mapping ---
    # dcgmi dmon outputs "GPU-I 0", "GPU-I 1", etc. in the order
    # the instances were added. Map by GI ID order → rank.
    # GI IDs are sorted by creation order which matches rank order
    # if MIG was set up rank 0 first, rank 3 last.
    _gi_index_to_rank = {}
    for dcgm_idx, gi_id in enumerate(sorted(gi_ids)):
        _gi_index_to_rank[dcgm_idx] = dcgm_idx  # rank == sorted position

    print(f"[MemMonitor] GPU-I index → rank mapping: {_gi_index_to_rank}")
    print(f"[MemMonitor] VERIFY THIS: rank 0=20GB, rank 1=10GB, rank 2=5GB, rank 3=5GB")


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

        # Emit once we have GPU 0 + all GPU-I entries
        gi_keys = sorted(k for k in sweep if k.startswith("GPU-I"))
        if "GPU 0" in sweep and len(gi_keys) >= NUM_MIG_INSTANCES:
            # Build per-rank memory array
            rank_mem = [0] * NUM_MIG_INSTANCES
            for idx, gi_key in enumerate(gi_keys):
                rank = _gi_index_to_rank.get(idx, idx)
                if rank < NUM_MIG_INSTANCES:
                    rank_mem[rank] = int(sweep[gi_key])

            _samples.append(
                (
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    _label,
                    int(sweep["GPU 0"]),
                    rank_mem[0],  # 20GB slice
                    rank_mem[1],  # 10GB slice
                    rank_mem[2],  # 5GB slice
                    rank_mem[3],  # 5GB slice
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
                "rank0_20gb_mb",
                "rank1_10gb_mb",
                "rank2_5gb_mb",
                "rank3_5gb_mb",
            ]
        )
        w.writerows(_samples)
    print(f"[MemMonitor] Saved {len(_samples)} samples → {path}")
