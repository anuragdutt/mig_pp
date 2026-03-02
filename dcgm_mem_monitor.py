import subprocess
import threading
import re
import csv
from datetime import datetime

DCGM_GROUP_NAME = "mig-bench"  # created earlier via setup
SAMPLE_INTERVAL_MS = 500
FIELD_FBUSED = "252"  # Framebuffer memory used (MiB)

# From your machine:
# dcgmi discovery -c shows:
#   -> I 0/1  GPU Instance (EntityID: 2)  == 1g.5gb slice (rank2)
GI_5GB_ENTITY_ID = 2

# Samples are now: [(timestamp, label, gpu0_mb, gi_5gb_mb)]
_samples = []
_label = "idle"
_stop = threading.Event()
_proc = None


def _parse_line(line: str):
    """
    Parse a dcgmi dmon line like:
      GPU-I 2   4096.0
      GPU 0     38000.0
    Returns: ("GPU-I 2", 4096.0) or ("GPU 0", 38000.0)
    """
    line = line.strip()
    m = re.match(r"(GPU(?:-I)?)\s+(\d+)\s+([\d.]+)", line)
    if not m:
        return None
    return f"{m.group(1)} {m.group(2)}", float(m.group(3))


def _get_group_id() -> int:
    """Look up the group ID for DCGM_GROUP_NAME."""
    out = subprocess.run(
        "sudo dcgmi group -l", shell=True, capture_output=True, text=True
    ).stdout
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if DCGM_GROUP_NAME in line:
            for j in range(max(0, i - 3), i):
                m = re.search(r"->\s*(\d+)", lines[j])
                if m:
                    return int(m.group(1))
    raise RuntimeError(
        f"DCGM group '{DCGM_GROUP_NAME}' not found. Run setup_dcgm_group() first."
    )


def setup_dcgm_group():
    """
    Create the DCGM monitor group with physical GPU + all MIG GPU instances.
    Call this once before start().
    """
    # Delete existing group with same name if present
    out = subprocess.run(
        "sudo dcgmi group -l", shell=True, capture_output=True, text=True
    ).stdout
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if DCGM_GROUP_NAME in line:
            for j in range(max(0, i - 3), i):
                m = re.search(r"->\s*(\d+)", lines[j])
                if m:
                    subprocess.run(f"sudo dcgmi group -d {m.group(1)}", shell=True)
            break

    # Discover GPU Instance EntityIDs
    discovery = subprocess.run(
        "sudo dcgmi discovery -c", shell=True, capture_output=True, text=True
    ).stdout
    entity_ids = re.findall(r"GPU Instance \(EntityID:\s*(\d+)\)", discovery)

    if not entity_ids:
        raise RuntimeError("No MIG instances found. Check: sudo dcgmi discovery -c")

    if str(GI_5GB_ENTITY_ID) not in entity_ids:
        raise RuntimeError(
            f"Expected 5GB GI EntityID {GI_5GB_ENTITY_ID} not found in discovery output. "
            f"Found: {entity_ids}"
        )

    print(f"[MemMonitor] Found MIG EntityIDs: {entity_ids}")
    print(f"[MemMonitor] Tracking 5GB slice: GPU-I {GI_5GB_ENTITY_ID}")

    entity_str = "0," + ",".join(f"i:{eid}" for eid in entity_ids)
    out = subprocess.run(
        f"sudo dcgmi group -c {DCGM_GROUP_NAME} -a {entity_str}",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout
    print(f"[MemMonitor] {out.strip()}")


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

    # Collect one sample once we have GPU 0 and the specific GPU-I entity we care about.
    sweep = {}
    gi_key = f"GPU-I {GI_5GB_ENTITY_ID}"

    for line in _proc.stdout:
        if _stop.is_set():
            break

        parsed = _parse_line(line)
        if not parsed:
            continue

        entity, value = parsed
        sweep[entity] = value

        if "GPU 0" in sweep and gi_key in sweep:
            _samples.append(
                (
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    _label,
                    int(sweep["GPU 0"]),
                    int(sweep[gi_key]),
                )
            )
            sweep = {}

    _proc.terminate()
    _proc.wait()


# --- PUBLIC API ---


def start():
    group_id = _get_group_id()
    _stop.clear()
    threading.Thread(target=_sample_loop, args=(group_id,), daemon=True).start()
    print(f"[MemMonitor] Sampling FB memory via DCGM group {group_id}.")


def stop():
    _stop.set()
    if _proc:
        _proc.terminate()
    print("[MemMonitor] Stopped.")


def set_label(phase: str):
    global _label
    _label = phase


def summary():
    print(f"\n{'Time':<15} {'Phase':<35} {'GPU MB':>9} {'GI-5GB MB':>10}")
    print("-" * 75)
    for row in _samples:
        print(f"{row[0]:<15} {row[1]:<35} {row[2]:>9} {row[3]:>10}")


def save_csv(path="mig_memory.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label", "gpu_mb", "gi_5gb_mb"])
        w.writerows(_samples)
    print(f"[MemMonitor] Saved {len(_samples)} samples → {path}")
