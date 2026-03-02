import subprocess
import threading
import re
import csv
from datetime import datetime
from typing import Optional, Tuple, List

DCGM_GROUP_NAME = "mig-bench"
SAMPLE_INTERVAL_MS = 500
FIELD_FBUSED = "252"  # DCGM's internal code for "Framebuffer Memory Used"


# Samples: (timestamp, label, gpu_mb, target_gi_mb)
_samples: List[Tuple[str, str, int, int]] = []
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
    Create ç group with:
      - physical GPU 0
      - ONLY the 5GB MIG slice (EntityID: 0)
    """

    # Delete existing group if present
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

    # Creating new group
    entity_str = "0,i:0"

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

    # Popen - runs continuously without freezing Python
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

        # Because the group has only ONE GPU-I, we emit once we have:
        # GPU 0 + any GPU-I line
        gi_keys = [k for k in sweep.keys() if k.startswith("GPU-I")]
        if "GPU 0" in sweep and len(gi_keys) >= 1:
            gi_key = sorted(gi_keys)[0]  # the only one in practice
            _samples.append(
                (
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    _label,
                    int(sweep["GPU 0"]),
                    int(sweep[gi_key]),
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
    print("[MemMonitor] Stopped.")


def set_label(phase: str):
    global _label
    _label = phase


def clear():
    _samples.clear()


def save_csv(path="mig_memory.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label", "gpu_mb", "target_5gb_mb"])
        w.writerows(_samples)
    print(f"[MemMonitor] Saved {len(_samples)} samples → {path}")
