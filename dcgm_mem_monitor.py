"""
Per-MIG-instance framebuffer memory sampling via DCGM.

Key difference from the previous version: the DCGM entity-ID -> rank mapping
is DISCOVERED at runtime by matching MIG UUIDs, instead of being a hardcoded
table derived by eyeballing idle memory footprints. That table was specific
to one physical card and one 4-slice topology; it silently mis-attributed
memory to the wrong ranks on any other box or slice layout.

Slice count is also no longer hardcoded — it follows the MIG_UUIDS list the
caller passes in.
"""

import subprocess
import threading
import re
import csv
import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import time

DCGM_GROUP_NAME = "mig-bench"
SAMPLE_INTERVAL_MS = 500
FIELD_FBUSED = "252"  # DCGM field: Framebuffer Memory Used (MB)

log = logging.getLogger(__name__)

# Populated by setup_dcgm_group(). Maps DCGM GPU-I entity id -> pipeline rank.
DCGM_ID_TO_RANK: Dict[int, int] = {}

# Number of MIG slices being monitored; set by setup_dcgm_group().
NUM_MIG_INSTANCES = 0

# Samples: (timestamp, label, gpu_mb, rank0_mb, rank1_mb, ... rankN_mb)
# Width follows NUM_MIG_INSTANCES, so downstream code must not assume 4.
_samples: List[Tuple] = []

# Slice sizes in GB, parallel to rank order — used only for CSV headers.
_slice_gb: List[int] = []

_label = "idle"
_stop = threading.Event()
_proc = None

# False when DCGM is unavailable — sampling becomes a no-op so a monitoring
# problem cannot abort a sweep whose primary measurement is latency.
_enabled = False


def _run(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def _run_checked(cmd: str) -> Tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


# ---------------------------------------------------------------------------
# HOST ENGINE
# ---------------------------------------------------------------------------


def ensure_hostengine() -> None:
    """
    DCGM needs nv-hostengine running. The apt package installs it but does
    not necessarily start it, which surfaces as:
        Error: unable to establish a connection to the specified host: localhost
    """
    rc, _ = _run_checked("sudo dcgmi discovery -l")
    if rc == 0:
        log.info("[MemMonitor] nv-hostengine already reachable")
        return

    # Try the service first, fall back to launching the daemon directly.
    _run_checked("sudo systemctl start nvidia-dcgm")
    time.sleep(2.0)
    rc, _ = _run_checked("sudo dcgmi discovery -l")
    if rc == 0:
        log.info("[MemMonitor] started nv-hostengine via systemd")
        return

    _run_checked("sudo nv-hostengine")
    time.sleep(2.0)
    rc, out = _run_checked("sudo dcgmi discovery -l")
    if rc != 0:
        raise RuntimeError(
            f"Could not start nv-hostengine. DCGM memory monitoring unavailable.\n{out}"
        )
    log.info("[MemMonitor] started nv-hostengine directly")


# ---------------------------------------------------------------------------
# ENTITY DISCOVERY
# ---------------------------------------------------------------------------


def _discover_gi_uuid_map() -> Dict[int, str]:
    """
    Map DCGM GPU-I entity id -> MIG UUID.

    `dcgmi discovery -l` lists GPU instances but not their UUIDs, so the
    UUIDs come from nvidia-smi and are matched positionally against DCGM's
    instance ordering. Both tools enumerate GPU instances by ascending
    GPU-Instance ID, which is what makes the positional match valid.
    """
    # nvidia-smi: GI id -> UUID
    out = _run("nvidia-smi -L")
    smi_uuids: List[str] = []
    for line in out.splitlines():
        m = re.search(r"MIG\s+\S+\s+Device\s+\d+:\s+\(UUID:\s+(MIG-[0-9a-f-]+)\)", line)
        if m:
            smi_uuids.append(m.group(1))

    # nvidia-smi -L lists devices in GI-id order; get the actual GI ids too.
    out_lgi = _run("sudo nvidia-smi mig -lgi")
    gi_ids: List[int] = []
    for line in out_lgi.splitlines():
        # rows look like: |  0  MIG 3g.20gb  |  0  |  2  |
        m = re.match(r"\|\s+(\d+)\s+MIG\s+\S+\s+\|\s+(\d+)\s+\|", line)
        if m:
            gi_ids.append(int(m.group(2)))

    # DCGM entity ids for GPU instances.
    #
    # dcgmi output format varies across DCGM versions (3.x vs 4.x) and
    # subcommands, so try several sources rather than trusting one regex.
    # The dmon probe is the most reliable because it prints the exact entity
    # labels the sampling loop will later parse.
    dcgm_ids: List[int] = []

    # Source 1: dmon probe — prints the real entity labels the sampling loop
    # will parse, so it is the most reliable source.
    #
    # `-c 2` makes dcgmi exit on its own after 2 samples; a Python-side
    # timeout guards the case where it ignores that and streams forever.
    # (Don't shell out to `timeout` — it isn't present everywhere, and when
    # missing the probe silently returns nothing and discovery falls through
    # to the weak 0..N-1 guess.)
    out_dmon = ""
    try:
        r = subprocess.run(
            f"sudo dcgmi dmon -e {FIELD_FBUSED} -c 2",
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        out_dmon = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        # Salvage whatever was printed before the timeout.
        out_dmon = (
            (e.stdout or b"").decode(errors="replace")
            if isinstance(e.stdout, bytes)
            else (e.stdout or "")
        )
        log.warning("[MemMonitor] dmon probe timed out; parsing partial output")

    for line in out_dmon.splitlines():
        m = re.match(r"\s*GPU-I\s+(\d+)", line)
        if m:
            dcgm_ids.append(int(m.group(1)))

    # Source 2: discovery listing, several known phrasings.
    if not dcgm_ids:
        out_disc = _run("sudo dcgmi discovery -l")
        for line in out_disc.splitlines():
            for pat in (
                r"GPU-I\s+(\d+)",
                r"GPU Instance\s+(?:ID\s*)?[:=]?\s*(\d+)",
                r"Instance\s+(\d+)",
                r"\bI\s+(\d+)\b",
            ):
                m = re.search(pat, line)
                if m:
                    dcgm_ids.append(int(m.group(1)))
                    break

    # Source 3: hierarchy view (DCGM 3.x+).
    if not dcgm_ids:
        out_h = _run("sudo dcgmi discovery --gpu-instance-hierarchy")
        for line in out_h.splitlines():
            m = re.search(r"GPU-I\s+(\d+)|instance\s+(\d+)", line, re.I)
            if m:
                val = m.group(1) or m.group(2)
                dcgm_ids.append(int(val))

    # Source 4: assume DCGM numbers instances 0..N-1 matching nvidia-smi's
    # count. Weakest option, but better than monitoring nothing — and the
    # mapping is logged so a wrong guess is visible rather than silent.
    if not dcgm_ids and smi_uuids:
        dcgm_ids = list(range(len(smi_uuids)))
        log.warning(
            "[MemMonitor] could not parse DCGM entity ids from any dcgmi "
            "output; assuming ids 0..%d. Verify memory columns before "
            "trusting them.",
            len(smi_uuids) - 1,
        )

    dcgm_ids = sorted(set(dcgm_ids))

    mapping: Dict[int, str] = {}
    for i, dcgm_id in enumerate(dcgm_ids):
        if i < len(smi_uuids):
            mapping[dcgm_id] = smi_uuids[i]

    log.info(
        "[MemMonitor] DCGM entities %s <-> MIG UUIDs %s",
        dcgm_ids,
        [u[:20] + "..." for u in smi_uuids],
    )
    return mapping


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


def setup_dcgm_group(
    mig_uuids: Optional[List[str]] = None, slice_gb: Optional[List[int]] = None
):
    """
    Create a DCGM group covering GPU 0 plus every MIG instance, and build
    the entity->rank mapping.

    mig_uuids: the caller's MIG_UUIDS list, in rank order. Passing it lets
    this function map each DCGM entity to the RIGHT rank rather than relying
    on a hardcoded table. If omitted, entities are assigned to ranks in
    ascending entity-id order, which is only correct if the caller's UUID
    order happens to match.
    """
    global DCGM_ID_TO_RANK, NUM_MIG_INSTANCES, _slice_gb

    # Record the slice sizes first so CSV headers stay right even if DCGM
    # setup fails below.
    _slice_gb = list(slice_gb) if slice_gb else []

    try:
        ensure_hostengine()
    except Exception as e:
        NUM_MIG_INSTANCES = 0
        DCGM_ID_TO_RANK = {}
        print(f"[MemMonitor] DISABLED — {e}", flush=True)
        print("[MemMonitor] sweep will continue; memory columns will be 0.", flush=True)
        return

    # Remove a stale group from a previous run.
    out = _run("sudo dcgmi group -l")
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if DCGM_GROUP_NAME in line:
            for j in range(max(0, i - 3), i):
                m = re.search(r"->\s*(\d+)", lines[j])
                if m:
                    gid = m.group(1)
                    _run(f"sudo dcgmi group -d {gid}")
                    log.info(f"[MemMonitor] Removed old group {gid}")
            break

    out = _run(f"sudo dcgmi group -c {DCGM_GROUP_NAME} -a 0")
    log.info(f"[MemMonitor] {out.strip()}")

    m = re.search(r"group ID of (\d+)", out)
    if not m:
        NUM_MIG_INSTANCES = 0
        DCGM_ID_TO_RANK = {}
        print(
            f"[MemMonitor] DISABLED — could not create DCGM group. "
            f"Output: {out.strip()}",
            flush=True,
        )
        print("[MemMonitor] sweep will continue; memory columns will be 0.", flush=True)
        return
    group_id = m.group(1)

    # --- build entity -> rank mapping ---
    entity_to_uuid = _discover_gi_uuid_map()

    DCGM_ID_TO_RANK = {}
    if mig_uuids:
        uuid_to_rank = {u: r for r, u in enumerate(mig_uuids)}
        for dcgm_id, uuid in entity_to_uuid.items():
            rank = uuid_to_rank.get(uuid)
            if rank is not None:
                DCGM_ID_TO_RANK[dcgm_id] = rank

        missing = set(range(len(mig_uuids))) - set(DCGM_ID_TO_RANK.values())
        if missing:
            log.warning(
                "[MemMonitor] could not map ranks %s to DCGM entities; "
                "falling back to positional order. Memory columns for those "
                "ranks may be wrong.",
                sorted(missing),
            )
            DCGM_ID_TO_RANK = {d: i for i, d in enumerate(sorted(entity_to_uuid))}
        NUM_MIG_INSTANCES = len(mig_uuids)
    else:
        DCGM_ID_TO_RANK = {d: i for i, d in enumerate(sorted(entity_to_uuid))}
        NUM_MIG_INSTANCES = len(DCGM_ID_TO_RANK)

    # Add each GPU instance to the group.
    for dcgm_id in sorted(DCGM_ID_TO_RANK):
        out = _run(f"sudo dcgmi group -g {group_id} -a i:{dcgm_id}")
        log.info(f"[MemMonitor] Added i:{dcgm_id} -> {out.strip()}")

    log.info(f"[MemMonitor] DCGM entity -> Rank: {DCGM_ID_TO_RANK}")

    if not DCGM_ID_TO_RANK:
        NUM_MIG_INSTANCES = 0
        print(
            "[MemMonitor] DISABLED — no MIG entities could be mapped. "
            "Latency still measured; memory columns will be 0.",
            flush=True,
        )
        return

    print(
        f"[MemMonitor] monitoring {NUM_MIG_INSTANCES} MIG slices, "
        f"entity->rank {DCGM_ID_TO_RANK}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------


def _parse_dmon_line(line: str) -> Optional[Tuple[str, int]]:
    """
    Parse dcgmi dmon rows:
      GPU-I 0   4096.0
      GPU 0     38000.0
    """
    line = line.strip()
    m = re.match(r"(GPU(?:-I)?)\s+(\d+)\s+([\d.]+)", line)
    if not m:
        return None
    entity = f"{m.group(1)} {m.group(2)}"
    return entity, int(float(m.group(3)))


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
                dcgm_id = int(gi_key.split()[1])
                rank = DCGM_ID_TO_RANK.get(dcgm_id)
                if rank is not None and rank < NUM_MIG_INSTANCES:
                    rank_mem[rank] = int(sweep[gi_key])

            _samples.append(
                (
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    _label,
                    int(sweep["GPU 0"]),
                    *rank_mem,
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
    """
    Begin sampling. Degrades to a no-op rather than raising: memory data is
    supplementary, and a DCGM problem must not abort a sweep whose primary
    output is latency. _enabled tells the caller whether samples will arrive.
    """
    global _enabled

    if NUM_MIG_INSTANCES == 0:
        _enabled = False
        print(
            "[MemMonitor] DISABLED — no MIG entities discovered. "
            "Latency will still be measured; memory columns will be 0.",
            flush=True,
        )
        log.warning(
            "[MemMonitor] start() called with 0 MIG instances; sampling disabled"
        )
        return

    # Clear orphaned dcgmi from a previous run.
    subprocess.run("sudo pkill -f 'dcgmi dmon'", shell=True, capture_output=True)
    time.sleep(0.5)

    try:
        group_id = _get_group_id()
    except Exception as e:
        _enabled = False
        print(f"[MemMonitor] DISABLED — {e}", flush=True)
        return

    _enabled = True
    _stop.clear()
    threading.Thread(target=_sample_loop, args=(group_id,), daemon=True).start()
    print(f"[MemMonitor] sampling FB memory via DCGM group {group_id}", flush=True)


def stop():
    _stop.set()
    if _proc:
        try:
            _proc.terminate()
        except Exception:
            pass
    subprocess.run("sudo pkill -f 'dcgmi dmon'", shell=True, capture_output=True)


def set_label(phase: str):
    global _label
    _label = phase


def clear():
    _samples.clear()


def save_csv(path="mig_memory.csv"):
    """
    Column count follows the slice count, not a fixed 4.

    Headers come from _slice_gb (the caller's topology) rather than
    NUM_MIG_INSTANCES, so a disabled monitor still writes a correctly-shaped
    file with zero rows instead of a truncated header.
    """
    width = len(_slice_gb) if _slice_gb else NUM_MIG_INSTANCES
    if _slice_gb:
        rank_cols = [f"rank{r}_{_slice_gb[r]}gb_mb" for r in range(width)]
    else:
        rank_cols = [f"rank{r}_mb" for r in range(width)]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label", "gpu_mb", *rank_cols])
        w.writerows(_samples)
    print(f"[MemMonitor] saved {len(_samples)} samples -> {path}", flush=True)
