"""
Per-MIG-slice memory sampler over NVML: a drop-in for the part of
dcgm_mem_monitor the benchmark uses (setup_dcgm_group once; per config
set_label, clear, start, stop twice; save_csv). row[3 + r] is rank r.

Why: dcgm_mem_monitor pkills every `dcgmi dmon` on the box, hardcodes GPU 0 and
maps DCGM entities to ranks by lining up tool orderings. Here rank r is read
from mig_uuids[r]'s own NVML handle, with no sudo, no subprocess and one NVML
session per process, so eight copies can run side by side. Memory is
supplementary to latency: setup never raises, a failure prints one DISABLED line.
"""

import csv
import ctypes
import threading
from datetime import datetime

SAMPLE_INTERVAL_MS = 500

NUM_MIG_INSTANCES = 0  # slices being sampled; 0 while disabled
_samples = []  # (timestamp, label, gpu_mb, rank0_mb, ...), ranks in mig_uuids order
_slice_gb = []  # names the CSV rank columns, nothing else
_label = "idle"
_enabled = False  # setup succeeded, so start() will produce rows

_off = False  # disable() was called; sticky
_nvml = None
_handles = []  # one per rank
_parent = None  # parent GPU handle; None means gpu_mb is the slices' sum
_thread = None  # the live sampler, if any
_halt = None  # its stop event


def _say(msg):
    print(f"[MemMonitor] {msg}", flush=True)


def _by_uuid(nvml, uuid):
    # Current nvidia-ml-py encodes str itself; older releases hand it straight
    # to ctypes, which takes only bytes. The package is unpinned, so try both.
    try:
        return nvml.nvmlDeviceGetHandleByUUID(uuid)
    except (TypeError, ctypes.ArgumentError):
        return nvml.nvmlDeviceGetHandleByUUID(uuid.encode())


def _used_mib(handle):
    return _nvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20


def setup_dcgm_group(mig_uuids=None, slice_gb=None):
    """Resolve each rank's slice by MIG UUID. DCGM's name, so either module fits."""
    global NUM_MIG_INSTANCES, _slice_gb, _enabled, _nvml, _handles, _parent
    _slice_gb = list(slice_gb or [])  # first: save_csv's header needs it regardless
    _enabled, NUM_MIG_INSTANCES = False, 0
    if _off:
        return
    if not mig_uuids:
        _say("DISABLED — no mig_uuids given")
        return

    step = "import pynvml"
    try:
        # Imported here, not at module level: every spawned rank re-imports
        # the benchmark, and so this module, but only the parent samples.
        import pynvml

        step = "nvmlInit"
        pynvml.nvmlInit()
        handles = []
        for r, uuid in enumerate(mig_uuids):
            step = f"rank {r} lookup of {uuid}"
            handles.append(_by_uuid(pynvml, uuid))
    except Exception as e:
        _say(f"DISABLED — {step} failed: {type(e).__name__}: {e}")
        return

    try:
        # A pipeline lives on one GPU: the first slice's parent is every slice's.
        parent = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handles[0])
        pynvml.nvmlDeviceGetMemoryInfo(parent)
        source = "the parent GPU"
    except Exception as e:
        # NVML gives a MIG-mode GPU's aggregate memory only to privileged
        # callers, and the benchmark runs without sudo.
        parent = None
        source = f"the slices' sum (parent GPU unreadable: {type(e).__name__}: {e})"
    _nvml, _handles, _parent = pynvml, handles, parent
    _enabled, NUM_MIG_INSTANCES = True, len(handles)
    _say(
        f"nvml: sampling {len(handles)} MIG slices by UUID every "
        f"{SAMPLE_INTERVAL_MS} ms; gpu_mb = {source}"
    )


setup_group = setup_dcgm_group  # neutral name for callers not replacing DCGM


def _sample(halt):
    printed = False
    while not halt.is_set():
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        try:
            ranks = [_used_mib(h) for h in _handles]
            gpu_mb = sum(ranks) if _parent is None else _used_mib(_parent)
        except Exception as e:
            # Skip the tick rather than record zeros, which would drag the
            # config's average down. Only a run's first failure is printed.
            if not printed:
                _say(f"nvml: read failed, skipping ticks: {type(e).__name__}: {e}")
                printed = True
        else:
            # A thread that stop() gave up on must not write into the next config.
            if not halt.is_set():
                _samples.append((ts, _label, gpu_mb, *ranks))
        halt.wait(SAMPLE_INTERVAL_MS / 1000)


def start():
    """Sample into _samples every SAMPLE_INTERVAL_MS until stop(); no-op if disabled."""
    global _thread, _halt
    if not _enabled or _thread is not None:
        return
    _halt = threading.Event()  # fresh per run, so start() after stop() works
    _thread = threading.Thread(target=_sample, args=(_halt,), daemon=True)
    _thread.start()


def stop():
    """Join the sampler, so _samples is final on return. Idempotent: called twice."""
    global _thread
    thread, _thread = _thread, None
    if thread is not None:
        _halt.set()
        thread.join(timeout=2.0)  # bounded: a hung NVML call must not hang the sweep


def set_label(phase):
    global _label
    _label = phase


def clear():
    _samples.clear()


def save_csv(path):
    # The header follows setup's topology, so a disabled monitor writes it too.
    ranks = [f"rank{r}_{gb}gb_mb" for r, gb in enumerate(_slice_gb)]
    ranks = ranks or [f"rank{r}_mb" for r in range(NUM_MIG_INSTANCES)]
    rows = list(_samples)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label", "gpu_mb", *ranks])
        w.writerows(rows)
    _say(f"saved {len(rows)} samples -> {path}")


def disable():
    """
    For mem_monitor="off": NVML is never touched and every later call is a no-op
    but save_csv, which writes the header alone. Silent: every rank calls it.
    """
    global _off, _enabled, NUM_MIG_INSTANCES
    stop()
    _samples.clear()
    _off, _enabled, NUM_MIG_INSTANCES = True, False, 0
