"""
Control channel between the two parents of a two-node pipeline. Stdlib only.

Each node runs its own benchmark parent, which spawns only its local ranks.
The parents meet here: the lead (the node holding rank 0) listens on its
private address, the other node connects. They agree they run the same code
and job (hello), run each configuration in lockstep (ready), and the other
node ships what its ranks produced to the lead (result), which writes the CSV.

Wire format: 4-byte big-endian length, then a UTF-8 JSON object with a "type".
Nothing is unpickled off the network.

Every failure surfaces as one of two exceptions, so a caller catching both
catches everything: PeerLost (the channel is unusable: a frame may be half
sent or half read, so close the socket) or Mismatch (the nodes disagree).
"""

import hashlib
import json
import socket
import struct
import time
from importlib import metadata
from pathlib import Path

# What a rank process imports. Not the whole repo: a node's own
# parallel_config.py must not abort a run.
CODE_FILES = (
    "benchmark_pipeline_microbatching.py",
    "mig_transport_pipeline.py",
    "async_handle_send.py",
    "async_handle_recv.py",
    "async_handle_net.py",
    "token_exchange.py",
    "helpers.py",
    "model_family.py",
    "experiment_config.py",
    "two_node_control.py",
    "nvml_mem_monitor.py",
)

# Job fields the two nodes must agree on. Node-specific ones (node, ifname,
# shm prefix, run id, model path) are left out.
FINGERPRINT_KEYS = (
    "model_key", "model_type", "num_layers", "hidden_size", "num_heads", "seq_len",
    "max_new_tokens", "max_runs", "batch_mb_pairs", "splits", "enforce_slice_ordering",
    "min_last_rank_layers", "rank_nodes", "slice_gb", "mig_uuids", "node_addrs",
    "master_port", "control_port", "layer_limits",
)

HELLO_TIMEOUT_S = 60
MAX_MESSAGE_BYTES = 256 * 2**20


class PeerLost(Exception):
    """The other parent closed the connection or went silent."""


class Mismatch(Exception):
    """The two parents disagree on code, versions, job or configuration."""


# --- identity ---------------------------------------------------------------
def code_hash(repo_dir, files=CODE_FILES):
    h = hashlib.sha256()
    for name in files:
        path = Path(repo_dir) / name
        h.update(name.encode() + b"\0")
        h.update(path.read_bytes() if path.exists() else b"<missing>")
        h.update(b"\0")
    return h.hexdigest()


def job_fingerprint(job):
    picked = {k: job.get(k) for k in FINGERPRINT_KEYS}
    return hashlib.sha256(json.dumps(picked, sort_keys=True).encode()).hexdigest()


def package_version(name):
    try:
        return metadata.version(name)  # does not import torch
    except metadata.PackageNotFoundError:
        return "none"


def make_hello(node, run_id, repo_dir, job, files=CODE_FILES):
    return {
        "type": "hello",
        "node": node,
        "run_id": run_id,
        "code": code_hash(repo_dir, files),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "job": job_fingerprint(job),
    }


# --- wire -------------------------------------------------------------------
def send_msg(sock, obj):
    data = json.dumps(obj).encode()
    try:
        sock.sendall(struct.pack(">I", len(data)) + data)
    except OSError as e:  # timeout (the last recv_msg's), reset, broken pipe
        raise PeerLost(f"send to the other node failed: {e}") from None


def _read_exact(sock, n):
    chunks = []
    while n:
        try:
            chunk = sock.recv(min(n, 1 << 20))
        except socket.timeout:
            raise PeerLost("timed out waiting for the other node") from None
        except OSError as e:
            raise PeerLost(f"connection error: {e}") from None
        if not chunk:
            raise PeerLost("other node closed the connection")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock, timeout):
    sock.settimeout(timeout)
    (size,) = struct.unpack(">I", _read_exact(sock, 4))
    if size > MAX_MESSAGE_BYTES:
        raise PeerLost(f"message of {size} bytes refused")
    try:
        msg = json.loads(_read_exact(sock, size))
    except (ValueError, RecursionError) as e:  # bad UTF-8 or JSON; absurd nesting
        raise PeerLost(f"unreadable message from the other node: {type(e).__name__}") from None
    if not isinstance(msg, dict):
        raise PeerLost(f"unreadable message from the other node: a {type(msg).__name__}")
    return msg


def expect(sock, kind, timeout):
    msg = recv_msg(sock, timeout)
    if msg.get("type") != kind:
        raise Mismatch(f"expected a '{kind}' message, got {msg.get('type')!r}")
    return msg


# --- connect ----------------------------------------------------------------
def listen_for_peer(bind_addr, port, timeout_s):
    """Lead side: bind on the private address (not every interface), accept one peer."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_addr, port))
        server.listen(1)
        server.settimeout(timeout_s)
        try:
            conn, _ = server.accept()
        except socket.timeout:
            raise PeerLost(f"no peer connected to {bind_addr}:{port} in {timeout_s}s") from None
    finally:
        server.close()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn


def connect_to_lead(lead_addr, port, timeout_s):
    """Other side: retry until the lead is listening."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((lead_addr, port), timeout=5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as e:
            last = e
            time.sleep(0.5)
    raise PeerLost(f"could not reach the lead at {lead_addr}:{port} in {timeout_s}s ({last})")


# --- protocol ---------------------------------------------------------------
def exchange_hello(sock, mine):
    """Send ours, read theirs, compare. Both sides raise Mismatch on a difference."""
    send_msg(sock, mine)
    theirs = expect(sock, "hello", HELLO_TIMEOUT_S)
    bad = [k for k in ("code", "torch", "transformers", "job") if mine[k] != theirs.get(k)]
    if bad:
        lines = [f"  {k}: {mine['node']}={mine[k]} {theirs.get('node')}={theirs.get(k)}" for k in bad]
        raise Mismatch("the two nodes differ:\n" + "\n".join(lines))
    return theirs


def hello_line(mine, theirs):
    """The log line the smoke verdict's peer check looks for."""
    return (
        f"[B09] peer {theirs.get('node')} matched: code {mine['code'][:8]} torch {mine['torch']} "
        f"transformers {mine['transformers']} job {mine['job'][:8]} run {mine['run_id']}"
    )


def agree_on_resume(sock, already_ok, timeout):
    """Right after hello. On a resume each node knows only its own job state, so
    a node that finished a job must not skip it alone: the other would rerun it
    and wait for a peer that never comes. Skip only if BOTH finished it ok;
    otherwise both rerun it. -> True to skip."""
    send_msg(sock, {"type": "resume", "ok": bool(already_ok)})
    theirs = expect(sock, "resume", timeout)
    return bool(already_ok) and theirs.get("ok") is True


def sync_ready(sock, index, label, timeout):
    """Lockstep: both announce configuration `index`, both wait for the other."""
    send_msg(sock, {"type": "ready", "i": index, "label": label})
    msg = expect(sock, "ready", timeout)
    if (msg.get("i"), msg.get("label")) != (index, label):
        raise Mismatch(
            f"out of step: this node is at config {index} {label}, "
            f"the other at {msg.get('i')} {msg.get('label')}"
        )


def send_result(sock, payload):
    send_msg(sock, {"type": "result", "payload": payload})


def recv_result(sock, timeout):
    msg = expect(sock, "result", timeout)
    if "payload" not in msg:
        raise Mismatch("a 'result' message without a payload")
    return msg["payload"]


def finish(sock, ok, timeout):
    """A job's last message: each node says whether it finished cleanly. The job
    is ok only if BOTH did, so a failure on either node fails it on both and the
    two nodes' job states always agree (a resume then reruns it on both).
    -> True if both ok."""
    send_msg(sock, {"type": "bye", "ok": bool(ok)})
    theirs = expect(sock, "bye", timeout)
    return bool(ok) and theirs.get("ok") is True


def send_bye(sock):
    try:
        send_msg(sock, {"type": "bye"})
    except PeerLost:
        pass


def recv_bye(sock, timeout):
    expect(sock, "bye", timeout)


def merge_queue_items(lead_items, other_items):
    """Union of both nodes' queue items. Keys carry the global rank, so they
    do not collide except 'oom' (ranks on both nodes OOM'd in one config);
    the lead's value wins."""
    return {**other_items, **lead_items}
