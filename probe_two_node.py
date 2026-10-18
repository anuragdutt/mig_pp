"""
Acceptance gate for the two-node pipeline transport. Run on BOTH nodes at about
the same time (either order; each waits up to peer_timeout_s for the other):

    MIG_NODE=A python3 probe_two_node.py        # on node A
    MIG_NODE=B python3 probe_two_node.py        # on node B

Reads NODES / PLACEMENT from two_node_config.py. Spawns this node's ranks (one
process per MIG slice, global rank numbers), runs the cases below in order and
prints one table with every rank's result from BOTH nodes, then a VERDICT. Paste
either node's output back (both print the same table).

  nic NODES[node]["addr"] is on ifname. gloo binds ifname's own address, so with
      the address on another NIC the traffic would take ifname's network and
      every case below would still pass
  N1  control channel up; code hash (this file included, so its CONFIG too) +
      torch/transformers versions match
  N2  gloo init across nodes; all_gather of every rank's hostname: one host per
      node, and the two nodes on different hosts (else the hops were loopback)
  --  cuda_device: torch.cuda.set_device on the rank's MIG slice (a bad UUID shows here)
  N3  raw gloo send/recv of pinned CPU tensors over each cross-node hop, bit-exact
  N4  TokenExchange across nodes (last rank -> rank 0, as in the benchmark)
  --  engine init (the shipped MIGPipelineTransport with rank_nodes)
  N5  shipped engine over ONE cross-node hop: steps x microbatches, tags reused,
      recv buffers reused, a slot pool of only n_mb+1 so slots are reused every
      step; bit-exact; zero slot spins
  N6  shipped engine over the WHOLE placement (SHM hops and network hops in one
      chain), each stage adds 1, bit-exact end to end
  N7  timing and transport counters (info only; fails only if reading them raises)
  exit  every rank process finished and exited 0 (judged by its parent)
  exchange  the two parents traded their rows (else each prints only its own)

Ranks report each case the moment it ends, so a rank that hangs later still
shows the rows (and the failure) it got to.

Order: the cheap raw cases first and the shipped engine last, so an early
failure points at the network or gloo setup instead of at the engine, and a
failure that poisons the process group (the first gloo probe's lesson) cannot
hide a case that would have passed. After a failed case the remaining ones are
SKIPped.

The benchmark is NOT modified until this prints VERDICT: PASS on both nodes.
"""

import multiprocessing
import os
import queue
import re
import runpy
import socket
import subprocess
import sys
import time
import traceback
from datetime import timedelta

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import experiment_config as ec  # noqa: E402  (torch-free)
import two_node_control as tc  # noqa: E402  (torch-free)

CASES = ["nic", "N1", "N2", "cuda_device", "N3", "N4", "engine_init", "N5", "N6", "N7", "exit", "exchange"]

# N1 compares this file too: the cases and their CONFIG (steps, microbatches,
# shapes) live here, and two different probes would pass the handshake and then
# hang in N5/N6 waiting for messages the other side never sends.
PROBE_CODE_FILES = tc.CODE_FILES + ("probe_two_node.py",)


# --- helpers shared by the parent and its ranks --------------------------------
def make_payload(seed, shape):
    """Deterministic fp16 integers 0..63 (exact in fp16), same on both nodes."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, shape, generator=g).half()


def first_cross_pair(rank_nodes):
    for r in range(len(rank_nodes) - 1):
        if rank_nodes[r] != rank_nodes[r + 1]:
            return [r, r + 1]
    return None


def token_source(rank_nodes):
    """The rank whose next_tokens reach rank 0 across nodes: the last rank (as in
    the benchmark) when it is on another node, else the first rank that is."""
    if rank_nodes[-1] != rank_nodes[0]:
        return len(rank_nodes) - 1
    for r, n in enumerate(rank_nodes):
        if n != rank_nodes[0]:
            return r
    return None


def host_problems(hosts, rank_nodes):
    """N2's placement proof. hosts: the hostname each rank reported, by rank.
    Every node's ranks must share one host, and the two nodes' hosts must
    differ: otherwise both "nodes" are one machine and the network hops ran
    over loopback, which every later case would happily pass."""
    by_node = {}
    for r, n in enumerate(rank_nodes):
        by_node.setdefault(n, set()).add(hosts[r])
    problems = [f"node {n}'s ranks report several hosts: {sorted(h)}"
                for n, h in by_node.items() if len(h) > 1]
    names = [next(iter(h)) for h in by_node.values() if len(h) == 1]
    if not problems and len(set(names)) < len(names):
        problems.append(f"nodes {sorted(by_node)} all run on host {names[0]}: "
                        "not two machines, so the cross-node hops never left it")
    return problems


def combine(entries):
    """entries: [(rank, node, name, ok, detail)] -> {name: (status, [lines])}.
    FAIL if any rank failed, SKIP if every rank skipped, else PASS."""
    out = {}
    for name in CASES:
        rows = [e for e in entries if e[2] == name]
        if not rows:
            continue
        oks = [e[3] for e in rows]
        status = "FAIL" if False in oks else "SKIP" if all(o is None for o in oks) else "PASS"
        out[name] = (status, [f"r{r}[{n}]: {d}" for r, n, _, _, d in sorted(rows, key=lambda e: (e[0], e[1]))])
    return out


def render(entries, title):
    """The table both nodes print, and the number of failed cases."""
    combined = combine(entries)
    lines, fails = [title], 0
    for name in CASES:
        if name not in combined:
            lines.append(f"SKIP  {name}: no result")
            continue
        status, rows = combined[name]
        fails += status == "FAIL"
        lines.append(f"{status}  {name}")
        lines += ["      " + row.replace("\n", "\n        ") for row in rows]
    lines.append(f"VERDICT: {'FAIL (%d)' % fails if fails else 'PASS'}")
    return lines, fails


def probe_hello(node, job, repo=REPO):
    return tc.make_hello(node, time.strftime("%Y%m%d_%H%M%S"), repo, job, files=PROBE_CODE_FILES)


def ifname_addrs(ifname):
    """IPv4 addresses `ip` lists on ifname ([] for none or an unknown NIC), or
    None where there is no `ip` (macOS)."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", ifname],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return re.findall(r"\binet (\d+\.\d+\.\d+\.\d+)/", out)


def nic_row(node, ifname, want, found):
    """The parent's 'nic' row. gloo binds ifname's own address, so with
    NODES[node]["addr"] on another NIC the traffic would take ifname's network
    and every case would still pass."""
    if found is None:
        return (-1, node, "nic", None, f"{ifname}: no `ip` here, NODES[{node!r}]['addr'] not checked")
    if want in found:
        return (-1, node, "nic", True, f"{ifname} carries {want}")
    return (-1, node, "nic", False,
            f"{ifname} carries {', '.join(found) or 'no IPv4 address'}, not NODES[{node!r}]['addr'] "
            f"{want}: gloo would bind {ifname}'s address and use that network")


def trade_rows(sock, entries, is_lead, wait_s):
    """The other node's rows, so both print the whole table. The lead sends first
    and the other node replies, so the two are never both blocked in sendall,
    however long the rows get."""
    sock.settimeout(wait_s)  # a send may wait for the other node to start reading
    if is_lead:
        tc.send_result(sock, entries)
        return [tuple(e) for e in tc.recv_result(sock, wait_s)]
    theirs = [tuple(e) for e in tc.recv_result(sock, wait_s)]
    tc.send_result(sock, entries)
    return theirs


def stop_process(p):
    """terminate(), then kill() if SIGTERM did not end it: a live child would
    block this parent at exit (multiprocessing joins its children)."""
    p.terminate()
    p.join(5)
    if p.is_alive():
        p.kill()
        p.join(5)


def collect_local(procs, local, q, node, deadline_s, poll_s=1.0):
    """
    This node's rank results, plus an 'exit' verdict per rank process. A rank
    puts one item per case as it ends and a final marker once it is done, so a
    rank that hangs or dies keeps the rows it already sent. A rank that finished
    and then exited non-zero (say, crashed in teardown) is a failure: the
    benchmark records such a config as 'crash', not 'ok'. A rank that died early
    is noticed at once, not after the deadline.
    """
    entries, finished = [], set()

    def take(item):
        rank, rnode, results, done = item
        entries.extend((rank, rnode, name, ok, detail) for name, ok, detail in results)
        if done:
            finished.add(rank)

    deadline = time.monotonic() + deadline_s
    while len(finished) < len(local) and time.monotonic() < deadline:
        try:
            take(q.get(timeout=poll_s))
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break  # all exited; what they put just before is drained below
    while True:  # rows still in the pipe, from ranks that just exited or that hang
        try:
            take(q.get(timeout=poll_s))
        except queue.Empty:
            break

    for p, r in zip(procs, local):
        p.join(10)
        hung = p.is_alive()
        if hung:
            stop_process(p)
        if r not in finished:
            why = "hung; terminated" if hung else f"exited with code {p.exitcode}"
            entries.append((r, node, "exit", False, f"{why} before finishing"))
        elif hung:
            entries.append((r, node, "exit", False, "finished, then hung at exit; terminated"))
        elif p.exitcode != 0:
            entries.append((r, node, "exit", False, f"finished, then exited with code {p.exitcode}"))
        else:
            entries.append((r, node, "exit", True, "exit code 0"))
    return entries


# --- one rank --------------------------------------------------------------------
def rank_main(rank, cfg, q):
    """Runs in its own process: one MIG slice, one global rank."""
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["uuids"][rank]
    os.environ["MASTER_ADDR"] = cfg["master_addr"]
    os.environ["MASTER_PORT"] = str(cfg["master_port"])
    os.environ["GLOO_SOCKET_IFNAME"] = cfg["ifname"]
    os.environ["MIG_SHM_PREFIX"] = f"migpp_probe_{cfg['node']}"
    os.environ.setdefault("MIG_LOG_STAMP", time.strftime("%Y%m%d_%H%M%S"))

    state = {"dead": None}  # name of the case after which nothing can run

    def report(name, ok, detail):
        # Put the moment a case ends, not at exit: a rank that hangs later still
        # leaves its rows, the failure that explains the hang included.
        q.put((rank, cfg["node"], [(name, ok, str(detail))], False))

    def run_case(name, fn):
        if state["dead"]:
            report(name, None, f"skipped: {state['dead']} failed")
            return
        try:
            detail = fn()
        except Exception as e:  # noqa: BLE001 - the probe reports, never raises
            report(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
            # Leaving this block frees the traceback and what the case held,
            # pending gloo sends included. On macOS (gloo's uv transport) that
            # free can spin forever, so give the queue's feeder thread time to
            # ship the row first.
            time.sleep(0.5)
            state["dead"] = name
            return
        report(name, True, detail)
        try:
            dist.barrier()
        except Exception as e:  # noqa: BLE001
            report(name, False, f"barrier after the case failed: {e}")
            state["dead"] = name

    try:
        import torch
        import torch.distributed as dist

        world, rank_nodes = len(cfg["uuids"]), cfg["rank_nodes"]
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=world,
            timeout=timedelta(seconds=cfg["timeout_s"]),
        )
        mb, seq, hidden, n_mb, steps = (cfg[k] for k in ("mb", "seq", "hidden", "n_mb", "steps"))

        def cuda_device():
            torch.cuda.set_device(torch.device("cuda:0"))
            return f"cuda:0 is {cfg['uuids'][rank]}"

        def n2():
            try:
                resolved = socket.gethostbyname(socket.gethostname())
            except OSError:
                resolved = "unresolvable"
            me = (rank, socket.gethostname(), resolved, os.environ["GLOO_SOCKET_IFNAME"])
            got = [None] * world
            dist.all_gather_object(got, me)
            if sorted(g[0] for g in got) != list(range(world)):
                raise RuntimeError(f"all_gather returned ranks {[g[0] for g in got]}")
            bad = host_problems([g[1] for g in sorted(got)], rank_nodes)
            if bad:
                raise RuntimeError("; ".join(bad))
            note = " (hostname resolves to loopback; gloo uses GLOO_SOCKET_IFNAME, informational)" \
                if me[2].startswith("127.") else ""
            return f"gloo up, world {world}; this rank: host {me[1]} resolves {me[2]} ifname {me[3]}{note}"

        def n3():
            pair = first_cross_pair(rank_nodes)
            if pair is None:
                raise RuntimeError("placement has no cross-node hop")
            src, dst = pair
            if rank not in pair:
                return "not on the hop"
            shapes = {"decode": (mb, 1, hidden), "prefill": (mb, seq, hidden)}
            parts = []
            for kind, shape in shapes.items():
                t0 = time.perf_counter()
                for i in range(10):
                    tag, seed = 500 + i, (1000 if kind == "decode" else 2000) + i  # not hash(): per-process
                    if rank == src:
                        dist.isend(make_payload(seed, shape).pin_memory(), dst=dst, tag=tag).wait()
                    else:
                        buf = torch.empty(shape, dtype=torch.float16).pin_memory()
                        dist.irecv(buf, src=src, tag=tag).wait()
                        if not torch.equal(buf, make_payload(seed, shape)):
                            raise RuntimeError(f"{kind} message {i} differs from what was sent")
                parts.append(f"{kind} {shape} x10 bit-exact in {(time.perf_counter() - t0) * 1000:.0f}ms")
            return "; ".join(parts)

        def n4():
            from token_exchange import TokenExchange

            src = token_source(rank_nodes)
            if src is None:
                raise RuntimeError("no rank outside rank 0's node")
            tokens = torch.zeros((8, 1), dtype=torch.long, device="cuda:0")
            tok = TokenExchange(tokens)
            if rank not in (0, src):
                return "not involved"
            for step in range(10):
                want = (torch.arange(8).reshape(8, 1) + step * 7).long()
                if rank == src:
                    tokens.copy_(want)
                    tok.send(dst=0, tag=900 + step)
                else:
                    tok.recv(src=src, tag=900 + step)
                    if not torch.equal(tokens.cpu(), want):
                        raise RuntimeError(f"step {step}: tokens {tokens.flatten().tolist()}")
            return f"rank {src} -> rank 0, 10 exchanges bit-exact; {tok.summary()}"

        def engine_init():
            import mig_transport_pipeline as mt

            log = mt.setup_transport_logging(log_dir=cfg["log_dir"])
            mt.register_hooks(
                mb_size=mb, seq_len=seq, hidden_size=hidden, num_slots=n_mb + 1,
                rank_nodes=rank_nodes,
            )
            e = mt._MIG_PIPE_ENGINE
            hops = ["net" if e._is_remote(p) else "shm" for p in (rank - 1, rank + 1) if 0 <= p < world]
            return f"engine up, {n_mb + 1} slots, hops {hops}, peer SHM slots for ranks {sorted(e.peer_slots)}, log {log or 'off'}"

        timings = {}

        def run_chain(chain, label):
            import mig_transport_pipeline as mt

            if rank not in chain:
                return "not in chain"
            pos = chain.index(rank)
            prev = chain[pos - 1] if pos > 0 else None
            nxt = chain[pos + 1] if pos + 1 < len(chain) else None
            shapes = {"prefill": (mb, seq, hidden), "decode": (mb, 1, hidden)}
            # Receive buffers are reused across steps, as in the benchmark.
            bufs = {k: [torch.empty(s, dtype=torch.float16, device="cuda:0") for _ in range(n_mb)]
                    for k, s in shapes.items()}
            spins0 = mt._MIG_PIPE_ENGINE.stat_slot_spins
            t0 = time.perf_counter()
            for step in range(steps):
                kind = "prefill" if step % 2 == 0 else "decode"
                shape, base_tag = shapes[kind], 20_000 + step * 10_000
                if prev is not None:
                    handles = [dist.irecv(bufs[kind][i], src=prev, tag=base_tag + i) for i in range(n_mb)]
                sends = []
                for i in range(n_mb):
                    base = make_payload(step * 1000 + i, shape)
                    if prev is None:
                        cur = base.to("cuda:0")
                    else:
                        handles[i].wait()
                        cur = bufs[kind][i]
                        # Each stage adds 1: the stage at chain position p sends base + p + 1.
                        if not torch.equal(cur.cpu(), base + pos):
                            raise RuntimeError(f"step {step} mb {i}: payload from rank {prev} differs")
                    out = cur + 1
                    if nxt is not None:
                        sends.append(dist.isend(out.clone(), dst=nxt, tag=base_tag + i))
                        if len(sends) > 1:
                            sends[-2].flush()
                if sends:
                    sends[-1].flush()
                for h in sends:
                    h.wait()
            elapsed = time.perf_counter() - t0
            spins = mt._MIG_PIPE_ENGINE.stat_slot_spins - spins0
            timings[label] = f"{elapsed / steps * 1000:.1f}ms/step"
            if spins:
                raise RuntimeError(f"{spins} slot spins: the pool of {n_mb + 1} was exhausted")
            return f"{steps} steps x {n_mb} mb bit-exact, {timings[label]}, 0 slot spins"

        def n5():
            pair = first_cross_pair(rank_nodes)
            if pair is None:
                raise RuntimeError("placement has no cross-node hop")
            return run_chain(pair, "N5")

        def n6():
            return run_chain(list(range(world)), "N6")

        def n7():
            import mig_transport_pipeline as mt

            stats = mt.log_summary(label=f"probe rank {rank}") or {}
            keep = ("net_send_count", "net_send_wait_mean_ms", "net_recv_count",
                    "net_recv_wait_mean_ms", "flush_block_mean_ms", "h2d_block_mean_ms", "slot_spins")
            return f"step times {timings}; " + ", ".join(f"{k}={stats.get(k, 0):.2f}" for k in keep)

        run_case("N2", n2)
        run_case("cuda_device", cuda_device)
        run_case("N3", n3)
        run_case("N4", n4)
        run_case("engine_init", engine_init)
        run_case("N5", n5)
        run_case("N6", n6)
        run_case("N7", n7)
    except Exception as e:  # noqa: BLE001 - torch import or gloo init failed: nothing ran
        report("N2", False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
    finally:
        # Teardown on every path, so a failed rank leaves no /dev/shm slots.
        _dist = sys.modules.get("torch.distributed")
        try:
            if _dist is not None and _dist.is_initialized():
                _dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        try:
            mt = sys.modules.get("mig_transport_pipeline")
            if mt is not None:
                mt.cleanup()
        except Exception:  # noqa: BLE001
            pass
        q.put((rank, cfg["node"], [], True))  # finished


# --- parent ----------------------------------------------------------------------
def main():
    CONFIG = dict(
        config_file=os.path.join(REPO, "two_node_config.py"),
        master_port=29990,
        control_port=29991,
        hidden=5120,  # slot sizing only; any model's hidden size works
        mb=4,
        seq=64,
        n_mb=6,
        steps=20,
        timeout_s=90,  # gloo timeout and per-case bound
        peer_timeout_s=600,
        log_dir="logs",
    )

    node = os.environ.get("MIG_NODE", "")
    ns = runpy.run_path(CONFIG["config_file"])
    nodes, placement = ns["NODES"], ns["PLACEMENT"]
    if node not in nodes:
        sys.exit(f"MIG_NODE={node!r}; set it to one of {sorted(nodes)}")
    rank_nodes = [p[0] for p in placement]
    uuids = [p[1] for p in placement]
    problems = ec.rank_nodes_problems(rank_nodes)
    problems += [f"PLACEMENT node {n!r} is not in NODES" for n in sorted(set(rank_nodes) - set(nodes), key=str)]
    if problems:
        sys.exit("two_node_config.py: " + "; ".join(problems))
    local = [r for r, n in enumerate(rank_nodes) if n == node]
    if not local:
        sys.exit(f"no rank of PLACEMENT is on node {node}")
    if any("PASTE" in str(x) for x in uuids + [nodes[n][k] for n in nodes for k in ("addr", "ifname")]):
        sys.exit("two_node_config.py still holds PASTE placeholders")
    lead = rank_nodes[0]
    addrs = {n: nodes[n]["addr"] for n in nodes}
    ifname = nodes[node]["ifname"]
    entries = [nic_row(node, ifname, addrs[node], ifname_addrs(ifname))]

    # N1: control channel and identity.
    job = dict(rank_nodes=rank_nodes, slice_gb=[p[2] for p in placement], mig_uuids=uuids,
               node_addrs=addrs, master_port=CONFIG["master_port"],
               control_port=CONFIG["control_port"])
    mine = probe_hello(node, job)
    sock = None
    n1 = None
    try:
        if node == lead:
            sock = tc.listen_for_peer(addrs[lead], CONFIG["control_port"], CONFIG["peer_timeout_s"])
        else:
            sock = tc.connect_to_lead(addrs[lead], CONFIG["control_port"], CONFIG["peer_timeout_s"])
        theirs = tc.exchange_hello(sock, mine)
        n1 = (True, tc.hello_line(mine, theirs))
    except tc.Mismatch as e:  # the channel works: still trade rows, so both print both sides
        n1 = (False, f"Mismatch: {e}")
    except (tc.PeerLost, OSError) as e:
        # The channel is dead (or holds a stray client): trading rows over it
        # would only wait out its timeout, so drop it and print this side alone.
        n1 = (False, f"{type(e).__name__}: {e}")
        if sock is not None:
            sock.close()
        sock = None
    entries.append((-1, node, "N1", n1[0], n1[1]))

    if n1[0]:
        cfg = dict(node=node, uuids=uuids, rank_nodes=rank_nodes, master_addr=addrs[lead],
                   ifname=ifname, **{k: CONFIG[k] for k in
                   ("master_port", "hidden", "mb", "seq", "n_mb", "steps", "timeout_s", "log_dir")})
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=rank_main, args=(r, cfg, q)) for r in local]
        for p in procs:
            p.start()
        try:
            entries += collect_local(procs, local, q, node, CONFIG["timeout_s"] * 12)
        finally:
            for p in procs:  # only if collect_local itself was interrupted
                if p.is_alive():
                    stop_process(p)

    if sock is not None:
        # The other node's collect deadline, up to 20 s per rank to stop it
        # (join 10 + terminate 5 + kill 5), and a margin.
        wait_s = CONFIG["timeout_s"] * 12 + 20 * len(rank_nodes) + 60
        try:
            entries += trade_rows(sock, entries, node == lead, wait_s)
            entries.append((-1, node, "exchange", True, "rows traded with the other node"))
        except Exception as e:  # noqa: BLE001 - the probe reports, never raises
            entries.append((-1, node, "exchange", False,
                            f"could not trade rows with the other node: {type(e).__name__}: {e}"))
        sock.close()

    title = (f"\n===== probe_two_node: node {node}, placement "
             f"{','.join(f'{n}{p[2]}' for n, p in zip(rank_nodes, placement))} =====")
    lines, fails = render(entries, title)
    print("\n".join(lines))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
