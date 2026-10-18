"""
One pipeline across two nodes: ranks on MIG slices of one GPU in each node.
Read by probe_two_node.py and by the planner:
    ./run_parallel.sh smoke -c two_node_config.py --node A    # on node A (mig1)
    ./run_parallel.sh smoke -c two_node_config.py --node B    # on node B (mig2)

Layout, fixed for every run: node A (mig1) holds the big slices, node B (mig2)
the small ones, on one A100-PCIE-40GB each (GPU 0):
    A: 3g.20gb + 2g.10gb  -> ranks 0, 1   (A is the lead: it holds rank 0)
    B: 1g.5gb  + 1g.5gb   -> ranks 2, 3

Steps on each node:
  1. create the MIG slices on that node's GPU, `nvidia-smi -L` for their UUIDs
  2. paste the UUIDs into PLACEMENT below (same file on both nodes)
  3. set NODES: the private address and NIC of each node
  4. open all TCP ports between the nodes on that NIC (gloo picks ephemeral ports)
Unknown keys are errors, so typos fail loudly.
"""
import os

MODEL_ROOT = os.path.expanduser("~/models")

# addr:   this node's IP on the private network (gloo binds and connects on it)
# ifname: the NIC carrying addr (exported to every rank as GLOO_SOCKET_IFNAME)
# gpu:    physical GPU index holding this node's slices
NODES = {
    "A": dict(addr="PASTE-IP-OF-MIG1", ifname="PASTE-NIC", gpu=0),
    "B": dict(addr="PASTE-IP-OF-MIG2", ifname="PASTE-NIC", gpu=0),
}

# Pipeline order: (node, MIG UUID, slice GB). The node holding rank 0 is the lead.
PLACEMENT = [
    ("A", "MIG-PASTE-3g.20gb-from-mig1", 20),
    ("A", "MIG-PASTE-2g.10gb-from-mig1", 10),
    ("B", "MIG-PASTE-1g.5gb-from-mig2", 5),
    ("B", "MIG-PASTE-1g.5gb-from-mig2-second", 5),
]

# Run in this order (one model at a time). layer_limits: one cap per rank,
# required. A slice's capacity does not depend on which node it sits in, so the
# single-GPU 20_10_5_5 caps of layer_limits.py carry over.
MODELS = {
    "vicuna_7b": dict(path=f"{MODEL_ROOT}/vicuna-7b-v1.5", layer_limits=[18, 12, 5, 5]),
}

SWEEP = dict(
    seq_len=64,
    max_new_tokens=512,
    max_runs=None,
    enforce_slice_ordering=True,
    batch_mb_pairs=[(8, 4), (8, 2)],
)

RUNNER = dict(
    base_port=29600,  # job k: master port base_port + 10k (single-node lanes use 29500-29570)
    control_port=29900,  # job k: control port control_port + k
    peer_timeout_s=1800,  # how long a parent waits for the other node to show up
    mem_monitor="nvml",  # "nvml" or "off"; DCGM is refused
    env={"MIG_LOG_LEVEL": "summary"},
)
