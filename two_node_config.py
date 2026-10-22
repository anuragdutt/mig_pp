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
    "A": dict(addr="10.191.130.146", ifname="enp161s0f0", gpu=0),
    "B": dict(addr="10.191.131.46", ifname="enp161s0f0", gpu=0),
}

# Pipeline order: (node, MIG UUID, slice GB). The node holding rank 0 is the lead.
PLACEMENT = [
    ("A", "MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866", 20),
    ("A", "MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5", 10),
    ("B", "MIG-952b3e2f-2d64-5456-9930-c90a85a45f62", 5),
    ("B", "MIG-175e1cbc-85ba-5f44-ac48-deac1590e175", 5),
]

# Run in this order (one model at a time). layer_limits: one cap per rank,
# required. A slice's capacity does not depend on which node it sits in, so the
# single-GPU 20_10_5_5 caps of layer_limits.py carry over.
MODELS = {
    # Done 2026-10-20 (run 20261018_203822): the widened reference sweep.
    # "vicuna_7b": dict(path=f"{MODEL_ROOT}/vicuna-7b-v1.5", layer_limits=[28, 12, 5, 5]),
    # Widened like Vicuna: rank 0 may take all but 1 + 1 layers (26 of 28; the
    # last rank can be head-only). Ranks 1-3 keep the 20_10_5_5 caps; the
    # 152k-vocab lm_head leaves the last 5GB slice room for 3 layers at most.
    # B16 + B32 done 2026-10-21; this is the B8 remainder.
    "qwen_7b": dict(path=f"{MODEL_ROOT}/Qwen2.5-7B", layer_limits=[26, 12, 5, 3],
                    batch_mb_pairs=[(8, 4), (8, 2)]),
    # Widened like Vicuna-7B (same 32-layer shape, GQA: 4x smaller KV). All 9 pairs.
    "mistral_7b": dict(path=f"{MODEL_ROOT}/Mistral-7B-Instruct-v0.3", layer_limits=[28, 12, 5, 5]),
    # Bigger models, widened too: rank 0 uncapped (all but 1 layer per other
    # rank). Many extreme splits do not fit a 20GB slice at B16/B32; those run
    # and record OOM_rank0 rows (the 13B/14B memory model is extrapolated, so
    # the OOM boundary is data too). 13B cannot fit B32 at all: B8 + B16 only.
    "vicuna_13b": dict(path=f"{MODEL_ROOT}/vicuna-13b-v1.5", layer_limits=[37, 12, 5, 5],
                       batch_mb_pairs=[(8, 4), (8, 2), (16, 8), (16, 4), (16, 2)]),
    # Head-only last rank (152k-vocab lm_head): the planner sets min_last_rank_layers=0.
    "qwen_14b": dict(path=f"{MODEL_ROOT}/Qwen2.5-14B", layer_limits=[46, 15, 7, 0]),
}

SWEEP = dict(
    seq_len=64,
    max_new_tokens=512,
    max_runs=None,
    enforce_slice_ordering=True,
    batch_mb_pairs=[  # default for every model; an entry's own batch_mb_pairs wins
        (8, 4), (8, 2),
        (16, 8), (16, 4), (16, 2),
        (32, 16), (32, 8), (32, 4), (32, 2),
    ],
)

RUNNER = dict(
    base_port=29600,  # job k: master port base_port + 10k (single-node lanes use 29500-29570)
    control_port=29900,  # job k: control port control_port + k
    peer_timeout_s=1800,  # how long a parent waits for the other node to show up
    mem_monitor="nvml",  # "nvml" or "off"; DCGM is refused
    env={"MIG_LOG_LEVEL": "summary"},
)
