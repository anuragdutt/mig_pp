"""
Measured hardware constants for the layer-placement ILPs (graph_placement.py, gpipe_placement.py).

To add a new configuration: run measure_constants.py on that machine, copy its numbers into a new
entry of CONSTANTS with the same keys, and pick it in the ILP's CONFIG:

    constants=hardware_constants.CONSTANTS["<name>"]

Probe sets, named by box (e.g. "mig1"): run the probes on the box (setup.md, "Probe runs"),
export the folder, run probe_constants.py on it; it writes probe_sets/<box>.json, loaded below as
CONSTANTS["<box>"]. Their slices hold generic GPU primitives with their measured points, from
which graph_placement composes every GPU time (layer_ops); they carry the CPU clock and thread
pool from the CPU probes. The sets written out below stay as they are.

New cards go into CARD_BW and MEMORY_SLICES.
"""

CONSTANTS = {
    # Measured by measure_constants.py on a whole A100 80GB PCIe (a100-05, job 2302084,
    # Vicuna-7B config, torch 2.14.1+cu126). eff_act is the launch-free value, matching terms(),
    # which keeps launches out of a layer's GPU time.
    "a100_80gb_pcie_whole_gpu": dict(
        eff_weight=0.769,  # achieved / peak bandwidth streaming weights
        eff_act=0.125,  # ... on per-token activation + KV traffic (launch-free form)
        launch_us=22.6,  # CPU dispatch per kernel inside a decode layer call
        kernels_per_layer=40,
        pcie_bw=1.33e9,  # pinned D2H, bytes/s, at the mb-8 decode activation (64 KiB)
        recv_h2d_bw=2.11e9,  # pinned H2D, same size
        shm_to_pinned_bw=2.18e10,  # receiver host memcpy
        memcpy_bw=2.64e10,  # sender pinned -> SHM
        handshake_ack_us=70.5,  # one handshake + one ACK over gloo (half charged to each side)
        tok_roundtrip_us=37.1,  # next_tokens gloo send/recv
        token_staging_us=47.6,  # its D2H + H2D
        step_start_us=73.4,  # position_ids sync at the start of a decode step
        barrier_us=99.9,  # gloo barrier, once after prefill (the final one is in END)
        rotary_kernels=9,  # extra launches per microbatch on every rank
        timing_event_calls=2,  # CPU-only CUDA calls per microbatch on every rank (not measured)
        send_queue_calls=2,  # ... and on sending ranks (not measured)
    ),
}

# mig1 / mig2: A100-PCIE-40GB, one GPU, MIG 20/10/5/5 (3g.20gb, 2g.10gb, 1g.5gb, 1g.5gb),
# schedutil governor. graph_placement.py reads `slices` per rank by slice GB.
# Per slice, measure_constants.py on these machines, Vicuna-7B config (measurements.md on branch
# test/intra-gpu-setup-2-A100-40gb: slice_3g20_v2, slice_2g10_v2, slice_1g5):
#   weight_rate      achieved bytes/s streaming the weights (mean of M = 1, 8, 32)
#   layer_fixed_us   decode layer kernel time - weight time = layer_fixed + mb x act_bytes(ctx) / act_rate,
#   act_rate         least squares over mb 1 / 8 / 32 x ctx 64..575 (worst point 6.5-8.2% off)
#   prefill_flops    prefill FLOP/s at mb 16, seq 64
#   pcie_bw          pinned D2H of a 64 KiB decode activation;  recv_h2d_bw  pinned H2D, same size
# launch_us: the harness's own CPU launch time per decode layer at full clock (run 10 events data,
#   0.67-0.70 ms per layer = 40 kernels x 17.2 us; the slice probe measured 19.8-20.1 us per kernel).
# Not measured on these machines yet, so from the whole-GPU set above: handshake_ack_us,
#   tok_roundtrip_us, token_staging_us, step_start_us, barrier_us, memcpy_bw, shm_to_pinned_bw.
# CPU clock, used with CONFIG clock="schedutil". Unlike the rest, fitted to measured data
# (run 3 events data, schedutil, Vicuna-7B):
#   clock_slow_factor     launch time per layer at 1.5 GHz / at boost: 1.56 / 0.69 ms
#   clock_busy_threshold  per rank: fast when N x GPU time per microbatch / step >= this, both as
#                         graph_placement predicts them with this set (so the ILP applies them
#                         consistently); Vicuna 99 / 94 / 89 / 89% of levels right, held-out
#                         Qwen-7B 99 / 93 / 71 / 81%
CONSTANTS["a100_40gb_pcie_mig"] = dict(
    CONSTANTS["a100_80gb_pcie_whole_gpu"],
    launch_us=17.2,
    slices={
        20: dict(weight_rate=6.36e11, layer_fixed_us=137.0, act_rate=2.03e11, prefill_flops=71.4e12,
                 pcie_bw=1.37e9, recv_h2d_bw=2.01e9),  # 3g.20gb, mig1
        10: dict(weight_rate=3.36e11, layer_fixed_us=142.8, act_rate=1.10e11, prefill_flops=46.2e12,
                 pcie_bw=1.34e9, recv_h2d_bw=1.97e9),  # 2g.10gb, mig1
        5: dict(weight_rate=1.76e11, layer_fixed_us=149.1, act_rate=5.50e10, prefill_flops=21.2e12,
                pcie_bw=1.28e9, recv_h2d_bw=1.90e9),  # 1g.5gb, mig2
    },
    clock_slow_factor=2.26,
    clock_busy_threshold=[0.31, 0.40, 0.55, 0.78],
)

# ---------------------------------------------------------------------------
# Inter-GPU setup (inter_gpu_measurements.md): two A100-PCIE-40GB nodes, MIG on each.
#   mig1 (node A): 3g.20gb (rank 0) + 2g.10gb (rank 1)
#   mig2 (node B): 1g.5gb  (rank 2) + 1g.5gb  (rank 3)
# Measured per slice, not on a whole GPU, so these do not fit the CONSTANTS form above:
# the ILPs take ONE set for every rank and do not read SLICE_CONSTANTS / NETWORK_CONSTANTS yet.
# ---------------------------------------------------------------------------

# per MIG slice, measure_constants.py on that slice, Vicuna-7B config unless noted.
# weight_rate / act_rate are achieved bytes/s (not efficiencies); act_rate is the ILP form
# (launches included), act_rate_no_launch the launch-free form terms() uses.
SLICE_CONSTANTS = {
    "a100_40gb_pcie_mig": {
        "3g.20gb": dict(  # mig1, slice_3g20_v2.log
            weight_rate=6.36e11,  # eff_weight vs card 0.409, vs mem share 0.819
            act_rate=4.46e11,
            act_rate_no_launch=1.48e11,
            pcie_bw=1.37e9,  # pinned D2H, 64 KiB decode activation
            recv_h2d_bw=2.01e9,
            launch_us=19.8,
            kernels_per_layer=40,
        ),
        "2g.10gb": dict(  # mig1, slice_2g10_v2.log
            weight_rate=3.36e11,  # eff_weight vs card 0.216, vs mem share 0.863
            act_rate=1.80e11,
            act_rate_no_launch=9.09e10,
            pcie_bw=1.34e9,
            recv_h2d_bw=1.97e9,
            launch_us=20.1,
            kernels_per_layer=40,
        ),
        "1g.5gb": dict(  # mig2, slice_1g5.log (first run: no launch-free act rate printed)
            weight_rate=1.76e11,  # eff_weight vs card 0.113, vs mem share 0.906
            act_rate=7.37e10,
            pcie_bw=1.28e9,
            recv_h2d_bw=1.90e9,
            launch_us=20.1,
            kernels_per_layer=40,
        ),
        "1g.5gb_qwen14b": dict(  # mig2, slice_1g5_v2.log, Qwen2.5-14B config (80 KiB activation)
            weight_rate=1.74e11,
            act_rate=1.14e11,
            pcie_bw=1.52e9,
            recv_h2d_bw=2.38e9,
            launch_us=20.1,
            kernels_per_layer=40,
        ),
    },
}

# cross-node hop mig1 -> mig2 over gloo (measure_constants.py --mode net, both nodes agree):
#   one way = net_fixed_us + bytes / net_bw;  next_tokens one way = tok_net_us
NETWORK_CONSTANTS = {
    "mig1_mig2": dict(
        net_fixed_us=115.75,  # node A 115.2, node B 116.3
        net_bw=1.035e9,  # bytes/s; node A 1.033e9, node B 1.038e9
        tok_net_us=114.1,  # worst batch; node A 114.1, node B 113.9
    ),
}

# both slices of one node copying at once (contention_A_v2.log / contention_B_v2.log):
# per-slice rate together / alone
COPY_CONTENTION = {
    "mig1": dict(d2h_decode=0.82, h2d_decode=1.04, d2h_prefill=0.08, h2d_prefill=0.47),
    "mig2": dict(d2h_decode=0.85, h2d_decode=1.07, d2h_prefill=0.09, h2d_prefill=0.57),
}

# used when a CONFIG does not pick a set
DEFAULT_NAME = "a100_80gb_pcie_whole_gpu"

# peak HBM bytes/s
CARD_BW = {"A100-40GB": 1555e9, "A100-80GB": 2039e9}

# slice GB -> memory slices of 8 on each card. NVIDIA MIG docs: each memory slice is 1/8 of
# the card's memory capacity AND bandwidth (3g.40gb = 4/8 memory, 3/7 SMs), so bandwidth
# follows memory slices, not GPCs.
MEMORY_SLICES = {
    "A100-40GB": {5: 1, 10: 2, 20: 4, 40: 8},
    "A100-80GB": {10: 1, 20: 2, 40: 4, 80: 8},
}


# ---------------------------------------------------------------------------
# Probe sets, one JSON per box, written by probe_constants.py
# ---------------------------------------------------------------------------
def _load_probe_sets():
    import glob
    import json
    import os

    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_sets")
    for path in sorted(glob.glob(os.path.join(here, "*.json"))):
        with open(path) as f:
            s = json.load(f)
        s["slices"] = {int(gb): prims for gb, prims in s["slices"].items()}
        if "clock_threshold_by_period" in s:
            s["clock_threshold_by_period"] = {int(p): v for p, v in s["clock_threshold_by_period"].items()}
        CONSTANTS[os.path.splitext(os.path.basename(path))[0]] = s


_load_probe_sets()
