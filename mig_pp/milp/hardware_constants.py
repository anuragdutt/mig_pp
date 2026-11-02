"""
Measured hardware constants for the layer-placement ILPs (graph_placement.py, gpipe_placement.py).

To add a new configuration: run measure_constants.py on that machine, copy its numbers into a new
entry of CONSTANTS with the same keys, and pick it in the ILP's CONFIG:

    constants=hardware_constants.CONSTANTS["<name>"]

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
