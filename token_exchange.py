"""
CPU-staged next_tokens exchange for the gloo process group.

Why this exists
---------------
next_tokens lives on the GPU, but the benchmark's process group is gloo,
which is CPU-only: it hands tensor.data_ptr() straight to writev(2), and a
device pointer there is EFAULT — "writev ...: Bad address" — which kills the
sending rank and takes the other ranks down with it as "Connection closed by
peer".

Activations dodge this because dist.isend is intercepted by the SHM engine
in mig_transport_pipeline and staged device-to-host. patched_send and
patched_recv deliberately bypass that machinery for this small control
tensor, so the staging has to happen at the call site instead.

Measured on the A100 MIG rig: ~0.16ms on the first copy, ~0.03ms
steady-state. The host buffer is pinned and allocated once per run, so the
decode loop never pays an allocation and the copies go by DMA.

This module is the single implementation of that staging: probe_gloo_send.py
exercises it, and benchmark_pipeline_microbatching.py uses the same code in
the real sweep. Do not inline a second copy of it.
"""

import time
from typing import Dict

import torch
import torch.distributed as dist


class TokenExchange:
    """
    Staged send/recv for a small CUDA tensor over a CPU-only backend.

    Holds one pinned host buffer matching the device tensor, and accumulates
    the time spent in the device-to-host and host-to-device copies so the
    cost stays visible in the benchmark's own numbers rather than being
    folded silently into the step.
    """

    def __init__(self, device_tensor: torch.Tensor):
        if device_tensor.device.type != "cuda":
            raise ValueError(
                f"TokenExchange expects a CUDA tensor, got device="
                f"{device_tensor.device}. On CPU the staging is pointless — "
                f"call dist.send/recv directly."
            )
        self.device_tensor = device_tensor
        self.host_buffer = torch.empty(
            device_tensor.shape,
            dtype=device_tensor.dtype,
            device="cpu",
        ).pin_memory()
        self.stats: Dict[str, float] = {"d2h_s": 0.0, "h2d_s": 0.0, "n": 0}

    def send(self, dst: int, tag: int) -> None:
        """Copy the device tensor to the pinned host buffer, then send that."""
        _t0 = time.perf_counter()
        self.host_buffer.copy_(self.device_tensor)
        torch.cuda.synchronize()
        self.stats["d2h_s"] += time.perf_counter() - _t0
        self.stats["n"] += 1
        dist.send(self.host_buffer, dst=dst, tag=tag)

    def recv(self, src: int, tag: int) -> None:
        """Receive into the pinned host buffer, then copy up to the device."""
        dist.recv(self.host_buffer, src=src, tag=tag)
        _t0 = time.perf_counter()
        self.device_tensor.copy_(self.host_buffer)
        torch.cuda.synchronize()
        self.stats["h2d_s"] += time.perf_counter() - _t0
        self.stats["n"] += 1

    # -- reporting ---------------------------------------------------------

    def total_ms(self) -> float:
        return (self.stats["d2h_s"] + self.stats["h2d_s"]) * 1000

    def mean_ms(self) -> float:
        n = self.stats["n"]
        return self.total_ms() / n if n else 0.0

    def summary(self) -> str:
        """One log line. This is the only place the cost is reported — it
        stays out of the results CSV deliberately, since it is harness
        overhead rather than a property of the configuration being swept."""
        return (
            f"d2h={self.stats['d2h_s'] * 1000:.2f}ms "
            f"h2d={self.stats['h2d_s'] * 1000:.2f}ms "
            f"total={self.total_ms():.2f}ms over {self.stats['n']} exchanges "
            f"(mean={self.mean_ms():.4f}ms)"
        )
