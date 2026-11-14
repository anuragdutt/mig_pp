"""
Generic GPU primitives on every MIG slice type of one GPU (measure_common.primitives: GEMM,
decode attention, KV append, elementwise, device copy, D2H / H2D, each as fixed + units / rate
over a shape sweep; no model), plus one decoder layer per validation model as a check only.
Edit CONFIG in main(), then on the box, MIG enabled, nothing else running on that GPU:

    python3 mig_pp/milp/measure_constants_mig.py

It finds the GPU's slices with nvidia-smi -L and measures one slice type at a time, each in
its own process (CUDA_VISIBLE_DEVICES=MIG-<uuid>); slices of the same type are measured once.
Prints a dict per slice and writes CONFIG["json"]. ~3-5 min per slice type.
"""

import json
import os
import re
import subprocess
import sys


def mig_slices(gpu_index):
    """[(profile, uuid)] of one GPU's MIG slices, from nvidia-smi -L."""
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    slices, gpu = [], None
    for line in out.splitlines():
        m = re.match(r"GPU (\d+):", line)
        if m:
            gpu = int(m[1])
            continue
        m = re.search(r"MIG (\d+g\.\d+gb)\s+Device\s+\d+: \(UUID: (MIG-[^)]+)\)", line)
        if m and gpu == gpu_index:
            slices.append((m[1], m[2]))
    return slices


def measure_one(c):
    """Child process: the visible device is one slice."""
    import torch
    import measure_common as mc

    name = torch.cuda.get_device_name(0)
    m = re.search(r"MIG (\d)g\.(\d+)gb", name)
    if not m:
        raise SystemExit(f"{name}: not a MIG slice")
    gpcs, gb = int(m[1]), int(m[2])
    card_gb = 80 if "80GB" in name else 40
    card = "A100-40GB" if card_gb == 40 else "A100-80GB"
    share = gb / card_gb  # NVIDIA: each memory slice is 1/8 of capacity and bandwidth
    peak = mc.CARD_BW[card] * share
    print(f"{name}: {gpcs} GPCs, {gb} GB = {share:.3f} of {card} (slice peak {peak / 1e9:.0f} GB/s)")
    g = mc.primitives(c, peak, gpcs)
    g.update(device=name, profile=f"{gpcs}g.{gb}gb", gb=gb, card=card, mem_share=share,
             validation=mc.validation_layers(c))
    with open(c["child_out"], "w") as f:
        json.dump(g, f)


def main():
    import measure_common as mc

    CONFIG = dict(
        mc.PRIMITIVE_CONFIG,  # the sweeps; override any key here
        gpu=0,  # the GPU whose slices to measure (nvidia-smi index)
        json="constants_mig.json",
    )
    c = CONFIG

    if os.environ.get("MEASURE_MIG_CHILD"):
        c["child_out"] = os.environ["MEASURE_MIG_CHILD"]
        return measure_one(c)

    slices = mig_slices(c["gpu"])
    if not slices:
        raise SystemExit(f"GPU {c['gpu']} has no MIG slices (nvidia-smi -L); enable MIG and create them first")
    once = {}
    for profile, uuid in slices:
        once.setdefault(profile, uuid)
    print(f"GPU {c['gpu']}: slices {[p for p, _ in slices]}; measuring {list(once)}")

    results = []
    for profile, uuid in once.items():
        out = f"/tmp/measure_mig_{profile}_{os.getpid()}.json"
        print(f"\n===== {profile} ({uuid}) =====", flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, MEASURE_MIG_CHILD=out)
        r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
        if r.returncode != 0 or not os.path.exists(out):
            print(f"  {profile}: FAILED (exit {r.returncode}); continuing with the other slices")
            continue
        results.append(json.load(open(out)))
        os.remove(out)

    keys = ("gemm_decode", "gemm_prefill", "attention_mha", "attention_gqa", "kv_append", "elementwise", "device_copy", "d2h", "h2d")
    print("\n# generic primitives per MIG slice: t = fixed_us + units / rate (units: bytes, or FLOPs for gemm_prefill)")
    print("SLICE_PRIMITIVES = {")
    for g in results:
        body = ", ".join(f"{k}_fixed_us={g[k + '_fixed_us']:.1f}, {k}_rate={g[k + '_rate']:.3g}" for k in keys)
        print(f"    {g['gb']}: dict({body}),  # {g['profile']}")
    print("}")
    for g in results:
        print(f"#   {g['profile']}: gemm_decode {g['gemm_decode_rate'] / g['peak']:.3f}, attention {g['attention_mha_rate'] / g['peak']:.3f} (GQA {g['attention_gqa_rate'] / g['peak']:.3f})"
              f", elementwise {g['elementwise_rate'] / g['peak']:.3f} of the slice's memory share; prefill "
              f"{g['gemm_prefill_rate'] / g['gpcs'] / 1e12:.1f} TFLOP/s per GPC")
    with open(c["json"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {c['json']}")


if __name__ == "__main__":
    main()
