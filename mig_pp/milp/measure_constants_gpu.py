"""
Generic GPU primitives of a WHOLE GPU (MIG disabled), the same sweeps as measure_constants_mig.py
(measure_common.primitives; no model), as fractions of the card, so a MIG slice's numbers can be
derived from them (memory share x card peak for the rates, GPC count for prefill) and checked
against measure_constants_mig.py. Edit CONFIG in main(), then on the box, with MIG off and
nothing else on that GPU:

    CUDA_VISIBLE_DEVICES=<gpu index> python3 mig_pp/milp/measure_constants_gpu.py

Prints a dict and writes CONFIG["json"]. ~5 min.
"""

import torch

import measure_common as mc


def main():
    CONFIG = dict(
        mc.PRIMITIVE_CONFIG,  # the sweeps; override any key here
        card=None,  # None: from the device memory; or "A100-40GB" / "A100-80GB"
        json="constants_gpu.json",
    )
    c = CONFIG

    name = torch.cuda.get_device_name(0)
    if "MIG" in name:
        raise SystemExit(f"{name} is a MIG slice: this script is for a whole GPU (MIG disabled); "
                         "use measure_constants_mig.py for slices")
    card = c["card"] or ("A100-40GB" if torch.cuda.get_device_properties(0).total_memory < 45e9 else "A100-80GB")
    peak = mc.CARD_BW[card]
    print(f"{name} as {card} (peak {peak / 1e9:.0f} GB/s, 7 GPCs)")

    g = mc.primitives(c, peak, gpcs=7)
    g["validation"] = mc.validation_layers(c)

    print(f"\n# measured on a whole {name} ({card}), steady state; a MIG slice of m memory slices (of 8) and")
    print("# g GPCs: *_eff x card peak x m/8 for the rates, gemm_prefill_flops_per_gpc x g, fixed times as they are")
    print(f"GPU_PRIMITIVES[\"{card}\"] = dict(")
    for k in [k[:-5] for k in g if k.endswith("_rate") and k not in ("gemm_prefill_rate", "d2h_rate", "h2d_rate")]:
        print(f"    {k}_fixed_us={g[k + '_fixed_us']:.1f}, {k}_eff={g[k + '_rate'] / peak:.3f},")
    print(f"    gemm_prefill_fixed_us={g['gemm_prefill_fixed_us']:.1f}, gemm_prefill_flops_per_gpc={g['gemm_prefill_rate'] / 7:.3g},")
    print(f"    d2h_fixed_us={g['d2h_fixed_us']:.1f}, d2h_rate={g['d2h_rate']:.3g}, "
          f"h2d_fixed_us={g['h2d_fixed_us']:.1f}, h2d_rate={g['h2d_rate']:.3g},")
    print(f")  # SM clock {g.get('sm_clock_mhz')}")
    mc.save_json(c["json"], dict(device=name, card=card, **g))


if __name__ == "__main__":
    main()
