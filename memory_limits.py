"""
Theoretical GPU memory per pipeline rank, and the layer capacity it implies.

    python3 memory_limits.py                  # every model in parallel_config.py
    python3 memory_limits.py --model qwen_7b
    python3 memory_limits.py --validate mig_benchmark_results_9_28.csv --model vicuna_7b

Per model: the most decoder layers each MIG slice can hold at every batch in
the sweep, the limits in layer_limits.py, and the share of configurations
predicted to OOM. Torch-free. Mirrors what run_pipeline() puts on a slice:

  CTX + l*W + KVF*l*B*KVSEQ + ACT(mb)   (+ embed on rank 0, + lm_head on the last)

W = fp16 layer weights; KVSEQ = k+v per sequence for seq_len + max_new_tokens
tokens; ACT = prefill hidden + MLP transients at one microbatch. Separately,
embed and lm_head are built fp32 on the GPU and then .half()'d, so each needs
6 bytes/param for a moment -- on the last rank that lands on top of its layers,
which is what caps Qwen's (152k-vocab) last 5GB slice.

CTX = 250 MiB and KVF = 1.10 are fit to the ok/OOM status of the 938
configurations in mig_benchmark_results_9_28.csv (Vicuna-7B, 20/10/5/5, A100-40GB):
933 predicted right; the 5 misses are rank 1 at exactly 9 layers, B64, mb 4.
Only B64 OOMed there, so other shapes (13B, GQA models) are extrapolations --
planning estimates for the smoke run to check, not measurements.
"""

import argparse
import ast
import csv
import json
import os
import runpy
import sys

import experiment_config
import layer_limits as ll
import model_family

MIB = 2**20
CTX_MIB = 250.0
KV_FACTOR = 1.10

# Usable MiB per MIG profile by nominal GB (A100-40GB; the 9/28 per-slice
# maxima 4863 / 9983 / 20085 MiB sit just under these).
SLICE_MIB = {5: 4864, 10: 9984, 20: 20096, 40: 40192}
# Layouts whose nominal sizes are other profiles: A100-80GB's 2g.20gb and 1g.10gb
# are smaller than the 40GB card's 3g.20gb and 2g.10gb (April 80GB maxima
# 40190 / 19965 / 9728 MiB).
LAYOUT_MIB = {"40_20_10_10": [40192, 19968, 9728, 9728]}

# Public config.json shapes, for when the weights are not on this machine.
REFERENCE_SHAPES = {
    "vicuna_7b": dict(L=32, H=4096, I=11008, heads=32, kv=32, hd=128, V=32000, bias=False),
    "llama_7b": dict(L=32, H=4096, I=11008, heads=32, kv=32, hd=128, V=32000, bias=False),
    "mistral_7b": dict(L=32, H=4096, I=14336, heads=32, kv=8, hd=128, V=32768, bias=False),
    "qwen_7b": dict(L=28, H=3584, I=18944, heads=28, kv=4, hd=128, V=152064, bias=True),
    "vicuna_13b": dict(L=40, H=5120, I=13824, heads=40, kv=40, hd=128, V=32000, bias=False),
    "llama_13b": dict(L=40, H=5120, I=13824, heads=40, kv=40, hd=128, V=32000, bias=False),
    "qwen_14b": dict(L=48, H=5120, I=13824, heads=40, kv=8, hd=128, V=152064, bias=True),
    "nemo_12b": dict(L=40, H=5120, I=14336, heads=32, kv=8, hd=128, V=131072, bias=False),
    "mistral_24b": dict(L=40, H=5120, I=32768, heads=32, kv=8, hd=128, V=131072, bias=False),
}


def shape_from_config(raw: dict) -> dict:
    heads, hidden = raw["num_attention_heads"], raw["hidden_size"]
    return dict(
        L=raw["num_hidden_layers"], H=hidden, I=raw["intermediate_size"], heads=heads,
        kv=raw.get("num_key_value_heads") or heads,
        hd=raw.get("head_dim") or hidden // heads,  # Mistral-Small-24B sets it explicitly
        V=raw["vocab_size"],
        bias=bool(raw.get("attention_bias", raw.get("model_type") == "qwen2")),  # Qwen2: q/k/v bias
    )


def slice_mib_for(gb) -> int:
    return SLICE_MIB.get(int(gb), int(gb * 1024 * 0.95))


def slice_mibs(slice_gb) -> list:
    """Usable MiB per rank of a layout."""
    return list(LAYOUT_MIB.get(ll.layout_key(slice_gb)) or [slice_mib_for(g) for g in slice_gb])


class MemoryModel:
    def __init__(self, s: dict, seq_len=64, max_new_tokens=512, ctx_mib=CTX_MIB, kv_factor=KV_FACTOR):
        q, kvd = s["heads"] * s["hd"], s["kv"] * s["hd"]
        params = s["H"] * q * 2 + 2 * s["H"] * kvd + 3 * s["H"] * s["I"] + 2 * s["H"] + ((q + 2 * kvd) if s["bias"] else 0)
        self.s, self.seq_len, self.ctx, self.kvf = s, seq_len, ctx_mib, kv_factor
        self.w = params * 2 / MIB
        self.kv_seq = 2 * (seq_len + max_new_tokens) * kvd * 2 / MIB
        self.emb = s["V"] * s["H"] * 2 / MIB
        self.build = s["V"] * s["H"] * 6 / MIB

    def need(self, rank, world, layers, batch, mb) -> float:
        run = self.ctx + layers * (self.w + self.kvf * self.kv_seq * batch)
        run += mb * self.seq_len * (2 * self.s["H"] + 2 * self.s["I"]) * 2 / MIB
        run += self.emb * ((rank == 0) + (rank == world - 1))
        # Build order in run_pipeline: rank 0's embed first (its fp32 copy is
        # freed by the first layer's empty_cache), then layers, then the last
        # rank's norm + lm_head on top of them.
        build = self.ctx + layers * self.w + (self.emb if rank == 0 else 0)
        if rank == 0:
            build = max(build, self.ctx + self.build)
        if rank == world - 1:
            build += self.build
        return max(run, build)

    def max_layers(self, rank, world, batch, mb, slice_mib) -> int:
        n = -1
        while n < 1000 and self.need(rank, world, n + 1, batch, mb) <= slice_mib:
            n += 1
        return n

    def oom_ranks(self, split, batch, mb, slice_gb):
        caps = slice_mibs(slice_gb)
        return [r for r in range(len(split)) if self.need(r, len(split), split[r], batch, mb) > caps[r]]


def shape_for(key, entry):
    try:
        d = model_family.resolve_model_path((entry or {})["path"])
        with open(os.path.join(d, "config.json")) as f:
            return shape_from_config(json.load(f)), "config.json"
    except (KeyError, OSError, ValueError, model_family.ModelPathError):
        return REFERENCE_SHAPES[key], "reference shape; weights not on this machine"


def report(key, entry, sweep, slice_gb):
    shape, source = shape_for(key, entry)
    eff = {**sweep, **{k: v for k, v in (entry or {}).items() if k in sweep}}
    mm = MemoryModel(shape, eff.get("seq_len", 64), eff.get("max_new_tokens", 512))
    pairs = [tuple(p) for p in eff.get("batch_mb_pairs", [(8, 4)])]
    batches = sorted({b for b, _ in pairs})
    world = len(slice_gb)
    print(f"\n{key}: {shape['L']} layers, {mm.w:.0f} MiB/layer, KV {mm.kv_seq:.2f} MiB/seq/layer, "
          f"lm_head build {mm.build:.0f} MiB ({source})")
    for b in batches:
        mb = max(m for bb, m in pairs if bb == b)
        caps = [mm.max_layers(r, world, b, mb, cap) for r, cap in enumerate(slice_mibs(slice_gb))]
        print(f"  B{b:<3d} max layers {caps}" + ("" if sum(max(c, 0) for c in caps) >= shape["L"] else "  <- model cannot fit"))
    try:
        limits, min_last = ll.limits_for(key, slice_gb)
    except KeyError as e:
        print(f"  {e.args[0]}")
        return
    splits = experiment_config.generate_layer_splits(shape["L"], limits, slice_gb, True, min_last)
    ooms = {b: [bool(mm.oom_ranks(s, b, mb, slice_gb)) for s in splits for bb, mb in pairs if bb == b] for b in batches}
    print(f"  layer_limits.py {limits} (head-only last rank: {min_last == 0}): {len(splits)} splits x "
          f"{len(pairs)} pairs = {len(splits) * len(pairs)} configs; predicted OOM: "
          + "  ".join(f"B{b} {100 * sum(v) / len(v):.0f}%" for b, v in ooms.items() if v))


def validate(csv_path, key, slice_gb):
    """(right, total) predicted ok/OOM outcomes for a results CSV."""
    mm = MemoryModel(REFERENCE_SHAPES[key])
    right = n = 0
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            st = r["status"]
            if st != "ok" and not st.startswith("OOM_rank"):
                continue
            ooms = mm.oom_ranks(ast.literal_eval(r["split"]), int(r["batch_size"]), int(r["microbatch_size"]), slice_gb)
            right += (not ooms) if st == "ok" else (int(st[len("OOM_rank"):]) in ooms)
            n += 1
    return right, n


def main(argv=None):
    ap = argparse.ArgumentParser(description="theoretical per-slice layer capacity")
    ap.add_argument("-c", "--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "parallel_config.py"))
    ap.add_argument("--model", action="append", help="model key (repeatable); default: all in the config")
    ap.add_argument("--validate", metavar="CSV", help="check the calibration against a results CSV")
    args = ap.parse_args(argv)
    if args.validate:
        key = (args.model or ["vicuna_7b"])[0]
        right, n = validate(args.validate, key, [20, 10, 5, 5])
        print(f"{key}: {right}/{n} ok/OOM outcomes predicted (CTX={CTX_MIB:.0f} MiB, KV factor={KV_FACTOR})")
        return 0
    ns = runpy.run_path(args.config)
    models, sweep = ns.get("MODELS", {}), ns.get("SWEEP", {})
    layouts = sorted({tuple(g["slice_gb"]) for g in ns.get("GPUS", []) if g.get("models")}) or [(20, 10, 5, 5)]
    for layout in layouts:
        print(f"=== layout {ll.layout_key(layout)} (slices {slice_mibs(layout)} MiB)")
        for key in args.model or list(models):
            report(key, models.get(key), sweep, list(layout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
