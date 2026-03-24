#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# repo layout:
#   ~/mig_pp/profiling/collect_poc_inputs.py
#   ~/mig_pp/profiles/
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "profiles"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, row: dict):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def get_decoder_layers(model):
    # Common for LLaMA/Vicuna HF: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if isinstance(layers, (list, torch.nn.ModuleList)):
            return list(layers)

    # Fallback: find a ModuleList named 'layers'
    for name, module in model.named_modules():
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            return list(module)

    raise RuntimeError("Could not locate decoder layers (expected model.model.layers).")


def tensor_param_bytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters(recurse=True):
        total += p.numel() * p.element_size()
    return int(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lmsys/vicuna-7b-v1.5", help="HF model id or local path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--input_len", type=int, default=64)
    ap.add_argument("--output_len", type=int, default=512)  # max_new_tokens
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--run_csv", default="poc_run.csv")
    ap.add_argument("--mem_csv", default="poc_layer_memory.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).expanduser()
    ensure_dir(out_dir)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # Load tokenizer + model (downloads via HF cache if needed)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # -------- Static model memory info (no inference) --------
    cfg = model.config
    layers = get_decoder_layers(model)
    n_layers = len(layers)

    hidden_size = int(getattr(cfg, "hidden_size"))
    n_heads = int(getattr(cfg, "num_attention_heads"))
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = hidden_size // n_heads
    peak_seq = args.input_len + args.output_len

    # Embeddings + LM head bytes (handle tying if present)
    inp_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    emb_bytes = tensor_param_bytes(inp_emb) if inp_emb is not None else 0
    out_bytes = tensor_param_bytes(out_emb) if out_emb is not None else 0
    tied = False
    if inp_emb is not None and out_emb is not None:
        try:
            tied = inp_emb.weight.data_ptr() == out_emb.weight.data_ptr()
        except Exception:
            tied = False
    shared_embed_lm_bytes = emb_bytes if tied else (emb_bytes + out_bytes)

    # KV per layer at peak seq (simple default): 2 * B * kv_heads * peak_seq * head_dim * dtype_bytes
    dtype_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    kv_bytes_peak_per_layer = 2 * args.batch * n_kv_heads * peak_seq * head_dim * dtype_bytes

    # Write per-layer memory CSV (overwrite each run for cleanliness)
    mem_path = out_dir / args.mem_csv
    with open(mem_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["component", "layer_id", "param_bytes", "kv_bytes_peak", "notes"],
        )
        w.writeheader()
        w.writerow({
            "component": "shared_embed_lm_head" if tied else "embed_plus_lm_head",
            "layer_id": -1,
            "param_bytes": int(shared_embed_lm_bytes),
            "kv_bytes_peak": 0,
            "notes": "tied_weights" if tied else "",
        })
        for i, layer in enumerate(layers):
            w.writerow({
                "component": "decoder_block",
                "layer_id": i,
                "param_bytes": int(tensor_param_bytes(layer)),
                "kv_bytes_peak": int(kv_bytes_peak_per_layer),
                "notes": "",
            })

    # -------- Exactly ONE inference call for timing --------
    vocab_size = int(cfg.vocab_size)
    input_ids = torch.randint(
        low=0, high=vocab_size,
        size=(args.batch, args.input_len),
        device=device, dtype=torch.long
    )
    attention_mask = torch.ones((args.batch, args.input_len), device=device, dtype=torch.long)

    torch.cuda.reset_peak_memory_stats(device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        torch.cuda.synchronize(device)
        start_wall = time.time()
        start_evt.record()

        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.output_len,
            do_sample=False,
            use_cache=True,   # default KV cache
        )

        end_evt.record()
        torch.cuda.synchronize(device)
        end_wall = time.time()

    total_ms = float(start_evt.elapsed_time(end_evt))
    wall_ms = (end_wall - start_wall) * 1000.0

    total_tokens = args.batch * (args.input_len + args.output_len)
    ms_per_total_token = total_ms / max(1, total_tokens)

    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    props = torch.cuda.get_device_properties(device)

    run_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "device_name": props.name,
        "device_total_mem_gb": f"{props.total_memory / (1024**3):.3f}",
        "batch": args.batch,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "peak_seq": peak_seq,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "total_time_ms_cuda_event": f"{total_ms:.3f}",
        "total_time_ms_wall": f"{wall_ms:.3f}",
        "ms_per_total_token": f"{ms_per_total_token:.6f}",
        "peak_mem_allocated_mb": f"{peak_alloc_mb:.2f}",
        "peak_mem_reserved_mb": f"{peak_reserved_mb:.2f}",
    }

    run_path = out_dir / args.run_csv
    write_csv(run_path, run_row)

    print(f"Wrote:\n  {run_path}\n  {mem_path}")
    print(f"Total (CUDA event): {total_ms:.3f} ms | ms/token (avg over all tokens): {ms_per_total_token:.6f}")
    print(f"Peak allocated: {peak_alloc_mb:.2f} MB | Peak reserved: {peak_reserved_mb:.2f} MB")


if __name__ == "__main__":
    main()
