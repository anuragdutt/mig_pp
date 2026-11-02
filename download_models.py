"""
Download the weights for every model in parallel_config.py.

    python3 download_models.py --check     # access, sizes and free disk; downloads nothing
    python3 download_models.py             # download all (resumes; skips complete models)
    python3 download_models.py --only nemo_12b vicuna_7b

Each model lands in its MODELS[...]["path"] from parallel_config.py, so the
config and the files cannot disagree. One weight format per model: the
HF-format safetensors if the repo has them, else .bin (the Vicuna v1.5 repos
have only .bin). Llama-2 repos carry both formats and Mistral repos a second
copy (consolidated.safetensors); fetching one set saves 13-26 GB per model.

Gated repos (meta-llama/*, mistralai/*): accept the license on the model's
huggingface.co page with your account, then `hf auth login` (or
`huggingface-cli login`, or export HF_TOKEN=...) before running this.
"""

import argparse
import fnmatch
import os
import runpy
import shutil
import sys

import model_family

HERE = os.path.dirname(os.path.abspath(__file__))

REPOS = {
    "vicuna_7b": "lmsys/vicuna-7b-v1.5",
    "llama_7b": "meta-llama/Llama-2-7b-hf",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen_7b": "Qwen/Qwen2.5-7B",
    "vicuna_13b": "lmsys/vicuna-13b-v1.5",
    "llama_13b": "meta-llama/Llama-2-13b-hf",
    "qwen_14b": "Qwen/Qwen2.5-14B",
    "nemo_12b": "mistralai/Mistral-Nemo-Base-2407",
    "mistral_24b": "mistralai/Mistral-Small-24B-Base-2501",
}

# config.json, generation_config.json and the tokenizer files, whatever the family.
AUX = ("*.json", "*.model", "*.txt", "*.tiktoken")


def pick_files(files):
    """Top-level config/tokenizer files plus exactly one HF-format weight set."""
    files = [f for f in files if "/" not in f]  # skips e.g. original/ in Meta repos
    st = [f for f in files if fnmatch.fnmatch(f, "model*.safetensors")]  # not consolidated.*
    weights = st or [f for f in files if fnmatch.fnmatch(f, "pytorch_model*.bin")]
    other_index = "pytorch_model.bin.index.json" if st else "model.safetensors.index.json"
    aux = [f for f in files if any(fnmatch.fnmatch(f, p) for p in AUX) and f != other_index]
    return sorted(set(aux) | set(weights)), bool(weights)


def complete(path):
    try:
        model_family.find_weight_files(path)
        return os.path.isfile(os.path.join(path, "config.json"))
    except (OSError, model_family.WeightFilesError):
        return False


def free_bytes(path):
    while not os.path.exists(path):
        path = os.path.dirname(path) or "/"
    return shutil.disk_usage(path).free


def explain(err, repo):
    text = f"{type(err).__name__}: {err}".splitlines()[0][:200]
    if any(k in text.lower() for k in ("gated", "401", "403", "authoriz", "restricted")):
        return (f"no access: accept the license at https://huggingface.co/{repo} with your account, "
                f"then `hf auth login` (or export HF_TOKEN=...) -- {text}")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "parallel_config.py"))
    ap.add_argument("--only", nargs="+", metavar="KEY", help="just these MODELS keys")
    ap.add_argument("--check", action="store_true", help="report access/sizes/disk, download nothing")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi, snapshot_download  # installed with transformers

    models = runpy.run_path(args.config)["MODELS"]
    keys = args.only or list(models)
    unknown = [k for k in keys if k not in models or k not in REPOS]
    if unknown:
        sys.exit(f"not in parallel_config.py MODELS and/or REPOS here: {', '.join(unknown)}")

    api, plan, failed = HfApi(), [], []
    for key in keys:
        path, repo = os.path.abspath(os.path.expanduser(models[key]["path"])), REPOS[key]
        if complete(path):
            print(f"present   {key:11s} {path}")
            continue
        try:
            info = api.model_info(repo, files_metadata=True)
            if getattr(info, "gated", False):
                # Gated repos list their files to anyone; only the download is
                # refused. Ask now, not 100 GB into the run.
                api.auth_check(repo)
            sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
            files, has_weights = pick_files(sizes)
            if not has_weights:
                raise RuntimeError("no HF-format weights in the repo")
        except Exception as e:  # network, auth, missing repo: report and go on
            failed.append((key, explain(e, repo)))
            print(f"FAILED    {key:11s} {failed[-1][1]}")
            continue
        gb = sum(sizes[f] for f in files) / 1e9
        plan.append((key, repo, path, files, gb))
        print(f"to fetch  {key:11s} {gb:6.1f} GB  {repo} -> {path}")

    need = sum(p[4] for p in plan)
    if plan:
        free = free_bytes(plan[0][2]) / 1e9
        print(f"\ntotal to download {need:.1f} GB; free at {os.path.dirname(plan[0][2])}: {free:.1f} GB")
        if need > free * 0.98:
            sys.exit("not enough disk space -- change MODEL_ROOT in parallel_config.py")
    if args.check:
        return 1 if failed else 0

    for key, repo, path, files, gb in plan:
        print(f"\ndownloading {key} ({gb:.1f} GB) -> {path}", flush=True)
        try:
            snapshot_download(repo_id=repo, local_dir=path, allow_patterns=files)
            if not complete(path):
                raise RuntimeError("download finished but the weights are incomplete")
            print(f"done      {key}")
        except Exception as e:
            failed.append((key, explain(e, repo)))
            print(f"FAILED    {key}: {failed[-1][1]}")

    print("\n" + ("all models present" if not failed else
                  "FAILED: " + "; ".join(f"{k} ({why})" for k, why in failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
