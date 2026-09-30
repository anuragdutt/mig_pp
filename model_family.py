"""
Model-architecture dispatch and checkpoint discovery. Torch-free at import.

The benchmark was written against LlamaDecoderLayer / LlamaRMSNorm /
LlamaRotaryEmbedding, and each other model lived on its own branch with those
three names swapped (qwen-7b-4mig -> Qwen2*, mistral-7b-40g-4mig -> Mistral*).
resolve() does that swap from config.json's model_type instead, so one
harness runs every family.

forward_through_layers() in the benchmark drives the layer's submodules
directly (input_layernorm -> self_attn -> post_attention_layernorm -> mlp),
so a family belongs in _FAMILIES only if its decoder layer has exactly that
shape. tests/test_model_family_torch.py checks that per family: the
benchmark's stage maths must reproduce the HF model's logits.

transformers is imported lazily inside resolve(); everything else here is
plain os/json so parallel_plan.py and tests/ run without torch.
"""

import importlib
import json
import os
import re
from collections import namedtuple
from typing import Dict, List, Optional

Family = namedtuple("Family", "model_type decoder_layer rms_norm rotary_embedding")

# model_type -> (module, decoder layer, RMSNorm, rotary embedding)
_FAMILIES: Dict[str, tuple] = {
    # Vicuna and Llama-2 both report model_type "llama".
    "llama": (
        "transformers.models.llama.modeling_llama",
        "LlamaDecoderLayer",
        "LlamaRMSNorm",
        "LlamaRotaryEmbedding",
    ),
    "mistral": (
        "transformers.models.mistral.modeling_mistral",
        "MistralDecoderLayer",
        "MistralRMSNorm",
        "MistralRotaryEmbedding",
    ),
    # Qwen2 and Qwen2.5. q/k/v carry a bias; the loader's generic attribute
    # walk fills it because Qwen2DecoderLayer constructs those Linears with
    # bias=True.
    "qwen2": (
        "transformers.models.qwen2.modeling_qwen2",
        "Qwen2DecoderLayer",
        "Qwen2RMSNorm",
        "Qwen2RotaryEmbedding",
    ),
}


class UnsupportedModelError(ValueError):
    pass


class ModelPathError(FileNotFoundError):
    pass


class WeightFilesError(FileNotFoundError):
    pass


def supported_model_types() -> List[str]:
    return sorted(_FAMILIES)


def family_spec(model_type: str) -> tuple:
    """The (module, class names...) entry, without importing anything."""
    if model_type not in _FAMILIES:
        raise UnsupportedModelError(
            f"model_type '{model_type}' is not supported by this harness "
            f"(supported: {', '.join(supported_model_types())}). Adding one "
            f"means an entry in model_family._FAMILIES — and a passing "
            f"tests/test_model_family_torch.py, because "
            f"forward_through_layers assumes the Llama layer shape."
        )
    return _FAMILIES[model_type]


def resolve(model_type: str) -> Family:
    """Import and return the decoder-layer / RMSNorm / rotary classes."""
    module_name, layer_name, norm_name, rope_name = family_spec(model_type)
    module = importlib.import_module(module_name)
    return Family(
        model_type,
        getattr(module, layer_name),
        getattr(module, norm_name),
        getattr(module, rope_name),
    )


# ---------------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------------

_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def hf_hub_cache_dir() -> str:
    """Same precedence huggingface_hub uses for its cache location."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return os.path.expanduser(os.environ[var])
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return os.path.join(os.path.expanduser(hf_home), "hub")


def resolve_model_path(spec: str, hub_cache: Optional[str] = None) -> str:
    """
    A local model directory for `spec`.

    Accepts a directory (as produced by `huggingface-cli download --local-dir`
    or `git clone`), or a repo id already in the HF cache ("lmsys/vicuna-7b-v1.5"
    -> <cache>/models--lmsys--vicuna-7b-v1.5/snapshots/<refs/main>). Never
    downloads: parallel runs must not race each other into the hub.
    """
    path = os.path.abspath(os.path.expanduser(spec))
    if os.path.isdir(path):
        return path

    if _REPO_ID_RE.match(spec):
        cache = hub_cache or hf_hub_cache_dir()
        repo_dir = os.path.join(cache, "models--" + spec.replace("/", "--"))
        snapshots = os.path.join(repo_dir, "snapshots")
        ref = os.path.join(repo_dir, "refs", "main")
        if os.path.isfile(ref):
            with open(ref) as f:
                cand = os.path.join(snapshots, f.read().strip())
            if os.path.isdir(cand):
                return cand
        if os.path.isdir(snapshots):
            snaps = sorted(os.listdir(snapshots))
            if len(snaps) == 1:
                return os.path.join(snapshots, snaps[0])
            if len(snaps) > 1:
                raise ModelPathError(
                    f"'{spec}' has {len(snaps)} snapshots in {snapshots} and no "
                    f"refs/main; give the snapshot directory explicitly"
                )
        raise ModelPathError(
            f"'{spec}' is not a directory and is not in the HF cache at {cache}"
        )

    raise ModelPathError(f"model path '{spec}' does not exist")


def read_model_config(model_dir: str) -> dict:
    """The fields the harness needs from config.json, plus the raw dict."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        raise ModelPathError(f"no config.json in {model_dir}")
    with open(path) as f:
        raw = json.load(f)
    missing = [
        k
        for k in ("model_type", "num_hidden_layers", "hidden_size", "num_attention_heads")
        if k not in raw
    ]
    if missing:
        raise ModelPathError(f"{path} lacks {', '.join(missing)}")
    return {
        "model_type": raw["model_type"],
        "num_layers": raw["num_hidden_layers"],
        "hidden_size": raw["hidden_size"],
        "num_heads": raw["num_attention_heads"],
        "vocab_size": raw.get("vocab_size"),
        # Llama, Mistral and Qwen2 configs all default this to False when
        # the key is absent.
        "tie_word_embeddings": bool(raw.get("tie_word_embeddings", False)),
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# WEIGHT FILES
# ---------------------------------------------------------------------------

# skipped: human-readable notes on candidates passed over because they were
# incomplete (e.g. a partial safetensors download next to a full .bin set).
WeightFiles = namedtuple(
    "WeightFiles", "folder fmt files index_path skipped", defaults=((),)
)

# Search order. safetensors first: it can be read tensor by tensor, so a rank
# pulls only its own layers off disk instead of whole multi-GB shards — which
# matters when 32 ranks load at once. Vicuna-7B v1.5 ships .bin only and
# lands on the third entry, as before.
_WEIGHT_CANDIDATES = (
    ("model.safetensors.index.json", "safetensors", True),
    ("model.safetensors", "safetensors", False),
    ("pytorch_model.bin.index.json", "bin", True),
    ("pytorch_model.bin", "bin", False),
)


def find_weight_files(model_dir: str) -> WeightFiles:
    """
    Locate the HF-format checkpoint in model_dir.

    A sharded candidate whose index names a shard that is not on disk (an
    interrupted download) is passed over for the next format — Vicuna-style
    dirs can hold a complete .bin set beside a partial safetensors one — and
    noted in `skipped`. Raises WeightFilesError only when no candidate is
    complete: the old loader would have silently skipped the load and run
    the sweep on random weights.
    Mistral's native consolidated.safetensors is deliberately not a
    candidate: its key names are not the HF ones.
    """
    skipped: List[str] = []
    for name, fmt, sharded in _WEIGHT_CANDIDATES:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            continue
        if not sharded:
            return WeightFiles(model_dir, fmt, [path], None, tuple(skipped))
        with open(path) as f:
            weight_map = json.load(f).get("weight_map") or {}
        if not weight_map:
            skipped.append(f"{name}: no weight_map")
            continue
        shards = sorted(set(weight_map.values()))
        files = [os.path.join(model_dir, s) for s in shards]
        absent = [s for s, p in zip(shards, files) if not os.path.isfile(p)]
        if absent:
            skipped.append(f"{name}: shards missing: {', '.join(absent)}")
            continue
        return WeightFiles(model_dir, fmt, files, path, tuple(skipped))

    detail = f" ({'; '.join(skipped)})" if skipped else ""
    raise WeightFilesError(
        f"no complete weights in {model_dir}{detail} (looked for "
        f"{', '.join(c[0] for c in _WEIGHT_CANDIDATES)})"
    )
