"""Torch-level check of the multi-model path: dispatch, loader, stage maths.

Skipped unless torch and transformers import. On the box (or any machine
with both) it runs on CPU in seconds, no GPU needed:
    python3 -m unittest discover -s tests -p 'test_model_family_torch.py' -v

For each family the harness supports (llama, mistral, qwen2) it builds a tiny
random model with transformers' own classes, saves it the ways real
checkpoints arrive (single safetensors, sharded safetensors + index, .bin),
and checks the SHIPPED code against the HF reference:

  1. helpers.load_specific_weights fills every parameter of a two-rank split
     with exactly the checkpoint's values, and reports missing=0 ([B08]);
  2. the benchmark's stage maths -- embed, forward_through_layers on each
     rank with its own DynamicCache, final norm, lm_head -- reproduces the
     full model's logits, for the prefill and for a cached decode step.

(2) is what makes a family safe to add to model_family._FAMILIES:
forward_through_layers drives the layer's submodules itself rather than
calling the layer's forward, so a family whose layer has a different shape
would silently compute something else.
"""

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import transformers  # noqa: F401
    from transformers import AutoModelForCausalLM, DynamicCache
except Exception as e:  # pragma: no cover - depends on the machine
    torch = None
    SKIP = f"needs torch + transformers ({type(e).__name__}: {e})"
else:
    SKIP = None


def import_benchmark():
    """Import the benchmark module for its functions, standalone (no job)."""
    os.environ.pop("MIG_EXP_CONFIG", None)
    # Only main() / get_wiki_sample use these; stub them if absent here.
    for name in ("datasets", "pandas"):
        try:
            importlib.import_module(name)
        except Exception:
            mod = types.ModuleType(name)
            if name == "datasets":
                def _no_dataset(*a, **k):
                    raise RuntimeError("datasets stubbed in test")
                mod.load_dataset = _no_dataset
            sys.modules[name] = mod
    import helpers
    import benchmark_pipeline_microbatching as bench
    import model_family
    return bench, helpers, model_family


TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
    num_attention_heads=4, max_position_embeddings=256, rms_norm_eps=1e-5,
    tie_word_embeddings=False,
)


def tiny_config(variant):
    from transformers import LlamaConfig, MistralConfig, Qwen2Config
    if variant == "llama":
        return LlamaConfig(num_key_value_heads=4, **TINY)
    if variant == "mistral":
        return MistralConfig(num_key_value_heads=2, sliding_window=None, **TINY)
    if variant == "mistral_head_dim":
        # Mistral-Small-24B's shape: head_dim set explicitly, 128 != 5120 / 32,
        # so q/o project to heads * head_dim, not to hidden_size.
        return MistralConfig(num_key_value_heads=2, sliding_window=None, head_dim=8, **TINY)
    return Qwen2Config(num_key_value_heads=2, **TINY)  # q/k/v bias on by default


@unittest.skipIf(SKIP, SKIP or "")
class ModelFamilyTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.bench, cls.helpers, cls.mf = import_benchmark()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.models = {}
        for mt in ("llama", "mistral", "mistral_head_dim", "qwen2"):
            cfg = tiny_config(mt)
            cfg._attn_implementation = "sdpa"
            model = AutoModelForCausalLM.from_config(cfg).float().eval()
            dirs = {}
            d = os.path.join(cls.tmp.name, mt, "safetensors")
            model.save_pretrained(d)
            dirs["safetensors"] = d
            d = os.path.join(cls.tmp.name, mt, "sharded")
            model.save_pretrained(d, max_shard_size="40KB")
            dirs["sharded"] = d
            # .bin with an index, the Vicuna v1.5 layout.
            d = os.path.join(cls.tmp.name, mt, "bin")
            os.makedirs(d)
            cfg.save_pretrained(d)
            sd = {k: v.clone() for k, v in model.state_dict().items()}
            half = sorted(sd)[: len(sd) // 2]
            shards = {"pytorch_model-00001-of-00002.bin": {k: sd[k] for k in half},
                      "pytorch_model-00002-of-00002.bin": {k: sd[k] for k in sd if k not in half}}
            wmap = {}
            for fname, part in shards.items():
                torch.save(part, os.path.join(d, fname))
                wmap.update({k: fname for k in part})
            import json
            with open(os.path.join(d, "pytorch_model.bin.index.json"), "w") as f:
                json.dump({"weight_map": wmap}, f)
            dirs["bin"] = d
            cls.models[mt] = (cfg, model, dirs)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build_stage(self, cfg, rank, world, layer_ids):
        fam = self.mf.resolve(cfg.model_type)
        layers = torch.nn.ModuleList(fam.decoder_layer(cfg, layer_idx=i) for i in layer_ids)
        comps = {}
        if rank == 0:
            comps["embed"] = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        if rank == world - 1:
            comps["norm"] = fam.rms_norm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            comps["lm_head"] = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        for m in [*layers, *comps.values()]:
            m.eval()
        return fam, layers, comps

    def test_loader_fills_every_parameter_exactly(self):
        for mt, (cfg, model, dirs) in self.models.items():
            ref = model.state_dict()
            for fmt, d in dirs.items():
                for rank, ids in ((0, [0, 1]), (1, [2, 3])):
                    with self.subTest(family=mt, fmt=fmt, rank=rank):
                        _, layers, comps = self.build_stage(cfg, rank, 2, ids)
                        stats = self.helpers.load_specific_weights(
                            rank, 2, d, layers, ids, comps, tie_word_embeddings=False
                        )
                        self.assertEqual(stats["missing"], [])
                        self.assertEqual(stats["loaded"], stats["expected"])
                        self.assertGreater(stats["expected"], 0)
                        for local, gid in enumerate(ids):
                            for name, p in layers[local].named_parameters():
                                torch.testing.assert_close(p, ref[f"model.layers.{gid}.{name}"], rtol=0, atol=0)
                        if rank == 0:
                            torch.testing.assert_close(comps["embed"].weight, ref["model.embed_tokens.weight"], rtol=0, atol=0)
                        else:
                            torch.testing.assert_close(comps["norm"].weight, ref["model.norm.weight"], rtol=0, atol=0)
                            torch.testing.assert_close(comps["lm_head"].weight, ref["lm_head.weight"], rtol=0, atol=0)

    def test_stage_maths_matches_hf_forward(self):
        B, S = 2, 8
        for mt, (cfg, model, dirs) in self.models.items():
            with self.subTest(family=mt):
                ids = torch.randint(0, cfg.vocab_size, (B, S))
                st = [self.build_stage(cfg, 0, 2, [0, 1]), self.build_stage(cfg, 1, 2, [2, 3])]
                for rank, (fam, layers, comps) in enumerate(st):
                    self.helpers.load_specific_weights(rank, 2, dirs["safetensors"], layers,
                                                       [0, 1] if rank == 0 else [2, 3], comps)
                rotary = st[0][0].rotary_embedding(config=cfg, device="cpu")
                caches = [DynamicCache(), DynamicCache()]  # one per rank, like the pipeline
                mask = torch.triu(torch.full((1, 1, S, S), torch.finfo(torch.float32).min), diagonal=1)

                def run(x_ids, pos, attn_mask):
                    h = st[0][2]["embed"](x_ids)
                    pe = rotary(h, pos)
                    for rank in (0, 1):
                        h = self.bench.forward_through_layers(st[rank][1], h, pe, attn_mask, caches[rank])
                    comps = st[1][2]
                    return comps["lm_head"](comps["norm"](h[:, -1:, :]))[:, -1, :]

                with torch.no_grad():
                    ours = run(ids, torch.arange(S).unsqueeze(0), mask)
                    ref = model(ids).logits[:, -1, :]
                    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)

                    nxt = ours.argmax(-1, keepdim=True)
                    ours_dec = run(nxt, torch.tensor([[S]]).expand(B, -1), None)
                    ref_dec = model(torch.cat([ids, nxt], dim=1)).logits[:, -1, :]
                    torch.testing.assert_close(ours_dec, ref_dec, rtol=1e-4, atol=1e-4)

    def test_family_resolves_to_hf_classes(self):
        for mt, (cfg, model, _) in self.models.items():
            fam = self.mf.resolve(cfg.model_type)
            self.assertIs(type(model.model.layers[0]), fam.decoder_layer)
            self.assertIs(type(model.model.norm), fam.rms_norm)


if __name__ == "__main__":
    unittest.main()
