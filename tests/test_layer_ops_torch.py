"""The ILP's kernel lists, written out from the model shape (layer_ops.layer_ops_from_shape /
extras_from_shape), against the model code itself, traced on the meta device (layer_ops.trace_ops /
trace_extras + classify). Needs torch + transformers (the boxes' 5.18.0); skips without them. On the
Mac, with a scratch install (not the venv):
    PYTHONPATH=<scratch>/tf518full .venv/bin/python -m unittest discover -s tests -p 'test_layer_ops_torch.py' -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mig_pp" / "milp"))

import memory_limits as ml  # noqa: E402
import layer_ops  # noqa: E402

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# REFERENCE_SHAPES key -> layer_ops.MODELS config (config.json values)
MODELS = {"vicuna_7b": "lmsys/vicuna-7b-v1.5", "qwen_7b": "Qwen/Qwen2.5-7B"}
# (mb, ctx, prefill seq): decode at a few contexts, prefill at the harness's prompt length
POINTS = [(1, 64, None), (8, 320, None), (16, 575, None), (2, 0, 64), (8, 0, 64)]


def kernels(classified):
    return [op for op in classified if op[0] != "free"]


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class OpListsMatchModelCode(unittest.TestCase):
    def test_shapes_match_configs(self):
        for key, name in MODELS.items():
            shape, cfg = ml.REFERENCE_SHAPES[key], layer_ops.MODELS[name]
            with self.subTest(model=key):
                self.assertEqual((shape["H"], shape["I"], shape["heads"], shape["kv"], shape["V"]),
                                 (cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"],
                                  cfg["num_key_value_heads"], cfg["vocab_size"]))
                self.assertEqual(shape["hd"], cfg["hidden_size"] // cfg["num_attention_heads"])

    def test_decoder_layer(self):
        for key, name in MODELS.items():
            for mb, ctx, seq in POINTS:
                traced = kernels(layer_ops.classify(layer_ops.trace_ops(layer_ops.MODELS[name], mb, ctx, seq)))
                with self.subTest(model=key, mb=mb, ctx=ctx, prefill=seq):
                    self.assertEqual(layer_ops.layer_ops_from_shape(ml.REFERENCE_SHAPES[key], mb, ctx, seq), traced)

    def test_work_outside_the_layers(self):
        for key, name in MODELS.items():
            for mb, ctx, seq in POINTS:
                traced = layer_ops.trace_extras(layer_ops.MODELS[name], mb, ctx, seq)
                written = layer_ops.extras_from_shape(ml.REFERENCE_SHAPES[key], mb, ctx, seq)
                self.assertEqual(set(written), {"rotary", "embed", "clone", "head"})
                for role, calls in traced.items():
                    with self.subTest(model=key, role=role, mb=mb, ctx=ctx, prefill=seq):
                        self.assertEqual(written[role], kernels(layer_ops.classify(calls)))


class KernelCounts(unittest.TestCase):  # torch-free
    def test_kernel_counts(self):
        """Decode: 40 per layer and 9 for rotary, as the probe counted on the GPU (torch.profiler);
        prefill: 42 (Vicuna) / 44 (Qwen, + repeat_kv's 2 copies), as the probe v4 per-kernel lists
        show: the empty cache's 2 small kernels, no copy for the attention output."""
        shape = ml.REFERENCE_SHAPES["vicuna_7b"]
        self.assertEqual(len(layer_ops.layer_ops_from_shape(shape, 1, 64)), 40)
        self.assertEqual(len(layer_ops.extras_from_shape(shape, 1, 64)["rotary"]), 9)
        self.assertEqual(len(layer_ops.layer_ops_from_shape(shape, 2, 0, 64)), 42)
        self.assertEqual(len(layer_ops.layer_ops_from_shape(ml.REFERENCE_SHAPES["qwen_7b"], 2, 0, 64)), 44)


if __name__ == "__main__":
    unittest.main()
