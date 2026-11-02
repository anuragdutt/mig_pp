"""Offline tests for download_models.py (no network, no huggingface_hub needed).

    python3 -m unittest discover -s tests -p 'test_download_models.py' -v
"""

import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import download_models as dm  # noqa: E402

# Top-level listings of the real repos (abridged to the files that matter).
VICUNA = ["README.md", "config.json", "generation_config.json", "pytorch_model-00001-of-00002.bin",
          "pytorch_model-00002-of-00002.bin", "pytorch_model.bin.index.json",
          "special_tokens_map.json", "tokenizer.model", "tokenizer_config.json"]
LLAMA2 = ["LICENSE.txt", "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
          "model.safetensors.index.json", "pytorch_model-00001-of-00002.bin",
          "pytorch_model-00002-of-00002.bin", "pytorch_model.bin.index.json", "tokenizer.json",
          "tokenizer.model", "tokenizer_config.json"]
MISTRAL = ["consolidated.safetensors", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
           "model-00003-of-00003.safetensors", "model.safetensors.index.json", "params.json",
           "tokenizer.json", "tokenizer_config.json"]
QWEN = ["config.json", "merges.txt", "model-00001-of-00004.safetensors", "model-00004-of-00004.safetensors",
        "model.safetensors.index.json", "tokenizer.json", "vocab.json"]
META3 = ["config.json", "model-00001-of-00004.safetensors", "model.safetensors.index.json",
         "original/consolidated.00.pth", "original/params.json", "tokenizer.json"]


class PickFiles(unittest.TestCase):
    def test_bin_only_repo_takes_bin(self):
        files, ok = dm.pick_files(VICUNA)
        self.assertTrue(ok)
        self.assertIn("pytorch_model.bin.index.json", files)
        self.assertIn("tokenizer.model", files)
        self.assertEqual(sum(f.endswith(".bin") for f in files), 2)

    def test_both_formats_takes_only_safetensors(self):
        files, _ = dm.pick_files(LLAMA2)
        self.assertFalse(any(f.startswith("pytorch_model") for f in files), files)
        self.assertIn("model.safetensors.index.json", files)
        self.assertEqual(sum(f.endswith(".safetensors") for f in files), 2)

    def test_mistral_skips_consolidated_copy(self):
        files, _ = dm.pick_files(MISTRAL)
        self.assertNotIn("consolidated.safetensors", files)
        self.assertEqual(sum(f.endswith(".safetensors") for f in files), 3)

    def test_qwen_keeps_bpe_tokenizer_files(self):
        files, _ = dm.pick_files(QWEN)
        self.assertTrue({"merges.txt", "vocab.json", "tokenizer.json"} <= set(files))

    def test_subfolders_skipped(self):
        files, _ = dm.pick_files(META3)
        self.assertFalse(any("/" in f for f in files))

    def test_no_weights_reported(self):
        self.assertEqual(dm.pick_files(["config.json", "README.md"])[1], False)


class ConfigAndCompleteness(unittest.TestCase):
    def test_every_configured_model_has_a_repo(self):
        models = runpy.run_path(str(ROOT / "parallel_config.py"))["MODELS"]
        self.assertEqual(set(models) - set(dm.REPOS), set())

    def test_complete_needs_config_and_every_shard(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(dm.complete(d))
            Path(d, "model.safetensors").write_bytes(b"")
            self.assertFalse(dm.complete(d), "no config.json yet")
            Path(d, "config.json").write_text("{}")
            self.assertTrue(dm.complete(d))
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text("{}")
            Path(d, "model.safetensors.index.json").write_text(
                '{"weight_map": {"a": "model-00001-of-00002.safetensors"}}')
            self.assertFalse(dm.complete(d), "shard listed in the index is missing")

    def test_free_bytes_walks_up_to_an_existing_dir(self):
        self.assertGreater(dm.free_bytes(os.path.join(tempfile.gettempdir(), "no", "such", "dir")), 0)


if __name__ == "__main__":
    unittest.main()
