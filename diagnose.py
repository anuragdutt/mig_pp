# diagnose.py
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import torch

MODEL_NAME = "lmsys/vicuna-7b-v1.5"
config = LlamaConfig.from_pretrained(MODEL_NAME)

print(f"BEFORE overrides:")
print(f"  hidden_size={config.hidden_size}")
print(f"  num_attention_heads={config.num_attention_heads}")
print(f"  head_dim={getattr(config, 'head_dim', 'NOT SET')}")
print(f"  rope_scaling={getattr(config, 'rope_scaling', 'NOT SET')}")

config.hidden_size = 4096
config.num_attention_heads = 32
config._attn_implementation = "eager"
config.head_dim = config.hidden_size // config.num_attention_heads

print(f"\nAFTER overrides:")
print(f"  head_dim={config.head_dim}")

rotary_emb = LlamaRotaryEmbedding(config=config)
print(f"\nRotary inv_freq shape: {rotary_emb.inv_freq.shape}")
# Should be (64,) for head_dim=128 → 128/2=64

dummy = torch.zeros(1, 512, 4096)
position_ids = torch.arange(0, 512).unsqueeze(0)
cos, sin = rotary_emb(dummy, position_ids)
print(f"cos shape: {cos.shape}")  # Should be (1, 512, 128)
print(f"sin shape: {sin.shape}")

from transformers.models.llama.modeling_llama import ROPE_INIT_FUNCTIONS
import inspect

print(inspect.getsource(ROPE_INIT_FUNCTIONS["default"]))
clear
