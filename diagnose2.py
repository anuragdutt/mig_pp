# check_internal_rope.py
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import inspect

config = LlamaConfig.from_pretrained("lmsys/vicuna-7b-v1.5")
config._attn_implementation = "eager"

layer = LlamaDecoderLayer(config, layer_idx=0)
print(inspect.getsource(layer.self_attn.forward))
