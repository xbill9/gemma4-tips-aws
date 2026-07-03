import torch
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nForCausalLM
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig
import json

config_path = "/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03/config.json"
with open(config_path, "r") as f:
    cfg_data = json.load(f)

# Instantiate the text config
text_cfg = Gemma3nTextConfig(**cfg_data["text_config"])
print("Loaded text config:")
print("altup_num_inputs:", text_cfg.altup_num_inputs)
print("laurel_rank:", text_cfg.laurel_rank)

# Instantiate model (meta device to avoid memory usage)
with torch.device("meta"):
    model = Gemma3nForCausalLM(text_cfg)

print("\nModel structure:")
print(model)
