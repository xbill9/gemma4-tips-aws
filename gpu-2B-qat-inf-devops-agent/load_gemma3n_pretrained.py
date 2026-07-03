import torch
from transformers import AutoConfig
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nForCausalLM
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig
from safetensors.torch import load_file
import traceback
import os

model_path = "/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03/"

try:
    print("Loading config...")
    cfg = AutoConfig.from_pretrained(model_path)
    text_config_dict = cfg.text_config.to_dict()
    text_config = Gemma3nTextConfig(**text_config_dict)
    
    print("Instantiating model on CPU...")
    model = Gemma3nForCausalLM(text_config)
    
    print("Loading state dict from safetensors...")
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    
    print("\nLoading weights into model (strict=False)...")
    mapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.language_model."):
            mapped_k = "model." + k.removeprefix("model.language_model.")
        elif k == "model.language_model.norm.weight":
            mapped_k = "model.norm.weight"
        else:
            mapped_k = k
        mapped_state_dict[mapped_k] = v

    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    print("\nSuccessfully called load_state_dict!")
    print("Missing keys count:", len(missing_keys))
    print("Unexpected keys count:", len(unexpected_keys))
    
    print("\nSample missing keys (first 15):")
    for mk in sorted(missing_keys)[:15]:
        print("  ", mk)
    
    print("\nSample unexpected keys (first 15):")
    for uk in sorted(unexpected_keys)[:15]:
        print("  ", uk)
    
except Exception as e:
    traceback.print_exc()
