from safetensors import safe_open
import sys

model_path = "/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03/model.safetensors"
print(f"Opening safetensors file: {model_path}")
try:
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Total keys: {len(keys)}")
        # Print embedding and lm_head keys
        for key in keys:
            if "embed" in key or "lm_head" in key:
                tensor = f.get_tensor(key)
                print(f"Key: {key}, Shape: {list(tensor.shape)}, Dtype: {tensor.dtype}")
        # Print a few MLP and attention layers
        for key in sorted(keys)[:20]:
            tensor = f.get_tensor(key)
            print(f"Key: {key}, Shape: {list(tensor.shape)}")
except Exception as e:
    print("Error:", e)
