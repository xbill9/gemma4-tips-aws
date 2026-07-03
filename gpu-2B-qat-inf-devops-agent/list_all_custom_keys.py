from safetensors import safe_open

model_path = "/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03/model.safetensors"
with safe_open(model_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    for k in sorted(keys):
        if any(w in k.lower() for w in ["per_layer", "projection", "gate", "laurel", "altup", "norm"]):
            if "audio_tower" not in k and "vision_tower" not in k:
                tensor = f.get_tensor(k)
                print(f"Key: {k}, Shape: {list(tensor.shape)}")
