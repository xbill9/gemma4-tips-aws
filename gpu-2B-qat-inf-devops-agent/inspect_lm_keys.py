from safetensors import safe_open

model_path = "/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03/model.safetensors"
with safe_open(model_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    for k in sorted(keys):
        if "lm_head" in k or "language_model" in k or k.startswith("lm_head") or k.endswith("weight") and not "audio_tower" in k and not "vision_tower" in k:
            tensor = f.get_tensor(k)
            print(f"Key: {k}, Shape: {list(tensor.shape)}")
