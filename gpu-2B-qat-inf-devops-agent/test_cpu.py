import torch
from transformers import AutoTokenizer, AutoModel
import sys

model_path = "/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03/"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model on CPU...")
try:
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16
    )
    print("Model loaded successfully!")
    print("Model class:", model.__class__.__name__)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

prompt = "is a hotdog a sandwich"
print(f"Prompt: {prompt}")
inputs = tokenizer(prompt, return_tensors="pt")

print("Generating...")
if hasattr(model, "generate"):
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"CPU Output: {decoded}")
else:
    print("Model has no generate() method. Running forward pass...")
    with torch.no_grad():
        out = model(**inputs)
    print("Forward pass complete. logits shape:", out.logits.shape if hasattr(out, "logits") else "No logits")
