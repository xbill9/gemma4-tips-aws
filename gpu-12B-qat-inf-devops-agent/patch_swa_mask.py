import re

file_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py"
with open(file_path, "r") as f:
    content = f.read()

target = """        # pad the attention mask if the KV cache is padded
        if prior_scores.shape[-1] > attention_mask.shape[-1] and self.neuron_config.apply_seq_ids_mask:
            attention_mask = F.pad(attention_mask, (0, prior_scores.shape[-1] - attention_mask.shape[-1]), "constant", 0)"""

replacement = target + """

        if getattr(self, "sliding_window", None) is not None:
            seq_len = prior_scores.shape[-1]
            batch_size = prior_scores.shape[0]
            idx = torch.arange(seq_len, device=prior_scores.device, dtype=position_ids.dtype).view(1, 1, 1, seq_len)
            pos = position_ids.view(batch_size, 1, 1, 1)
            swa_mask = idx >= (pos - self.sliding_window + 1)
            attention_mask = attention_mask & swa_mask"""

if target in content and "swa_mask = idx >=" not in content:
    content = content.replace(target, replacement)
    with open(file_path, "w") as f:
        f.write(content)
    print("SWA mask logic successfully patched!")
elif "swa_mask = idx >=" in content:
    print("SWA mask logic already present.")
else:
    print("Target not found for SWA mask patch.")
