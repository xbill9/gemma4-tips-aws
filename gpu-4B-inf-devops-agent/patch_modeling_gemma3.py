init_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py'
with open(init_file, 'r') as f:
    content = f.read()

target = '''        if text_config is not None:
            for attribute in self.attributes:
                setattr(self, attribute, getattr(text_config, attribute))'''

replacement = '''        if text_config is not None:
            for attribute in self.attributes:
                val = getattr(text_config, attribute, None)
                if val is None and attribute == "query_pre_attn_scalar":
                    val = getattr(text_config, "head_dim", 256)
                setattr(self, attribute, val)'''

if target in content:
    content = content.replace(target, replacement)
    with open(init_file, 'w') as f:
        f.write(content)
    print("Successfully patched modeling_gemma3.py")
else:
    print("Error: target string not found in modeling_gemma3.py")
