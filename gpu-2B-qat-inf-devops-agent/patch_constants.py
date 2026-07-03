init_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/utils/constants.py'
with open(init_file, 'r') as f:
    content = f.read()

target = '    "gemma3": {"causal-lm": NeuronGemma3ForCausalLM},'
replacement = target + '\n    "gemma4unified": {"causal-lm": NeuronGemma3ForCausalLM},'

if target in content and "gemma4unified" not in content:
    content = content.replace(target, replacement)
    with open(init_file, 'w') as f:
        f.write(content)
    print("Successfully patched constants.py with gemma4unified mapping")
else:
    print("constants.py already patched or target not found")
