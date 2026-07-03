init_file = '/opt/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py'
with open(init_file, 'r') as f:
    content = f.read()

target = '''    if architecture in NEURON_MULTI_MODAL_MODELS:
        config = getattr(config, "text_config", None)'''

replacement = '''    if architecture in NEURON_MULTI_MODAL_MODELS or hasattr(config, "text_config"):
        config = getattr(config, "text_config", None) or config'''

if target in content:
    content = content.replace(target, replacement)
    with open(init_file, 'w') as f:
        f.write(content)
    print("Successfully patched neuronx_distributed_model_loader.py")
else:
    print("Error: target string not found in neuronx_distributed_model_loader.py")
