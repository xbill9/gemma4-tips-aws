init_file = '/opt/vllm/vllm_neuron/worker/neuron_worker.py'
with open(init_file, 'r') as f:
    content = f.read()

target = 'ensure_kv_transfer_initialized(vllm_config)'
replacement = 'ensure_kv_transfer_initialized(vllm_config, vllm_config.cache_config)'

if target in content:
    content = content.replace(target, replacement)
    with open(init_file, 'w') as f:
        f.write(content)
    print("Successfully patched neuron_worker.py")
else:
    print("Error: target string not found in neuron_worker.py")
