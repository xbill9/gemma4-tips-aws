init_file = '/opt/conda/lib/python3.12/site-packages/vllm/utils/torch_utils.py'
with open(init_file, 'r') as f:
    content = f.read()

target = '''        from vllm.platforms import current_platform

        current_platform.manual_seed_all(seed)'''

replacement = '''        from vllm.platforms import current_platform
        try:
            current_platform.manual_seed_all(seed)
        except NotImplementedError:
            pass'''

if target in content:
    content = content.replace(target, replacement)
    with open(init_file, 'w') as f:
        f.write(content)
    print("Successfully patched torch_utils.py")
else:
    print("Error: target string not found in torch_utils.py")
