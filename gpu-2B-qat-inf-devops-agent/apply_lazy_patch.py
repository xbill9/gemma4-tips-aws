import sys

# 1. Clean and patch generation/__init__.py
init_file = '/opt/conda/lib/python3.12/site-packages/transformers/generation/__init__.py'
with open(init_file, 'r') as f:
    content = f.read()

# Revert any previous patch attempt first
target_bad_patch = '''    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    setattr(sys.modules[__name__], "SampleDecoderOnlyOutput", sys.modules[__name__].GenerateDecoderOnlyOutput)
    setattr(sys.modules[__name__], "SampleEncoderDecoderOutput", sys.modules[__name__].GenerateEncoderDecoderOutput)'''

if target_bad_patch in content:
    content = content.replace(target_bad_patch, '    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)')

# Now add to _import_structure['utils']
target_import = '"GenerateDecoderOnlyOutput",'
replacement_import = target_import + '\n        "SampleDecoderOnlyOutput",\n        "SampleEncoderDecoderOutput",'

if target_import in content and "SampleDecoderOnlyOutput" not in content:
    content = content.replace(target_import, replacement_import)
    with open(init_file, 'w') as f:
        f.write(content)
    print("Successfully patched generation/__init__.py with lazy structures")
else:
    # Try writing it back if we just cleaned it
    with open(init_file, 'w') as f:
        f.write(content)
    print("Cleaned generation/__init__.py, structural check needed")

# 2. Append aliases to generation/utils.py
utils_file = '/opt/conda/lib/python3.12/site-packages/transformers/generation/utils.py'
with open(utils_file, 'r') as f:
    utils_content = f.read()

alias_definitions = '\n\n# Compatibility aliases\nSampleDecoderOnlyOutput = GenerateDecoderOnlyOutput\nSampleEncoderDecoderOutput = GenerateEncoderDecoderOutput\n'
if "SampleDecoderOnlyOutput =" not in utils_content:
    with open(utils_file, 'a') as f:
        f.write(alias_definitions)
    print("Successfully appended compatibility aliases to generation/utils.py")
else:
    print("generation/utils.py already has aliases")
