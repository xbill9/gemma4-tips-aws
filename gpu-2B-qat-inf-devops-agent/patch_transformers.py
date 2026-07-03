init_file = '/opt/conda/lib/python3.12/site-packages/transformers/generation/__init__.py'
with open(init_file, 'r') as f:
    content = f.read()

target = '    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)'
replacement = target + '\n    setattr(sys.modules[__name__], "SampleDecoderOnlyOutput", sys.modules[__name__].GenerateDecoderOnlyOutput)\n    setattr(sys.modules[__name__], "SampleEncoderDecoderOutput", sys.modules[__name__].GenerateEncoderDecoderOutput)'

if target in content:
    content = content.replace(target, replacement)
    with open(init_file, 'w') as f:
        f.write(content)
    print('Successfully patched generation/__init__.py')
else:
    print('Error: target string not found in generation/__init__.py')
