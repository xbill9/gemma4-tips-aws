try:
    from vllm.model_executor.models.registry import ModelRegistry
    for arch in ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM", "Gemma4UnifiedForConditionalGeneration"]:
        try:
            model_cls = ModelRegistry.resolve_model_cls(arch)
            print(f"Arch: {arch} -> Class: {model_cls.__name__} in module {model_cls.__module__}")
        except Exception as e:
            print(f"Arch: {arch} -> Failed to resolve: {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
