import sys
try:
    from vllm.model_executor.models.registry import ModelRegistry
    print("ARCHS:", ModelRegistry.get_supported_archs())
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
