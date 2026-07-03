try:
    from vllm.model_executor.models.registry import ModelRegistry
    # Check if 'models' is the attribute
    if hasattr(ModelRegistry, "models"):
        model_cls = ModelRegistry.models["Gemma4ForConditionalGeneration"]
    else:
        model_cls = ModelRegistry._models["Gemma4ForConditionalGeneration"]
    print("CLASS:", model_cls)
    import inspect
    print("FILE:", inspect.getfile(model_cls))
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
