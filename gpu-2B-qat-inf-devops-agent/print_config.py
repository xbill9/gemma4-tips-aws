from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained("/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03/")
    print("CONFIG CLASS:", cfg.__class__.__name__)
    text_cfg = cfg.text_config
    print("TEXT CONFIG CLASS:", text_cfg.__class__.__name__)
    for attr in ["altup_num_inputs", "laurel_rank", "hidden_size_per_layer_input", "vocab_size_per_layer_input"]:
        print(f"{attr}: {getattr(text_cfg, attr, None)}")
except Exception as e:
    import traceback
    traceback.print_exc()
