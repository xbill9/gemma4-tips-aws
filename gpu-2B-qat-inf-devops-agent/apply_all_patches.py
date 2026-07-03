import sys
import os

def patch_file(filepath, target, replacement):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    with open(filepath, 'r') as f:
        content = f.read()
    if replacement in content:
        print(f"Patch already applied to {filepath}")
        return True
    if target in content:
        content = content.replace(target, replacement)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Successfully patched {filepath}")
        return True
    else:
        print(f"Target not found in {filepath}")
        return False

# 1. Patch neuron_worker.py
patch_file(
    '/opt/vllm/vllm_neuron/worker/neuron_worker.py',
    'ensure_kv_transfer_initialized(vllm_config)',
    'ensure_kv_transfer_initialized(vllm_config, vllm_config.cache_config)'
)

# 2. Patch modeling_gemma3.py
gemma3_target = '''        if text_config is not None:
            for attribute in self.attributes:
                setattr(self, attribute, getattr(text_config, attribute))'''
gemma3_repl = '''        if text_config is not None:
            for attribute in self.attributes:
                val = getattr(text_config, attribute, None)
                if val is None and attribute == "query_pre_attn_scalar":
                    val = getattr(text_config, "head_dim", 256)
                if attribute == "sliding_window" and val is not None:
                    val = getattr(text_config, "sliding_window", 1024)
                setattr(self, attribute, val)'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_target,
    gemma3_repl
)

# 2_attr. Patch Gemma3InferenceConfig attributes to include final_logit_softcapping
gemma3_attr_target = '''        self.attributes = [
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "query_pre_attn_scalar",
            "sliding_window",
        ]'''
gemma3_attr_repl = '''        self.attributes = [
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "query_pre_attn_scalar",
            "sliding_window",
            "final_logit_softcapping",
            "attn_logit_softcapping",
            "use_double_wide_mlp",
        ]'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_attr_target,
    gemma3_attr_repl
)

# 2_softcap. Patch lm_head definition to include register_forward_hook for logit softcapping
gemma3_lm_head_target = '''        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )'''
gemma3_lm_head_repl = '''        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Logit softcapping monkeypatch for Gemma 4
        final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            orig_forward = self.lm_head.forward
            def softcap_forward(*args, **kwargs):
                output = orig_forward(*args, **kwargs)
                return torch.tanh(output / final_logit_softcapping) * final_logit_softcapping
            self.lm_head.forward = softcap_forward'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_lm_head_target,
    gemma3_lm_head_repl
)


# 2a. Patch get_updated_configs in modeling_gemma3.py
gemma3_get_configs_target = '''def get_updated_configs(config: InferenceConfig):
    """
    Generate a list of configurations for each hidden layer in a Gemma3 model.

    Args:
    config (InferenceConfig): The inference configuration for the model.

    Returns:
    list[InferenceConfig]: A list of InferenceConfig objects, one for each layer in the model.
                           Each config may be either the original config or a modified version.
    """
    updated_configs = []

    for i in range(config.num_hidden_layers):
        updated_config = copy.deepcopy(config)

        swa_layer = (i + 1) % 6 != 0

        if not swa_layer:
            updated_config.sliding_window = None

        updated_configs.append(updated_config)

    return updated_configs'''

gemma3_get_configs_repl = '''def get_updated_configs(config: InferenceConfig):
    """
    Generate a list of configurations for each hidden layer in a Gemma3 model.

    Args:
    config (InferenceConfig): The inference configuration for the model.

    Returns:
    list[InferenceConfig]: A list of InferenceConfig objects, one for each layer in the model.
                           Each config may be either the original config or a modified version.
    """
    updated_configs = []

    for i in range(config.num_hidden_layers):
        updated_config = copy.deepcopy(config)

        # Gemma 4 uses a double-wide MLP (intermediate_size = 12288) for layers 15 to 34
        if getattr(config, "use_double_wide_mlp", False) and i >= 15:
            updated_config.intermediate_size = 12288

        swa_layer = (i + 1) % 5 != 0

        if not swa_layer:
            updated_config.sliding_window = None
            updated_config.head_dim = getattr(config, "global_head_dim", 512)
            updated_config.num_key_value_heads = getattr(config, "num_global_key_value_heads", 1)
            updated_config.query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", None) or getattr(config, "head_dim", 256)
            updated_config.neuron_config.fused_qkv = False
        else:
            updated_config.sliding_window = getattr(config, "sliding_window", 1024)
            updated_config.head_dim = getattr(config, "head_dim", 256)
            updated_config.num_key_value_heads = getattr(config, "num_key_value_heads", 8)
            updated_config.query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", None) or getattr(config, "head_dim", 256)

        updated_configs.append(updated_config)

    return updated_configs'''

patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_get_configs_target,
    gemma3_get_configs_repl
)

# 2b. Patch convert_hf_to_neuron_state_dict in modeling_gemma3.py
gemma3_convert_target = '''        if "model.norm.weight" in state_dict.keys():
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}'''
gemma3_convert_repl = '''        if "model.language_model.norm.weight" in state_dict.keys():
            state_dict = {k.removeprefix("model.language_model."): v for k, v in state_dict.items()}
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_convert_target,
    gemma3_convert_repl
)

gemma3_convert_target_patched = '''        if "model.language_model.norm.weight" in state_dict.keys():
            state_dict = {k.removeprefix("model.language_model."): v for k, v in state_dict.items()}
        elif "model.norm.weight" in state_dict.keys():
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_convert_target_patched,
    gemma3_convert_repl
)

# 2c. Patch convert_hf_to_neuron_state_dict in modeling_gemma3.py to rename q/k/v proj to qkv_proj when fused_qkv is False
gemma3_fused_target = '''            if config.neuron_config.fused_qkv:
                attr = "weight"  # Will have to set this to "scale" if we pursue quantized weights

                state_dict[f"layers.{i}.self_attn.Wqkv.{attr}"] = torch.cat(
                    [
                        state_dict[f"layers.{i}.self_attn.q_proj.{attr}"],
                        state_dict[f"layers.{i}.self_attn.k_proj.{attr}"],
                        state_dict[f"layers.{i}.self_attn.v_proj.{attr}"],
                    ],
                )
                del state_dict[f"layers.{i}.self_attn.q_proj.{attr}"]
                del state_dict[f"layers.{i}.self_attn.k_proj.{attr}"]
                del state_dict[f"layers.{i}.self_attn.v_proj.{attr}"]'''

gemma3_fused_repl = '''            if f"layers.{i}.self_attn.k_proj.weight" in state_dict and f"layers.{i}.self_attn.v_proj.weight" not in state_dict:
                state_dict[f"layers.{i}.self_attn.v_proj.weight"] = state_dict[f"layers.{i}.self_attn.k_proj.weight"].clone()
            
            # Determine if this layer is a sliding window layer
            swa_layer = (i + 1) % 5 != 0
            is_fused_layer = config.neuron_config.fused_qkv and swa_layer

            if is_fused_layer:
                attr = "weight"  # Will have to set this to "scale" if we pursue quantized weights

                state_dict[f"layers.{i}.self_attn.Wqkv.{attr}"] = torch.cat(
                    [
                        state_dict[f"layers.{i}.self_attn.q_proj.{attr}"],
                        state_dict[f"layers.{i}.self_attn.k_proj.{attr}"],
                        state_dict[f"layers.{i}.self_attn.v_proj.{attr}"],
                    ],
                )
                del state_dict[f"layers.{i}.self_attn.q_proj.{attr}"]
                del state_dict[f"layers.{i}.self_attn.k_proj.{attr}"]
                del state_dict[f"layers.{i}.self_attn.v_proj.{attr}"]
            else:
                for proj in ["q_proj", "k_proj", "v_proj"]:
                    if f"layers.{i}.self_attn.{proj}.weight" in state_dict:
                        weight = state_dict.pop(f"layers.{i}.self_attn.{proj}.weight")
                        setattr(weight, "tensor_model_parallel", True)
                        setattr(weight, "partition_dim", 0)
                        setattr(weight, "partition_stride", 1)
                        setattr(weight, "num_partitions", config.neuron_config.tp_degree)
                        state_dict[f"layers.{i}.self_attn.qkv_proj.{proj}.weight"] = weight

            if f"layers.{i}.self_attn.o_proj.weight" in state_dict:
                weight = state_dict.pop(f"layers.{i}.self_attn.o_proj.weight")
                setattr(weight, "tensor_model_parallel", True)
                setattr(weight, "partition_dim", 1)
                setattr(weight, "partition_stride", 1)
                setattr(weight, "num_partitions", config.neuron_config.tp_degree)
                state_dict[f"layers.{i}.self_attn.o_proj.o_proj.weight"] = weight'''

patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_fused_target,
    gemma3_fused_repl
)

# 2d. Patch NeuronGemma3Attention and NeuronGemma3DecoderLayer in modeling_gemma3.py to support heterogeneous head dimensions
gemma3_attn_init_target = '''class NeuronGemma3Attention(NeuronAttentionBase):
    def __init__(self, config: Gemma3InferenceConfig):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)'''

gemma3_attn_init_repl = '''class NeuronGemma3Attention(NeuronAttentionBase):
    def __init__(self, config: Gemma3InferenceConfig, layer_idx: int = None):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if layer_idx is not None and (layer_idx + 1) % 5 == 0:
            head_dim = getattr(config, "global_head_dim", 512)
        self.is_sliding_window_attention = config.sliding_window is not None and (layer_idx is None or (layer_idx + 1) % 5 != 0)
        self.layer_idx = layer_idx'''

patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_attn_init_target,
    gemma3_attn_init_repl
)

gemma3_decoder_attn_target = '''        self.self_attn = NeuronGemma3Attention(config)'''
gemma3_decoder_attn_repl = '''        self.self_attn = NeuronGemma3Attention(config, layer_idx=layer_idx)'''

patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_decoder_attn_target,
    gemma3_decoder_attn_repl
)

# 2e. Patch rotary_emb selection in modeling_gemma3.py
gemma3_rotary_target = '''        rotary_emb = local_rotary_emb
        if config.sliding_window is None:
            rotary_emb = global_rotary_emb'''

gemma3_rotary_repl = '''        rotary_emb = local_rotary_emb
        if config.sliding_window is None or (layer_idx is not None and (layer_idx + 1) % 5 == 0):
            rotary_emb = global_rotary_emb'''

patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_rotary_target,
    gemma3_rotary_repl
)

# 2f. Patch is_sliding_window_attention in modeling_gemma3.py
gemma3_swa_attention_target = '''class NeuronGemma3DecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Gemma3InferenceConfig, layer_idx: int):
        super().__init__()

        self.is_sliding_window_attention = config.sliding_window is not None'''

gemma3_swa_attention_repl = '''class NeuronGemma3DecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Gemma3InferenceConfig, layer_idx: int):
        super().__init__()

        self.is_sliding_window_attention = config.sliding_window is not None and (layer_idx + 1) % 5 != 0'''

patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_swa_attention_target,
    gemma3_swa_attention_repl
)

# 2f2. Patch super().__init__ of NeuronGemma3Attention to conditionally pass sliding_window
gemma3_super_init_target = '''            use_scaled_rope=None,
            sliding_window=config.sliding_window,
            softmax_scale=(config.query_pre_attn_scalar**(.5))'''
gemma3_super_init_repl = '''            use_scaled_rope=None,
            sliding_window=None if (layer_idx is not None and (layer_idx + 1) % 5 == 0) else config.sliding_window,
            softmax_scale=(config.query_pre_attn_scalar**(.5))'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_super_init_target,
    gemma3_super_init_repl
)

# 2g. Patch global_rotary_emb dimension in modeling_gemma3.py
gemma3_global_rope_dim_target = '''        global_rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.global_rope_theta,
            factor=config.rope_scaling,
        )'''
gemma3_global_rope_dim_repl = '''        global_rotary_emb = RotaryEmbedding(
            dim=getattr(config, "global_head_dim", 512),
            max_position_embeddings=config.max_position_embeddings,
            base=config.global_rope_theta,
            factor=config.rope_scaling,
            head_dim=getattr(config, "global_head_dim", 512),
            partial_rotary_factor=0.25,
        )'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_global_rope_dim_target,
    gemma3_global_rope_dim_repl
)

# 2h. Patch max_position_embeddings and rope_scaling in modeling_gemma3.py
gemma3_rope_scaling_target = '        setattr(self, "rope_scaling", 8.0)'
gemma3_rope_scaling_repl = '        setattr(self, "rope_scaling", None)'
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_rope_scaling_target,
    gemma3_rope_scaling_repl
)

gemma3_max_pos_target = '        setattr(self, "max_position_embeddings", 131072)'
gemma3_max_pos_repl = '        setattr(self, "max_position_embeddings", 262144)'
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_max_pos_target,
    gemma3_max_pos_repl
)

# 3. Patch model_loader.py
loader_target = '''    if architecture in NEURON_MULTI_MODAL_MODELS:
        config = getattr(config, "text_config", None)'''
loader_repl = '''    if architecture in NEURON_MULTI_MODAL_MODELS or hasattr(config, "text_config"):
        config = getattr(config, "text_config", None) or config'''
patch_file(
    '/opt/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py',
    loader_target,
    loader_repl
)

# 4. Patch constants.py
constants_target = '    "gemma3": {"causal-lm": NeuronGemma3ForCausalLM},'
constants_repl = """    \"gemma3\": {\"causal-lm\": NeuronGemma3ForCausalLM},
    \"gemma4\": {\"causal-lm\": NeuronGemma3ForCausalLM},
    \"gemma4unified\": {\"causal-lm\": NeuronGemma3ForCausalLM},
    \"gemma4_unified\": {\"causal-lm\": NeuronGemma3ForCausalLM},"""
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/utils/constants.py',
    constants_target,
    constants_repl
)

# 5. Patch torch_utils.py
torch_target = '''        from vllm.platforms import current_platform

        current_platform.manual_seed_all(seed)'''
torch_repl = '''        from vllm.platforms import current_platform
        try:
            current_platform.manual_seed_all(seed)
        except NotImplementedError:
            pass'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/vllm/utils/torch_utils.py',
    torch_target,
    torch_repl
)

# 6. Patch generation/__init__.py and utils.py
init_file = '/opt/conda/lib/python3.12/site-packages/transformers/generation/__init__.py'
if os.path.exists(init_file):
    with open(init_file, 'r') as f:
        init_content = f.read()
    # Clean previous lazy patch if exists
    bad_lazy = '''    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    setattr(sys.modules[__name__], "SampleDecoderOnlyOutput", sys.modules[__name__].GenerateDecoderOnlyOutput)
    setattr(sys.modules[__name__], "SampleEncoderDecoderOutput", sys.modules[__name__].GenerateEncoderDecoderOutput)'''
    if bad_lazy in init_content:
        init_content = init_content.replace(bad_lazy, '    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)')
    
    target_import = '"GenerateDecoderOnlyOutput",'
    repl_import = """\"GenerateDecoderOnlyOutput\",
        \"SampleDecoderOnlyOutput\",
        \"SampleEncoderDecoderOutput\","""
    if target_import in init_content and "SampleDecoderOnlyOutput" not in init_content:
        init_content = init_content.replace(target_import, repl_import)
        with open(init_file, 'w') as f:
            f.write(init_content)
        print("Patched generation/__init__.py")
        
utils_file = '/opt/conda/lib/python3.12/site-packages/transformers/generation/utils.py'
if os.path.exists(utils_file):
    with open(utils_file, 'r') as f:
        utils_content = f.read()
    alias_defs = """

# Compatibility aliases
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput
"""
    if "SampleDecoderOnlyOutput =" not in utils_content:
        with open(utils_file, 'a') as f:
            f.write(alias_defs)
        print("Appended aliases to generation/utils.py")

# 7. Add auto-patching for Transformers 4.x if applicable, otherwise create dummy fx.py for Transformers 5.x
import transformers
print(f"Detected Transformers version: {transformers.__version__}")

if transformers.__version__.startswith('4.'):
    config_auto_file = '/opt/conda/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py'
    if os.path.exists(config_auto_file):
        patch_file(
            config_auto_file,
            '("aimv2", "AIMv2"),',
            '("aimv2", "AIMv2"),\n        ("gemma4", "Gemma 4"),\n        ("gemma4_text", "Gemma 4 Text"),\n        ("gemma4_unified", "Gemma 4 Unified"),\n        ("gemma4_unified_text", "Gemma 4 Unified Text"),\n        ("gemma4_unified_vision", "Gemma 4 Unified Vision"),\n        ("gemma4_unified_audio", "Gemma 4 Unified Audio"),'
        )
        patch_file(
            config_auto_file,
            '("gemma3", "Gemma3Config"),',
            '("gemma3", "Gemma3Config"),\n        ("gemma4", "Gemma3Config"),\n        ("gemma4_text", "Gemma3TextConfig"),\n        ("gemma4_unified", "Gemma3Config"),\n        ("gemma4_unified_text", "Gemma3TextConfig"),'
        )
        patch_file(
            config_auto_file,
            '("gemma3_text", "gemma3"),',
            '("gemma3_text", "gemma3"),\n        ("gemma4", "gemma3"),\n        ("gemma4_text", "gemma3"),\n        ("gemma4_unified", "gemma3"),\n        ("gemma4_unified_text", "gemma3"),\n        ("gemma4_unified_vision", "gemma3"),\n        ("gemma4_unified_audio", "gemma3"),'
        )
        patch_file(
            config_auto_file,
            '("gemma3", "Gemma3ForConditionalGeneration"),',
            '("gemma3", "Gemma3ForConditionalGeneration"),\n        ("gemma4", "Gemma4Unified"),\n        ("gemma4_unified", "Gemma4Unified"),\n        ("gemma4_unified_text", "Gemma4UnifiedText"),'
        )

    modeling_auto_file = '/opt/conda/lib/python3.12/site-packages/transformers/models/auto/modeling_auto.py'
    if os.path.exists(modeling_auto_file):
        patch_file(
            modeling_auto_file,
            '        ("gemma3", "Gemma3Model"),\n        ("gemma3_text", "Gemma3TextModel"),',
            '        ("gemma3", "Gemma3Model"),\n        ("gemma3_text", "Gemma3TextModel"),\n        ("gemma4", "Gemma3TextModel"),\n        ("gemma4_text", "Gemma3TextModel"),\n        ("gemma4_unified", "Gemma3TextModel"),\n        ("gemma4_unified_text", "Gemma3TextModel"),'
        )
        patch_file(
            modeling_auto_file,
            '        ("gemma3", "Gemma3ForConditionalGeneration"),\n        ("gemma3_text", "Gemma3ForCausalLM"),',
            '        ("gemma3", "Gemma3ForConditionalGeneration"),\n        ("gemma3_text", "Gemma3ForCausalLM"),\n        ("gemma4", "Gemma3ForCausalLM"),\n        ("gemma4_text", "Gemma3ForCausalLM"),\n        ("gemma4_unified", "Gemma3ForCausalLM"),\n        ("gemma4_unified_text", "Gemma3ForCausalLM"),'
        )
else:
    # Transformers 5.x
    fx_dir = '/opt/conda/lib/python3.12/site-packages/transformers/utils'
    if os.path.exists(fx_dir):
        fx_file = os.path.join(fx_dir, 'fx.py')
        if not os.path.exists(fx_file):
            with open(fx_file, 'w') as f:
                f.write('''# Dummy file created to satisfy imports from neuronx-distributed in transformers v5
class HFTracer:
    pass
class HFProxy:
    pass
''')
            print("Created dummy transformers.utils.fx for transformers v5 compatibility")

# 8. Patch attention_base.py to fallback when head_dim > 128
attention_base_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py'
if os.path.exists(attention_base_file):
    # Patch get_flash_attention_strategy_cp
    target_cp = '    def get_flash_attention_strategy_cp(self, q_len):'
    repl_cp = '''    def get_flash_attention_strategy_cp(self, q_len):
        if getattr(self, "head_dim", 0) > 128:
            return FlashAttentionStrategy.NONE'''
    patch_file(attention_base_file, target_cp, repl_cp)

    # Patch get_flash_attention_strategy
    target_strategy = '    def get_flash_attention_strategy(self, q_len, has_attention_mask) -> FlashAttentionStrategy:'
    repl_strategy = '''    def get_flash_attention_strategy(self, q_len, has_attention_mask) -> FlashAttentionStrategy:
        if getattr(self, "head_dim", 0) > 128:
            return FlashAttentionStrategy.NONE'''
    patch_file(attention_base_file, target_strategy, repl_strategy)

    # Patch apply_rotary_embedding for heterogeneous head dimensions (Q: 512, K: 256)
    target_rope = '''    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)'''
    repl_rope = '''    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            from .utils import _rotate_half
            is_swa_layer = True
            if hasattr(self, "layer_idx") and self.layer_idx is not None:
                if (self.layer_idx + 1) % 5 == 0:
                    is_swa_layer = False
            elif hasattr(self, "is_sliding_window_attention"):
                is_swa_layer = self.is_sliding_window_attention
            partial_factor = 1.0 if is_swa_layer else 0.25
            rotary_dim = int((256 if is_swa_layer else 512) * partial_factor)
            half_head_dim = cos_cache.shape[-1] // 2
            half_rot_dim = rotary_dim // 2
            if rotary_dim < Q.shape[-1]:
                cos_q = torch.cat([
                    cos_cache[..., :half_rot_dim],
                    cos_cache[..., half_head_dim : half_head_dim + half_rot_dim]
                ], dim=-1).unsqueeze(1)
                sin_q = torch.cat([
                    sin_cache[..., :half_rot_dim],
                    sin_cache[..., half_head_dim : half_head_dim + half_rot_dim]
                ], dim=-1).unsqueeze(1)
                Q_rot = Q[..., :rotary_dim]
                Q_pass = Q[..., rotary_dim:]
                Q_rot = (Q_rot * cos_q) + (_rotate_half(Q_rot) * sin_q)
                Q = torch.cat([Q_rot, Q_pass], dim=-1)
            else:
                cos_q = cos_cache[..., :Q.shape[-1]].unsqueeze(1)
                sin_q = sin_cache[..., :Q.shape[-1]].unsqueeze(1)
                Q = (Q * cos_q) + (_rotate_half(Q) * sin_q)
            if rotary_dim < K.shape[-1]:
                cos_k = torch.cat([
                    cos_cache[..., :half_rot_dim],
                    cos_cache[..., half_head_dim : half_head_dim + half_rot_dim]
                ], dim=-1).unsqueeze(1)
                sin_k = torch.cat([
                    sin_cache[..., :half_rot_dim],
                    sin_cache[..., half_head_dim : half_head_dim + half_rot_dim]
                ], dim=-1).unsqueeze(1)
                K_rot = K[..., :rotary_dim]
                K_pass = K[..., rotary_dim:]
                K_rot = (K_rot * cos_k) + (_rotate_half(K_rot) * sin_k)
                K = torch.cat([K_rot, K_pass], dim=-1)
            else:
                cos_k = cos_cache[..., :K.shape[-1]].unsqueeze(1)
                sin_k = sin_cache[..., :K.shape[-1]].unsqueeze(1)
                K = (K * cos_k) + (_rotate_half(K) * sin_k)'''
    patch_file(attention_base_file, target_rope, repl_rope)



    # Patch prep_qkv_tensors to pad Q, K, V from 256 to 512
    target_prep = '        return Q, K, V, cos_cache, sin_cache, residual'
    repl_prep = '''        if Q.shape[-1] < 512:
            import torch.nn.functional as F
            Q = F.pad(Q, (0, 512 - Q.shape[-1]))
            K = F.pad(K, (0, 512 - K.shape[-1]))
            V = F.pad(V, (0, 512 - V.shape[-1]))
        return Q, K, V, cos_cache, sin_cache, residual'''
    patch_file(attention_base_file, target_prep, repl_prep)

    # Patch 12: Trim KV cache prior tensors back to true head_dim for token generation warmup crash
    target_token_gen = '''        if getattr(self, "head_dim", 0) == 256:
            if Q.shape[-1] == 512:
                Q = Q[..., :256]
            if K.shape[-1] == 512:
                K = K[..., :256]
            if V.shape[-1] == 512:
                V = V[..., :256]'''
    repl_token_gen = '''        if getattr(self, "head_dim", 0) == 256:
            if Q.shape[-1] == 512: Q = Q[..., :256]
            if K.shape[-1] == 512: K = K[..., :256]
            if V.shape[-1] == 512: V = V[..., :256]
            if past_key_value is not None and len(past_key_value) > 0 and past_key_value[0] is not None:
                K_prior, V_prior = past_key_value[0], past_key_value[1]
                if K_prior.shape[-1] == 512 and not self.k_cache_transposed:
                    K_prior = K_prior[..., :256]
                elif K_prior.shape[-2] == 512 and self.k_cache_transposed:
                    K_prior = K_prior[..., :256, :]
                if V_prior.shape[-1] == 512:
                    V_prior = V_prior[..., :256]
                past_key_value = (K_prior, V_prior)'''
    # We will apply target_token_gen later in the file after target_tokengen patch is applied.

    # Patch attn_output to slice back from 512 to 256 before merge multi head hidden
    target_merge = '''        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)'''
    repl_merge = '''        if attn_output.shape[-1] == 512 and self.head_dim == 256:
            attn_output = attn_output[..., :256]
        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)'''
    patch_file(attention_base_file, target_merge, repl_merge)

    # SWA mask logic is not needed inside compute_for_token_gen because modulo index cache is wrapped.

    # Patch compute_for_token_gen to slice Q, K, V back to 256 for sliding attention layers
    target_tokengen = '''    def compute_for_token_gen(
        self,
        Q,
        K,
        V,
        position_ids,
        past_key_value,
        attention_mask,
        active_mask,
        is_prefix_caching=False,
    ) -> Tensor:'''
    repl_tokengen = '''    def compute_for_token_gen(
        self,
        Q,
        K,
        V,
        position_ids,
        past_key_value,
        attention_mask,
        active_mask,
        is_prefix_caching=False,
    ) -> Tensor:
        if getattr(self, "head_dim", 0) == 256:
            if Q.shape[-1] == 512:
                Q = Q[..., :256]
            if K.shape[-1] == 512:
                K = K[..., :256]
            if V.shape[-1] == 512:
                V = V[..., :256]
        if past_key_value is not None and len(past_key_value) > 0 and past_key_value[0] is not None:
            k_prior_heads = past_key_value[0].shape[1]
            if Q.shape[1] % k_prior_heads == 0:
                self.num_key_value_groups = Q.shape[1] // k_prior_heads
        elif K is not None:
            if Q.shape[1] % K.shape[1] == 0:
                self.num_key_value_groups = Q.shape[1] // K.shape[1]'''
    patch_file(attention_base_file, target_tokengen, repl_tokengen)
    # Apply token_gen prior slicing now that compute_for_token_gen is patched
    patch_file(attention_base_file, target_token_gen, repl_token_gen)

    # Patch K_prior and V_prior repetition in compute_for_token_gen to avoid head count mismatch
    target_prior_repeat = '''        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)'''
    repl_prior_repeat = '''        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        prior_repeat = Q.shape[1] // K_prior.shape[1] if (K_prior is not None and K_prior.shape[1] > 0) else getattr(self, "num_key_value_groups", 1)
        K_prior = repeat_kv(K_prior, prior_repeat)
        V_prior = repeat_kv(V_prior, prior_repeat)'''
    patch_file(attention_base_file, target_prior_repeat, repl_prior_repeat)

    # Patch K_active and V_active repetition in compute_for_token_gen to avoid head count mismatch
    target_active_repeat = '''        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)'''
    repl_active_repeat = '''        # ii. active (current/new) KV
        active_repeat = Q.shape[1] // K.shape[1] if (K is not None and K.shape[1] > 0) else getattr(self, "num_key_value_groups", 1)
        K_active = repeat_kv(K, active_repeat)
        V_active = repeat_kv(V, active_repeat)'''
    patch_file(attention_base_file, target_active_repeat, repl_active_repeat)

    # Patch 8b: Patch scaled_qk in attention_base.py to support attention logit softcapping
    target_scaled_qk = '''    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / self.softmax_scale'''
    repl_scaled_qk = '''    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / self.softmax_scale
        softcap = getattr(self.config, "attn_logit_softcapping", None) or getattr(self.config, "attention_logit_cap", None) or 50.0
        import sys
        sys.stderr.write(f"SCALED_QK_DEBUG: softmax_scale={self.softmax_scale}, softcap={softcap}, Q.shape={list(Q.shape)}, K.shape={list(K.shape)}\\n")
        sys.stderr.flush()
        if softcap is not None:
            QK = torch.tanh(QK / softcap) * softcap
        if attention_mask is not None:
            QK = torch.where(attention_mask.to(torch.bool), QK, torch.finfo(QK.dtype).min)
        return QK'''
    patch_file(attention_base_file, target_scaled_qk, repl_scaled_qk)

    # Patch 8c: Patch perform_prefix_prefill in attention_base.py to support attention logit softcapping
    target_prefix_prefill = '''            # Attention computation: softmax((Q.K/√dkv) + mask).V
            # i. prior (cached) KV
            if not self.k_cache_transposed:
                K_prior = K_prior.transpose(2, 3)
            prior_scores = torch.matmul(Q, K_prior) / self.softmax_scale
            prior_scores = prior_scores.to(torch.float32)

            # ii. active (current/new) KV
            active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / self.softmax_scale'''
    repl_prefix_prefill = '''            # Attention computation: softmax((Q.K/√dkv) + mask).V
            # i. prior (cached) KV
            if not self.k_cache_transposed:
                K_prior = K_prior.transpose(2, 3)
            prior_scores = torch.matmul(Q, K_prior) / self.softmax_scale
            softcap = getattr(self.config, "attn_logit_softcapping", None) or getattr(self.config, "attention_logit_cap", None) or 50.0
            if softcap is not None:
                prior_scores = torch.tanh(prior_scores / softcap) * softcap
            prior_scores = prior_scores.to(torch.float32)

            # ii. active (current/new) KV
            active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / self.softmax_scale
            if softcap is not None:
                active_scores = torch.tanh(active_scores / softcap) * softcap'''
    patch_file(attention_base_file, target_prefix_prefill, repl_prefix_prefill)

    # Patch 8d: Patch compute_for_token_gen in attention_base.py to support attention logit softcapping
    target_token_gen_softcap = '''        if not self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)
        prior_scores = torch.matmul(Q, K_prior) / self.softmax_scale

        # pad the attention mask if the KV cache is padded'''
    repl_token_gen_softcap = '''        if not self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)
        prior_scores = torch.matmul(Q, K_prior) / self.softmax_scale
        softcap = getattr(self.config, "attn_logit_softcapping", None) or getattr(self.config, "attention_logit_cap", None) or 50.0
        if softcap is not None:
            prior_scores = torch.tanh(prior_scores / softcap) * softcap

        # pad the attention mask if the KV cache is padded'''
    patch_file(attention_base_file, target_token_gen_softcap, repl_token_gen_softcap)

    target_token_gen_active_softcap = '''        # ii. active (current/new) KV
        active_repeat = Q.shape[1] // K.shape[1] if (K is not None and K.shape[1] > 0) else getattr(self, "num_key_value_groups", 1)
        K_active = repeat_kv(K, active_repeat)
        V_active = repeat_kv(V, active_repeat)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / self.softmax_scale'''
    repl_token_gen_active_softcap = '''        # ii. active (current/new) KV
        active_repeat = Q.shape[1] // K.shape[1] if (K is not None and K.shape[1] > 0) else getattr(self, "num_key_value_groups", 1)
        K_active = repeat_kv(K, active_repeat)
        V_active = repeat_kv(V, active_repeat)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / self.softmax_scale
        softcap = getattr(self.config, "attn_logit_softcapping", None) or getattr(self.config, "attention_logit_cap", None) or 50.0
        if softcap is not None:
            active_scores = torch.tanh(active_scores / softcap) * softcap'''
    patch_file(attention_base_file, target_token_gen_active_softcap, repl_token_gen_active_softcap)

    target_flash_decoding = '''    def compute_for_flash_decoding(
        self, Q, K, V, past_key_value, attention_mask, active_mask
    ) -> Tensor:
        # TODO: refactor/decompose this to reduce duplication with compute_for_token_gen
        # active attention
        n_repeat = Q.shape[1]
        K_active = repeat_kv(K, n_repeat)
        V_active = repeat_kv(V, n_repeat)
        active_scores = (torch.matmul(Q, K_active.transpose(2, 3)) / self.softmax_scale).to(
            torch.float32
        )
        active_scores = torch.where(
            active_mask, active_scores, torch.finfo(active_scores.dtype).min
        )

        # prior attention
        K_prior = repeat_kv(past_key_value[0], n_repeat)
        V_prior = repeat_kv(past_key_value[1], n_repeat)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / self.softmax_scale
        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)'''

    repl_flash_decoding = '''    def compute_for_flash_decoding(
        self, Q, K, V, past_key_value, attention_mask, active_mask
    ) -> Tensor:
        # TODO: refactor/decompose this to reduce duplication with compute_for_token_gen
        # active attention
        n_repeat = Q.shape[1]
        K_active = repeat_kv(K, n_repeat)
        V_active = repeat_kv(V, n_repeat)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / self.softmax_scale
        softcap = getattr(self.config, "attn_logit_softcapping", None) or getattr(self.config, "attention_logit_cap", None) or 50.0
        if softcap is not None:
            active_scores = torch.tanh(active_scores / softcap) * softcap
        active_scores = active_scores.to(torch.float32)
        active_scores = torch.where(
            active_mask, active_scores, torch.finfo(active_scores.dtype).min
        )

        # prior attention
        K_prior = repeat_kv(past_key_value[0], n_repeat)
        V_prior = repeat_kv(past_key_value[1], n_repeat)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / self.softmax_scale
        if softcap is not None:
            prior_scores = torch.tanh(prior_scores / softcap) * softcap
        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)'''
    patch_file(attention_base_file, target_flash_decoding, repl_flash_decoding)


# 9. Patch vllm/transformers_utils/config.py to support nested gemma4 rope_parameters
vllm_config_file = '/opt/conda/lib/python3.12/site-packages/vllm/transformers_utils/config.py'
if os.path.exists(vllm_config_file):
    # Patch 9a: is_rope_parameters_nested support for full_attention/sliding_attention
    target_nested = '''def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:
    """Check if rope_parameters is nested by layer types."""
    # Cannot be nested if rope_parameters is empty
    if not rope_parameters:
        return False
    return set(rope_parameters.keys()).issubset(ALLOWED_ATTENTION_LAYER_TYPES)'''
    repl_nested = '''def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:
    """Check if rope_parameters is nested by layer types."""
    # Cannot be nested if rope_parameters is empty
    if not rope_parameters:
        return False
    if "full_attention" in rope_parameters or "sliding_attention" in rope_parameters:
        return True
    return set(rope_parameters.keys()).issubset(ALLOWED_ATTENTION_LAYER_TYPES)'''
    patch_file(vllm_config_file, target_nested, repl_nested)

    # Patch 9b: prevent legacy fields from polluting nested rope_parameters
    target_legacy = '''        # Patch legacy fields into rope_parameters
        if rope_theta is not None:
            config.rope_parameters["rope_theta"] = rope_theta
        if partial_rotary_factor is not None:
            config.rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        if ompe is not None:
            config.rope_parameters["original_max_position_embeddings"] = ompe'''
    repl_legacy = '''        # Patch legacy fields into rope_parameters
        if not is_rope_parameters_nested(getattr(config, "rope_parameters", None)):
            if rope_theta is not None:
                config.rope_parameters["rope_theta"] = rope_theta
            if partial_rotary_factor is not None:
                config.rope_parameters["partial_rotary_factor"] = partial_rotary_factor
            if ompe is not None:
                config.rope_parameters["original_max_position_embeddings"] = ompe'''
    patch_file(vllm_config_file, target_legacy, repl_legacy)

# 10. Patch vllm/model_executor/models/registry.py to register Gemma4UnifiedForConditionalGeneration as CausalLM
vllm_registry_file = '/opt/conda/lib/python3.12/site-packages/vllm/model_executor/models/registry.py'
if os.path.exists(vllm_registry_file):
    target_reg = '    "Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),  # noqa: E501'
    repl_reg = '''    "Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),  # noqa: E501
    "Gemma4ForConditionalGeneration": ("gemma3", "Gemma3ForCausalLM"),
    "Gemma4ForCausalLM": ("gemma3", "Gemma3ForCausalLM"),
    "Gemma4UnifiedForConditionalGeneration": ("gemma3", "Gemma3ForCausalLM"),'''
    patch_file(vllm_registry_file, target_reg, repl_reg)

# 11. Patch vllm/model_executor/layers/quantization/__init__.py to register neuron_quant
vllm_quant_init = '/opt/conda/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__init__.py'
if os.path.exists(vllm_quant_init):
    target_quant = '''__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]'''
    repl_quant = '''__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]

import torch
from typing import Any
@register_quantization_config("neuron_quant")
class NeuronQuantConfig(QuantizationConfig):
    def get_name(self) -> str:
        return "neuron_quant"
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]
    @classmethod
    def get_min_capability(cls) -> int:
        return 0
    @staticmethod
    def get_config_filenames() -> list[str]:
        return []
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NeuronQuantConfig":
        return cls()
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return None'''
    patch_file(vllm_quant_init, target_quant, repl_quant)

# 12. Patch transformers/tokenization_utils_base.py to support list in _set_model_specific_special_tokens
transformers_tok_base = '/opt/conda/lib/python3.12/site-packages/transformers/tokenization_utils_base.py'
if os.path.exists(transformers_tok_base):
    target_tok = '        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())'
    repl_tok = '''        if not hasattr(special_tokens, "keys") or not hasattr(special_tokens, "items"):
            if isinstance(special_tokens, (list, tuple)):
                new_dict = {}
                for item in special_tokens:
                    if isinstance(item, dict):
                        new_dict.update(item)
                    elif isinstance(item, str):
                        new_dict[item] = item
                    elif hasattr(item, "content"):
                        new_dict[getattr(item, "content")] = item
                    else:
                        new_dict[str(item)] = item
                special_tokens = new_dict
            else:
                special_tokens = {}
        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())'''
    patch_file(transformers_tok_base, target_tok, repl_tok)

# 13. Patch transformers/models/auto/image_processing_auto.py to support gemma4_unified
transformers_img_auto = '/opt/conda/lib/python3.12/site-packages/transformers/models/auto/image_processing_auto.py'
if os.path.exists(transformers_img_auto):
    target_img = '            ("gemma3", ("Gemma3ImageProcessor", "Gemma3ImageProcessorFast")),'
    repl_img = '''            ("gemma3", ("Gemma3ImageProcessor", "Gemma3ImageProcessorFast")),
            ("gemma4_unified", ("Gemma3ImageProcessor", "Gemma3ImageProcessorFast")),'''
    patch_file(transformers_img_auto, target_img, repl_img)

    target_debug = '''        raise ValueError(
            f"Unrecognized image processor in {pretrained_model_name_or_path}. Should have a "'''
    repl_debug = '''        print("DEBUG INFO FOR GEMMA4:")
        print("config:", config)
        print("type(config):", type(config))
        print("in mapping:", type(config) in IMAGE_PROCESSOR_MAPPING)
        print("mapping keys:", [k.__name__ for k in IMAGE_PROCESSOR_MAPPING.keys() if hasattr(k, "__name__")])
        raise ValueError(
            f"Unrecognized image processor in {pretrained_model_name_or_path}. Should have a "'''
    patch_file(transformers_img_auto, target_debug, repl_debug)

    target_cls_name = 'def get_image_processor_class_from_name(class_name: str):'
    repl_cls_name = '''def get_image_processor_class_from_name(class_name: str):
    if "Gemma4UnifiedImageProcessor" in class_name:
        class_name = class_name.replace("Gemma4UnifiedImageProcessor", "Gemma3ImageProcessor")'''
    patch_file(transformers_img_auto, target_cls_name, repl_cls_name)

# 14. Patch transformers/models/gemma3/processing_gemma3.py to handle missing tokenizer attributes
gemma3_processing_file = '/opt/conda/lib/python3.12/site-packages/transformers/models/gemma3/processing_gemma3.py'
if os.path.exists(gemma3_processing_file):
    target_proc = '''        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.image_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = f"\\n\\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\\n\\n"'''
    repl_proc = '''        self.image_seq_length = image_seq_length
        self.image_token_id = getattr(tokenizer, "image_token_id", getattr(tokenizer, "image_token_index", 258880))
        if self.image_token_id is None:
            self.image_token_id = 258880
        self.boi_token = getattr(tokenizer, "boi_token", "<|image>")
        self.image_token = getattr(tokenizer, "image_token", "<|image|>")
        self.eoi_token = getattr(tokenizer, "eoi_token", "<image|>")
        image_tokens_expanded = "".join([self.image_token] * image_seq_length)
        self.full_image_sequence = f"\\n\\n{self.boi_token}{image_tokens_expanded}{self.eoi_token}\\n\\n"'''
    patch_file(gemma3_processing_file, target_proc, repl_proc)

# 15. Patch transformers/models/gemma3/processing_gemma3.py to escape boi_token in re.finditer
if os.path.exists(gemma3_processing_file):
    target_escape = 'image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]'
    repl_escape = 'image_indexes = [m.start() for m in re.finditer(re.escape(self.boi_token), prompt)]'
    patch_file(gemma3_processing_file, target_escape, repl_escape)

# 16. Patch vllm/model_executor/models/gemma3_mm.py to support missing tokenizer.image_token
vllm_gemma3_mm = '/opt/conda/lib/python3.12/site-packages/vllm/model_executor/models/gemma3_mm.py'
if os.path.exists(vllm_gemma3_mm):
    target_mm = '        image_token_id = vocab[tokenizer.image_token]'
    repl_mm = '''        image_token = getattr(tokenizer, "image_token", None)
        if not isinstance(image_token, str):
            image_token = "<|image|>"
        image_token_id = vocab.get(image_token, 258880)'''
    patch_file(vllm_gemma3_mm, target_mm, repl_mm)

# 17. Patch modeling_gemma3.py to use global_head_dim for q_layernorm and k_layernorm - COMMENTED OUT as attention __init__ correctly computes layer-specific head_dim
gemma3_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py'
if os.path.exists(gemma3_file):
#     target_norm = '''        self.q_layernorm = get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps)
#         self.k_layernorm = get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps)'''
#     repl_norm = '''        self.q_layernorm = get_rmsnorm_cls()(hidden_size=getattr(config, "global_head_dim", head_dim), eps=config.rms_norm_eps)
#         self.k_layernorm = get_rmsnorm_cls()(hidden_size=getattr(config, "global_head_dim", head_dim), eps=config.rms_norm_eps)'''
#     patch_file(gemma3_file, target_norm, repl_norm)

    # 17b. Patch gemma3 load_config vocab_size hardcode to use text_config.vocab_size if available
    target_vocab = '        setattr(self, "vocab_size", 262208)'
    repl_vocab = '''        vocab = 262208
        if text_config is not None:
            vocab = getattr(text_config, "vocab_size", 262208)
        setattr(self, "vocab_size", vocab)'''
    patch_file(gemma3_file, target_vocab, repl_vocab)

# 18. Patch vllm/multimodal/processing/context.py to support Gemma4UnifiedConfig
vllm_context_file = '/opt/conda/lib/python3.12/site-packages/vllm/multimodal/processing/context.py'
if os.path.exists(vllm_context_file):
    target_ctx = '''        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, typ):
            raise TypeError(
                "Invalid type of HuggingFace config. "
                f"Expected type: {typ}, but "
                f"found type: {type(hf_config)}"
            )'''
    repl_ctx = '''        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, typ):
            if typ.__name__ == "Gemma3Config" and type(hf_config).__name__ == "Gemma4UnifiedConfig":
                pass
            else:
                raise TypeError(
                    "Invalid type of HuggingFace config. "
                    f"Expected type: {typ}, but "
                    f"found type: {type(hf_config)}"
                )'''
    patch_file(vllm_context_file, target_ctx, repl_ctx)

# 19. Patch configuration_gemma4_unified.py to add image_size property
gemma4_config_file = '/opt/conda/lib/python3.12/site-packages/transformers/models/gemma4_unified/configuration_gemma4_unified.py'
if os.path.exists(gemma4_config_file):
    target_cfg = '''    @property
    def model_patch_size(self):'''
    repl_cfg = '''    @property
    def image_size(self):
        return self.mm_posemb_size

    @property
    def model_patch_size(self):'''
    patch_file(gemma4_config_file, target_cfg, repl_cfg)

# 20. Patch trace.py to add torch.nn.Linear to supported sharded modules
trace_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed/trace/trace.py'
if os.path.exists(trace_file):
    trace_target = '''__SUPPORTED_SHARDED_MODULES = (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    OutputChannelParallelConv2d,
    InputChannelParallelConv2d,
    QuantizedRowParallel,
    QuantizedColumnParallel,
    BaseParallelLinear,
    SPMDRank
)'''
    trace_repl = '''import torch
__SUPPORTED_SHARDED_MODULES = (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    OutputChannelParallelConv2d,
    InputChannelParallelConv2d,
    QuantizedRowParallel,
    QuantizedColumnParallel,
    BaseParallelLinear,
    SPMDRank,
    torch.nn.Linear
)'''
    patch_file(trace_file, trace_target, trace_repl)

# 20b. Patch trace.py to fallback to checkpoint attributes for sharding
if os.path.exists(trace_file):
    trace_sharding_target = '''        if hasattr(module_parameter, "tensor_model_parallel") and module_parameter.tensor_model_parallel:
            partition_dim = module_parameter.partition_dim
            stride = module_parameter.partition_stride
            num_partitions = module_parameter.num_partitions
            per_partition_size = tensor.shape[partition_dim] // num_partitions
            partition_rank = rank % num_partitions'''
    trace_sharding_repl = '''        is_tensor_mp = (hasattr(module_parameter, "tensor_model_parallel") and module_parameter.tensor_model_parallel) or (hasattr(tensor, "tensor_model_parallel") and tensor.tensor_model_parallel)
        if is_tensor_mp:
            partition_dim = getattr(module_parameter, "partition_dim", getattr(tensor, "partition_dim", 0))
            stride = getattr(module_parameter, "partition_stride", getattr(tensor, "partition_stride", 1))
            num_partitions = getattr(module_parameter, "num_partitions", getattr(tensor, "num_partitions", 2))
            per_partition_size = tensor.shape[partition_dim] // num_partitions
            partition_rank = rank % num_partitions
            if "layers.30.mlp.gate_proj" in parameter_name or "layers.15.mlp.gate_proj" in parameter_name:
                print(f"[DEBUG_SHARD] param={parameter_name} tensor_shape={list(tensor.shape)} mod_param_shape={list(module_parameter.shape)} num_partitions={num_partitions} per_partition_size={per_partition_size} partition_dim={partition_dim} is_tensor_mp={is_tensor_mp}", flush=True)
                import traceback
                traceback.print_stack()'''
    patch_file(trace_file, trace_sharding_target, trace_sharding_repl)

# 20c. Patch trace.py to automatically pad checkpoint shape mismatches for hybrid attention head_dims
if os.path.exists(trace_file):
    trace_shape_target = '''        if checkpoint[parameter_name].shape != module_parameter.shape and not is_lora_cpu_shard:
            raise RuntimeError(f"expected shape {module_parameter.shape} for {parameter_name} but found {checkpoint[parameter_name].shape}")'''
    trace_shape_repl = '''        if checkpoint[parameter_name].shape != module_parameter.shape and not is_lora_cpu_shard:
            src_tensor = checkpoint[parameter_name]
            tgt_shape = list(module_parameter.shape)
            src_shape = list(src_tensor.shape)
            if len(tgt_shape) == len(src_shape):
                padded_tensor = torch.zeros(tgt_shape, dtype=src_tensor.dtype, device=src_tensor.device)
                slices = tuple(slice(0, min(t, s)) for t, s in zip(tgt_shape, src_shape))
                padded_tensor[slices] = src_tensor[slices]
                checkpoint[parameter_name] = padded_tensor
                print(f"[PATCH] Auto-padded mismatch for {parameter_name} from {src_shape} to {tgt_shape}", flush=True)
            else:
                raise RuntimeError(f"expected shape {module_parameter.shape} for {parameter_name} but found {checkpoint[parameter_name].shape}")'''
    patch_file(trace_file, trace_shape_target, trace_shape_repl)


# 21. Patch gqa.py to expose sharding attributes on CPU linear fallbacks
gqa_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/gqa.py'
if os.path.exists(gqa_file):
    gqa_target = '''        else:
            if self.fused_qkv:
                self.Wqkv = nn.Linear(
                    self.hidden_size,
                    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
                    bias=self.bias,
                )
            else:
                self.q_proj = nn.Linear(
                    self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.bias
                )
                self.k_proj = nn.Linear(
                    self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias
                )
                self.v_proj = nn.Linear(
                    self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias
                )'''
    gqa_repl = '''        else:
            if self.fused_qkv:
                self.Wqkv = nn.Linear(
                    self.hidden_size,
                    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
                    bias=self.bias,
                )
                for p_name in ["weight", "bias"]:
                    param = getattr(self.Wqkv, p_name, None)
                    if param is not None:
                        setattr(param, "tensor_model_parallel", True)
                        setattr(param, "partition_dim", 0)
                        setattr(param, "partition_stride", 1)
                        setattr(param, "num_partitions", tp_degree)
                        setattr(param, "fused_qkv", True)
                        setattr(param, "num_attention_heads", self.num_attention_heads)
                        setattr(param, "num_key_value_heads", self.num_key_value_heads)
                        setattr(param, "head_dim", self.head_dim)
            else:
                self.q_proj = nn.Linear(
                    self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.bias
                )
                self.k_proj = nn.Linear(
                    self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias
                )
                self.v_proj = nn.Linear(
                    self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias
                )
                for proj in [self.q_proj, self.k_proj, self.v_proj]:
                    for p_name in ["weight", "bias"]:
                        param = getattr(proj, p_name, None)
                        if param is not None:
                            setattr(param, "tensor_model_parallel", True)
                            setattr(param, "partition_dim", 0)
                            setattr(param, "partition_stride", 1)
                            setattr(param, "num_partitions", tp_degree)'''
    patch_file(gqa_file, gqa_target, gqa_repl)

# 22. Patch parallel ColumnParallelLinear inside GroupQueryAttention_QKV in gqa.py to expose sharding attributes
if os.path.exists(gqa_file):
    gqa_parallel_target = '''                self.v_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                    rank_ordering=rank_ordering,
                )'''
    gqa_parallel_repl = '''                self.v_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                    rank_ordering=rank_ordering,
                )
                for proj in [self.q_proj, self.k_proj, self.v_proj]:
                    for p_name in ["weight", "bias"]:
                        param = getattr(proj, p_name, None)
                        if param is not None:
                            setattr(param, "tensor_model_parallel", True)
                            setattr(param, "partition_dim", 0)
                            setattr(param, "partition_stride", 1)
                            setattr(param, "num_partitions", tp_degree)'''
    patch_file(gqa_file, gqa_parallel_target, gqa_parallel_repl)

# 23. Patch Gemma3Attention in transformers/models/gemma3/modeling_gemma3.py to support heterogeneous head dimensions on CPU
hf_gemma3_file = '/opt/conda/lib/python3.12/site-packages/transformers/models/gemma3/modeling_gemma3.py'
if os.path.exists(hf_gemma3_file):
    hf_gemma3_target = '''        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads'''
    hf_gemma3_repl = '''        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if not self.is_sliding:
            self.head_dim = getattr(config, "global_head_dim", 512)
        self.num_key_value_heads = (config.num_key_value_heads or config.num_attention_heads) if self.is_sliding else (getattr(config, "num_global_key_value_heads", 1) or 1)
        self.num_key_value_groups = config.num_attention_heads // self.num_key_value_heads'''
    patch_file(hf_gemma3_file, hf_gemma3_target, hf_gemma3_repl)

    hf_norm_target = '''        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)'''
    hf_norm_repl = '''        self.q_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)'''
    patch_file(hf_gemma3_file, hf_norm_target, hf_norm_repl)

    hf_proj_target = '''        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )'''
    hf_proj_repl = '''        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )'''
    patch_file(hf_gemma3_file, hf_proj_target, hf_proj_repl)

    # Patch 23b: Patch Gemma3DecoderLayer to support double-wide MLP on layers >= 15
    hf_mlp_target = '''        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)'''
    hf_mlp_repl = '''        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        if getattr(config, "use_double_wide_mlp", False) and layer_idx >= 15:
            import copy
            config = copy.deepcopy(config)
            config.intermediate_size = 12288
        self.mlp = Gemma3MLP(config)'''
    patch_file(hf_gemma3_file, hf_mlp_target, hf_mlp_repl)

# 24. Patch update_cache_const_indices in utils.py to pad updates to match cache head dimension
utils_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/utils.py'
if os.path.exists(utils_file):
    utils_target = '''def update_cache_const_indices(cache: torch.Tensor, updates: torch.Tensor, sequence_ids: Tensor):
    """
    Use constants for head and position indices, so that compiler just needs to compute the offset for batch dimension.
    This is needed to avoid inefficient DMAs, since compiler is not able to const-prop a constant address offset and treats it a dynamic offset.
    NCC-6227
    """
    max_batch_size, kv_heads, max_sequence_length, d_head = cache.shape
    batch_size, _, bucket_length, _ = updates.shape'''
    
    utils_repl = '''def update_cache_const_indices(cache: torch.Tensor, updates: torch.Tensor, sequence_ids: Tensor):
    """
    Use constants for head and position indices, so that compiler just needs to compute the offset for batch dimension.
    This is needed to avoid inefficient DMAs, since compiler is not able to const-prop a constant address offset and treats it a dynamic offset.
    NCC-6227
    """
    max_batch_size, kv_heads, max_sequence_length, d_head = cache.shape
    if updates.shape[-1] < d_head:
        updates = torch.nn.functional.pad(updates, (0, d_head - updates.shape[-1]))
    batch_size, _, bucket_length, _ = updates.shape'''
    patch_file(utils_file, utils_target, utils_repl)

    # 24b. Patch dynamic_update_slice in utils.py to pad updates to match cache/tensor head dimension
    utils_slice_target = '''def dynamic_update_slice(
    tensor: torch.Tensor, update: torch.Tensor, start_indices: List[torch.Tensor]
):'''
    utils_slice_repl = '''def dynamic_update_slice(
    tensor: torch.Tensor, update: torch.Tensor, start_indices: List[torch.Tensor]
):
    if update.shape[-1] < tensor.shape[-1]:
        update = torch.nn.functional.pad(update, (0, tensor.shape[-1] - update.shape[-1]))'''
    patch_file(utils_file, utils_slice_target, utils_slice_repl)

    # 24c. Patch pos_indices in update_cache_const_indices to wrap with modulo to avoid 1006 Memory Out-Of-Bounds
    utils_pos_target = '''    pos_indices = torch.arange(bucket_length).view(1, 1, -1).expand(batch_size, kv_heads, -1).to(torch.int32)'''
    utils_pos_repl = '''    pos_indices = torch.arange(bucket_length).view(1, 1, -1).expand(batch_size, kv_heads, -1)
    if bucket_length > max_sequence_length:
        pos_indices = pos_indices % max_sequence_length
    pos_indices = pos_indices.to(torch.int32)'''
    patch_file(utils_file, utils_pos_target, utils_pos_repl)


# 25. Patch _get_hidden_dim_per_head in kv_cache_manager.py and gpt_oss_kv_cache_manager.py
kv_mgr_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py'
if os.path.exists(kv_mgr_file):
    kv_target_dim = '''    def _get_hidden_dim_per_head(self, config: InferenceConfig):
        hidden_size = config.hidden_size
        num_atten_head = config.num_attention_heads
        hidden_dim_per_head = getattr(config, "head_dim", None) or hidden_size // num_atten_head
        return hidden_dim_per_head'''
    kv_repl_dim = '''    def _get_hidden_dim_per_head(self, config: InferenceConfig):
        hidden_size = config.hidden_size
        num_atten_head = config.num_attention_heads
        hidden_dim_per_head = getattr(config, "head_dim", None) or hidden_size // num_atten_head
        global_dim = None
        if hasattr(config, "text_config") and config.text_config is not None:
            global_dim = getattr(config.text_config, "global_head_dim", None)
        if global_dim is None:
            global_dim = getattr(config, "global_head_dim", None)
        if global_dim is not None:
            return global_dim
        return hidden_dim_per_head'''
    patch_file(kv_mgr_file, kv_target_dim, kv_repl_dim)

gpt_mgr_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/gpt_oss_kv_cache_manager.py'
if os.path.exists(gpt_mgr_file):
    gpt_target_dim = '''    def _get_hidden_dim_per_head(self, config: InferenceConfig):
        hidden_size = config.hidden_size
        num_atten_head = config.num_attention_heads
        hidden_dim_per_head = getattr(config, "head_dim", hidden_size // num_atten_head)
        return hidden_dim_per_head'''
    gpt_repl_dim = '''    def _get_hidden_dim_per_head(self, config: InferenceConfig):
        hidden_size = config.hidden_size
        num_atten_head = config.num_attention_heads
        hidden_dim_per_head = getattr(config, "head_dim", hidden_size // num_atten_head)
        global_dim = None
        if hasattr(config, "text_config") and config.text_config is not None:
            global_dim = getattr(config.text_config, "global_head_dim", None)
        if global_dim is None:
            global_dim = getattr(config, "global_head_dim", None)
        if global_dim is not None:
            return global_dim
        return hidden_dim_per_head'''
    patch_file(gpt_mgr_file, gpt_target_dim, gpt_repl_dim)

# 26. Patch get_kv_by_layer_id in managers to slice the returned cache from 512 to 256 for sliding-window layers
if os.path.exists(kv_mgr_file):
    kv_target_get = '''        if windowed_context_encoding_window_idx >= 1:
            if not self.sliding_window:
                k_cache = k_cache[:, :, 0 : windowed_context_encoding_window_idx * self.windowed_context_encoding_size, :]
                v_cache = v_cache[:, :, 0 : windowed_context_encoding_window_idx * self.windowed_context_encoding_size, :]
        return k_cache, v_cache'''
    kv_repl_get = '''        if windowed_context_encoding_window_idx >= 1:
            if not self.sliding_window:
                k_cache = k_cache[:, :, 0 : windowed_context_encoding_window_idx * self.windowed_context_encoding_size, :]
                v_cache = v_cache[:, :, 0 : windowed_context_encoding_window_idx * self.windowed_context_encoding_size, :]
        if (idx + 1) % 5 != 0:
            if k_cache.shape[-1] == 512:
                k_cache = k_cache[..., :256]
            if v_cache.shape[-1] == 512:
                v_cache = v_cache[..., :256]
        return k_cache, v_cache'''
    patch_file(kv_mgr_file, kv_target_get, kv_repl_get)

if os.path.exists(gpt_mgr_file):
    gpt_target_get = '''        # slice for partial view
        if not skip_slice:
            k_cache = _slice_kv_cacheline(self.padding_side, seq_len, k_cache, is_k_cache_transposed)
            v_cache = _slice_kv_cacheline(self.padding_side, seq_len, v_cache, False)

        return k_cache, v_cache'''
    gpt_repl_get = '''        # slice for partial view
        if not skip_slice:
            k_cache = _slice_kv_cacheline(self.padding_side, seq_len, k_cache, is_k_cache_transposed)
            v_cache = _slice_kv_cacheline(self.padding_side, seq_len, v_cache, False)

        if (idx + 1) % 5 != 0:
            if k_cache.shape[-1] == 512:
                k_cache = k_cache[..., :256]
            if v_cache.shape[-1] == 512:
                v_cache = v_cache[..., :256]
        return k_cache, v_cache'''
    patch_file(gpt_mgr_file, gpt_target_get, gpt_repl_get)

block_mgr_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/block_kv_cache_manager.py'
if os.path.exists(block_mgr_file):
    block_target_get = '''        else:
            raise ValueError("Can't find a proper way to read block KV cache.")

        return key_state, value_state'''
    block_repl_get = '''        else:
            raise ValueError("Can't find a proper way to read block KV cache.")

        if (idx + 1) % 5 != 0:
            if key_state.shape[-1] == 512:
                key_state = key_state[..., :256]
            if value_state.shape[-1] == 512:
                value_state = value_state[..., :256]
        return key_state, value_state'''
    patch_file(block_mgr_file, block_target_get, block_repl_get)

    block_target_update_kv = '''    def update_kv_by_layer_id(
        self,
        idx,
        kv_per_layer: List[Tensor],
        scatter_index=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

        # Quantize before writing to cache
        if self.kv_quant_config:
            latest_k = self._quantize_cache(latest_k, idx, is_key=True)
            latest_v = self._quantize_cache(latest_v, idx, is_key=False)

        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer=kvcache_buffer)
        slot_mapping = scatter_index
        k_cache = self._update_cache_into_block_layout(
            latest=latest_k,
            cache=k_cache,
            slot_mapping=slot_mapping,
        )
        v_cache = self._update_cache_into_block_layout(
            latest=latest_v,
            cache=v_cache,
            slot_mapping=slot_mapping,
        )
        return k_cache, v_cache'''

    block_repl_update_kv = '''    def update_kv_by_layer_id(
        self,
        idx,
        kv_per_layer: List[Tensor],
        scatter_index=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

        # Quantize before writing to cache
        if self.kv_quant_config:
            latest_k = self._quantize_cache(latest_k, idx, is_key=True)
            latest_v = self._quantize_cache(latest_v, idx, is_key=False)

        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer=kvcache_buffer)
        slot_mapping = scatter_index
        k_cache = self._update_cache_into_block_layout(
            latest=latest_k,
            cache=k_cache,
            slot_mapping=slot_mapping,
            layer_idx=idx,
        )
        v_cache = self._update_cache_into_block_layout(
            latest=latest_v,
            cache=v_cache,
            slot_mapping=slot_mapping,
            layer_idx=idx,
        )
        return k_cache, v_cache'''
    patch_file(block_mgr_file, block_target_update_kv, block_repl_update_kv)

    block_target_update = '''    def _update_cache_into_block_layout(self, latest, cache, slot_mapping, padding_id=-1):
        if self.is_prefix_caching:'''
    block_repl_update = '''    def _update_cache_into_block_layout(self, latest, cache, slot_mapping, padding_id=-1, layer_idx=None):
        if latest.shape[-1] < cache.shape[-1]:
            latest = torch.nn.functional.pad(latest, (0, cache.shape[-1] - latest.shape[-1]))
        if self.is_prefix_caching:'''
    patch_file(block_mgr_file, block_target_update, block_repl_update)

    block_target_tokengen = '''    row_indices = torch.arange(B, dtype=position_ids.dtype, device=position_ids.device)
    block_indices = (position_ids // block_size).squeeze(dim=1)'''
    block_repl_tokengen = '''    # Wrap position_ids to avoid out-of-bounds index access in block_table
    max_positions = block_table.shape[1] * block_size
    position_ids = position_ids % max_positions

    row_indices = torch.arange(B, dtype=position_ids.dtype, device=position_ids.device)
    block_indices = (position_ids // block_size).squeeze(dim=1)'''
    patch_file(block_mgr_file, block_target_tokengen, block_repl_tokengen)

    block_target_fusedspec = '''    relative_speculative_positions = torch.arange(speculation_length, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(dim=0)
    expanded_positions = position_ids + relative_speculative_positions

    row_indices = torch.arange(B, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(dim=1)'''
    block_repl_fusedspec = '''    relative_speculative_positions = torch.arange(speculation_length, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(dim=0)
    expanded_positions = position_ids + relative_speculative_positions

    # Wrap expanded_positions to avoid out-of-bounds index access in block_table
    max_positions = block_table.shape[1] * block_size
    expanded_positions = expanded_positions % max_positions

    row_indices = torch.arange(B, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(dim=1)'''
    patch_file(block_mgr_file, block_target_fusedspec, block_repl_fusedspec)

    block_target_index_put = '''        pad_dest_index = torch.tensor(num_blocks * block_size - 1, device=device, dtype=dtype)

        slot_mapping = torch.where(
            slot_mapping == padding_id,
            pad_dest_index,
            slot_mapping,
        )

        block_id = slot_mapping // self.pa_block_size'''

    block_repl_index_put = '''        pad_dest_index = torch.tensor(num_blocks * block_size - 1, device=device, dtype=dtype)

        slot_mapping = torch.where(
            slot_mapping == padding_id,
            pad_dest_index,
            slot_mapping,
        )

        # Wrap slot_mapping to avoid out-of-bounds index access in block_table/cache
        max_slots = num_blocks * block_size
        slot_mapping = slot_mapping % max_slots

        block_id = slot_mapping // self.pa_block_size'''
    patch_file(block_mgr_file, block_target_index_put, block_repl_index_put)

# 27. Patch sliding window attention checks in gpt_oss_kv_cache_manager.py
if os.path.exists(gpt_mgr_file):
    # Fix layer check
    patch_file(
        gpt_mgr_file,
        'is_swa_layer = layer % 2 == 0',
        'is_swa_layer = (layer + 1) % 5 != 0'
    )
    # Fix idx checks (there are two instances, we'll patch with AllowMultiple or single patches)
    # First idx % 2 == 0 check
    gpt_idx_target = '''        is_swa_layer = idx % 2 == 0
        is_k_cache_transposed = self.k_cache_transposed and not is_swa_layer'''
    gpt_idx_repl = '''        is_swa_layer = (idx + 1) % 5 != 0
        is_k_cache_transposed = self.k_cache_transposed and not is_swa_layer'''
    patch_file(gpt_mgr_file, gpt_idx_target, gpt_idx_repl)

    # Second idx % 2 == 0 check in update_kv_by_layer_id
    gpt_idx_update_target = '''        is_swa_layer = idx % 2 == 0
        is_k_cache_transposed = self.k_cache_transposed and not is_swa_layer
        dp_degree = self.swa_dp_degree if is_swa_layer else self.dp_degree

        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]'''
    gpt_idx_update_repl = '''        is_swa_layer = (idx + 1) % 5 != 0
        is_k_cache_transposed = self.k_cache_transposed and not is_swa_layer
        dp_degree = self.swa_dp_degree if is_swa_layer else self.dp_degree

        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]
        if latest_k.shape[-1] < 512:
            latest_k = torch.nn.functional.pad(latest_k, (0, 512 - latest_k.shape[-1]))
        if latest_v.shape[-1] < 512:
            latest_v = torch.nn.functional.pad(latest_v, (0, 512 - latest_v.shape[-1]))'''
    patch_file(gpt_mgr_file, gpt_idx_update_target, gpt_idx_update_repl)

    # Fix layer_idx check in _get_index_to_update_new_position with robust physical size modulo
    gpt_get_index_target_1 = '''    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int):
        is_swa_layer = layer_idx % 2 == 0
        if is_swa_layer:
            position_ids = position_ids % (self.sliding_window)
        index = position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)'''

    gpt_get_index_target_2 = '''    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int):
        is_swa_layer = (layer_idx + 1) % 5 != 0
        if is_swa_layer:
            position_ids = position_ids % (self.sliding_window)
        index = position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)'''

    gpt_get_index_repl = '''    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int):
        is_swa_layer = (layer_idx + 1) % 5 != 0
        cache_shape = self.k_shapes[layer_idx] if (hasattr(self, "k_shapes") and self.k_shapes) else self.k_shape
        seq_dim_size = cache_shape[-1] if transposed else cache_shape[-2]
        if is_swa_layer:
            limit = min(self.sliding_window, seq_dim_size)
            position_ids = position_ids % limit
        else:
            position_ids = position_ids % seq_dim_size
        index = position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)'''
    gpt_get_index_target_old = '''    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int):
        is_swa_layer = (layer_idx + 1) % 5 != 0
        seq_dim_size = full_k.shape[-1] if transposed else full_k.shape[-2]
        if is_swa_layer:
            limit = min(self.sliding_window, seq_dim_size)
            position_ids = position_ids % limit
        else:
            position_ids = position_ids % seq_dim_size
        index = position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)'''
    if not patch_file(gpt_mgr_file, gpt_get_index_target_1, gpt_get_index_repl):
        if not patch_file(gpt_mgr_file, gpt_get_index_target_2, gpt_get_index_repl):
            patch_file(gpt_mgr_file, gpt_get_index_target_old, gpt_get_index_repl)

# 28. Patch early padding in kv_cache_manager.py
if os.path.exists(kv_mgr_file):
    kv_update_target = '''    def update_kv_by_layer_id(
        self,
        idx,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        kv_per_layer: Tuple[Tensor, Tensor],
        seq_len: int,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        windowed_context_encoding_window_idx: int = -1,
        is_valid_window_kv: Tensor = None,
        **kwargs,
    ):
        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]'''
    
    kv_update_repl = '''    def update_kv_by_layer_id(
        self,
        idx,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        kv_per_layer: Tuple[Tensor, Tensor],
        seq_len: int,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        windowed_context_encoding_window_idx: int = -1,
        is_valid_window_kv: Tensor = None,
        **kwargs,
    ):
        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]
        if latest_k.shape[-1] < 512:
            latest_k = torch.nn.functional.pad(latest_k, (0, 512 - latest_k.shape[-1]))
        if latest_v.shape[-1] < 512:
            latest_v = torch.nn.functional.pad(latest_v, (0, 512 - latest_v.shape[-1]))
        is_swa_layer = (idx + 1) % 5 != 0
        if is_swa_layer and self.sliding_window:
            seq_len_dim = 2 if not self.k_cache_transposed else 3
            current_seq_len = latest_k.shape[seq_len_dim]
            if current_seq_len > self.sliding_window:
                if not self.k_cache_transposed:
                    latest_k = latest_k[:, :, -self.sliding_window:, :]
                else:
                    latest_k = latest_k[:, :, :, -self.sliding_window:]
                latest_v = latest_v[:, :, -self.sliding_window:, :]
                position_ids = position_ids[:, -self.sliding_window:]
                if scatter_index is not None:
                    scatter_index = scatter_index[:, -self.sliding_window:]'''
    patch_file(kv_mgr_file, kv_update_target, kv_update_repl)

    # Patch sliding window position ID check in kv_cache_manager.py to avoid 1006 memory out-of-bounds in prefill
    kv_swa_target = '''        elif self.sliding_window:
            position_ids = position_ids % (self.sliding_window - 1)'''

    kv_swa_repl = '''        elif self.sliding_window:
            is_swa_layer = (layer_idx + 1) % 5 != 0
            cache_shape = self.k_shapes[layer_idx] if (hasattr(self, "k_shapes") and self.k_shapes) else self.k_shape
            seq_dim_size = cache_shape[-1] if transposed else cache_shape[-2]
            if is_swa_layer:
                limit = min(self.sliding_window, seq_dim_size)
                position_ids = position_ids % limit
            else:
                position_ids = position_ids % seq_dim_size'''
    kv_swa_target_old = '''        elif self.sliding_window:
            is_swa_layer = (layer_idx + 1) % 5 != 0
            seq_dim_size = full_k.shape[-1] if transposed else full_k.shape[-2]
            if is_swa_layer:
                limit = min(self.sliding_window, seq_dim_size)
                position_ids = position_ids % limit
            else:
                position_ids = position_ids % seq_dim_size'''
    if not patch_file(kv_mgr_file, kv_swa_target, kv_swa_repl):
        patch_file(kv_mgr_file, kv_swa_target_old, kv_swa_repl)

    # =========================================================================
    # 31. CRITICAL FIX: 1006 DGE scatter fault in context_encoding_model (prefill)
    #
    # Root cause (confirmed via container source inspection):
    #   fill_prefix() calls DynamicUpdateSlice(tensor, update, 0, 0, 0, 0).
    #   HLO requires update.shape[i] <= tensor.shape[i] for ALL dims.
    #   For SWA layers: cache.shape[2] = sliding_window (512), but
    #   latest_k.shape[2] = prefill bucket_length (e.g. 1024) → 1006 OOB fault.
    #   This is baked into the compiled NEFF at XLA trace time — Python-level
    #   runtime modulo guards in patch 24c DO NOT help because fill_prefix
    #   bypasses update_cache_const_indices entirely, going straight to HLO.
    #
    # Fix: Patch fill_prefix itself to clip prefix_cache's seq dim (dim=-2) and
    # head_dim (dim=-1) to match cache capacity BEFORE the HLO is traced.
    # This is the same approach already used in the 4B Inferentia agent.
    # =========================================================================
    utils_fill_target = '''def fill_prefix(cache, prefix_cache):
    if cpu_mode():'''
    utils_fill_repl = '''def fill_prefix(cache, prefix_cache):
    # PATCH 31: Clip head_dim (dim=-1) to match cache — SWA layers have 256, cache allocated for 512
    if prefix_cache.shape[-1] < cache.shape[-1]:
        prefix_cache = torch.nn.functional.pad(prefix_cache, (0, cache.shape[-1] - prefix_cache.shape[-1]))
    if prefix_cache.shape[-1] > cache.shape[-1]:
        prefix_cache = prefix_cache[..., :cache.shape[-1]]
    # PATCH 31: Clip sequence dimension (dim=-2) to cache capacity.
    # Prevents 1006 OOB in context_encoding_model NEFF (_bk0, _bk1) when
    # prefill bucket_length > SWA cache max_sequence_length (sliding_window).
    # DynamicUpdateSlice requires update.shape[i] <= tensor.shape[i] for all dims.
    if prefix_cache.ndim >= 3 and prefix_cache.shape[-2] > cache.shape[-2]:
        prefix_cache = prefix_cache[..., :cache.shape[-2], :]
    if cpu_mode():'''
    patch_file(utils_file, utils_fill_target, utils_fill_repl)

    # 29. Patch gpt_oss_kv_cache_manager.py update_kv_by_layer_id to slice sequence inputs to SWA window size
    if os.path.exists(gpt_mgr_file):
        gpt_update_target = '''        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)'''

        gpt_update_repl = '''        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]
        if latest_k.shape[-1] < 512:
            latest_k = torch.nn.functional.pad(latest_k, (0, 512 - latest_k.shape[-1]))
        if latest_v.shape[-1] < 512:
            latest_v = torch.nn.functional.pad(latest_v, (0, 512 - latest_v.shape[-1]))
        if is_swa_layer and self.sliding_window:
            seq_len_dim = 2 if not is_k_cache_transposed else 3
            current_seq_len = latest_k.shape[seq_len_dim]
            if current_seq_len > self.sliding_window:
                if not is_k_cache_transposed:
                    latest_k = latest_k[:, :, -self.sliding_window:, :]
                else:
                    latest_k = latest_k[:, :, :, -self.sliding_window:]
                latest_v = latest_v[:, :, -self.sliding_window:, :]
                position_ids = position_ids[:, -self.sliding_window:]
                if scatter_index is not None:
                    scatter_index = scatter_index[:, -self.sliding_window:]

        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)'''
        patch_file(gpt_mgr_file, gpt_update_target, gpt_update_repl)

    # 30. Patch get_last_kv_window in utils.py to return immediately if seq_len <= window_size
    utils_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/utils.py'
    if os.path.exists(utils_file):
        utils_target = '''def get_last_kv_window(window_size, position_ids, latest_k, latest_v, windowed_context_encoding_window_idx=-1, spec_len=0):
    batch_size, num_head, _, head_dim = latest_k.shape'''
        utils_repl = '''def get_last_kv_window(window_size, position_ids, latest_k, latest_v, windowed_context_encoding_window_idx=-1, spec_len=0):
    batch_size, num_head, seq_len, head_dim = latest_k.shape
    if seq_len <= window_size:
        return latest_k, latest_v'''
        patch_file(utils_file, utils_target, utils_repl)

# 32. Patch vLLM serving to prevent list index out of range for logprobs
vllm_serving_file = '/opt/conda/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/serving.py'
if os.path.exists(vllm_serving_file):
    logprobs_target = '''        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:'''
    logprobs_repl = '''        for i, token_id in enumerate(token_ids):
            if i >= len(top_logprobs):
                continue
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:'''
    patch_file(vllm_serving_file, logprobs_target, logprobs_repl)

    # Disable NKI attention kernels to force stable PyTorch/XLA sliding window attention path
    print("Disabling attn_block_tkg_nki_kernel_enabled to stabilize attention routing...")
    patch_file(gpt_mgr_file, 'attn_kernel_enabled = self.neuron_config.attn_block_tkg_nki_kernel_enabled', 'attn_kernel_enabled = False')
    
    kv_mgr_file_path = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py'
    if os.path.exists(kv_mgr_file_path):
        patch_file(kv_mgr_file_path, 'attn_kernel_enabled = self.neuron_config.attn_block_tkg_nki_kernel_enabled', 'attn_kernel_enabled = False')
        
    patch_file(attention_base_file, 'self.attn_block_tkg_nki_kernel_enabled = self.neuron_config.attn_block_tkg_nki_kernel_enabled', 'self.attn_block_tkg_nki_kernel_enabled = False')

    # 33. Patch RotaryEmbedding in attention/utils.py to scale frequencies by head_dim
    if os.path.exists(utils_file):
        rope_target = '''    def __init__(self, dim, max_position_embeddings=2048, base=10000, factor : float = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)
        self.factor = factor

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
        inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))'''
        rope_repl = '''    def __init__(self, dim, max_position_embeddings=2048, base=10000, factor : float = None, head_dim = None, partial_rotary_factor = 1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)
        self.factor = factor
        self.head_dim = head_dim or dim
        self.partial_rotary_factor = partial_rotary_factor

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        rope_angles = int(self.partial_rotary_factor * self.head_dim // 2)
        inv_freq_rotated = 1.0 / (
            self.base
            ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float, device=device) / self.head_dim)
        )
        if self.factor is not None:
            inv_freq_rotated = inv_freq_rotated / self.factor

        nope_angles = self.head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                (
                    inv_freq_rotated,
                    torch.zeros(nope_angles, dtype=torch.float, device=device),
                ),
                dim=0,
            )
        else:
            inv_freq = inv_freq_rotated
        return inv_freq'''
        patch_file(utils_file, rope_target, rope_repl)

        # Patch 34: Reset cos/sin cache when head_dim changes between local/global layers in model_base.py
        model_base_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/model_base.py'
        loop_target = '''        for idx, decoder_layer in enumerate(self.layers):
            if self.config.neuron_config.layer_boundary_markers:
                hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer('''
        loop_repl = '''        for idx, decoder_layer in enumerate(self.layers):
            if self.config.neuron_config.layer_boundary_markers:
                hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Reset cos/sin cache if head dimension mismatches the layer's expected head_dim (hybrid attention)
            if cos_cache is not None:
                expected_head_dim = getattr(getattr(decoder_layer, "self_attn", None), "head_dim", None)
                if expected_head_dim is not None and cos_cache.shape[-1] != expected_head_dim:
                    cos_cache, sin_cache = None, None

            layer_outputs = decoder_layer('''
        patch_file(model_base_file, loop_target, loop_repl)

        # Patch 36: Safe check for past_key_values bound inside model_base.py for shared layers to avoid XLA lower errors
        loop_target2 = '''            past_key_value = past_key_values[idx] if past_key_values is not None else None'''
        loop_repl2 = '''            if past_key_values is not None and idx < len(past_key_values):
                past_key_value = past_key_values[idx]
            else:
                past_key_value = None'''
        patch_file(model_base_file, loop_target2, loop_repl2)

        # Patch 37: Return unmodified cache for idx >= 20 to bypass updates without returning duplicates
        mgr_files = [
            '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py',
            '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/gpt_oss_kv_cache_manager.py',
            '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/block_kv_cache_manager.py'
        ]
        for mgr_f in mgr_files:
            if not os.path.exists(mgr_f):
                continue
            update_target = '''    def update_kv_by_layer_id('''
            update_repl = '''    def update_kv_by_layer_id(
        self,
        idx,
        *args,
        **kwargs,
    ):
        import os
        import torch
        if (os.environ.get("IS_NEURON_COMPILING") == "1") or torch.jit.is_tracing():
            return self.update_kv_by_layer_id_original(idx, *args, **kwargs)
        if idx >= 20:
            is_swa_layer = (idx + 1) % 5 != 0
            source_idx = 18 if is_swa_layer else 19
            return self.update_kv_by_layer_id_original(source_idx, *args, **kwargs)
        return self.update_kv_by_layer_id_original(idx, *args, **kwargs)

    def update_kv_by_layer_id_original('''
            patch_file(mgr_f, update_target, update_repl)

        # Patch 37b: Redirect get_kv_by_layer_id for shared layers >= 20 to physical layers 18 or 19
        for mgr_f in mgr_files:
            if not os.path.exists(mgr_f):
                continue
            get_target = '''    def get_kv_by_layer_id('''
            get_repl = '''    def get_kv_by_layer_id(
        self,
        idx,
        *args,
        **kwargs,
    ):
        import os
        import torch
        if (os.environ.get("IS_NEURON_COMPILING") == "1") or torch.jit.is_tracing():
            return self.get_kv_by_layer_id_original(idx, *args, **kwargs)
        if idx >= 20:
            is_swa_layer = (idx + 1) % 5 != 0
            source_idx = 18 if is_swa_layer else 19
            return self.get_kv_by_layer_id_original(source_idx, *args, **kwargs)
        return self.get_kv_by_layer_id_original(idx, *args, **kwargs)

    def get_kv_by_layer_id_original('''
            patch_file(mgr_f, get_target, get_repl)

        # Patch 38: Intercept compile in application_base.py to set/unset IS_NEURON_COMPILING env var
        app_base_f = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/application_base.py'
        app_base_target = '''    def compile('''
        app_base_repl = '''    def compile(
        self,
        compiled_model_path,
        debug=False,
        pre_shard_weights_hook=None,
        dry_run=False,
        disable_fail_fast=False,
    ):
        import os
        os.environ["IS_NEURON_COMPILING"] = "1"
        try:
            return self.compile_original(compiled_model_path, debug, pre_shard_weights_hook, dry_run, disable_fail_fast)
        finally:
            os.environ.pop("IS_NEURON_COMPILING", None)

    def compile_original('''
        patch_file(app_base_f, app_base_target, app_base_repl)

    # 39. Patch modeling_gemma3.py to implement Per-Layer Embeddings (PLE)
    gemma3_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py'
    if os.path.exists(gemma3_file):
        # 39a. Add hidden_size_per_layer_input and vocab_size_per_layer_input to config attributes
        gemma3_attr_target = '''        self.attributes = [
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "query_pre_attn_scalar",
            "sliding_window",
            "final_logit_softcapping",
            "attn_logit_softcapping",
            "use_double_wide_mlp",
        ]'''
        gemma3_attr_repl = '''        self.attributes = [
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "query_pre_attn_scalar",
            "sliding_window",
            "final_logit_softcapping",
            "attn_logit_softcapping",
            "use_double_wide_mlp",
            "hidden_size_per_layer_input",
            "vocab_size_per_layer_input",
        ]'''
        patch_file(gemma3_file, gemma3_attr_target, gemma3_attr_repl)

        # 39b. Update NeuronGemma3DecoderLayer constructor to accept parent and init PLE layers
        gemma3_layer_init_target = '''class NeuronGemma3DecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Gemma3InferenceConfig, layer_idx: int):
        super().__init__()

        self.is_sliding_window_attention = config.sliding_window is not None and (layer_idx + 1) % 5 != 0
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size'''
        gemma3_layer_init_repl = '''class NeuronGemma3DecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Gemma3InferenceConfig, layer_idx: int, parent=None):
        super().__init__()
        self.__dict__['parent'] = parent

        self.is_sliding_window_attention = config.sliding_window is not None and (layer_idx + 1) % 5 != 0
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act] if hasattr(config, "hidden_act") else ACT2FN[getattr(config, "hidden_activation", "gelu_pytorch_tanh")]
        hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", None)
        if hidden_size_per_layer_input is not None:
            self.per_layer_input_gate = nn.Linear(self.hidden_size, hidden_size_per_layer_input, bias=False, dtype=config.neuron_config.torch_dtype)
            self.per_layer_projection = nn.Linear(hidden_size_per_layer_input, self.hidden_size, bias=False, dtype=config.neuron_config.torch_dtype)
            self.post_per_layer_input_norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None'''
        patch_file(gemma3_file, gemma3_layer_init_target, gemma3_layer_init_repl)

        # 39c. Update layers instantiation in NeuronGemma3TextModel.init_model
        gemma3_instantiate_target = '''        self.layers = nn.ModuleList(
            [NeuronGemma3DecoderLayer(conf, idx) for idx, conf in enumerate(updated_configs)]
        )'''
        gemma3_instantiate_repl = '''        self.layers = nn.ModuleList(
            [NeuronGemma3DecoderLayer(conf, idx, parent=self) for idx, conf in enumerate(updated_configs)]
        )'''
        patch_file(gemma3_file, gemma3_instantiate_target, gemma3_instantiate_repl)

        # 39d. Update PLE layers initialization inside init_model
        gemma3_model_init_target = '''        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [NeuronGemma3DecoderLayer(conf, idx, parent=self) for idx, conf in enumerate(updated_configs)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)'''
        gemma3_model_init_repl = '''        # PLE layers
        hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", None)
        vocab_size_per_layer_input = getattr(config, "vocab_size_per_layer_input", None)
        if hidden_size_per_layer_input is not None and vocab_size_per_layer_input is not None:
            self.embed_tokens_per_layer = ParallelEmbedding(
                vocab_size_per_layer_input,
                config.num_hidden_layers * hidden_size_per_layer_input,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * hidden_size_per_layer_input,
                bias=False,
                dtype=config.neuron_config.torch_dtype,
            )
            self.per_layer_projection_norm = get_rmsnorm_cls()(
                hidden_size_per_layer_input,
                eps=config.rms_norm_eps,
            )
            self.register_buffer("per_layer_projection_scale", torch.tensor(config.hidden_size**-0.5), persistent=False)
            self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [NeuronGemma3DecoderLayer(conf, idx, parent=self) for idx, conf in enumerate(updated_configs)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)'''
        patch_file(gemma3_file, gemma3_model_init_target, gemma3_model_init_repl)

        # 39e. Insert forward method in NeuronGemma3TextModel
        gemma3_model_forward_target = '''    def setup_attr_for_model(self, config: Gemma3InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets'''
        gemma3_model_forward_repl = '''    def setup_attr_for_model(self, config: Gemma3InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds=None,
        kv_cache=None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
        deepstack_vision_embeds=None,
    ):
        hidden_size_per_layer_input = getattr(self.config, "hidden_size_per_layer_input", None)
        vocab_size_per_layer_input = getattr(self.config, "vocab_size_per_layer_input", None)
        if hidden_size_per_layer_input is not None and vocab_size_per_layer_input is not None:
            actual_embeds = inputs_embeds
            if actual_embeds is None or actual_embeds.numel() == 0:
                actual_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.embed_tokens_per_layer(input_ids)
            per_layer_inputs = per_layer_inputs.reshape(
                *input_ids.shape,
                self.config.num_hidden_layers,
                hidden_size_per_layer_input
            )
            per_layer_projection = self.per_layer_model_projection(actual_embeds)
            per_layer_projection *= self.per_layer_projection_scale.to(actual_embeds.dtype)
            per_layer_projection = per_layer_projection.reshape(
                *actual_embeds.shape[:-1],
                self.config.num_hidden_layers,
                hidden_size_per_layer_input
            )
            per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
            self.current_per_layer_inputs = (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(actual_embeds.dtype)
        else:
            self.current_per_layer_inputs = None

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            prev_hidden=prev_hidden,
            adapter_ids=adapter_ids,
            accepted_indices=accepted_indices,
            current_length=current_length,
            medusa_mask=medusa_mask,
            scatter_index=scatter_index,
            slot_mapping=slot_mapping,
            active_block_table=active_block_table,
            num_queries=num_queries,
            computed_context_lens=computed_context_lens,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
            active_mask=active_mask,
            rotary_position_id=rotary_position_id,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            deepstack_vision_embeds=deepstack_vision_embeds,
        )'''
        patch_file(gemma3_file, gemma3_model_forward_target, gemma3_model_forward_repl)

        # 39f. Patch NeuronGemma3DecoderLayer.forward to use PLE gating
        gemma3_layer_forward_target = '''        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # # End module marker
        # hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)'''
        gemma3_layer_forward_repl = '''        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        
        per_layer_inputs = getattr(self.parent, "current_per_layer_inputs", None)
        if per_layer_inputs is not None and self.per_layer_input_gate is not None:
            per_layer_input = per_layer_inputs[:, :, self.layer_idx, :]
            ple_signal = self.per_layer_input_gate(hidden_states)
            ple_signal = self.act_fn(ple_signal)
            ple_signal = torch.multiply(ple_signal, per_layer_input)
            ple_signal = self.per_layer_projection(ple_signal)
            ple_signal = self.post_per_layer_input_norm(ple_signal)
            hidden_states = hidden_states + ple_signal

        # # End module marker
        # hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)'''
        patch_file(gemma3_file, gemma3_layer_forward_target, gemma3_layer_forward_repl)

        # 39g. Patch convert_hf_to_neuron_state_dict to map PLE weights
        gemma3_state_dict_target = '''        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        neuron_config = config.neuron_config'''
        gemma3_state_dict_repl = '''        # Dynamically inject PLE keys from local model.safetensors to bypass HF filtering
        try:
            import glob
            from safetensors import safe_open
            snap_dirs = glob.glob('/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*')
            if snap_dirs:
                safetensors_file = glob.glob(snap_dirs[0] + "/*.safetensors")[0]
                print(f"LOADING PLE KEYS FROM {safetensors_file}...")
                with safe_open(safetensors_file, framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        if "per_layer" in k or "embed_tokens_per_layer" in k:
                            state_dict[k] = sf.get_tensor(k)
                print(f"SUCCESSFULLY LOADED PLE KEYS. TOTAL KEYS NOW: {len(state_dict)}")
            else:
                print("WARNING: NO SNAPSHOT DIRECTORY FOUND FOR PLE KEY INJECTION!")
        except Exception as e:
            print(f"ERROR DURING PLE KEY INJECTION: {e}")

        # Remove both model. and model.language_model. / language_model. prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("model.language_model."):
                new_key = new_key.removeprefix("model.language_model.")
            elif new_key.startswith("language_model."):
                new_key = new_key.removeprefix("language_model.")
            elif new_key.startswith("model."):
                new_key = new_key.removeprefix("model.")
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        neuron_config = config.neuron_config
        print("DEBUG STATE DICT ALL KEYS CONTAINS EMBED:", [k for k in state_dict.keys() if "embed" in k])
        print("DEBUG STATE DICT ALL KEYS CONTAINS NORM:", [k for k in state_dict.keys() if "norm" in k])
        print("DEBUG STATE DICT PLE KEYS BEFORE:", [k for k in state_dict.keys() if "per_layer" in k or "embed_tokens_per_layer" in k])
        if "embed_tokens_per_layer.weight" in state_dict:
            if neuron_config.vocab_parallel:
                state_dict["embed_tokens_per_layer.rank_util.rank"] = torch.arange(
                    0, neuron_config.local_ranks_size
                )
        if "per_layer_projection_norm.weight" in state_dict:
            state_dict["per_layer_projection_norm.weight"] += 1.0
        print("DEBUG STATE DICT PLE KEYS AFTER:", [k for k in state_dict.keys() if "per_layer" in k or "embed_tokens_per_layer" in k])'''
        patch_file(gemma3_file, gemma3_state_dict_target, gemma3_state_dict_repl)

        # 39h. Patch convert_hf_to_neuron_state_dict layer loop to apply post_per_layer_input_norm offset
        gemma3_state_dict_layer_target = '''            state_dict[f"layers.{i}.input_layernorm.weight"] += 1.0
            state_dict[f"layers.{i}.post_attention_layernorm.weight"] += 1.0
            state_dict[f"layers.{i}.post_feedforward_layernorm.weight"] += 1.0
            state_dict[f"layers.{i}.pre_feedforward_layernorm.weight"] += 1.0'''
        gemma3_state_dict_layer_repl = '''            state_dict[f"layers.{i}.input_layernorm.weight"] += 1.0
            state_dict[f"layers.{i}.post_attention_layernorm.weight"] += 1.0
            state_dict[f"layers.{i}.post_feedforward_layernorm.weight"] += 1.0
            state_dict[f"layers.{i}.pre_feedforward_layernorm.weight"] += 1.0
            if f"layers.{i}.post_per_layer_input_norm.weight" in state_dict:
                state_dict[f"layers.{i}.post_per_layer_input_norm.weight"] += 1.0'''
        patch_file(gemma3_file, gemma3_state_dict_layer_target, gemma3_state_dict_layer_repl)



