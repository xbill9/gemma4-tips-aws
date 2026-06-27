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
                setattr(self, attribute, val)'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_target,
    gemma3_repl
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
            swa_layer = (i + 1) % 6 != 0
            is_fused_layer = config.neuron_config.fused_qkv and swa_layer

            if not swa_layer:
                # Pad global layer key/value weights from 512 (1 global head * 512 dim) to 4096 (8 heads * 512 dim)
                for proj in ["k_proj", "v_proj"]:
                    key = f"layers.{i}.self_attn.{proj}.weight"
                    if key in state_dict:
                        weight = state_dict[key]
                        if weight.shape[0] < 4096:
                            padded_weight = torch.zeros((4096, weight.shape[1]), dtype=weight.dtype, device=weight.device)
                            padded_weight[:weight.shape[0], :] = weight
                            state_dict[key] = padded_weight

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
                        state_dict[f"layers.{i}.self_attn.qkv_proj.{proj}.weight"] = weight'''

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
        if layer_idx is not None and (layer_idx + 1) % 6 == 0:
            head_dim = getattr(config, "global_head_dim", 512)'''

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
        if config.sliding_window is None or (layer_idx is not None and (layer_idx + 1) % 6 == 0):
            rotary_emb = global_rotary_emb'''

patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_rotary_target,
    gemma3_rotary_repl
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
            max_dim = max(Q.shape[-1], K.shape[-1])
            if cos_cache is None or sin_cache is None or cos_cache.shape[-1] < max_dim:
                if Q.shape[-1] == max_dim:
                    cos_cache, sin_cache = self.rotary_emb(Q, position_ids)
                else:
                    cos_cache, sin_cache = self.rotary_emb(K, position_ids)
            from .utils import _rotate_half
            q_cos = cos_cache[..., :Q.shape[-1]]
            q_sin = sin_cache[..., :Q.shape[-1]]
            k_cos = cos_cache[..., :K.shape[-1]]
            k_sin = sin_cache[..., :K.shape[-1]]
            cos_q = q_cos.unsqueeze(1)
            sin_q = q_sin.unsqueeze(1)
            Q = (Q * cos_q) + (_rotate_half(Q) * sin_q)
            cos_k = k_cos.unsqueeze(1)
            sin_k = k_sin.unsqueeze(1)
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

    # Patch attn_output to slice back from 512 to 256 before merge multi head hidden
    target_merge = '''        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)'''
    repl_merge = '''        if attn_output.shape[-1] == 512 and self.head_dim == 256:
            attn_output = attn_output[..., :256]
        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)'''
    patch_file(attention_base_file, target_merge, repl_merge)

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
                V = V[..., :256]'''
    patch_file(attention_base_file, target_tokengen, repl_tokengen)


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
            partition_rank = rank % num_partitions'''
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
        self.num_key_value_heads = config.num_key_value_heads if self.is_sliding else getattr(config, "num_global_key_value_heads", 1)
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
        if (idx + 1) % 6 != 0:
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

        if (idx + 1) % 6 != 0:
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

        if (idx + 1) % 6 != 0:
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
        if layer_idx is not None and (layer_idx + 1) % 6 != 0:
            slot_mapping = torch.where(
                slot_mapping == padding_id,
                slot_mapping,
                slot_mapping % 512
            )
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
        'is_swa_layer = (layer + 1) % 6 != 0'
    )
    # Fix idx checks (there are two instances, we'll patch with AllowMultiple or single patches)
    # First idx % 2 == 0 check
    gpt_idx_target = '''        is_swa_layer = idx % 2 == 0
        is_k_cache_transposed = self.k_cache_transposed and not is_swa_layer'''
    gpt_idx_repl = '''        is_swa_layer = (idx + 1) % 6 != 0
        is_k_cache_transposed = self.k_cache_transposed and not is_swa_layer'''
    patch_file(gpt_mgr_file, gpt_idx_target, gpt_idx_repl)

    # Second idx % 2 == 0 check in update_kv_by_layer_id
    gpt_idx_update_target = '''        is_swa_layer = idx % 2 == 0
        is_k_cache_transposed = self.k_cache_transposed and not is_swa_layer
        dp_degree = self.swa_dp_degree if is_swa_layer else self.dp_degree

        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]'''
    gpt_idx_update_repl = '''        is_swa_layer = (idx + 1) % 6 != 0
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
        is_swa_layer = (layer_idx + 1) % 6 != 0
        if is_swa_layer:
            position_ids = position_ids % (self.sliding_window)
        index = position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)'''

    gpt_get_index_repl = '''    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int):
        is_swa_layer = (layer_idx + 1) % 6 != 0
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
        patch_file(gpt_mgr_file, gpt_get_index_target_2, gpt_get_index_repl)

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
        is_swa_layer = (idx + 1) % 6 != 0
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
            is_swa_layer = (layer_idx + 1) % 6 != 0
            seq_dim_size = full_k.shape[-1] if transposed else full_k.shape[-2]
            if is_swa_layer:
                limit = min(self.sliding_window, seq_dim_size)
                position_ids = position_ids % limit
            else:
                position_ids = position_ids % seq_dim_size'''
    patch_file(kv_mgr_file, kv_swa_target, kv_swa_repl)

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






