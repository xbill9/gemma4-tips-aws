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
patch_file(
    '/opt/vllm/vllm_neuron/worker/neuron_worker.py',
    '    def load_model(self):\n        self.model_runner.load_model()',
    '    def load_model(self):\n        import traceback\n        try:\n            self.model_runner.load_model()\n        except Exception as e:\n            with open("/home/ubuntu/worker_error.txt", "w") as f:\n                traceback.print_exc(file=f)\n                f.write(f"\\\\nException: {str(e)}\\\\n")\n            raise'
)

# 2. Patch modeling_gemma3.py (loader adjustments ONLY - NO state dict padding)
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

# 2c. Patch NeuronGemma3Attention and NeuronGemma3DecoderLayer in modeling_gemma3.py to support heterogeneous head dimensions
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

# 2d. Patch rotary_emb selection in modeling_gemma3.py
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

# 2e. Patch global_rotary_emb dimension in modeling_gemma3.py
gemma3_global_rope_dim_target = '''        global_rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.global_rope_theta,
            factor=config.rope_scaling,
        )'''
gemma3_global_rope_dim_repl = '''        global_dim = head_dim
        if config.sliding_window is not None:
            global_dim = int(getattr(config, "global_head_dim", 512) * 0.25)
        global_rotary_emb = RotaryEmbedding(
            dim=global_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.global_rope_theta,
            factor=config.rope_scaling,
        )'''
patch_file(
    '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py',
    gemma3_global_rope_dim_target,
    gemma3_global_rope_dim_repl
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

# 7. Auto-patching for Transformers 4.x
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

# 8. Patch attention_base.py to fallback when head_dim > 128 (but NO shape-forcing padding!)
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
            rotary_dim = cos_cache.shape[-1]
            if rotary_dim < Q.shape[-1]:
                Q_rot = Q[..., :rotary_dim]
                Q_pass = Q[..., rotary_dim:]
                cos_q = cos_cache.unsqueeze(1)
                sin_q = sin_cache.unsqueeze(1)
                Q_rot = (Q_rot * cos_q) + (_rotate_half(Q_rot) * sin_q)
                Q = torch.cat([Q_rot, Q_pass], dim=-1)
            else:
                cos_q = cos_cache[..., :Q.shape[-1]].unsqueeze(1)
                sin_q = sin_cache[..., :Q.shape[-1]].unsqueeze(1)
                Q = (Q * cos_q) + (_rotate_half(Q) * sin_q)
            if rotary_dim < K.shape[-1]:
                K_rot = K[..., :rotary_dim]
                K_pass = K[..., rotary_dim:]
                cos_k = cos_cache.unsqueeze(1)
                sin_k = sin_cache.unsqueeze(1)
                K_rot = (K_rot * cos_k) + (_rotate_half(K_rot) * sin_k)
                K = torch.cat([K_rot, K_pass], dim=-1)
            else:
                cos_k = cos_cache[..., :K.shape[-1]].unsqueeze(1)
                sin_k = sin_cache[..., :K.shape[-1]].unsqueeze(1)
                K = (K * cos_k) + (_rotate_half(K) * sin_k)'''
    patch_file(attention_base_file, target_rope, repl_rope)

    # Patch compute_for_token_gen for heterogeneous head dimensions (Q: 512, K_prior/V_prior: 256/512)
    target_token_gen = '''        if not self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)
        prior_scores = torch.matmul(Q, K_prior) / self.softmax_scale'''
    repl_token_gen = '''        min_dim = min(Q.shape[-1], K_prior.shape[-1])
        if Q.shape[-1] > min_dim: Q = Q[..., :min_dim]
        if K_prior.shape[-1] > min_dim: K_prior = K_prior[..., :min_dim]
        if V_prior.shape[-1] > min_dim: V_prior = V_prior[..., :min_dim]
        if not self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)
        prior_scores = torch.matmul(Q, K_prior) / self.softmax_scale'''
    patch_file(attention_base_file, target_token_gen, repl_token_gen)

# 9. Patch vllm/transformers_utils/config.py to support nested gemma4 rope_parameters
vllm_config_file = '/opt/conda/lib/python3.12/site-packages/vllm/transformers_utils/config.py'
if os.path.exists(vllm_config_file):
    target_9 = '        config = _maybe_remap_hf_config_attrs(config)'
    repl_9 = '''        if hasattr(config, "text_config") and hasattr(config.text_config, "rope_parameters"):
            rp = config.text_config.rope_parameters
            if isinstance(rp, dict) and "full_attention" in rp:
                config.text_config.rope_parameters = rp["full_attention"]
        if hasattr(config, "rope_parameters"):
            rp = config.rope_parameters
            if isinstance(rp, dict) and "full_attention" in rp:
                config.rope_parameters = rp["full_attention"]
        config = _maybe_remap_hf_config_attrs(config)'''
    patch_file(vllm_config_file, target_9, repl_9)

# 10. Register gemma4_unified in transformers
trans_models_dir = '/opt/conda/lib/python3.12/site-packages/transformers/models'
if os.path.exists(trans_models_dir):
    # Create the directory for gemma4_unified if it doesn't exist
    gemma4_dir = os.path.join(trans_models_dir, 'gemma4_unified')
    os.makedirs(gemma4_dir, exist_ok=True)
    
    # Write init file
    with open(os.path.join(gemma4_dir, '__init__.py'), 'w') as f:
        f.write('''from typing import TYPE_CHECKING
from transformers.utils import _LazyModule

_import_structure = {
    "configuration_gemma4_unified": ["Gemma4UnifiedConfig"],
    "modeling_gemma4_unified": ["Gemma4UnifiedForCausalLM", "Gemma4UnifiedModel", "Gemma4UnifiedPreTrainedModel"],
}

if TYPE_CHECKING:
    from .configuration_gemma4_unified import Gemma4UnifiedConfig
    from .modeling_gemma4_unified import Gemma4UnifiedForCausalLM, Gemma4UnifiedModel, Gemma4UnifiedPreTrainedModel
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
''')
    
    # Write configuration file
    with open(os.path.join(gemma4_dir, 'configuration_gemma4_unified.py'), 'w') as f:
        f.write('''from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Gemma4UnifiedConfig(PretrainedConfig):
    model_type = "gemma4_unified"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=26,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        sliding_window=512,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.sliding_window = sliding_window
        super().__init__(**kwargs)
''')
    print("Registered gemma4_unified architecture in transformers")

# 11. Patch vllm/model_executor/models/registry.py to register Gemma4UnifiedForConditionalGeneration as CausalLM
vllm_registry_file = '/opt/conda/lib/python3.12/site-packages/vllm/model_executor/models/registry.py'
if os.path.exists(vllm_registry_file):
    target_reg = '    "Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),  # noqa: E501'
    repl_reg = '''    "Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),  # noqa: E501
    "Gemma4ForConditionalGeneration": ("gemma3", "Gemma3ForCausalLM"),
    "Gemma4ForCausalLM": ("gemma3", "Gemma3ForCausalLM"),
    "Gemma4UnifiedForConditionalGeneration": ("gemma3", "Gemma3ForCausalLM"),'''
    patch_file(vllm_registry_file, target_reg, repl_reg)

# 12. Patch tokenization_utils_base.py to handle list extra_special_tokens
tokenization_utils_file = '/opt/conda/lib/python3.12/site-packages/transformers/tokenization_utils_base.py'
if os.path.exists(tokenization_utils_file):
    target_tok = '    def _set_model_specific_special_tokens(self, special_tokens: list[str]):'
    repl_tok = '''    def _set_model_specific_special_tokens(self, special_tokens: list[str]):
        if isinstance(special_tokens, list):
            special_tokens = {tok: tok for tok in special_tokens}'''
    patch_file(tokenization_utils_file, target_tok, repl_tok)

# 13. Patch kvcache/utils.py to support heterogeneous head dimensions padding in dynamic_update_slice and update_cache_const_indices
kvcache_utils_file = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/utils.py'
if os.path.exists(kvcache_utils_file):
    target_dus = '''def dynamic_update_slice(
    tensor: torch.Tensor, update: torch.Tensor, start_indices: List[torch.Tensor]
):'''
    repl_dus = '''def dynamic_update_slice(
    tensor: torch.Tensor, update: torch.Tensor, start_indices: List[torch.Tensor]
):
    if update.shape[-1] > tensor.shape[-1]:
        update = update[..., :tensor.shape[-1]]
    elif update.shape[-1] < tensor.shape[-1]:
        update = torch.nn.functional.pad(update, (0, tensor.shape[-1] - update.shape[-1]))'''
    patch_file(kvcache_utils_file, target_dus, repl_dus)

    target_const = '''def update_cache_const_indices(cache: torch.Tensor, updates: torch.Tensor, sequence_ids: Tensor):
    """
    Use constants for head and position indices, so that compiler just needs to compute the offset for batch dimension.
    This is needed to avoid inefficient DMAs, since compiler is not able to const-prop a constant address offset and treats it a dynamic offset.
    NCC-6227
    """
    max_batch_size, kv_heads, max_sequence_length, d_head = cache.shape'''
    repl_const = '''def update_cache_const_indices(cache: torch.Tensor, updates: torch.Tensor, sequence_ids: Tensor):
    """
    Use constants for head and position indices, so that compiler just needs to compute the offset for batch dimension.
    This is needed to avoid inefficient DMAs, since compiler is not able to const-prop a constant address offset and treats it a dynamic offset.
    NCC-6227
    """
    max_batch_size, kv_heads, max_sequence_length, d_head = cache.shape
    if updates.shape[-1] > d_head:
        updates = updates[..., :d_head]
    elif updates.shape[-1] < d_head:
        updates = torch.nn.functional.pad(updates, (0, d_head - updates.shape[-1]))'''
    patch_file(kvcache_utils_file, target_const, repl_const)

print("All mixed shape (unpadded SWA) patches prepared successfully!")

