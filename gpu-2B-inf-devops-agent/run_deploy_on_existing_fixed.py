import boto3
import asyncio
import os
import server

async def main():
    hf_token = await server.get_secret() or ""
    ssm = boto3.client('ssm', region_name='us-east-1')
    
    patch_transformers_py = """import os
import re

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

# 1. Patch transformers/utils/fx.py
fx_path = "/opt/conda/lib/python3.12/site-packages/transformers/utils/fx.py"
os.makedirs(os.path.dirname(fx_path), exist_ok=True)
with open(fx_path, "w") as f:
    f.write('''import torch.fx

class HFTracer(torch.fx.Tracer):
    pass

def symbolic_trace(model, *args, **kwargs):
    return torch.fx.symbolic_trace(model)
''')
print("Patched fx.py")

# 2. Patch transformers/generation/utils.py
gen_utils_path = "/opt/conda/lib/python3.12/site-packages/transformers/generation/utils.py"
if os.path.exists(gen_utils_path):
    with open(gen_utils_path, "r") as f:
        content = f.read()
    if "SampleDecoderOnlyOutput" not in content:
        with open(gen_utils_path, "a") as f:
            f.write("\\n\\nSampleDecoderOnlyOutput = GenerateDecoderOnlyOutput\\nSampleEncoderDecoderOutput = GenerateEncoderDecoderOutput\\n")
        print("Patched generation/utils.py")
    else:
        print("generation/utils.py already patched")

# 3. Patch transformers/generation/__init__.py
gen_init_path = "/opt/conda/lib/python3.12/site-packages/transformers/generation/__init__.py"
if os.path.exists(gen_init_path):
    with open(gen_init_path, "r") as f:
        content = f.read()
    if "if TYPE_CHECKING:" in content:
        content = content.replace(
            "if TYPE_CHECKING:",
            '_import_structure["utils"].extend(["SampleDecoderOnlyOutput", "SampleEncoderDecoderOutput"])\\n\\nif TYPE_CHECKING:'
        )
        print("Injected into _import_structure")
    else:
        content += '\\n_import_structure["utils"].extend(["SampleDecoderOnlyOutput", "SampleEncoderDecoderOutput"])\\n'
    with open(gen_init_path, "w") as f:
        f.write(content)
    print("Patched generation/__init__.py")

# 4. Patch vllm_neuron constants to include Gemma4UnifiedForConditionalGeneration
constants_path = "/opt/vllm/vllm_neuron/worker/constants.py"
if os.path.exists(constants_path):
    with open(constants_path, "r") as f:
        content = f.read()
    if "Gemma4UnifiedForConditionalGeneration" not in content:
        content = content.replace(
            "NEURON_MULTI_MODAL_MODELS = [",
            "NEURON_MULTI_MODAL_MODELS = [\\n    'Gemma4UnifiedForConditionalGeneration',"
        )
        with open(constants_path, "w") as f:
            f.write(content)
        print("Patched NEURON_MULTI_MODAL_MODELS in constants.py")

# 5. Patch neuronx_distributed_inference constants to register gemma4unified
constants_py_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/utils/constants.py"
if os.path.exists(constants_py_path):
    with open(constants_py_path, "r") as f:
        content = f.read()
    if "gemma4unified" not in content:
        with open(constants_py_path, "a") as f:
            f.write("\\n\\nMODEL_TYPES['gemma4unified'] = MODEL_TYPES['gemma3']\\nMODEL_TYPES['gemma4_unified'] = MODEL_TYPES['gemma3']\\n")
        print("Patched neuronx_distributed_inference constants.py")

# 6. Patch gemma3 modeling to handle missing query_pre_attn_scalar (defaults to head_dim)
gemma3_modeling_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py"
if os.path.exists(gemma3_modeling_path):
    with open(gemma3_modeling_path, "r") as f:
        content = f.read()
    
    old_target = "setattr(self, attribute, getattr(text_config, attribute))"
    new_target = "setattr(self, attribute, getattr(text_config, attribute, None))"
    if old_target in content:
        content = content.replace(old_target, new_target)
        
    old_derived = "self.add_derived_config()"
    new_derived = "if getattr(self, 'query_pre_attn_scalar', None) is None:\\n            self.query_pre_attn_scalar = self.head_dim\\n        self.add_derived_config()"
    if old_derived in content and "query_pre_attn_scalar = self.head_dim" not in content:
        content = content.replace(old_derived, new_derived)
        
    with open(gemma3_modeling_path, "w") as f:
        f.write(content)
    print("Patched modeling_gemma3.py query_pre_attn_scalar fallback")

    # 6b. Patch gemma3 modeling convert_hf_to_neuron_state_dict to handle multimodal model keys
    with open(gemma3_modeling_path, "r") as f:
        content = f.read()
    old_convert = 'if "model.norm.weight" in state_dict.keys():\\n            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}'
    new_convert = 'if "model.language_model.norm.weight" in state_dict.keys():\\n            state_dict = {k.removeprefix("model.language_model."): v for k, v in state_dict.items()}\\n        elif "model.norm.weight" in state_dict.keys():\\n            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}'
    if old_convert in content:
        content = content.replace(old_convert, new_convert)
        with open(gemma3_modeling_path, "w") as f:
            f.write(content)
        print("Patched convert_hf_to_neuron_state_dict for multimodal keys")

# 7. Patch attention_base.py to disable flash attention if head_dim > 128 and support dynamic rope
attention_base_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py"
if os.path.exists(attention_base_path):
    with open(attention_base_path, "r") as f:
        content = f.read()
    
    old_code = "if self.attn_kernel_enabled is False:"
    new_code = "if self.attn_kernel_enabled is False or self.head_dim > 128:"
    if old_code in content and new_code not in content:
        content = content.replace(old_code, new_code)
        with open(attention_base_path, "w") as f:
            f.write(content)
        print("Patched attention_base.py to disable flash attention for head_dim > 128")

    target_rope = \"\"\"    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)\"\"\"
    repl_rope = \"\"\"    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
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
            K = (K * cos_k) + (_rotate_half(K) * sin_k)\"\"\"
    patch_file(attention_base_path, target_rope, repl_rope)

# 8. Patch model loader for multimodal configuration fallback
loader_path = '/opt/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py'
if os.path.exists(loader_path):
    patch_file(
        loader_path,
        'if architecture in NEURON_MULTI_MODAL_MODELS:\\n        config = getattr(config, "text_config", None)',
        'if architecture in NEURON_MULTI_MODAL_MODELS or hasattr(config, "text_config"):\\n        config = getattr(config, "text_config", None) or config'
    )

# 9. Patch transformers auto-mappings (Crucial for gemma4_unified validation)
config_auto_file = '/opt/conda/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py'
if os.path.exists(config_auto_file):
    patch_file(
        config_auto_file,
        '("gemma3", "Gemma3Config"),',
        '("gemma3", "Gemma3Config"),\\n        ("gemma4_unified", "Gemma3Config"),\\n        ("gemma4_unified_text", "Gemma3TextConfig"),'
    )
    patch_file(
        config_auto_file,
        '("gemma3_text", "gemma3"),',
        '("gemma3_text", "gemma3"),\\n        ("gemma4_unified", "gemma3"),\\n        ("gemma4_unified_text", "gemma3"),\\n        ("gemma4_unified_vision", "gemma3"),\\n        ("gemma4_unified_audio", "gemma3"),'
    )
    patch_file(
        config_auto_file,
        '("gemma3", "Gemma3ForConditionalGeneration"),',
        '("gemma3", "Gemma3ForConditionalGeneration"),\\n        ("gemma4_unified", "Gemma4Unified"),\\n        ("gemma4_unified_text", "Gemma4UnifiedText"),'
    )

modeling_auto_file = '/opt/conda/lib/python3.12/site-packages/transformers/models/auto/modeling_auto.py'
if os.path.exists(modeling_auto_file):
    patch_file(
        modeling_auto_file,
        '        ("gemma3", "Gemma3Model"),\\n        ("gemma3_text", "Gemma3TextModel"),',
        '        ("gemma3", "Gemma3Model"),\\n        ("gemma3_text", "Gemma3TextModel"),\\n        ("gemma4_unified", "Gemma3Model"),\\n        ("gemma4_unified_text", "Gemma3TextModel"),'
    )
    patch_file(
        modeling_auto_file,
        '        ("gemma3", "Gemma3ForConditionalGeneration"),\\n        ("gemma3_text", "Gemma3ForCausalLM"),',
        '        ("gemma3", "Gemma3ForConditionalGeneration"),\\n        ("gemma3_text", "Gemma3ForCausalLM"),\\n        ("gemma4_unified", "Gemma3ForConditionalGeneration"),\\n        ("gemma4_unified_text", "Gemma3ForCausalLM"),'
    )

# 10. Patch neuronx_distributed_inference kvcache utils & manager
kvcache_utils_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/utils.py"
if os.path.exists(kvcache_utils_path):
    with open(kvcache_utils_path, "r") as f:
        content = f.read()
    old_dynamic = "def dynamic_update_slice(\\n    tensor: torch.Tensor, update: torch.Tensor, start_indices: List[torch.Tensor]\\n):"
    new_dynamic = "def dynamic_update_slice(\\n    tensor: torch.Tensor, update: torch.Tensor, start_indices: List[torch.Tensor]\\n):\\n    if update.shape[-1] < tensor.shape[-1]:\\n        update = torch.nn.functional.pad(update, (0, tensor.shape[-1] - update.shape[-1]))"
    old_const = "    batch_indices = sequence_ids.view(-1, 1, 1).expand(-1, kv_heads, bucket_length).to(torch.int32)"
    new_const = "    if updates.shape[-1] < d_head:\\n        updates = torch.nn.functional.pad(updates, (0, d_head - updates.shape[-1]))\\n    batch_indices = sequence_ids.view(-1, 1, 1).expand(-1, kv_heads, bucket_length).to(torch.int32)"
    if old_dynamic in content and "torch.nn.functional.pad" not in content:
        content = content.replace(old_dynamic, new_dynamic)
    if old_const in content and "updates.shape[-1] < d_head" not in content:
        content = content.replace(old_const, new_const)
    with open(kvcache_utils_path, "w") as f:
        f.write(content)
    print("Patched kvcache/utils.py")

kv_cache_mgr_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py"
if os.path.exists(kv_cache_mgr_path):
    with open(kv_cache_mgr_path, "r") as f:
        content = f.read()
    target_fetch = "        if self.is_kv_cache_tiled:\\n            k_cache = untile_cache(cache=k_cache, transposed=self.k_cache_transposed)\\n            v_cache = untile_cache(cache=v_cache, transposed=False)\\n\\n        return k_cache, v_cache"
    replacement_fetch = "        if self.is_kv_cache_tiled:\\n            k_cache = untile_cache(cache=k_cache, transposed=self.k_cache_transposed)\\n            v_cache = untile_cache(cache=v_cache, transposed=False)\\n\\n        if (idx + 1) % 6 != 0:\\n            if k_cache.shape[-1] == 512:\\n                k_cache = k_cache[..., :256]\\n            if v_cache.shape[-1] == 512:\\n                v_cache = v_cache[..., :256]\\n        return k_cache, v_cache"
    if target_fetch in content and "k_cache.shape[-1] == 512" not in content:
        content = content.replace(target_fetch, replacement_fetch)
        with open(kv_cache_mgr_path, "w") as f:
            f.write(content)
        print("Patched kv_cache_manager.py")

# 11. Patch vllm/transformers_utils/config.py to support nested gemma4 rope_parameters
vllm_config_file = '/opt/conda/lib/python3.12/site-packages/vllm/transformers_utils/config.py'
if os.path.exists(vllm_config_file):
    # Patch is_rope_parameters_nested support for full_attention/sliding_attention
    target_nested = "def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:\\n    \\\"\\\"\\\"Check if rope_parameters is nested by layer types.\\\"\\\"\\\"\\n    # Cannot be nested if rope_parameters is empty\\n    if not rope_parameters:\\n        return False\\n    return set(rope_parameters.keys()).issubset(ALLOWED_ATTENTION_LAYER_TYPES)"
    repl_nested = "def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:\\n    \\\"\\\"\\\"Check if rope_parameters is nested by layer types.\\\"\\\"\\\"\\n    # Cannot be nested if rope_parameters is empty\\n    if not rope_parameters:\\n        return False\\n    if \\\"full_attention\\\" in rope_parameters or \\\"sliding_attention\\\" in rope_parameters:\\n        return True\\n    return set(rope_parameters.keys()).issubset(ALLOWED_ATTENTION_LAYER_TYPES)"
    patch_file(vllm_config_file, target_nested, repl_nested)

    # Patch legacy fields
    target_legacy = "        # Patch legacy fields into rope_parameters\\n        if rope_theta is not None:\\n            config.rope_parameters[\\\"rope_theta\\\"] = rope_theta\\n        if partial_rotary_factor is not None:\\n            config.rope_parameters[\\\"partial_rotary_factor\\\"] = partial_rotary_factor\\n        if ompe is not None:\\n            config.rope_parameters[\\\"original_max_position_embeddings\\\"] = ompe"
    repl_legacy = "        # Patch legacy fields into rope_parameters\\n        if not is_rope_parameters_nested(getattr(config, \\\"rope_parameters\\\", None)):\\n            if rope_theta is not None:\\n                config.rope_parameters[\\\"rope_theta\\\"] = rope_theta\\n            if partial_rotary_factor is not None:\\n                config.rope_parameters[\\\"partial_rotary_factor\\\"] = partial_rotary_factor\\n            if ompe is not None:\\n                config.rope_parameters[\\\"original_max_position_embeddings\\\"] = ompe"
    patch_file(vllm_config_file, target_legacy, repl_legacy)

# 12. Patch vllm/model_executor/models/registry.py to register Gemma4UnifiedForConditionalGeneration
vllm_registry_file = '/opt/conda/lib/python3.12/site-packages/vllm/model_executor/models/registry.py'
if os.path.exists(vllm_registry_file):
    target_reg = '    "Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),  # noqa: E501'
    repl_reg = '    "Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),  # noqa: E501\\n    "Gemma4UnifiedForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),'
    patch_file(vllm_registry_file, target_reg, repl_reg)

# 13. Patch transformers/tokenization_utils_base.py to support list in _set_model_specific_special_tokens
transformers_tok_base = '/opt/conda/lib/python3.12/site-packages/transformers/tokenization_utils_base.py'
if os.path.exists(transformers_tok_base):
    target_tok = '        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())'
    repl_tok = \"\"\"        if not hasattr(special_tokens, "keys") or not hasattr(special_tokens, "items"):
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
        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())\"\"\"
    patch_file(transformers_tok_base, target_tok, repl_tok)
"""


    container_script = """#!/bin/bash
set -e
echo "Ensuring correct transformers version..."
pip install "transformers==4.57.6"

echo "Running python patcher for transformers..."
python3 /patch_transformers.py

echo "Registering neuron_quant quantization method in vLLM..."
cat << 'INNER_EOF' >> /opt/conda/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__init__.py

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import register_quantization_config
import torch

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
    def from_config(cls, config: dict) -> "NeuronQuantConfig":
        return cls()

    def get_quant_method(self, layer, prefix):
        return None
INNER_EOF

echo "Starting vLLM Server..."
python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E2B-it \
  --quantization neuron_quant \
  --max-model-len 1024 \
  --tensor-parallel-size 2 \
  --max-num-seqs 2 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 512 \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 8080
"""

    host_commands = [
        "docker stop vllm-server || true",
        "docker rm vllm-server || true",
        "sudo rm -rf /var/tmp/neuron-compile-cache || true",
        "sudo rm -rf /home/ubuntu/.cache/neuron/* || true",
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_transformers.py\n{patch_transformers_py}\nOUTER_EOF",
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_and_run.sh\n{container_script}\nOUTER_EOF",
        "chmod +x /home/ubuntu/patch_and_run.sh",
        f"docker run -d --name vllm-server --device /dev/neuron0 --ipc=host --restart always -p 8080:8080 -e HF_TOKEN=\"{hf_token}\" -e NEURON_CC_FLAGS=\"--model-type transformer\" -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface -v /home/ubuntu/.cache/neuron:/root/.cache/neuron -v /home/ubuntu/patch_transformers.py:/patch_transformers.py -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 bash /patch_and_run.sh"
    ]

    print("Sending SSM deployment command to instance i-0230cf22bbebf1814...")
    res = ssm.send_command(
        InstanceIds=['i-0230cf22bbebf1814'],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': host_commands}
    )
    print("Command ID:", res['Command']['CommandId'])

if __name__ == "__main__":
    asyncio.run(main())
