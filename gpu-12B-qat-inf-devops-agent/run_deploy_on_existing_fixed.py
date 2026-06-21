import boto3
import asyncio
import os
import server

async def main():
    hf_token = await server.get_secret() or ""
    ssm = boto3.client('ssm', region_name='us-west-2')
    
    patch_transformers_py = """import os

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
            f.write("\\n\\nMODEL_TYPES['gemma4unified'] = MODEL_TYPES['gemma3']\\n")
        print("Patched neuronx_distributed_inference constants.py")

# 6. Patch gemma3 modeling to handle missing query_pre_attn_scalar (defaults to head_dim)
gemma3_modeling_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py"
if os.path.exists(gemma3_modeling_path):
    with open(gemma3_modeling_path, "r") as f:
        content = f.read()
    
    # We replace the line-by-line getattr lookup to use a fallback and add fallback for query_pre_attn_scalar
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

# 7. Patch attention_base.py to disable flash attention if head_dim > 128
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
"""

    container_script = """#!/bin/bash
set -e
echo "Upgrading transformers..."
pip install --upgrade transformers

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
  --model google/gemma-4-12B-it \
  --quantization neuron_quant \
  --max-model-len 16384 \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
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
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_transformers.py\n{patch_transformers_py}\nOUTER_EOF",
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_and_run.sh\n{container_script}\nOUTER_EOF",
        "chmod +x /home/ubuntu/patch_and_run.sh",
        f"docker run -d --name vllm-server --device /dev/neuron0 --ipc=host --restart always -p 8080:8080 -e HF_TOKEN=\"{hf_token}\" -e NEURON_CC_FLAGS=\"--model-type transformer\" -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface -v /home/ubuntu/.cache/neuron:/root/.cache/neuron -v /home/ubuntu/patch_transformers.py:/patch_transformers.py -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 bash /patch_and_run.sh"
    ]

    print("Sending SSM deployment command to instance i-06c1e95201777d40e...")
    res = ssm.send_command(
        InstanceIds=['i-06c1e95201777d40e'],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': host_commands}
    )
    print("Command ID:", res['Command']['CommandId'])

if __name__ == "__main__":
    asyncio.run(main())
