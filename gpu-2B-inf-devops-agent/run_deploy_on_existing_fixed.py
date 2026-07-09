import boto3
import asyncio
import os
import server

async def main():
    hf_token = await server.get_secret() or ""
    
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v
                    
    ssm = boto3.client(
        'ssm', 
        region_name='us-east-1',
        aws_access_key_id=creds.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=creds.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=creds.get("AWS_SESSION_TOKEN")
    )
    
    with open("apply_all_patches.py", "r") as f:
        patch_transformers_py = f.read()

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
  --max-model-len 1024 \
  --tensor-parallel-size 2 \
  --max-num-seqs 2 \
  --swap-space 0 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 512 \
  --block-size 16 \
  --enable-auto-tool-choice \
  --tool-call-parser functiongemma \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 8080
"""

    host_commands = [
        "docker stop vllm-server || true",
        "docker rm vllm-server || true",
        "sudo rm -rf /var/tmp/neuron-compile-cache || true",
        "sudo rm -rf /home/ubuntu/.cache/neuron/* || true",
        "DEVICES=\"\"",
        "for dev in /dev/neuron*; do if [ -e \"$dev\" ]; then DEVICES=\"$DEVICES --device $dev\"; fi; done",
        "if [ -z \"$DEVICES\" ]; then DEVICES=\"--device /dev/neuron0\"; fi",
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_transformers.py\n{patch_transformers_py}\nOUTER_EOF",
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_and_run.sh\n{container_script}\nOUTER_EOF",
        "chmod +x /home/ubuntu/patch_and_run.sh",
        f"docker run -d --name vllm-server $DEVICES --ipc=host --restart no -p 8080:8080 -e HF_TOKEN=\"{hf_token}\" -e NEURON_CC_FLAGS=\"--model-type=gemma4 --enable-mixed-shapes=False --target=inf2 --hbm-scratchpad-page-size=1024\" -e NEURON_SCRATCHPAD_PAGE_SIZE=1024 -e NEURON_CORES_PER_WORKER=2 -e NEURON_COMPILER_WORKERS=1 -e VLLM_ENGINE_READY_TIMEOUT_S=1800 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=600 -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface -v /home/ubuntu/.cache/neuron:/root/.cache/neuron -v /home/ubuntu/patch_transformers.py:/patch_transformers.py -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 bash /patch_and_run.sh"
    ]

    print("Sending SSM deployment command to instance i-04b87113ec6c17762...")
    res = ssm.send_command(
        InstanceIds=['i-04b87113ec6c17762'],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': host_commands}
    )
    print("Command ID:", res['Command']['CommandId'])

if __name__ == "__main__":
    asyncio.run(main())
