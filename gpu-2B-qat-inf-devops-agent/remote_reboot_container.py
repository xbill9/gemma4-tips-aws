import boto3
import time
import os

def main():
    creds = {}
    if os.path.exists("/home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/.aws_creds"):
        with open("/home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/.aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v
    for k, v in creds.items():
        os.environ[k] = v

    instance_id = "i-0af2ceb15e7807e96"
    region = "us-east-1"
    ssm = boto3.client("ssm", region_name=region)
    with open("apply_all_patches.py", "r") as f:
        apply_all_patches_content = f.read()

    script_content = """#!/bin/bash
set -e
echo "Ensuring correct transformers version..."
pip install "transformers==4.57.6"

echo "Running python patcher for transformers..."
python3 /home/ubuntu/patch_transformers.py

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

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "NeuronQuantConfig":
        return cls()

    def get_quant_method(self, layer, prefix):
        return None
INNER_EOF

echo "Starting vLLM Server with memory optimizations..."
if [ -z "$tensor_parallel_size" ]; then
  tensor_parallel_size=2
fi
python3 -m vllm.entrypoints.openai.api_server \\
  --model google/gemma-4-12B-it \\
  --max-model-len 1024 \\
  --tensor-parallel-size $tensor_parallel_size \\
  --max-num-seqs 2 \\
  --swap-space 0 \\
  --no-enable-prefix-caching \\
  --max-num-batched-tokens 512 \\
  --block-size 16 \\
  --enable-auto-tool-choice \\
  --tool-call-parser functiongemma \\
  --host 0.0.0.0 \\
  --port 8080
"""

    docker_cmd = """docker run -d --name vllm-server \\
  --no-healthcheck \\
  --device /dev/neuron0 \\
  --ipc=host \\
  --restart no \\
  -p 8080:8080 \\
  -e HF_TOKEN="$(aws ssm get-parameter --name /vllm/HF_TOKEN --with-decryption --query Parameter.Value --output text 2>/dev/null || echo '')" \\
  -e tensor_parallel_size="2" \\
  -e NEURON_CC_FLAGS="--model-type=gemma4 --enable-mixed-shapes=False --target=inf2 --hbm-scratchpad-page-size=1024" \\
  -e NEURON_SCRATCHPAD_PAGE_SIZE=1024 \\
  -e NEURON_CORES_PER_WORKER=2 \\
  -e NEURON_COMPILER_WORKERS=1 \\
  -e VLLM_USE_V1=0 \\
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \\
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=600 \\
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \\
  -v /home/ubuntu/.cache/neuron:/root/.cache/neuron \\
  -v /home/ubuntu/patch_transformers.py:/home/ubuntu/patch_transformers.py \\
  -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh \\
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 \\
  bash /patch_and_run.sh"""

    # We escape double quotes and backslashes for the remote bash shell
    commands = [
        "docker stop vllm-server || true",
        "docker rm -f vllm-server || true",
        "sudo rm -rf /var/tmp/neuron-compile-cache || true",
        "sudo rm -rf /home/ubuntu/.cache/neuron/* || true",
        f"cat << 'EOF' > /home/ubuntu/patch_transformers.py\n{apply_all_patches_content}\nEOF",
        f"cat << 'EOF' > /home/ubuntu/patch_and_run.sh\n{script_content}\nEOF",
        "chmod +x /home/ubuntu/patch_and_run.sh",
        docker_cmd
    ]

    print("Sending SSM command...")
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands}
    )
    command_id = response["Command"]["CommandId"]
    print(f"Command sent successfully! Command ID: {command_id}")

    # Wait for the command to execute and print output
    time.sleep(5)
    result = ssm.get_command_invocation(
        CommandId=command_id,
        InstanceId=instance_id
    )
    print("Command Status:", result["Status"])
    print("Stdout:", result.get("StandardOutputContent", ""))
    print("Stderr:", result.get("StandardErrorContent", ""))

if __name__ == "__main__":
    main()
