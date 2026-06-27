import boto3
import asyncio
import os
import server

async def main():
    # Read from .aws_creds if present
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v
    for k, v in creds.items():
        os.environ[k] = v

    hf_token = await server.get_secret() or ""
    # Use the active region us-east-1
    ssm = boto3.client('ssm', region_name='us-east-1')
    ec2 = boto3.client('ec2', region_name='us-east-1')
    
    # Query all running instances under our service tag
    instances_resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": ["inferentia-12b-devops-agent"]},
            {"Name": "instance-state-name", "Values": ["running"]}
        ]
    )
    instance_ids = []
    for reservation in instances_resp.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            instance_ids.append(instance["InstanceId"])
            
    if not instance_ids:
        print("No running instances found!")
        return
    print(f"Found running instances: {instance_ids}")

    
    # Dynamically read the comprehensive apply_all_patches.py from local directory
    with open("apply_all_patches.py", "r") as f:
        apply_all_patches_content = f.read()

    container_script = """#!/bin/bash
set -e
echo "Preserving pre-installed transformers 4.57.6..."

echo "Running comprehensive python patcher..."
python3 /apply_all_patches.py

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

# Dynamic TP detection inside container based on exposed neuron devices
device_count=0
for dev in /dev/neuron*; do
    if [ -e "$dev" ]; then
        device_count=$((device_count + 1))
    fi
done
if [ $device_count -eq 0 ]; then
    device_count=1
fi
TP_SIZE=$((device_count * 2))
echo "Detected $device_count Neuron device(s). Using --tensor-parallel-size $TP_SIZE"

echo "Starting vLLM Server with optimized parameters..."
python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-12B-it \
  --quantization neuron_quant \
  --max-model-len 1024 \
  --tensor-parallel-size $TP_SIZE \
  --max-num-seqs 2 \
  --num-gpu-blocks-override 128 \
  --swap-space 0 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 512 \
  --block-size 16 \
  --kv-cache-dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser functiongemma \
  --async-scheduling \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --host 0.0.0.0 \
  --port 8080
"""

    # We write a deployment shell script that runs on the host to avoid python-to-SSM variable/quoting issues
    deploy_sh_content = f"""#!/bin/bash
set -e

echo "Stopping and removing existing vllm-server container..."
docker stop vllm-server || true
docker rm vllm-server || true

echo "Actively clearing corrupt compiler cache on host to force correct JIT graph compilation..."
sudo rm -rf /var/tmp/neuron-compile-cache || true
sudo rm -rf /home/ubuntu/.cache/neuron/* || true


# Dynamic device mapping on the host
DEVICES=""
for dev in /dev/neuron*; do
    if [ -e "$dev" ]; then
        DEVICES="$DEVICES --device $dev"
    fi
done
if [ -z "$DEVICES" ]; then
    DEVICES="--device /dev/neuron0"
fi

echo "Launching vLLM container with devices: $DEVICES"

docker run -d --name vllm-server \\
  --no-healthcheck \\
  $DEVICES \\
  --ipc=host \\
  --restart no \\
  -p 8080:8080 \\
  -e HF_TOKEN="{hf_token}" \\
  -e NEURON_CC_FLAGS="--model-type=gemma4 --enable-mixed-shapes=False --target=inf2 --hbm-scratchpad-page-size=1024" \\
  -e NEURON_SCRATCHPAD_PAGE_SIZE=1024 \\
  -e NEURON_CORES_PER_WORKER=2 \\
  -e NEURON_COMPILER_WORKERS=1 \\
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \\
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \\
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \\
  -v /home/ubuntu/.cache/neuron:/root/.cache/neuron \\
  -v /home/ubuntu/apply_all_patches.py:/apply_all_patches.py \\
  -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh \\
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 \\
  bash /patch_and_run.sh
"""

    host_commands = [
        f"cat << 'OUTER_EOF' > /home/ubuntu/apply_all_patches.py\n{apply_all_patches_content}\nOUTER_EOF",
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_and_run.sh\n{container_script}\nOUTER_EOF",
        "chmod +x /home/ubuntu/patch_and_run.sh",
        f"cat << 'OUTER_EOF' > /home/ubuntu/deploy.sh\n{deploy_sh_content}\nOUTER_EOF",
        "chmod +x /home/ubuntu/deploy.sh",
        "bash /home/ubuntu/deploy.sh"
    ]

    print(f"Sending SSM deployment command to active instances {instance_ids} in us-east-1...")
    res = ssm.send_command(
        InstanceIds=instance_ids,
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': host_commands}
    )
    command_id = res['Command']['CommandId']
    print("Command ID:", command_id)
    return command_id

if __name__ == "__main__":
    asyncio.run(main())
