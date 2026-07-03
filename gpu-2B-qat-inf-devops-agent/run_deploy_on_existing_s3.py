import boto3
import asyncio
import os
import server

async def main():
    # 1. Resolve Region
    region = "us-east-1"
    if os.path.exists("active_deployment_region.txt"):
        with open("active_deployment_region.txt", "r") as f:
            region = f.read().strip()

    print(f"Using Region: {region}")

    # Read credentials
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
    
    s3_client = boto3.client(
        's3',
        region_name='us-east-1', # Bucket is in us-east-1
        aws_access_key_id=creds.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=creds.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=creds.get("AWS_SESSION_TOKEN")
    )
    
    ssm = boto3.client('ssm', region_name=region)
    ec2 = boto3.client('ec2', region_name=region)
    
    # Query running instances under our service tag
    instances_resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
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

    # Upload the latest local apply_all_patches.py to S3 first
    bucket_name = "xbill-gemma4-patches-2b"
    print(f"Uploading apply_all_patches.py to S3 bucket {bucket_name}...")
    s3_client.upload_file(
        Filename="apply_all_patches.py",
        Bucket=bucket_name,
        Key="apply_all_patches.py"
    )

    # Generate a presigned URL or public URL from our bucket 'xbill-gemma4-patches-2b'
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': 'apply_all_patches.py'},
        ExpiresIn=31536000  # 1 year
    )
    print(f"Using S3 Presigned URL for deployment: {presigned_url}")

    container_script = """#!/bin/bash
set -e
echo "Preserving pre-installed transformers 4.57.6..."

echo "Running comprehensive python patcher from S3..."
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
  --model google/gemma-4-E2B-it \
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

    deploy_sh_content = f"""#!/bin/bash
set -e

echo "Stopping and removing existing vllm-server container..."
docker stop vllm-server || true
docker rm vllm-server || true

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
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \\
  -e VLLM_USE_V1=0 \\
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \\
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \\
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \\
  -v /home/ubuntu/neuron-compile-cache:/var/tmp/neuron-compile-cache \\
  -v /home/ubuntu/apply_all_patches.py:/apply_all_patches.py \\
  -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh \\
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 \\
  bash /patch_and_run.sh
"""

    host_commands = [
        "echo '=== Purging compiler cache and cleaning storage ==='",
        "rm -rf /home/ubuntu/neuron-compile-cache/*",
        "rm -rf /home/ubuntu/.cache/neuron/*",
        "docker volume prune -f",
        "docker system prune -f || true",
        f"curl -s -S -o /home/ubuntu/apply_all_patches.py '{presigned_url}'",
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_and_run.sh\n{container_script}\nOUTER_EOF",
        "chmod +x /home/ubuntu/patch_and_run.sh",
        f"cat << 'OUTER_EOF' > /home/ubuntu/deploy.sh\n{deploy_sh_content}\nOUTER_EOF",
        "chmod +x /home/ubuntu/deploy.sh",
        "bash /home/ubuntu/deploy.sh"
    ]

    print(f"Sending SSM S3-based deployment command to active instances {instance_ids} in {region}...")
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
