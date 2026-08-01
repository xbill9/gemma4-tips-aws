import boto3
import os
import time

creds = {}
if os.path.exists(".aws_creds"):
    with open(".aws_creds", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v

for k, v in creds.items():
    os.environ[k] = v

# Read active token
import asyncio
import server
hf_token = asyncio.run(server.get_secret()) or ""
ssm = boto3.client("ssm", region_name="us-east-1")


# Read actual patch script
with open("apply_all_patches.py", "r") as f:
    patch_transformers_py = f.read()

# Define container startup script
container_script = """#!/bin/bash
set -e
echo "Ensuring correct transformers version..."
pip install "transformers==4.57.6"

echo "Running python patcher..."
python3 /patch_transformers.py

# Disable Triton flash attention fallback to prevent looping
export VLLM_USE_TRITON_FLASH_ATTN=0

echo "Starting vLLM Server with enforce-eager..."
python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-12B-it \
  --quantization neuron_quant \
  --max-model-len 1024 \
  --tensor-parallel-size 2 \
  --max-num-seqs 2 \
  --swap-space 0 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 512 \
  --num-gpu-blocks-override 128 \
  --block-size 16 \
  --kv-cache-dtype auto \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser functiongemma \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 8080
"""

instance_id = "i-0b82d784a76f89a79"

host_commands = [
    # 1. Correctly detect and mount the secondary cache volume
    'ROOT_DISK=$(lsblk -no PKNAME $(findmnt -n -o SOURCE /) | head -n 1)',
    'CACHE_DEV=$(lsblk -dln -o NAME,TYPE | awk \'$2=="disk" {print $1}\' | grep -v "$ROOT_DISK" | head -n 1)',
    'if [ -n "$CACHE_DEV" ]; then ',
    '  CACHE_DEV="/dev/$CACHE_DEV"; ',
    '  PART=$(lsblk -ln -o NAME,TYPE | grep "^${CACHE_DEV##*/}" | awk \'$2=="part" {print "/dev/"$1}\' | head -n 1); '
    '  if [ -n "$PART" ]; then CACHE_DEV="$PART"; fi; '
    '  echo "Mounting $CACHE_DEV on /home/ubuntu/.cache"; '
    '  umount /home/ubuntu/.cache || true; '
    '  mkdir -p /home/ubuntu/.cache; '
    '  mount "$CACHE_DEV" /home/ubuntu/.cache || { mkfs -t ext4 "$CACHE_DEV" && mount "$CACHE_DEV" /home/ubuntu/.cache; }; '
    '  chown -R ubuntu:ubuntu /home/ubuntu/.cache; '
    '  chmod -R 777 /home/ubuntu/.cache; '
    'fi',
    
    # 2. Stop and clean any existing container
    "docker stop vllm-server || true",
    "docker rm vllm-server || true",
    
    # 3. Clean failed compile caches (disabled to preserve cache)
    # "sudo rm -rf /var/tmp/neuron-compile-cache || true",
    # "sudo rm -rf /home/ubuntu/.cache/neuron/* || true",
    
    # 4. Write patch and run files
    "rm -rf /home/ubuntu/patch_transformers.py || true",
    f"cat << 'OUTER_EOF' > /home/ubuntu/patch_transformers.py\n{patch_transformers_py}\nOUTER_EOF",
    f"cat << 'OUTER_EOF' > /home/ubuntu/patch_and_run.sh\n{container_script}\nOUTER_EOF",
    "chmod +x /home/ubuntu/patch_and_run.sh",
    "chown ubuntu:ubuntu /home/ubuntu/patch_transformers.py /home/ubuntu/patch_and_run.sh",

    
    # 5. Resolve Neuron Devices
    "DEVICES=\"\"",
    "for dev in /dev/neuron*; do if [ -e \"$dev\" ]; then DEVICES=\"$DEVICES --device $dev\"; fi; done",
    "if [ -z \"$DEVICES\" ]; then DEVICES=\"--device /dev/neuron0\"; fi",
    
    # 6. Start the docker run command
    f'docker run -d --name vllm-server $DEVICES --ipc=host --restart no -p 8080:8080 -e HF_TOKEN="{hf_token}" -e NEURON_CC_FLAGS="--model-type=gemma4 --enable-mixed-shapes=False --target=inf2 --hbm-scratchpad-page-size=1024" -e NEURON_SCRATCHPAD_PAGE_SIZE=1024 -e NEURON_CORES_PER_WORKER=2 -e NEURON_COMPILER_WORKERS=1 -e VLLM_USE_TRITON_FLASH_ATTN=0 -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface -v /home/ubuntu/.cache/neuron:/root/.cache/neuron -v /home/ubuntu/neuron-compile-cache:/var/tmp/neuron-compile-cache -v /home/ubuntu/patch_transformers.py:/patch_transformers.py -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 bash /patch_and_run.sh'
]

print(f"Sending SSM deployment command to instance {instance_id}...")
res = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': host_commands}
)
command_id = res['Command']['CommandId']
print("Command ID:", command_id)

print("Waiting for command to execute...")
while True:
    time.sleep(2)
    result = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
    status = result["Status"]
    if status in ["Success", "Failed", "Cancelled", "TimedOut"]:
        print(f"--- STATUS: {status} ---")
        print("--- STDOUT ---")
        print(result.get("StandardOutputContent", ""))
        print("--- STDERR ---")
        print(result.get("StandardErrorContent", ""))
        break
