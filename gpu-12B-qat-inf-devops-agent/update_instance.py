import boto3

ssm = boto3.client('ssm', region_name='us-east-1')
script = f'''#!/bin/bash
echo "Restarting vllm-server container..."
docker stop vllm-server || true
docker rm vllm-server || true
sudo rm -rf /home/ubuntu/.cache/neuron/*
sudo rm -rf /var/tmp/neuron-compile-cache

devices=""
for dev in /dev/neuron*; do
    if [ -e "$dev" ]; then
        devices="$devices --device $dev"
    fi
done

hf_token_expr=$(aws ssm get-parameter --name /vllm/HF_TOKEN --with-decryption --query Parameter.Value --output text 2>/dev/null || echo '')

docker run -d --name vllm-server \\
  --no-healthcheck \\
  $devices \\
  --ipc=host \\
  --restart no \\
  -p 8080:8080 \\
  -e HF_TOKEN="$hf_token_expr" \\
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
  bash /patch_and_run.sh
'''

resp = ssm.send_command(
    InstanceIds=["i-0af2ceb15e7807e96"],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
print(f"Command sent! Command ID: {resp['Command']['CommandId']}")
