import boto3
import asyncio
import os
import server

async def main():
    hf_token = await server.get_secret() or ""
    ssm = boto3.client('ssm', region_name='us-west-2')
    
    container_script = """#!/bin/bash
set -e
echo "Upgrading transformers..."
pip install --upgrade transformers

echo "Patching transformers..."
mkdir -p /opt/conda/lib/python3.12/site-packages/transformers/utils
cat << 'INNER_EOF' > /opt/conda/lib/python3.12/site-packages/transformers/utils/fx.py
import torch.fx

class HFTracer(torch.fx.Tracer):
    pass

def symbolic_trace(model, *args, **kwargs):
    return torch.fx.symbolic_trace(model)
INNER_EOF

cat << 'INNER_EOF' >> /opt/conda/lib/python3.12/site-packages/transformers/generation/__init__.py

from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput
INNER_EOF

echo "Starting vLLM Server..."
python3 -m vllm.entrypoints.openai.api_server \\
  --model google/gemma-4-12B-it \\
  --quantization neuron_quant \\
  --max-model-len 16384 \\
  --tensor-parallel-size 2 \\
  --max-num-seqs 8 \\
  --async-scheduling \\
  --block-size 16 \\
  --host 0.0.0.0 \\
  --port 8080
"""

    host_commands = [
        "docker stop vllm-server || true",
        "docker rm vllm-server || true",
        f"cat << 'OUTER_EOF' > /home/ubuntu/patch_and_run.sh\n{container_script}\nOUTER_EOF",
        "chmod +x /home/ubuntu/patch_and_run.sh",
        f"docker run -d --name vllm-server --device /dev/neuron0 --ipc=host --restart always -p 8080:8080 -e HF_TOKEN=\"{hf_token}\" -e NEURON_CC_FLAGS=\"--model-type transformer\" -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface -v /home/ubuntu/.cache/neuron:/root/.cache/neuron -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 bash /patch_and_run.sh"
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
