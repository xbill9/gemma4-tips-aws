import boto3
import os

region = "us-east-1"
if os.path.exists("active_deployment_region.txt"):
    with open("active_deployment_region.txt", "r") as f:
        region = f.read().strip()

creds = {}
if os.path.exists(".aws_creds"):
    with open(".aws_creds", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v
for k, v in creds.items():
    os.environ[k] = v

ec2 = boto3.client('ec2', region_name=region)
ssm = boto3.client('ssm', region_name=region)

resp = ec2.describe_instances(
    Filters=[
        {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
        {"Name": "instance-state-name", "Values": ["running"]}
    ]
)
instance_id = resp["Reservations"][0]["Instances"][0]["InstanceId"]

script = '''#!/bin/bash
docker exec vllm-server python3 -c "
import gc
import torch
from transformers import AutoConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.gemma3.config import Gemma3InferenceConfig
from neuronx_distributed_inference.utils.constants import Gemma3NeuronConfig

# Initialize dummy configs to inspect
import os
cache_dir = '/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/'
snapshot = os.listdir(cache_dir)[0]
model_path = os.path.join(cache_dir, snapshot)
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
neuron_config = Gemma3NeuronConfig(tp_degree=2)
inf_config = Gemma3InferenceConfig(neuron_config=neuron_config)
# Copy attributes
for k, v in config.__dict__.items():
    setattr(inf_config, k, v)

# Initialize model (without weights to be fast)
# Wait, we can just print the updated configs from get_updated_configs
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import get_updated_configs
updated = get_updated_configs(inf_config)
for idx, uc in enumerate(updated):
    print(f'Layer {idx} intermediate_size:', uc.intermediate_size)
"
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
command_id = resp['Command']['CommandId']
import time
time.sleep(3)
out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
print("STATUS:", out.get('Status'))
print("STDOUT:")
print(out.get('StandardOutputContent'))
print("STDERR:")
print(out.get('StandardErrorContent'))
