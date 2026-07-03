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
import os
import torch
from transformers import AutoConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM
from neuronx_distributed_inference.models.gemma3.config import Gemma3InferenceConfig
from neuronx_distributed_inference.utils.constants import Gemma3NeuronConfig

# Initialize dummy configs to inspect
cache_dir = '/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/'
snapshot = os.listdir(cache_dir)[0]
model_path = os.path.join(cache_dir, snapshot)
config = AutoConfig.from_pretrained(model_path, local_files_only=True)

neuron_config = Gemma3NeuronConfig(tp_degree=2)
inf_config = Gemma3InferenceConfig(neuron_config=neuron_config)
for k, v in config.__dict__.items():
    setattr(inf_config, k, v)
# Also copy nested text_config properties if any
if hasattr(config, 'text_config'):
    for k, v in config.text_config.__dict__.items():
        setattr(inf_config, k, v)

# Print properties of inf_config to verify they are loaded
print('final_logit_softcapping:', getattr(inf_config, 'final_logit_softcapping', None))
print('use_double_wide_mlp:', getattr(inf_config, 'use_double_wide_mlp', None))

# Initialize the model structure without executing xla
model = NeuronGemma3ForCausalLM(inf_config)
print('Layer 0 mlp gate_proj weight shape:', model.model.layers[0].mlp.gate_proj.weight.shape)
print('Layer 15 mlp gate_proj weight shape:', model.model.layers[15].mlp.gate_proj.weight.shape)
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
