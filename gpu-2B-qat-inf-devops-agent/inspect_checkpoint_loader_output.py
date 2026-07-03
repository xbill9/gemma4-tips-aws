import boto3
import os

region = "us-east-2"
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
import torch
import os
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM, Gemma3InferenceConfig
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

# Setup a dummy config
neuron_config = NeuronConfig(
    tp_degree=2,
    quantized=False,
    model_name_or_path='google/gemma-4-E2B-it'
)
config = Gemma3InferenceConfig(
    neuron_config=neuron_config,
    num_hidden_layers=36,
    hidden_size=1536,
    intermediate_size=6144,
    num_attention_heads=12,
    num_key_value_heads=4,
    vocab_size=256000,
    use_double_wide_mlp=True
)

print('Calling get_state_dict...')
state_dict = NeuronGemma3ForCausalLM.get_state_dict('google/gemma-4-E2B-it', config)
key = 'layers.30.mlp.gate_proj.weight'
if key in state_dict:
    print(f'{key} shape in state_dict: {list(state_dict[key].shape)}')
else:
    # search prefix model.layers.30.mlp.gate_proj.weight
    for k in state_dict.keys():
        if 'layers.30.mlp.gate_proj.weight' in k:
            print(f'Found key: {k} with shape {list(state_dict[k].shape)}')
"
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
command_id = resp['Command']['CommandId']
import time
for poll in range(30):
    try:
        out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        status = out.get('Status')
        if status in ['Success', 'Failed', 'Cancelled', 'TimedOut']:
            print(f"FINAL STATUS: {status}")
            print("STDOUT:")
            print(out.get('StandardOutputContent'))
            print("STDERR:")
            print(out.get('StandardErrorContent'))
            break
    except Exception as e:
        pass
    time.sleep(2)
