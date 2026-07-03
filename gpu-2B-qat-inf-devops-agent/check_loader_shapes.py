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
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

model_path = '/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03'
hf_config = AutoConfig.from_pretrained(model_path)
print('hf_config class:', type(hf_config).__name__)

neuron_config_cls = NeuronGemma3ForCausalLM.get_neuron_config_cls()
neuron_config = neuron_config_cls()
neuron_config.tp_degree = 2

# Instantiate Gemma3InferenceConfig
config_cls = NeuronGemma3ForCausalLM.get_config_cls()
config = config_cls(neuron_config=neuron_config, load_config=load_pretrained_config(model_path))
print('vocab_size in config:', config.vocab_size)
print('intermediate_size in config:', config.intermediate_size)

# Load state dict
state_dict = NeuronGemma3ForCausalLM.get_state_dict(model_path, config)
print('Keys in state dict containing layers.15.mlp:')
for k, v in sorted(state_dict.items()):
    if 'layers.15.mlp' in k:
        print(f'{k}: {v.shape}')

# Instantiate model
model = NeuronGemma3ForCausalLM(model_path, config)
print('Model parameter shapes for layers.15.mlp:')
if hasattr(model, 'context_encoding_model') and model.context_encoding_model is not None:
    mw = model.context_encoding_model
    if hasattr(mw, 'model') and mw.model is not None:
        for name, param in mw.model.named_parameters():
            if 'layers.15.mlp' in name:
                print(f'context_encoding_model.{name}: {param.shape}')
if hasattr(model, 'token_generation_model') and model.token_generation_model is not None:
    mw = model.token_generation_model
    if hasattr(mw, 'model') and mw.model is not None:
        for name, param in mw.model.named_parameters():
            if 'layers.15.mlp' in name:
                print(f'token_generation_model.{name}: {param.shape}')
"
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
command_id = resp['Command']['CommandId']
import time
import time
for i in range(25):
    try:
        out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        status = out.get('Status')
        print(f"Poll {i}: STATUS = {status}")
        if status not in ['InProgress', 'Pending']:
            break
    except Exception as e:
        print(f"Poll {i}: exception {e}")
    time.sleep(2)
print("FINAL STATUS:", out.get('Status') if 'out' in locals() else 'Unknown')
if 'out' in locals():
    print("STDOUT:")
    print(out.get('StandardOutputContent'))
    print("STDERR:")
    print(out.get('StandardErrorContent'))
