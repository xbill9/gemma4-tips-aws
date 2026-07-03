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
import torch
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

model_path = '/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03/'

neuron_config_cls = NeuronGemma3ForCausalLM.get_neuron_config_cls()
neuron_config = neuron_config_cls()
neuron_config.tp_degree = 4

config_cls = NeuronGemma3ForCausalLM.get_config_cls()
config = config_cls(neuron_config=neuron_config, load_config=load_pretrained_config(model_path))

# Instantiate model on meta device using NxD trace helpers
from neuronx_distributed.trace.model_builder import mock_distributed, init_on_device
import torch.distributed as dist
from neuronx_distributed.parallel_layers import parallel_state

with mock_distributed(world_size=4), init_on_device(torch.device('meta'), force_custom_init_on_device=True):
    dist.init_process_group(backend='xla', rank=0, world_size=4)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=4, skip_collective_init=True)
    
    model = NeuronGemma3ForCausalLM(model_path, config)
    
    print('Prefill model parameter shapes:')
    if hasattr(model, 'context_encoding_model') and model.context_encoding_model is not None:
        mw = model.context_encoding_model
        if hasattr(mw, 'model') and mw.model is not None:
            for name, param in mw.model.named_parameters():
                if 'layers.14.mlp.gate_proj' in name or 'layers.15.mlp.gate_proj' in name:
                    print(f'prefill.{name}: {param.shape}')
                    
    print('TokenGen model parameter shapes:')
    if hasattr(model, 'token_generation_model') and model.token_generation_model is not None:
        mw = model.token_generation_model
        if hasattr(mw, 'model') and mw.model is not None:
            for name, param in mw.model.named_parameters():
                if 'layers.14.mlp.gate_proj' in name or 'layers.15.mlp.gate_proj' in name:
                    print(f'tokengen.{name}: {param.shape}')

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
"
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
command_id = resp['Command']['CommandId']
import time
time.sleep(10)
out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
print("STATUS:", out.get('Status'))
print("STDOUT:")
print(out.get('StandardOutputContent'))
print("STDERR:")
print(out.get('StandardErrorContent'))
