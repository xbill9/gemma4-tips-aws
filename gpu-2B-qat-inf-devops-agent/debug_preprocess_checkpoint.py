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
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM, Gemma3InferenceConfig
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed.trace.trace import check_for_duplicate_tensors, invoke_preshard_hook

# Setup a dummy config
neuron_config = NeuronConfig(
    tp_degree=2,
    quantized=False
)
neuron_config.model_name_or_path = 'google/gemma-4-E2B-it'

def custom_load_config(self):
    self.num_hidden_layers = 36
    self.hidden_size = 1536
    self.intermediate_size = 6144
    self.num_attention_heads = 12
    self.num_key_value_heads = 4
    self.vocab_size = 256000
    self.use_double_wide_mlp = True
    self.sliding_window = 512
    self.query_pre_attn_scalar = 22
    self.final_logit_softcapping = 30.0
    self.head_dim = 256

config = Gemma3InferenceConfig(
    neuron_config=neuron_config,
    load_config=custom_load_config
)
config.vocab_size = 256000
config.rms_norm_eps = 1e-06
config.hidden_act = 'gelu_pytorch_tanh'

state_dict = NeuronGemma3ForCausalLM.get_state_dict('google/gemma-4-E2B-it', config)
key = 'layers.30.mlp.gate_proj.weight'
print('Shape after get_state_dict:', list(state_dict[key].shape))

# Instantiate a mock model structure
model = NeuronGemma3ForCausalLM(config)

state_dict = check_for_duplicate_tensors(state_dict, False)
print('Shape after check_for_duplicate_tensors:', list(state_dict[key].shape))

invoke_preshard_hook(model.context_encoding_model, state_dict, '')
print('Shape after invoke_preshard_hook:', list(state_dict[key].shape))
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
