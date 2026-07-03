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
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM, Gemma3InferenceConfig
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

neuron_config = NeuronConfig(tp_degree=2, quantized=False)
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

config = Gemma3InferenceConfig(neuron_config=neuron_config, load_config=custom_load_config)
model = NeuronGemma3ForCausalLM(config).context_encoding_model

def search_hooks(module, prefix=''):
    if hasattr(module, 'preshard_hook'):
        print(f'{prefix} has preshard_hook!')
    for name, child in module._modules.items():
        if child is not None:
            search_hooks(child, prefix + name + '.')

search_hooks(model)
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
