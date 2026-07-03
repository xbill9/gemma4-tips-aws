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

script = """#!/bin/bash
docker exec vllm-server python3 -c "
filepath = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py'
with open(filepath, 'r') as f:
    content = f.read()

target = '''            partial_factor = 1.0 if is_swa_layer else 0.25
            rotary_dim = int((256 if is_swa_layer else 512) * partial_factor)
            half_head_dim = cos_cache.shape[-1] // 2
            half_rot_dim = rotary_dim // 2'''

replacement = '''            partial_factor = 1.0 if is_swa_layer else 0.25
            rotary_dim = int((256 if is_swa_layer else 512) * partial_factor)
            half_head_dim = cos_cache.shape[-1] // 2
            half_rot_dim = rotary_dim // 2
            print(f'[ROPE_DEBUG] idx={getattr(self, \\"layer_idx\\", None)} is_swa={is_swa_layer} Q={Q.shape} K={K.shape} cos={cos_cache.shape} rotary_dim={rotary_dim} half_head_dim={half_head_dim} half_rot_dim={half_rot_dim}', flush=True)'''

if target in content:
    content = content.replace(target, replacement)
    with open(filepath, 'w') as f:
        f.write(content)
    print('Successfully added prints!')
else:
    print('Target not found in attention_base.py')
"
"""

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
