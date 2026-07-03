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
import os, glob
from safetensors import safe_open
# Find model.safetensors in huggingface cache
cache_dir = '/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots'
paths = glob.glob(os.path.join(cache_dir, '*', 'model.safetensors'))
if not paths:
    print('model.safetensors not found!')
else:
    print('Found safetensors:', paths[0])
    with safe_open(paths[0], framework='pt', device='cpu') as f:
        keys = list(f.keys())
    # Find k_proj and v_proj indices
    k_indices = set()
    v_indices = set()
    q_indices = set()
    for k in keys:
        if 'language_model.model.layers' in k or 'model.layers' in k:
            if 'self_attn.k_proj.weight' in k:
                parts = k.split('.')
                for p in parts:
                    if p.isdigit():
                        k_indices.add(int(p))
            elif 'self_attn.v_proj.weight' in k:
                parts = k.split('.')
                for p in parts:
                    if p.isdigit():
                        v_indices.add(int(p))
            elif 'self_attn.q_proj.weight' in k:
                parts = k.split('.')
                for p in parts:
                    if p.isdigit():
                        q_indices.add(int(p))
    print('q_proj layer indices:', sorted(list(q_indices)))
    print('k_proj layer indices:', sorted(list(k_indices)))
    print('v_proj layer indices:', sorted(list(v_indices)))
    print('Total keys:', len(keys))
"
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
command_id = resp['Command']['CommandId']
import time
time.sleep(5)
out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
print("STATUS:", out.get('Status'))
print("STDOUT:")
print(out.get('StandardOutputContent'))
print("STDERR:")
print(out.get('StandardErrorContent'))
