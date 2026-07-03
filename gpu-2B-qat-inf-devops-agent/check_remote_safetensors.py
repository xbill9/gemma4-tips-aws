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
import glob
import os
from safetensors import safe_open

cache_dir = '/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots'
if not os.path.exists(cache_dir):
    print(f'Cache dir {cache_dir} not found')
    # search anywhere in /root/.cache
    files = glob.glob('/root/.cache/**/*.safetensors', recursive=True)
else:
    files = glob.glob(cache_dir + '/**/*.safetensors', recursive=True)

print('Found safetensors:', files)
if files:
    with safe_open(files[0], framework='pt', device='cpu') as f:
        for k in sorted(f.keys()):
            if 'layers.15.mlp' in k or 'layers.14.mlp' in k:
                print(f'{k}: {f.get_tensor(k).shape}')
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
