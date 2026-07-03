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
import os
import glob
from safetensors import safe_open

cache_dir = '/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots'
snapshot_dirs = glob.glob(cache_dir + '/*')
if not snapshot_dirs:
    print('No snapshots found in cache_dir')
    exit(1)
latest_snapshot = snapshot_dirs[0]
print('Latest snapshot path:', latest_snapshot)

layers_found = set()
for f in glob.glob(latest_snapshot + '/*.safetensors'):
    with safe_open(f, framework='pt') as sf:
        for k in sf.keys():
            if 'mlp.gate_proj.weight' in k:
                # e.g., model.layers.15.mlp.gate_proj.weight
                parts = k.split('.')
                for part in parts:
                    if part.isdigit():
                        layers_found.add(int(part))
print('Layers with gate_proj.weight:', sorted(list(layers_found)))
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
