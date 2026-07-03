import boto3
import os
import sys

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
if not resp["Reservations"] or not resp["Reservations"][0]["Instances"]:
    print("No running instance found!")
    sys.exit(1)

instance_id = resp["Reservations"][0]["Instances"][0]["InstanceId"]
print(f"Targeting Instance ID: {instance_id}")

script = '''#!/bin/bash
cat << 'EOF' > /tmp/inspect_safetensors.py
import os
import glob
import struct
import json

cache_dir = '/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots'
snapshot_dirs = glob.glob(cache_dir + '/*')
if not snapshot_dirs:
    print('No snapshots found in cache_dir')
    exit(1)
latest_snapshot = snapshot_dirs[0]
print('Latest snapshot path:', latest_snapshot)

safetensor_files = glob.glob(latest_snapshot + '/*.safetensors')
print('Safetensor files found:', safetensor_files)

for f in safetensor_files:
    print('--- Keys in file:', os.path.basename(f))
    try:
        with open(f, 'rb') as fp:
            header_size_bytes = fp.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            header_json_bytes = fp.read(header_size)
            header = json.loads(header_json_bytes.decode('utf-8'))
            
            keys = [k for k in header.keys() if k != '__metadata__']
            print('Total keys in this file:', len(keys))
            
            for key in keys:
                if 'embed_tokens' in key or 'layers.' not in key:
                    meta = header[key]
                    print(f'  Key: {key}, Shape: {meta.get("shape")}, Dtype: {meta.get("data_type")}')

    except Exception as e:
        print('Error parsing safetensors:', e)
EOF
python3 /tmp/inspect_safetensors.py
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
