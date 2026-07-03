import boto3
import time
import os
import json

region = "us-east-1"
if os.path.exists("active_deployment_region.txt"):
    with open("active_deployment_region.txt", "r") as f:
        region = f.read().strip()

print(f"Using Region: {region}")

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
instance_ids = []
for reservation in resp.get("Reservations", []):
    for instance in reservation.get("Instances", []):
        instance_ids.append(instance["InstanceId"])

if not instance_ids:
    print("No running instance found!")
    exit(1)

instance_id = instance_ids[0]
print(f"Active Instance: {instance_id}")

script = '''#!/bin/bash
curl -s -X POST http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "google/gemma-4-E2B-it", "messages": [{"role": "user", "content": "is a hotdog a sandwich"}], "temperature": 0.0, "max_tokens": 100}'
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
command_id = resp['Command']['CommandId']
time.sleep(3)

for _ in range(15):
    try:
        out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        if out.get('Status') in ['Success', 'Failed']:
            stdout = out.get('StandardOutputContent', '')
            try:
                # Pretty print JSON if returned
                parsed = json.loads(stdout)
                print(json.dumps(parsed, indent=2))
            except Exception:
                print(stdout)
            break
    except Exception as e:
        pass
    time.sleep(2)
