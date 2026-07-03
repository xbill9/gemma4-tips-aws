import boto3
import time
import os

# 1. Resolve Region
region = "us-east-1"
if os.path.exists("active_deployment_region.txt"):
    with open("active_deployment_region.txt", "r") as f:
        region = f.read().strip()

print(f"Using Region: {region}")

# Load credentials
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

# 2. Resolve Instance ID
resp = ec2.describe_instances(
    Filters=[
        {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]}
    ]
)

instance_ids = []
for reservation in resp.get("Reservations", []):
    for instance in reservation.get("Instances", []):
        instance_ids.append(instance["InstanceId"])

if not instance_ids:
    print("No pending/running instance found!")
    # Fallback to stopped if none is running/pending
    resp_stopped = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
            {"Name": "instance-state-name", "Values": ["stopped"]}
        ]
    )
    for reservation in resp_stopped.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            instance_ids.append(instance["InstanceId"])

if not instance_ids:
    print("Fatal: No instances found at all under 'inferentia-2b-devops-agent' tag.")
    exit(1)

instance_id = instance_ids[0]
print(f"Resolved Active Instance ID: {instance_id}")

script = '''#!/bin/bash
if docker ps -a --filter name=vllm-server | grep -q vllm-server; then
    echo "=== CONTAINER STATUS ==="
    docker ps -a --filter name=vllm-server
    echo "=== RESTART COUNT ==="
    docker inspect vllm-server --format '{{.RestartCount}}' 2>/dev/null || echo "N/A"
    echo "=== LAST 25 LOG LINES ==="
    docker logs vllm-server 2>&1 | grep -v 'weight' | tail -n 25
else
    echo "=== CLOUD-INIT STATUS ==="
    tail -n 30 /var/log/cloud-init-output.log 2>/dev/null || echo "cloud-init-output.log not found"
fi
'''

try:
    resp = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [script]}
    )
except Exception as e:
    print(f"Failed to send SSM command: {e}")
    exit(1)

command_id = resp['Command']['CommandId']
time.sleep(3)

for _ in range(10):
    try:
        out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        if out.get('Status') in ['Success', 'Failed']:
            stdout = out.get('StandardOutputContent', '')
            print(stdout)
            break
    except Exception as e:
        pass
    time.sleep(2)
