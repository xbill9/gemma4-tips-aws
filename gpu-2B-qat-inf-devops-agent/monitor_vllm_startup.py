import boto3
import os
import time

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

print(f"Monitoring vLLM startup logs on instance {instance_id}...")

# Poll for up to 5 minutes
for i in range(20):
    script = '''#!/bin/bash
docker logs vllm-server 2>&1 | tail -n 25
'''
    resp = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [script]}
    )
    command_id = resp['Command']['CommandId']
    time.sleep(4)
    out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
    stdout = out.get('StandardOutputContent', '')
    print(f"\n--- Poll {i+1} ---")
    print(stdout)
    
    if "Uvicorn running on http://0.0.0.0:8080" in stdout or "Application startup complete" in stdout:
        print("Server is fully up and running!")
        break
    time.sleep(11)
