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
with open('/opt/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py', 'r') as f:
    content = f.read()

# Print class definitions and load_weights
import re
for m in re.finditer(r'(class \w+|def load_weights)', content):
    start = max(0, m.start() - 100)
    end = min(len(content), m.end() + 500)
    print('MATCH:', content[m.start():m.end()])
    print(content[start:end])
    print('='*40)
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
