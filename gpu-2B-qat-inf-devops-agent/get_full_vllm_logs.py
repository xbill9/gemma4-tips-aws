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
docker logs vllm-server > /home/ubuntu/vllm_temp.log 2>&1
echo "=== FILTERED LAST 80 LINES ==="
grep -v -E "UserWarning|warnings.warn|Some weights|were not initialized|'model.layers|STATE DICT|DEBUG STATE|embed_tokens_per_layer|per_layer_input_gate|per_layer_projection" /home/ubuntu/vllm_temp.log | tail -n 80
echo "=== ABSOLUTE LAST 30 LINES ==="
tail -n 30 /home/ubuntu/vllm_temp.log
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
