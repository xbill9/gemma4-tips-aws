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
if not resp["Reservations"]:
    print("No running instance!")
    exit(1)

instance_id = resp["Reservations"][0]["Instances"][0]["InstanceId"]

# Filter out the verbose model layers listing in bash before returning
script = '''#!/bin/bash
docker logs vllm-server > /home/ubuntu/vllm_current_all.log 2>&1
echo "=== Filtered Logs ==="
grep -v -E "model\\.layers\\.[0-9]+|Some weights of Gemma3ForCausalLM|lm_head\\.weight" /home/ubuntu/vllm_current_all.log | tail -n 150
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
command_id = resp['Command']['CommandId']

for _ in range(15):
    try:
        out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        if out.get('Status') in ['Success', 'Failed']:
            stdout = out.get('StandardOutputContent', '')
            with open("vllm_filtered_logs.txt", "w") as f:
                f.write(stdout)
            print("Successfully wrote filtered logs to vllm_filtered_logs.txt")
            break
    except Exception as e:
        pass
    time.sleep(2)
