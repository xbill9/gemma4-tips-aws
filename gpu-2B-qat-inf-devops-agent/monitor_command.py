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

command_id = "16f64ee4-438b-4112-b968-5ea45ad843ae"

print(f"Monitoring command {command_id} on instance {instance_id}...")

for _ in range(30):
    out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
    status = out.get('Status')
    print(f"Status: {status}")
    if status in ['Success', 'Failed', 'Cancelled', 'TimedOut']:
        print("STDOUT:")
        print(out.get('StandardOutputContent'))
        print("STDERR:")
        print(out.get('StandardErrorContent'))
        break
    time.sleep(2)
