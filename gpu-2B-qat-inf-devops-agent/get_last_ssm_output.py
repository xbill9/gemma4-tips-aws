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

# Get the most recent command
commands = ssm.list_commands(InstanceId=instance_id, MaxResults=1)
if commands['Commands']:
    cmd = commands['Commands'][0]
    cmd_id = cmd['CommandId']
    print(f"Latest Command ID: {cmd_id}")
    print(f"Status: {cmd['Status']}")
    try:
        out = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
        print("STDOUT:")
        print(out.get('StandardOutputContent'))
        print("STDERR:")
        print(out.get('StandardErrorContent'))
    except Exception as e:
        print(f"Error fetching invocation: {e}")
else:
    print("No commands found.")
