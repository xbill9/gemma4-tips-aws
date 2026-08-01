import boto3
import os
import time

# Load cached credentials
creds = {}
if os.path.exists(".aws_creds"):
    with open(".aws_creds", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v

for k, v in creds.items():
    os.environ[k] = v

ssm = boto3.client("ssm", region_name="us-east-1")
instance_id = "i-0af2ceb15e7807e96"

commands = [
    "docker ps -a",
    "docker logs vllm-server 2>&1 | grep -v 'weight' | tail -n 150"
]

print(f"Sending SSM command to instance {instance_id}...")
response = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": commands}
)

command_id = response["Command"]["CommandId"]
print(f"Command sent. ID: {command_id}. Waiting for output...")

while True:
    time.sleep(2)
    result = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
    status = result["Status"]
    if status in ["Success", "Failed", "Cancelled", "TimedOut"]:
        print(f"--- STATUS: {status} ---")
        print("--- STDOUT ---")
        print(result.get("StandardOutputContent", ""))
        print("--- STDERR ---")
        print(result.get("StandardErrorContent", ""))
        break
