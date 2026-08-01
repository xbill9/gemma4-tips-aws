import boto3
import os
import sys
import time

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
instance_id = "i-03a304a0cd999dcef"

commands = ["docker logs --tail 150 vllm-server"]
if len(sys.argv) > 1:
    commands = [sys.argv[1]]

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": commands}
)
command_id = resp["Command"]["CommandId"]

# Wait for command completion
while True:
    time.sleep(1)
    out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
    status = out["Status"]
    if status in ["Success", "Failed", "Cancelled", "TimedOut"]:
        print(out.get("StandardOutputContent", ""))
        stderr = out.get("StandardErrorContent", "")
        if stderr:
            print("--- STDERR ---", file=sys.stderr)
            print(stderr, file=sys.stderr)
        break
