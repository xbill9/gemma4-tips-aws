import boto3
import os
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
instance_id = "i-08dc36bcfb8241ee5"

commands = [
    "docker exec vllm-server grep -r -n 'update_cache_const_indices' /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/ || echo 'not found'"
]

response = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": commands}
)

command_id = response["Command"]["CommandId"]
while True:
    time.sleep(1)
    result = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
    status = result["Status"]
    if status in ["Success", "Failed", "Cancelled", "TimedOut"]:
        print("--- STDOUT ---")
        print(result.get("StandardOutputContent", ""))
        print("--- STDERR ---")
        print(result.get("StandardErrorContent", ""))
        break
