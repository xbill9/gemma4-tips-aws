import boto3
import os
import sys
import time
import base64

creds = {}
if os.path.exists(".aws_creds"):
    with open(".aws_creds", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v

for k, v in creds.items():
    os.environ[k] = v

region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ec2 = boto3.client('ec2', region_name=region)
ssm = boto3.client('ssm', region_name=region)

# Find the active instance
service_name = "inferentia-12b-devops-agent"
response = ec2.describe_instances(
    Filters=[
        {"Name": "tag:Name", "Values": [service_name]},
        {"Name": "instance-state-name", "Values": ["running"]},
    ]
)
instances = []
for reservation in response.get("Reservations", []):
    instances.extend(reservation.get("Instances", []))

if not instances:
    print("No running EC2 instance found.")
    sys.exit(1)

instance_id = instances[0]["InstanceId"]
print(f"Fetching from instance {instance_id}")

def run_ssm_command(commands):
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
            return status, result.get("StandardOutputContent", ""), result.get("StandardErrorContent", "")

# We will fetch in chunks of 15000 bytes
chunk_size = 15000
start = 0
file_data = b""

print("Downloading remote gqa.py in chunks...")
while True:
    print(f"Fetching chunk starting at byte {start}...")
    cmd = f"docker exec vllm-server python3 -c \"import base64; f = open('/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/gqa.py', 'rb'); f.seek({start}); chunk = f.read({chunk_size}); print(base64.b64encode(chunk).decode()); f.close()\""
    status, stdout, stderr = run_ssm_command([cmd])
    if status != "Success":
        print(f"Failed to fetch chunk at {start}. Status: {status}")
        print("Error:", stderr)
        sys.exit(1)
    
    b64_data = stdout.strip()
    chunk_bytes = base64.b64decode(b64_data)
    if not chunk_bytes:
        break
    file_data += chunk_bytes
    if len(chunk_bytes) < chunk_size:
        break
    start += chunk_size

with open("gqa_remote.py", "wb") as f:
    f.write(file_data)

print(f"Saved gqa_remote.py successfully. Total size: {len(file_data)} bytes")
