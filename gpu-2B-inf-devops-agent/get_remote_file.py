import sys
import boto3
import time
import base64

if len(sys.argv) != 3:
    print("Usage: python3 get_remote_file.py <remote_path> <local_path>")
    sys.exit(1)

remote_path = sys.argv[1]
local_path = sys.argv[2]

ssm = boto3.client('ssm', region_name='us-east-1')
instance_id = "i-0af2ceb15e7807e96"

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

chunk_size = 15000
start = 0
file_data = b""

print(f"Downloading {remote_path} to {local_path} in chunks...")
while True:
    cmd = f"docker exec vllm-server python3 -c \"import base64; f = open('{remote_path}', 'rb'); f.seek({start}); chunk = f.read({chunk_size}); print(base64.b64encode(chunk).decode()); f.close()\""
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

with open(local_path, "wb") as f:
    f.write(file_data)

print(f"Saved {local_path} successfully. Total size: {len(file_data)} bytes")
