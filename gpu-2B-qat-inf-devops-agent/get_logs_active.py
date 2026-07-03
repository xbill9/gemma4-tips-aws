import boto3
import time
import os

# Load credentials
creds = {}
if os.path.exists(".aws_creds"):
    with open(".aws_creds", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v
for k, v in creds.items():
    os.environ[k] = v

ssm = boto3.client('ssm', region_name='us-east-1')
instance_id = "i-04b87113ec6c17762"

script = '''#!/bin/bash
echo "=== CONTAINER LOGS (LAST 40 LINES) ==="
docker logs --tail 40 vllm-server 2>&1
echo "=== PS AUX ==="
ps aux | grep vllm
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)

command_id = resp['Command']['CommandId']
print(f"Sent SSM logs command: {command_id}")
time.sleep(3)

for _ in range(5):
    try:
        out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        if out.get('Status') in ['Success', 'Failed']:
            print("OUTPUT:")
            print(out.get('StandardOutputContent'))
            print("ERR:")
            print(out.get('StandardErrorContent'))
            break
    except Exception as e:
        print("Waiting for invocation...", e)
    time.sleep(2)
