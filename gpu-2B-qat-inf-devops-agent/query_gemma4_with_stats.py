import boto3
import time
import os
import sys
import json
import shlex

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

prompt = " ".join(sys.argv[1:]) or "is a hotdog a sandwich"

ec2 = boto3.client('ec2', region_name=region)
ssm = boto3.client('ssm', region_name=region)

resp = ec2.describe_instances(
    Filters=[
        {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
        {"Name": "instance-state-name", "Values": ["running"]}
    ]
)
instance_ids = [
    inst["InstanceId"]
    for r in resp.get("Reservations", [])
    for inst in r.get("Instances", [])
]
if not instance_ids:
    print("No running instance found!")
    sys.exit(1)
instance_id = instance_ids[0]
print(f"Region: {region}  Instance: {instance_id}")
print(f"Prompt: {prompt!r}")

payload = json.dumps({
    "model": "google/gemma-4-E2B-it",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0,
    "max_tokens": 200,
})

script = f'''#!/bin/bash
START=$(date +%s.%N)
RESP=$(curl -s -X POST http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d {shlex.quote(payload)})
END=$(date +%s.%N)
echo "$RESP"
echo "WALL_SECONDS:$(echo "$END - $START" | bc)"
'''

resp = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
command_id = resp['Command']['CommandId']
time.sleep(3)

stdout = None
for _ in range(30):
    try:
        out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        if out.get('Status') in ['Success', 'Failed', 'TimedOut']:
            stdout = out.get('StandardOutputContent', '')
            if out.get('Status') != 'Success':
                print("SSM status:", out.get('Status'))
                print(out.get('StandardErrorContent', ''))
            break
    except ssm.exceptions.InvocationDoesNotExist:
        pass
    time.sleep(2)

if not stdout:
    print("No output from SSM command.")
    sys.exit(1)

wall = None
body_lines = []
for line in stdout.splitlines():
    if line.startswith("WALL_SECONDS:"):
        wall = float(line.split(":", 1)[1])
    else:
        body_lines.append(line)

try:
    parsed = json.loads("\n".join(body_lines).strip())
except Exception:
    print(stdout)
    sys.exit(0)

if "choices" in parsed:
    content = parsed["choices"][0]["message"]["content"]
    usage = parsed.get("usage", {})
    print("\n----- RESPONSE -----")
    print(content)
    print("\n----- STATS -----")
    print(f"finish_reason:     {parsed['choices'][0].get('finish_reason')}")
    print(f"prompt_tokens:     {usage.get('prompt_tokens')}")
    print(f"completion_tokens: {usage.get('completion_tokens')}")
    print(f"total_tokens:      {usage.get('total_tokens')}")
    if wall is not None:
        print(f"wall_time_s:       {wall:.2f}")
        if usage.get("completion_tokens"):
            print(f"tokens_per_s:      {usage['completion_tokens'] / wall:.2f} (incl. prefill+HTTP)")
else:
    print(json.dumps(parsed, indent=2))
