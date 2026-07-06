import boto3, os, time, subprocess, base64
REGION = open("active_deployment_region.txt").read().strip()
subprocess.run("bash save-aws-creds.sh", cwd="/home/xbill/gemma4-tips-aws", shell=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()
key = open("/home/xbill/.anthropic_key").read().strip()
KEYB64 = base64.b64encode(key.encode()).decode()

ec2 = boto3.client("ec2", region_name=REGION); ssm = boto3.client("ssm", region_name=REGION)
iid = ec2.describe_instances(Filters=[
    {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
    {"Name": "instance-state-name", "Values": ["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]

# single-quoted inner scripts so the HOST shell does not substitute $()/redirects
remote = (
  "set +e\n"
  "docker exec autoport bash -lc 'echo " + KEYB64 + " | base64 -d > /root/.anthropic_key; chmod 600 /root/.anthropic_key'\n"
  "docker exec autoport bash -lc 'echo keyfile_bytes=$(wc -c < /root/.anthropic_key); head -c 11 /root/.anthropic_key; echo'\n"
  "echo '=== claude smoke test (root bypass + auth + Fable) ==='\n"
  "docker exec -e HOME=/root -e IS_SANDBOX=1 autoport bash -lc '"
  "export CLAUDE_CODE_OAUTH_TOKEN=$(cat /root/.anthropic_key); "
  "export ANTHROPIC_MODEL=claude-fable-5; "
  "timeout 150 /root/.local/bin/claude -p \"Reply with exactly the single word: PONG\" "
  "--dangerously-skip-permissions --output-format text 2>&1 | head -40'\n"
)
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]}, TimeoutSeconds=300)["Command"]["CommandId"]
for _ in range(50):
    time.sleep(4)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", "")[-5000:])
if o.get("StandardErrorContent","").strip():
    print("STDERR:", o["StandardErrorContent"][-1500:])
