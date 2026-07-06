import boto3, os, time, subprocess, base64
REGION = open("active_deployment_region.txt").read().strip()
subprocess.run("bash save-aws-creds.sh", cwd="/home/xbill/gemma4-tips-aws", shell=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()

key = open("/home/xbill/.anthropic_key").read().strip()
assert key.startswith("sk-ant-"), "key file missing/looks wrong"
KEYB64 = base64.b64encode(key.encode()).decode()

if key.startswith("sk-ant-oat"):
    auth_line = 'unset ANTHROPIC_API_KEY; export CLAUDE_CODE_OAUTH_TOKEN=$(cat /root/.anthropic_key)'
    print("AUTH: CLAUDE_CODE_OAUTH_TOKEN (setup-token)")
else:
    auth_line = 'unset CLAUDE_CODE_OAUTH_TOKEN; export ANTHROPIC_API_KEY=$(cat /root/.anthropic_key)'
    print("AUTH: ANTHROPIC_API_KEY")

launch_sh = ('#!/bin/bash\n'
             'export HOME=/root\n'
             'export PATH=/root/.local/bin:$PATH\n'
             'export IS_SANDBOX=1\n'
             + auth_line + '\n'
             'export ANTHROPIC_MODEL=claude-fable-5\n'
             'cd /workspace/port\n'
             'exec /root/.local/bin/claude -p "$(cat /workspace/port/INVOKE.txt)" '
             '--dangerously-skip-permissions --verbose --output-format stream-json\n')
LAUNCHB64 = base64.b64encode(launch_sh.encode()).decode()

ec2 = boto3.client("ec2", region_name=REGION); ssm = boto3.client("ssm", region_name=REGION)
iid = ec2.describe_instances(Filters=[
    {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
    {"Name": "instance-state-name", "Values": ["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
print("INSTANCE:", iid)

# All inner scripts single-quoted so the HOST shell never substitutes $() / redirects.
remote = (
  "set +e\n"
  "docker exec autoport bash -lc 'echo " + KEYB64 + " | base64 -d > /root/.anthropic_key; chmod 600 /root/.anthropic_key'\n"
  "docker exec autoport bash -lc 'echo " + LAUNCHB64 + " | base64 -d > /root/launch_autoport.sh; chmod +x /root/launch_autoport.sh'\n"
  "docker exec -e HOME=/root autoport bash -lc 'rm -f /workspace/port/run.log; setsid nohup /root/launch_autoport.sh > /workspace/port/run.log 2>&1 < /dev/null & echo LAUNCH_PID $!'\n"
  "sleep 30\n"
  "echo === run.log size ===\n"
  "docker exec autoport bash -lc 'wc -c /workspace/port/run.log; echo ---head---; head -c 3500 /workspace/port/run.log'\n"
  "echo; echo === claude alive? ===\n"
  "docker exec autoport bash -lc \"ps aux | grep -E 'claude|launch_autoport' | grep -v grep | head\"\n"
)
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]}, TimeoutSeconds=600)["Command"]["CommandId"]
for _ in range(50):
    time.sleep(4)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", "")[-6000:])
if o.get("StandardErrorContent","").strip():
    print("STDERR:", o["StandardErrorContent"][-1500:])
