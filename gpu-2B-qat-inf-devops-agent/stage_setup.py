import boto3, os, time, subprocess
REGION = open("active_deployment_region.txt").read().strip()
subprocess.run("bash save-aws-creds.sh", cwd="/home/xbill/gemma4-tips-aws", shell=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()
ec2 = boto3.client("ec2", region_name=REGION); ssm = boto3.client("ssm", region_name=REGION)
iid = ec2.describe_instances(Filters=[
    {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
    {"Name": "instance-state-name", "Values": ["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
print("INSTANCE:", iid)

remote = r'''
set +e
echo "=== 0. prevent container auto-restart from re-grabbing the neuron device later ==="
docker update --restart=no vllm-server && echo "restart policy -> no"

echo "=== 1. install Claude Code inside the container (HOME=/root) ==="
docker exec -e HOME=/root vllm-server bash -lc '
if which claude >/dev/null 2>&1; then echo "claude already present: $(which claude)"; else
  curl -fsSL https://claude.ai/install.sh -o /tmp/claude_install.sh 2>/tmp/cerr && bash /tmp/claude_install.sh </dev/null >/tmp/clog 2>&1
  tail -5 /tmp/clog
fi
# ensure on PATH
export PATH="$HOME/.local/bin:$PATH"
which claude && claude --version 2>&1 | head -1 || echo "NO claude on PATH after install"
'

echo "=== 2. deploy the neuron agentic-development skills into ~/.claude ==="
docker exec -e HOME=/root vllm-server bash -lc '
/opt/conda/bin/deploy-neuron-agentic-development-to-claude </dev/null 2>&1 | tail -15
echo "--- skills landed? ---"
ls ~/.claude/skills 2>/dev/null
ls ~/.claude/skills/neuron-framework-autoport/SKILL.md 2>/dev/null && echo "SKILL.md OK" || echo "SKILL.md MISSING"
'

echo "=== 3. create persistent work dir + write copy-paste invocation ==="
docker exec -e HOME=/root vllm-server bash -lc '
mkdir -p /workspace/port
REF=/tmp/refenv/lib/python3.12/site-packages/transformers/models/gemma4
cat > /workspace/port/INVOKE.txt <<EOF
/neuron-framework-autoport Gemma4ForConditionalGeneration $REF modeling_gemma4.py configuration_gemma4.py google/gemma-4-E2B-it /workspace/vllm/local-models/google/gemma-4-E2B-it /tmp/refenv
EOF
echo "wrote /workspace/port/INVOKE.txt:"; cat /workspace/port/INVOKE.txt
'

echo "=== 4. sanity: refenv still produces the golden reference ==="
docker exec vllm-server bash -lc "/tmp/refenv/bin/python - <<PY 2>&1 | tail -3
from transformers import pipeline
p=pipeline('text-generation', model='/workspace/vllm/local-models/google/gemma-4-E2B-it', device='cpu', max_new_tokens=6)
print(repr(p('The capital of France is')[0]['generated_text']))
PY" || echo "(reference check skipped/failed — non-fatal for staging)"
echo "=== DONE staging ==="
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]}, TimeoutSeconds=1800)["Command"]["CommandId"]
for _ in range(120):
    time.sleep(5)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", "")[-6000:])
if o.get("StandardErrorContent","").strip():
    print("STDERR:", o["StandardErrorContent"][-2500:])
