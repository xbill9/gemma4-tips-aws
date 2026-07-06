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

WSVOL = "29ec3ad0a2dc2147a8c0258ba7c830b7f8318f03521323834ab69f19bbda388c"

remote = r'''
set +e
if docker ps -a --format '{{.Names}}' | grep -qx autoport; then
  echo "autoport container already exists:"; docker ps -a --filter name=autoport --format '{{.Names}} {{.Status}}'
else
  echo "=== 1. stop the (gibberish) serving container -> frees /dev/neuron0 ==="
  docker stop vllm-server && echo "vllm-server stopped (retained for rollback: docker start vllm-server)"
  echo "=== 2. commit its writable layer (claude, skills, /tmp/refenv, NxD libs) ==="
  docker commit vllm-server gemma-autoport:latest && echo "committed gemma-autoport:latest"
  echo "=== 3. launch idle device-free container (no vLLM), same device + workspace volume ==="
  docker run -d --name autoport \
    --device /dev/neuron0 --shm-size 8g \
    -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
    -v /home/ubuntu/.cache/neuron:/root/.cache/neuron \
    -v /home/ubuntu/.cache/neuron:/var/tmp/neuron-compile-cache \
    -v ''' + WSVOL + r''':/workspace \
    --entrypoint bash gemma-autoport:latest -lc 'sleep infinity' \
    && echo "autoport container up"
fi

echo "=== 4. verify environment inside autoport ==="
docker exec -e HOME=/root autoport bash -lc '
export PATH=$HOME/.local/bin:$PATH
echo "claude: $(which claude)  $(claude --version 2>&1 | head -1)"
echo -n "skill: "; ls /root/.claude/skills/neuron-framework-autoport/SKILL.md 2>/dev/null || echo MISSING
echo -n "refenv py: "; ls /tmp/refenv/bin/python 2>/dev/null || echo MISSING
echo -n "weights: "; ls -d /workspace/vllm/local-models/google/gemma-4-E2B-it 2>/dev/null || echo MISSING
echo "invoke line:"; cat /workspace/port/INVOKE.txt 2>/dev/null
'
echo "=== 5. is the Neuron device now FREE / claimable? ==="
docker exec autoport bash -lc 'neuron-ls 2>&1 | head -20; echo "--- runtime check ---"; python3 -c "import torch_neuronx" 2>&1 | head -3 || true'
echo "=== DONE cutover ==="
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
