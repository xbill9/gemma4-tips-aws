import boto3, os, time, subprocess
REGION = open("active_deployment_region.txt").read().strip()
subprocess.run("bash save-aws-creds.sh", cwd="/home/xbill/gemma4-tips-aws", shell=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()
ec2 = boto3.client("ec2", region_name=REGION); ssm = boto3.client("ssm", region_name=REGION)
r = ec2.describe_instances(Filters=[
    {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
    {"Name": "instance-state-name", "Values": ["running"]}])["Reservations"][0]["Instances"][0]
iid = r["InstanceId"]
life = r.get("InstanceLifecycle", "on-demand")
itype = r["InstanceType"]
print("INSTANCE:", iid, itype, "lifecycle=", life, "region=", REGION)

remote = r'''
echo "=== HOST: who holds the neuron device ==="
ls -l /dev/neuron0 2>/dev/null || echo "no /dev/neuron0 on host"
echo "--- neuron-top / processes ---"
(neuron-ls 2>/dev/null | head -30) || echo "neuron-ls n/a on host"
echo "=== docker containers ==="
docker ps --format '{{.Names}}  {{.Status}}  {{.Image}}'
echo "=== disk ==="
df -h / /home 2>/dev/null | grep -vE '^tmpfs'
echo "=== claude CLI present on host? ==="
which claude || echo "no host claude"
echo "=== INSIDE vllm-server container ==="
docker exec vllm-server bash -lc '
echo "-- refenv transformers gemma4 dir --"
ls -d /tmp/refenv/lib/python*/site-packages/transformers/models/gemma4 2>/dev/null || echo "NO gemma4 dir in refenv"
echo "-- refenv transformers version --"
/tmp/refenv/bin/python -c "import transformers,os;print(transformers.__version__, os.path.dirname(transformers.__file__))" 2>&1 | head -3
echo "-- gemma4 modeling/config files --"
ls /tmp/refenv/lib/python*/site-packages/transformers/models/gemma4/*.py 2>/dev/null | sed "s#.*/##"
echo "-- weights dir (HF cache + vllm local-models) --"
ls -d /root/.cache/huggingface/hub/*gemma-4* 2>/dev/null
ls -d /workspace/vllm/local-models/google/gemma-4-E2B-it 2>/dev/null
find / -maxdepth 6 -type d -name "gemma-4-E2B-it" 2>/dev/null | head
echo "-- neuron device visible in container? --"
ls -l /dev/neuron0 2>/dev/null || echo "no neuron dev in container"
echo "-- claude CLI in container? --"
which claude || echo "no container claude"
echo "-- agentic dev tooling present? --"
ls /opt/conda/lib/python3.12/site-packages/neuron_agentic_development/ 2>/dev/null | head
which deploy-neuron-agentic-development-to-claude 2>/dev/null || echo "no deploy cli on PATH"
'
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]}, TimeoutSeconds=600)["Command"]["CommandId"]
for _ in range(60):
    time.sleep(4)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip():
    print("STDERR:", o["StandardErrorContent"][:2000])
