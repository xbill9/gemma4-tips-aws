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
remote = r'''
set -e
AB=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py
docker cp vllm-server:$AB /tmp/ab_edit.py
python3 - <<'PYEOF'
T="/tmp/ab_edit.py"
s=open(T).read()
if "SCALE_DBG" in s:
    print("already"); raise SystemExit
anchor="self.softmax_scale = math.sqrt(self.head_dim) if softmax_scale is None else softmax_scale"
assert anchor in s, "anchor missing"
inj=anchor+"\n        import sys as _sd; _sd.stderr.write('[SCALE_DBG] ss=%s hd=%s\\n' % (self.softmax_scale, self.head_dim)); _sd.stderr.flush()"
s=s.replace(anchor, inj)
open(T,"w").write(s)
print("instrumented softmax_scale print")
PYEOF
docker cp /tmp/ab_edit.py vllm-server:$AB
echo "=== restart (print fires at module build, early) ==="
docker restart vllm-server
sleep 2; docker ps --filter name=vllm-server --format 'STATE={{.Status}}'
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
