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
MG=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py
docker cp vllm-server:$MG /tmp/mg_ple.py
python3 - <<'PYEOF'
T="/tmp/mg_ple.py"; s=open(T).read()
anchor='        per_layer_inputs = getattr(self.parent, "current_per_layer_inputs", None)'
if "PLE_DBG" in s: print("already"); raise SystemExit
assert anchor in s, "anchor missing"
inj=anchor+"\n        import sys as _p; _p.stderr.write('[PLE_DBG] L%s per_layer_inputs_is_None=%s gate_is_None=%s\\n' % (getattr(self,'layer_idx','?'), per_layer_inputs is None, self.per_layer_input_gate is None)); _p.stderr.flush()"
s=s.replace(anchor, inj, 1)
open(T,"w").write(s); print("instrumented PLE check")
PYEOF
docker cp /tmp/mg_ple.py vllm-server:$MG
docker restart vllm-server
sleep 2; docker ps --filter name=vllm-server --format 'S={{.Status}}'
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"]); print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
