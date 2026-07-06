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
TP=/opt/conda/lib/python3.12/site-packages/neuronx_distributed/trace/trace.py
docker cp vllm-server:$TP /tmp/trace_edit.py
python3 - <<'PYEOF'
T="/tmp/trace_edit.py"
s=open(T).read()
if "SHARD_DBG" in s:
    print("already instrumented"); raise SystemExit
lines=s.split("\n"); out=[]
anchor="is_tensor_mp = (hasattr(module_parameter,"
n=0
for l in lines:
    out.append(l)
    if anchor in l:
        ind=l[:len(l)-len(l.lstrip())]
        out.append(ind+'if "qkv_proj.q_proj" in parameter_name:')
        out.append(ind+'    import sys; sys.stderr.write("[SHARD_DBG] %s mod=%s ckpt=%s mp=%s mtmp=%s ttmp=%s pdim=%s\\n" % (parameter_name, tuple(module_parameter.shape), tuple(tensor.shape), is_tensor_mp, getattr(module_parameter,"tensor_model_parallel",None), getattr(tensor,"tensor_model_parallel",None), getattr(module_parameter,"partition_dim",None))); sys.stderr.flush()')
        n+=1
open(T,"w").write("\n".join(out))
print("instrumented anchor sites:", n)
PYEOF
docker cp /tmp/trace_edit.py vllm-server:$TP
echo "verify:"; docker exec -i vllm-server grep -n "SHARD_DBG" $TP 2>/dev/null | head || grep -n SHARD_DBG /tmp/trace_edit.py | head
echo "=== docker start (reuse compiled cache) ==="
docker start vllm-server
sleep 2
docker ps --filter name=vllm-server --format 'STATE={{.Status}}'
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:2000])
