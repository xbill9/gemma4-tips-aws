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
echo "=== LATEST NORM_DBG (tail) for input_layernorm/q_layernorm ==="
docker logs vllm-server 2>&1 | grep "NORM_DBG" | grep -E "layers.0.input_layernorm|layers.0.self_attn.q_layernorm|norm.weight mean" | tail -4
echo "=== logprobs now (uniform-check) ==="
docker exec vllm-server bash -lc 'curl -s -m 60 http://localhost:8080/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"google/gemma-4-E2B-it\",\"prompt\":\"The capital of France is\",\"max_tokens\":4,\"temperature\":0,\"logprobs\":6}"' | python3 -c "
import sys,json
d=json.load(sys.stdin); c=d[\"choices\"][0]
tl=(c.get(\"logprobs\") or {}).get(\"top_logprobs\")
if tl:
    for i,step in enumerate(tl):
        items=sorted(step.items(),key=lambda kv:-kv[1])[:6]
        print(f\"step{i}:\",[(repr(k),round(v,2)) for k,v in items])
"
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"]); print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
