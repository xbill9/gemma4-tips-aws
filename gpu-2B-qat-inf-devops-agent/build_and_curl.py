import boto3, os, time, subprocess
REGION = open("active_deployment_region.txt").read().strip()
def refresh():
    subprocess.run("bash save-aws-creds.sh", cwd="/home/xbill/gemma4-tips-aws", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()
def run(ssm, iid, script):
    cid=ssm.send_command(InstanceIds=[iid],DocumentName="AWS-RunShellScript",Parameters={"commands":[script]})["Command"]["CommandId"]
    for _ in range(40):
        time.sleep(3)
        oo=ssm.get_command_invocation(CommandId=cid,InstanceId=iid)
        if oo["Status"] not in ("Pending","InProgress"): return oo.get("StandardOutputContent","")
    return "(timeout)"
HEALTH='curl -s -m4 -o /dev/null -w "%{http_code}" http://localhost:8080/health; echo; docker ps -a --filter name=vllm-server --format "S={{.Status}}"'
CURL=r'''
echo "=== NORM_DBG (should be ~10 now, not 0) ==="
docker logs vllm-server 2>&1 | grep "NORM_DBG" | grep -E "layers.0.input_layernorm|layers.0.self_attn.q_layernorm|layers.34.post_feedforward" | head -3
echo "=== PROMPT 1: The capital of France is ==="
docker exec vllm-server bash -lc 'curl -s -m 60 http://localhost:8080/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"google/gemma-4-E2B-it\",\"prompt\":\"The capital of France is\",\"max_tokens\":25,\"temperature\":0}" | python3 -c "import sys,json;d=json.load(sys.stdin);print(repr(d[\"choices\"][0][\"text\"]))"'
echo "=== PROMPT 2 (chat): What is 2+2? ==="
docker exec vllm-server bash -lc 'curl -s -m 60 http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"google/gemma-4-E2B-it\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer briefly.\"}],\"max_tokens\":25,\"temperature\":0}" | python3 -c "import sys,json;d=json.load(sys.stdin);print(repr(d[\"choices\"][0][\"message\"][\"content\"]))"'
'''
deadline=time.time()+12*60; i=0
while time.time()<deadline:
    i+=1
    try:
        refresh()
        ec2=boto3.client("ec2",region_name=REGION); ssm=boto3.client("ssm",region_name=REGION)
        iid=ec2.describe_instances(Filters=[{"Name":"tag:Name","Values":["inferentia-2b-devops-agent"]},{"Name":"instance-state-name","Values":["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
        h=run(ssm,iid,HEALTH)
    except Exception as e:
        print(f"[{i}] transient {type(e).__name__}",flush=True); time.sleep(25); continue
    code=h.split(chr(10))[0].strip() if h else "?"
    st=next((l for l in (h or "").splitlines() if l.startswith("S=")),"")
    print(f"[{i}] health={code} {st[:40]}",flush=True)
    if code=="200":
        print("=== READY — DIRECT CURL (bypassing MCP) ===",flush=True)
        print(run(ssm,iid,CURL),flush=True); break
    if "Exited" in st:
        print("EXITED",flush=True); break
    time.sleep(30)
else:
    print("timed out",flush=True)
