import boto3, os, time, subprocess
REGION = open("active_deployment_region.txt").read().strip()
def refresh():
    subprocess.run("bash save-aws-creds.sh", cwd="/home/xbill/gemma4-tips-aws", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()

REMOTE = r'''
docker ps -a --filter name=vllm-server --format 'STATE={{.Status}}'
echo "---DBG---"
docker logs vllm-server 2>&1 | grep "SHARD_DBG" | head -30
echo "---ERR---"
docker logs --tail 120 vllm-server 2>&1 | grep -iE "expected shape|Compiler status PASS|Recompiling|Uvicorn running|Application startup complete|Engine core init" | tail -4
'''
deadline = time.time() + 12*60
i=0
while time.time() < deadline:
    i+=1
    try:
        refresh()
        ec2=boto3.client("ec2",region_name=REGION); ssm=boto3.client("ssm",region_name=REGION)
        iid=ec2.describe_instances(Filters=[{"Name":"tag:Name","Values":["inferentia-2b-devops-agent"]},{"Name":"instance-state-name","Values":["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
        cid=ssm.send_command(InstanceIds=[iid],DocumentName="AWS-RunShellScript",Parameters={"commands":[REMOTE]})["Command"]["CommandId"]
        out=None
        for _ in range(30):
            time.sleep(3)
            oo=ssm.get_command_invocation(CommandId=cid,InstanceId=iid)
            if oo["Status"] not in ("Pending","InProgress"): out=oo.get("StandardOutputContent",""); break
    except Exception as e:
        print(f"[{i}] transient {type(e).__name__}", flush=True); time.sleep(25); continue
    dbg = [l for l in (out or "").splitlines() if "SHARD_DBG" in l]
    state = next((l for l in (out or "").splitlines() if l.startswith("STATE=")), "")
    err = "expected shape" in (out or "")
    print(f"[{i}] {state[:36]} dbg_lines={len(dbg)} err={err}", flush=True)
    if dbg:
        print("=== SHARD_DBG ===", flush=True)
        for l in dbg: print(l, flush=True)
        print("=== state:", state, flush=True)
        break
    if "Exited" in state:
        print("EXITED without SHARD_DBG. Recent:", flush=True)
        print(out, flush=True); break
    time.sleep(30)
else:
    print("watch timed out", flush=True)
