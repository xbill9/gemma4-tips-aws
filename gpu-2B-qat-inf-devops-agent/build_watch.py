import boto3, os, time, subprocess

REGION = open("active_deployment_region.txt").read().strip()
CREDS = "/home/xbill/gemma4-tips-aws/.aws_creds"
REPO = "/home/xbill/gemma4-tips-aws"

def refresh():
    subprocess.run("bash save-aws-creds.sh", cwd=REPO, shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for line in open(CREDS):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()

def clients():
    refresh()
    return (boto3.client("ec2", region_name=REGION),
            boto3.client("ssm", region_name=REGION))

def get_iid(ec2):
    return ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
        {"Name": "instance-state-name", "Values": ["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]

REMOTE = r'''
curl -s -m 4 -o /dev/null -w "HEALTH=%{http_code}" http://localhost:8080/health 2>/dev/null; echo
docker ps -a --filter name=vllm-server --format 'STATE={{.Status}}'
docker logs --tail 250 vllm-server 2>&1 | grep -viE "not initialized|newly initialized|Will convert to torch|warnings.warn|DeprecationWarning|UserWarning|nki_jit|blockwise|libcuda" | grep -iE "error|traceback|exception|Application startup complete|Uvicorn running|Started server|api_server.*started|compil|bucket|Traced|Generating|worker_error|Neuron|FAILED|assert" | tail -6
'''

def run_check(ssm, iid):
    cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                           Parameters={"commands": [REMOTE]})["Command"]["CommandId"]
    for _ in range(30):
        time.sleep(3)
        o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
        if o["Status"] not in ("Pending", "InProgress"):
            return o.get("StandardOutputContent", "")
    return "(check timed out)"

deadline = time.time() + 35 * 60
i = 0
while time.time() < deadline:
    i += 1
    try:
        ec2, ssm = clients()
        iid = get_iid(ec2)
        out = run_check(ssm, iid)
    except Exception as e:
        print(f"[iter {i}] transient: {type(e).__name__}: {str(e)[:120]}", flush=True)
        time.sleep(30); continue
    health = ""
    state = ""
    for ln in out.splitlines():
        if ln.startswith("HEALTH="): health = ln.split("=",1)[1].strip()
        if ln.startswith("STATE="): state = ln.split("=",1)[1].strip()
    extra = " | ".join(l.strip() for l in out.splitlines() if not l.startswith(("HEALTH=","STATE=")))
    print(f"[iter {i}] health={health} state={state[:40]} :: {extra[:220]}", flush=True)
    if health == "200":
        print("READY: /health returned 200", flush=True); break
    if "Exited" in state:
        print("CONTAINER EXITED — build failed. Full recent log follows:", flush=True)
        # dump error tail
        err = run_check(ssm, iid)
        print(err, flush=True); break
    time.sleep(45)
else:
    print("WATCH TIMED OUT after 35 min", flush=True)
