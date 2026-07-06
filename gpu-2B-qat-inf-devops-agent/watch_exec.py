import boto3, os, time, subprocess
REGION = open("active_deployment_region.txt").read().strip()
def refresh():
    subprocess.run("bash save-aws-creds.sh", cwd="/home/xbill/gemma4-tips-aws", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()
REMOTE='docker logs vllm-server 2>&1 | grep -oE "\\[EXEC [A-Z_]+\\]" | sort | uniq -c'
deadline=time.time()+8*60; i=0
while time.time()<deadline:
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
        print(f"[{i}] transient {type(e).__name__}",flush=True); time.sleep(20); continue
    lines=[l for l in (out or "").splitlines() if l.strip()]
    print(f"[{i}] exec_tags={len(lines)}",flush=True)
    if lines:
        print("=== EXECUTED PATHS ===",flush=True)
        for l in lines: print(l,flush=True)
        break
    time.sleep(25)
else:
    print("timed out",flush=True)
