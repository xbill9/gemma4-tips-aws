import boto3, os, time, subprocess, json
REGION=open("active_deployment_region.txt").read().strip()
for k in ("AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_SESSION_TOKEN"): os.environ.pop(k,None)
subprocess.run("bash -lc 'env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN bash save-aws-creds.sh'",
               cwd="/home/xbill/gemma4-tips-aws", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for ln in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
    if "=" in ln and not ln.strip().startswith("#"):
        k,v=ln.strip().split("=",1); os.environ[k.strip()]=v.strip()
ec2=boto3.client("ec2",region_name=REGION); ssm=boto3.client("ssm",region_name=REGION)
iid=ec2.describe_instances(Filters=[{"Name":"tag:Name","Values":["inferentia-2b-devops-agent"]},
    {"Name":"instance-state-name","Values":["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
remote=(
 "docker exec autoport bash -lc \"ps aux | grep -E 'claude -p|compile_gemma4' | grep -v grep\"\n"
 "echo ===LASTEVENTS===\n"
 "docker exec autoport bash -lc 'tail -c 20000 /workspace/port/run.log'\n"
)
cid=ssm.send_command(InstanceIds=[iid],DocumentName="AWS-RunShellScript",
    Parameters={"commands":[remote]},TimeoutSeconds=300)["Command"]["CommandId"]
for _ in range(50):
    time.sleep(4)
    o=ssm.get_command_invocation(CommandId=cid,InstanceId=iid)
    if o["Status"] not in ("Pending","InProgress"): break
out=o.get("StandardOutputContent","")
head,_,tail=out.partition("===LASTEVENTS===")
print("PROCS:\n"+head.strip()+"\n")
# print last events, decoding stream-json into readable lines
last=[]
for ln in tail.splitlines():
    ln=ln.strip()
    if not ln.startswith("{"):
        if ln: last.append(("raw",ln))
        continue
    try: e=json.loads(ln)
    except: continue
    t=e.get("type")
    if t=="assistant":
        for b in e.get("message",{}).get("content",[]):
            if b.get("type")=="text" and b.get("text","").strip(): last.append(("TEXT",b["text"].strip()))
            elif b.get("type")=="tool_use":
                inp=b.get("input",{}); c=inp.get("command") or inp.get("file_path") or inp.get("description") or ""
                last.append(("TOOL",b.get("name")+": "+str(c)[:200]))
    elif t=="result": last.append(("RESULT","is_error=%s subtype=%s :: %s"%(e.get("is_error"),e.get("subtype"),str(e.get("result",""))[:1500])))
    elif t=="system" and e.get("subtype")=="error": last.append(("ERR",str(e)[:400]))
    elif t=="rate_limit_event": last.append(("RATELIMIT",str(e.get("rate_limit_info",""))[:300]))
    elif t=="error": last.append(("ERROR",str(e)[:400]))
for tag,txt in last[-25:]:
    print("[%s] %s"%(tag, txt.replace("\n"," ")[:400]))
