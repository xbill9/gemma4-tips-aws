import boto3, os, time, subprocess, json, sys
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

# grab: process state, log size, last ~120 lines of run.log
remote = (
  "docker exec autoport bash -lc \"ps aux | grep -E 'launch_autoport|claude -p' | grep -v grep | wc -l | sed 's/^/PROC_ALIVE=/'\"\n"
  "docker exec autoport bash -lc 'echo LOGBYTES=$(wc -c < /workspace/port/run.log); echo NPORT=$(ls -1 /workspace/port 2>/dev/null | wc -l)'\n"
  "echo ===TAIL===\n"
  "docker exec autoport bash -lc 'tail -c 60000 /workspace/port/run.log'\n"
)
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]}, TimeoutSeconds=300)["Command"]["CommandId"]
for _ in range(50):
    time.sleep(4)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
out = o.get("StandardOutputContent", "")
head, _, tail = out.partition("===TAIL===")
print(head.strip())

# parse stream-json events from tail
events = []
for ln in tail.splitlines():
    ln = ln.strip()
    if not ln.startswith("{"): continue
    try: events.append(json.loads(ln))
    except Exception: pass

models=set(); tools=[]; texts=[]; errors=[]; result=None; fallbacks=0
for e in events:
    t=e.get("type")
    if t=="assistant":
        m=e.get("message",{})
        if m.get("model"): models.add(m["model"])
        for b in m.get("content",[]):
            bt=b.get("type")
            if bt=="fallback": fallbacks+=1
            elif bt=="text" and b.get("text","").strip(): texts.append(b["text"].strip())
            elif bt=="tool_use":
                inp=b.get("input",{})
                cmd=inp.get("command") or inp.get("file_path") or inp.get("prompt") or inp.get("description") or ""
                tools.append("%s: %s" % (b.get("name"), str(cmd)[:160]))
    elif t=="result":
        result=e
    elif t in ("error","system") and e.get("subtype")=="error":
        errors.append(str(e)[:300])

low=tail.lower()
signals=[k for k in ["token match","token-match","% match","validation","passed","failed","compile","inference","neuron_port","95%"] if k in low]

print("\n=== SUMMARY ===")
print("models seen (recent window):", models, "| fallbacks:", fallbacks)
print("signals in log:", signals)
print("\n-- last 6 tool actions --")
for x in tools[-6:]: print("  ", x)
print("\n-- last 3 assistant texts --")
for x in texts[-3:]: print("  >", x[:400].replace("\n"," "))
if result:
    print("\n*** RESULT EVENT ***")
    print("  is_error:", result.get("is_error"), "| subtype:", result.get("subtype"))
    print("  ", str(result.get("result",""))[:800])
