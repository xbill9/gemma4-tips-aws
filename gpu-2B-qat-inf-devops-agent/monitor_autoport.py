import boto3, os, time, subprocess, json, datetime, sys

REPO="/home/xbill/gemma4-tips-aws"
REGION=open("active_deployment_region.txt").read().strip()
MONLOG=open("/home/xbill/gemma4-tips-aws/gpu-2B-qat-inf-devops-agent/monitor.log","a")

def log(s):
    line="[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), s)
    print(line, flush=True); MONLOG.write(line+"\n"); MONLOG.flush()

def clients():
    # refresh from the SSO profile with stale temp creds explicitly unset (proven-working form)
    for k in ("AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_SESSION_TOKEN"):
        os.environ.pop(k, None)
    # login shell so ~/.profile (AWS_PROFILE / SSO config) is sourced, matching a fresh terminal
    subprocess.run("bash -lc 'env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN "
                   "bash save-aws-creds.sh'", cwd=REPO, shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for ln in open(REPO+"/.aws_creds"):
        if "=" in ln and not ln.strip().startswith("#"):
            k,v=ln.strip().split("=",1); os.environ[k.strip()]=v.strip()
    ec2=boto3.client("ec2",region_name=REGION); ssm=boto3.client("ssm",region_name=REGION)
    iid=ec2.describe_instances(Filters=[
        {"Name":"tag:Name","Values":["inferentia-2b-devops-agent"]},
        {"Name":"instance-state-name","Values":["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
    return ssm,iid

def poll(ssm,iid):
    remote=("docker exec autoport bash -lc \"ps aux | grep -E 'claude -p' | grep -v grep | wc -l | sed 's/^/ALIVE=/'\"\n"
            "docker exec autoport bash -lc 'echo LOGBYTES=$(wc -c < /workspace/port/run.log)'\n"
            "echo ===TAIL===\n"
            "docker exec autoport bash -lc 'tail -c 40000 /workspace/port/run.log'\n")
    cid=ssm.send_command(InstanceIds=[iid],DocumentName="AWS-RunShellScript",
                         Parameters={"commands":[remote]},TimeoutSeconds=300)["Command"]["CommandId"]
    for _ in range(50):
        time.sleep(4)
        o=ssm.get_command_invocation(CommandId=cid,InstanceId=iid)
        if o["Status"] not in ("Pending","InProgress"): break
    return o.get("StandardOutputContent","")

def summarize(out):
    head,_,tail=out.partition("===TAIL===")
    alive="ALIVE=1" in head
    logb=[x for x in head.split() if x.startswith("LOGBYTES=")]
    logb=logb[0] if logb else "LOGBYTES=?"
    models=set(); last_tool=""; last_text=""; result=None; fb=0
    for ln in tail.splitlines():
        ln=ln.strip()
        if not ln.startswith("{"): continue
        try: e=json.loads(ln)
        except: continue
        t=e.get("type")
        if t=="assistant":
            m=e.get("message",{})
            if m.get("model"): models.add(m["model"])
            for b in m.get("content",[]):
                if b.get("type")=="fallback": fb+=1
                elif b.get("type")=="text" and b.get("text","").strip(): last_text=b["text"].strip()
                elif b.get("type")=="tool_use":
                    inp=b.get("input",{})
                    c=inp.get("command") or inp.get("file_path") or inp.get("description") or ""
                    last_tool="%s: %s"%(b.get("name"),str(c)[:120])
        elif t=="result": result=e
    low=tail.lower()
    sig=[k for k in ["token match","% match","validation","passed","failed","95%","compil","inference","neuron_port"] if k in low]
    return alive,logb,models,fb,last_tool,last_text,sig,result

MAXIT=48   # ~4h at 5min
fails=0
for i in range(MAXIT):
    try:
        ssm,iid=clients()
        out=poll(ssm,iid)
        fails=0
    except Exception as ex:
        fails+=1
        log("POLL ERROR #%d (transient creds?): %s" % (fails, str(ex)[:160]))
        if fails>=8:
            log("*** 8 consecutive poll failures — SSO session likely fully expired; need 'aws sso login'. run continues on box."); break
        time.sleep(75); continue
    alive,logb,models,fb,last_tool,last_text,sig,result=summarize(out)
    log("alive=%s %s models=%s fb=%s sig=%s | tool[%s] | %s" %
        (alive,logb,",".join(models) or "-",fb,sig,last_tool,(last_text[:90].replace(chr(10)," ")) ))
    if result is not None:
        log("*** RESULT is_error=%s: %s" % (result.get("is_error"), str(result.get("result",""))[:1200]))
        break
    if not alive:
        log("*** claude process no longer running (no result event yet — check run.log)")
        break
    time.sleep(300)
else:
    log("monitor hit MAXIT without completion; run still going — relaunch monitor to keep watching")
log("monitor exiting")
