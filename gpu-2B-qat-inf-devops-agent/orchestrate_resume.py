import boto3, os, time, subprocess, base64, datetime, re
REPO="/home/xbill/gemma4-tips-aws"; REGION=open("active_deployment_region.txt").read().strip()
SESSION="7b19b2a3-d58b-4026-a926-4fe735dee4e2"; MODEL="claude-opus-4-8"
MON=open("/home/xbill/gemma4-tips-aws/gpu-2B-qat-inf-devops-agent/orchestrate3.log","a")
def log(s):
    line="[%s] %s"%(datetime.datetime.now().strftime("%H:%M:%S"),s); print(line,flush=True); MON.write(line+"\n"); MON.flush()

CONTINUE=("CONTINUE the Gemma-4 NxD autoport. Measured state: the on-device port compiles and runs; latest validation is "
 "Token match rate 70% (7/10) vs the >=95% gate. Artifacts in /workspace/port (neuron_port/modeling_gemma4.py, "
 "gemma4_hf_reference.py, agent_artifacts/). The 3 failing prompts share a pattern; the fix is the per-layer 1/head_dim "
 "attention scale (1/256 sliding, 1/512 global) applied where it survives QK-norm and the traced kernel path.\n\n"
 "NON-INTERACTIVE HEADLESS RUN — no callback. Never background a step and end your turn to wait to be notified. Run EVERY "
 "step FOREGROUND with a `timeout`, block until done, proceed in the SAME turn. Loop yourself: diagnose the 3 failing prompts "
 "-> fix -> recompile(foreground,wait) -> run the validation script -> read 'Token match rate'. If <95%, iterate again this "
 "turn. Only end when validation prints RESULT: PASSED / rate>=95% (state rate + neuron_port/ path) or you are truly blocked "
 "(state exactly what). Do not end on a promise of future work.")
PROMPTB64=base64.b64encode(CONTINUE.encode()).decode()
launch_sh=('#!/bin/bash\nexport HOME=/root\nexport PATH=/root/.local/bin:$PATH\nexport IS_SANDBOX=1\n'
 'unset ANTHROPIC_API_KEY; export CLAUDE_CODE_OAUTH_TOKEN=$(cat /root/.anthropic_key)\n'
 'export ANTHROPIC_MODEL='+MODEL+'\ncd /workspace/port\n'
 'exec /root/.local/bin/claude -p "$(cat /workspace/port/RESUME.txt)" --resume '+SESSION+
 ' --dangerously-skip-permissions --verbose --output-format stream-json\n')
LAUNCHB64=base64.b64encode(launch_sh.encode()).decode()

def clients():
    for k in ("AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_SESSION_TOKEN"): os.environ.pop(k,None)
    subprocess.run("bash -lc 'env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN bash save-aws-creds.sh'",
                   cwd=REPO, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for ln in open(REPO+"/.aws_creds"):
        if "=" in ln and not ln.strip().startswith("#"):
            k,v=ln.strip().split("=",1); os.environ[k.strip()]=v.strip()
    ec2=boto3.client("ec2",region_name=REGION); ssm=boto3.client("ssm",region_name=REGION)
    iid=ec2.describe_instances(Filters=[{"Name":"tag:Name","Values":["inferentia-2b-devops-agent"]},
        {"Name":"instance-state-name","Values":["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
    return ssm,iid
def run(ssm,iid,cmd,to=400):
    cid=ssm.send_command(InstanceIds=[iid],DocumentName="AWS-RunShellScript",Parameters={"commands":[cmd]},TimeoutSeconds=to)["Command"]["CommandId"]
    for _ in range(90):
        time.sleep(4); o=ssm.get_command_invocation(CommandId=cid,InstanceId=iid)
        if o["Status"] not in ("Pending","InProgress"): break
    return o.get("StandardOutputContent","")
def state(ssm,iid):
    out=run(ssm,iid,
      "docker exec autoport bash -lc \"ps aux | grep -E 'claude -p' | grep -v grep | grep -q . && echo ALIVE=1 || echo ALIVE=0\"\n"
      "docker exec autoport bash -lc 'echo LOGB=$(wc -c < /workspace/port/run.log)'\n"
      # AUTHORITATIVE pass/fail: newest accuracy result filename encodes _pass_/_fail_
      "docker exec autoport bash -lc \"ls -t /workspace/port/agent_artifacts/results/accuracy/gemma-4-E2B-it/*.json 2>/dev/null | head -1 | sed 's#.*/##; s/^/ACC=/'\"\n"
      "docker exec autoport bash -lc \"grep -hoE 'Token match rate: [0-9.]+%' /workspace/port/agent_artifacts/tmp/*.log 2>/dev/null | tail -1 | sed 's/^/LASTRATE=/'\"\n"
      "docker exec autoport bash -lc \"tail -c 3000 /workspace/port/run.log | grep -oiE 'hit your session limit|rate limit|resets [0-9].*' | tail -1 | sed 's/^/LIMIT=/'\"")
    alive="ALIVE=1" in out
    acc=re.search(r"ACC=(\S+)", out); acc=acc.group(1) if acc else ""
    passed = "_pass_" in acc.lower()
    logb=re.search(r"LOGB=(\d+)",out); logb=int(logb.group(1)) if logb else -1
    rate=re.search(r"LASTRATE=Token match rate: ([0-9.]+)%",out); rate=rate.group(1) if rate else "?"
    limited = "LIMIT=" in out and ("session limit" in out.lower() or "rate limit" in out.lower())
    resetmsg=re.search(r"LIMIT=(.*)", out); resetmsg=resetmsg.group(1).strip() if resetmsg else ""
    return dict(alive=alive, passed=passed, acc=acc, logb=logb, rate=rate, limited=limited, reset=resetmsg)
def relaunch(ssm,iid,part):
    run(ssm,iid,
      "docker exec autoport bash -lc 'echo "+PROMPTB64+" | base64 -d > /workspace/port/RESUME.txt'\n"
      "docker exec autoport bash -lc 'echo "+LAUNCHB64+" | base64 -d > /root/resume_autoport.sh; chmod +x /root/resume_autoport.sh'\n"
      "docker exec -e HOME=/root autoport bash -lc 'cp -f /workspace/port/run.log /workspace/port/run_r%d.log 2>/dev/null; "
      ": > /workspace/port/run.log; setsid nohup /root/resume_autoport.sh > /workspace/port/run.log 2>&1 < /dev/null & echo GO $!'\n" % part)

resumes=0; MAXRESUME=8; dead=0; last_logb=-1; part=1; start=time.time(); MAXWALL=5*3600
log("orchestrate3 start on %s (rate-limit-aware, authoritative pass detect)"%MODEL)
while time.time()-start < MAXWALL:
    try: ssm,iid=clients()
    except Exception as ex:
        log("creds err %s"%str(ex)[:100]); time.sleep(90); continue
    st=state(ssm,iid)
    log("alive=%s passed=%s rate=%s acc=%s limited=%s %s logb=%s res=%d"%(
        st["alive"],st["passed"],st["rate"],st["acc"][-40:],st["limited"],("["+st["reset"]+"]" if st["reset"] else ""),st["logb"],resumes))
    if st["passed"]:
        log("*** VALIDATION PASSED (authoritative): %s"%st["acc"]); break
    if st["alive"]:
        dead=0; last_logb=st["logb"]; time.sleep(180); continue
    if st["limited"]:
        log("rate-limited; backing off 900s then retrying (%s)"%st["reset"]); time.sleep(900); continue
    # dead & not limited -> confirm idle then resume
    if st["logb"]==last_logb: dead+=1
    else: dead=1; last_logb=st["logb"]
    if dead<2: time.sleep(60); continue
    if resumes>=MAXRESUME: log("MAXRESUME reached (rate=%s)"%st["rate"]); break
    resumes+=1; log("resume #%d (rate=%s)"%(resumes,st["rate"]))
    relaunch(ssm,iid,part); part+=1; dead=0; last_logb=-1; time.sleep(120)
else:
    log("MAXWALL reached")
log("orchestrate3 exiting")
