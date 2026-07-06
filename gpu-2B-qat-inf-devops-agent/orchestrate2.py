import boto3, os, time, subprocess, base64, datetime, re
REPO="/home/xbill/gemma4-tips-aws"; REGION=open("active_deployment_region.txt").read().strip()
SESSION="7b19b2a3-d58b-4026-a926-4fe735dee4e2"
MODEL="claude-opus-4-8"
MONLOG=open("/home/xbill/gemma4-tips-aws/gpu-2B-qat-inf-devops-agent/orchestrate2.log","a")
def log(s):
    line="[%s] %s"%(datetime.datetime.now().strftime("%H:%M:%S"),s); print(line,flush=True); MONLOG.write(line+"\n"); MONLOG.flush()

CONTINUE=("CONTINUE the Gemma-4 NxD autoport. Current measured state: the on-device port compiles and runs, and the latest "
 "validation is Token match rate 70% (7/10 prompts) vs the >=95% gate. Artifacts are in /workspace/port "
 "(neuron_port/modeling_gemma4.py, gemma4_hf_reference.py, agent_artifacts/). The 3 failing prompts shared a pattern. "
 "The fix is the per-layer 1/head_dim attention scale (1/256 sliding, 1/512 global) applied where it survives QK-norm and "
 "the traced kernel path.\n\n"
 "THIS IS A NON-INTERACTIVE HEADLESS RUN — there is no callback. Do NOT background any step and end your turn to wait to be "
 "notified. Run EVERY step in the FOREGROUND with an explicit `timeout`, block on it until it finishes, then proceed within "
 "the SAME turn. Do the full loop yourself: diagnose the 3 failing prompts -> fix -> recompile (foreground, wait) -> run the "
 "validation script -> read the 'Token match rate' result. If below 95%, iterate again in the same turn. Keep going. Only end "
 "your turn when validation prints RESULT: PASSED / Token match rate >=95% (then state the rate and the neuron_port/ path), "
 "or you hit a hard blocker you cannot resolve (state exactly what). Do not end on a promise of future work.")
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
      "docker exec autoport bash -lc \"grep -hoE 'Token match rate: [0-9.]+%|RESULT: (PASSED|FAILED)' /workspace/port/agent_artifacts/tmp/*.log /workspace/port/run*.log 2>/dev/null | tail -6\"")
    alive="ALIVE=1" in out
    rates=re.findall(r"Token match rate: ([0-9.]+)%", out)
    passed=("RESULT: PASSED" in out) or any(float(r)>=95 for r in rates)
    best=max([float(r) for r in rates], default=None)
    return alive,best,passed,out
def relaunch(ssm,iid,part):
    run(ssm,iid,
      "docker exec autoport bash -lc 'echo "+PROMPTB64+" | base64 -d > /workspace/port/RESUME.txt'\n"
      "docker exec autoport bash -lc 'echo "+LAUNCHB64+" | base64 -d > /root/resume_autoport.sh; chmod +x /root/resume_autoport.sh'\n"
      "docker exec -e HOME=/root autoport bash -lc 'cp -f /workspace/port/run.log /workspace/port/run_r%d.log 2>/dev/null; "
      ": > /workspace/port/run.log; setsid nohup /root/resume_autoport.sh > /workspace/port/run.log 2>&1 < /dev/null & echo GO $!'\n" % part)

resumes=0; MAXRESUME=6; dead_streak=0; last_logb=-1; part=1
log("orchestrate2 start on %s"%MODEL)
while resumes<=MAXRESUME:
    try: ssm,iid=clients()
    except Exception as ex:
        log("creds err %s"%str(ex)[:100]); time.sleep(90); continue
    alive,best,passed,raw=state(ssm,iid)
    logb=re.search(r"LOGB=(\d+)",raw); logb=int(logb.group(1)) if logb else -1
    log("alive=%s best_match=%s passed=%s logbytes=%s resumes=%d"%(alive,best,passed,logb,resumes))
    if passed:
        log("*** VALIDATION PASSED — best match=%s"%best); break
    if alive:
        dead_streak=0; last_logb=logb; time.sleep(180); continue
    # not alive: confirm it's really idle (log unchanged) before resuming
    if logb==last_logb: dead_streak+=1
    else: dead_streak=1; last_logb=logb
    if dead_streak<2:
        time.sleep(60); continue
    if resumes==MAXRESUME:
        log("reached MAXRESUME without pass (best=%s)"%best); break
    resumes+=1; log("resume #%d (best so far=%s)"%(resumes,best))
    relaunch(ssm,iid,part); part+=1; dead_streak=0; last_logb=-1
    time.sleep(120)  # give the new turn time to spin up and start working
log("orchestrate2 exiting")
