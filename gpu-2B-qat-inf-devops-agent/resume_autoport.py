import boto3, os, time, subprocess, base64
REPO="/home/xbill/gemma4-tips-aws"
REGION=open("active_deployment_region.txt").read().strip()
for k in ("AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_SESSION_TOKEN"): os.environ.pop(k,None)
subprocess.run("bash -lc 'env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN bash save-aws-creds.sh'",
               cwd=REPO, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for ln in open(REPO+"/.aws_creds"):
    if "=" in ln and not ln.strip().startswith("#"):
        k,v=ln.strip().split("=",1); os.environ[k.strip()]=v.strip()

SESSION="7b19b2a3-d58b-4026-a926-4fe735dee4e2"

resume_prompt = (
 "CONTINUE the Gemma-4 NxD autoport. Your previous turn ended right after launching a background compile. "
 "CRITICAL: this is a non-interactive headless run — there is NO callback or notification mechanism, so NEVER end your "
 "turn to 'wait to be notified' by a background monitor/task. Instead wait SYNCHRONOUSLY inside your own turn (sleep in a "
 "loop and re-check) until each long-running step finishes, then proceed.\n\n"
 "Do this now, in order:\n"
 "1. Verify the most recent compile (agent_artifacts/tmp/compile2.log, process name compile_gemma4) actually succeeded. "
 "If it is still running, wait for it synchronously (sleep+grep loop) until it reports success.\n"
 "2. Run the on-device inference check using the current k_proj / per-layer (1/head_dim) scale fix.\n"
 "3. Run the FULL validation against the config-correct golden reference — the >=95% token-match gate. "
 "You were last at 7/10 prompts matching perfectly; the 3 failures shared a 'neuron converging' pattern.\n"
 "4. If below the gate, diagnose the remaining mismatching prompts, apply the fix where it survives QK-norm and the traced "
 "kernel path (bake into weights if code overrides are inert), recompile (waiting synchronously), and re-validate. "
 "Iterate until the token-match gate passes or you hit a hard blocker.\n"
 "5. End your turn ONLY when validation PASSES (report the exact match rate and the deployable neuron_port/ path) or you "
 "are genuinely blocked (state exactly what and why). Do not end on a promise of future work.\n"
)
open("/tmp/_resume_prompt.txt","w").write(resume_prompt)
PROMPTB64=base64.b64encode(resume_prompt.encode()).decode()

launch_sh = ('#!/bin/bash\n'
             'export HOME=/root\n'
             'export PATH=/root/.local/bin:$PATH\n'
             'export IS_SANDBOX=1\n'
             'unset ANTHROPIC_API_KEY; export CLAUDE_CODE_OAUTH_TOKEN=$(cat /root/.anthropic_key)\n'
             'export ANTHROPIC_MODEL=claude-fable-5\n'
             'cd /workspace/port\n'
             'exec /root/.local/bin/claude -p "$(cat /workspace/port/RESUME.txt)" '
             '--resume ' + SESSION + ' '
             '--dangerously-skip-permissions --verbose --output-format stream-json\n')
LAUNCHB64=base64.b64encode(launch_sh.encode()).decode()

ec2=boto3.client("ec2",region_name=REGION); ssm=boto3.client("ssm",region_name=REGION)
iid=ec2.describe_instances(Filters=[{"Name":"tag:Name","Values":["inferentia-2b-devops-agent"]},
    {"Name":"instance-state-name","Values":["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
print("INSTANCE:",iid)

remote=(
 "set +e\n"
 "docker exec autoport bash -lc 'echo "+PROMPTB64+" | base64 -d > /workspace/port/RESUME.txt'\n"
 "docker exec autoport bash -lc 'echo "+LAUNCHB64+" | base64 -d > /root/resume_autoport.sh; chmod +x /root/resume_autoport.sh'\n"
 "docker exec -e HOME=/root autoport bash -lc 'cp -f /workspace/port/run.log /workspace/port/run_part1.log 2>/dev/null; "
 ": > /workspace/port/run.log; setsid nohup /root/resume_autoport.sh > /workspace/port/run.log 2>&1 < /dev/null & echo RESUME_PID $!'\n"
 "sleep 30\n"
 "echo === run.log head ===\n"
 "docker exec autoport bash -lc 'wc -c /workspace/port/run.log; head -c 2500 /workspace/port/run.log'\n"
 "echo; echo === alive? ===\n"
 "docker exec autoport bash -lc \"ps aux | grep -E 'claude -p' | grep -v grep | head\"\n"
)
cid=ssm.send_command(InstanceIds=[iid],DocumentName="AWS-RunShellScript",
    Parameters={"commands":[remote]},TimeoutSeconds=600)["Command"]["CommandId"]
for _ in range(50):
    time.sleep(4)
    o=ssm.get_command_invocation(CommandId=cid,InstanceId=iid)
    if o["Status"] not in ("Pending","InProgress"): break
print("STATUS:",o["Status"])
print(o.get("StandardOutputContent","")[-5000:])
if o.get("StandardErrorContent","").strip(): print("STDERR:",o["StandardErrorContent"][-1500:])
