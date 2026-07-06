import boto3, os, time, subprocess
REPO="/home/xbill/gemma4-tips-aws"; REGION=open("active_deployment_region.txt").read().strip()
for k in ("AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_SESSION_TOKEN"): os.environ.pop(k,None)
subprocess.run("bash -lc 'env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN bash save-aws-creds.sh'",
               cwd=REPO, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for ln in open(REPO+"/.aws_creds"):
    if "=" in ln and not ln.strip().startswith("#"):
        k,v=ln.strip().split("=",1); os.environ[k.strip()]=v.strip()
ec2=boto3.client("ec2",region_name=REGION); ssm=boto3.client("ssm",region_name=REGION)
iid=ec2.describe_instances(Filters=[{"Name":"tag:Name","Values":["inferentia-2b-devops-agent"]},
    {"Name":"instance-state-name","Values":["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
remote=r'''
echo "=== claude -p alive? ==="
docker exec autoport bash -lc "ps aux | grep -E 'claude -p' | grep -v grep | wc -l"
echo "=== best match evidence across all logs (MATCH / x/10 / match rate) ==="
docker exec autoport bash -lc "grep -hoE 'MATCH[^ ]*|[0-9]+/10 (prompts )?match|match rate[:= ]+[0-9.]+%?|[0-9]+/10 match|PASS|FAIL' /workspace/port/run_part*.log /workspace/port/run.log 2>/dev/null | sort | uniq -c | sort -rn | head -30"
echo "=== validation artifact files ==="
docker exec autoport bash -lc "ls -t /workspace/port/agent_artifacts/tmp/*.log 2>/dev/null | head; echo ---; ls -t /workspace/port/*.md /workspace/port/neuron_port/*.py 2>/dev/null | head"
echo "=== latest validation-ish log content ==="
docker exec autoport bash -lc "for f in \$(ls -t /workspace/port/agent_artifacts/tmp/*valid* /workspace/port/agent_artifacts/tmp/*match* 2>/dev/null | head -2); do echo \"## \$f\"; grep -iE 'match|PASS|FAIL|/10|prompt' \$f 2>/dev/null | tail -20; done"
'''
cid=ssm.send_command(InstanceIds=[iid],DocumentName="AWS-RunShellScript",
    Parameters={"commands":[remote]},TimeoutSeconds=300)["Command"]["CommandId"]
for _ in range(50):
    time.sleep(4); o=ssm.get_command_invocation(CommandId=cid,InstanceId=iid)
    if o["Status"] not in ("Pending","InProgress"): break
print("STATUS:",o["Status"]); print(o.get("StandardOutputContent","")[-5000:])
