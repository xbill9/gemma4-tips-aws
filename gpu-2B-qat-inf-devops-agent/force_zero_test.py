import boto3, os, time, subprocess
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
host_py = r'''
P="/home/ubuntu/patch_and_run.sh"
s=open(P).read()
add='a=a.replace("            is_swa_layer = True\\n", "            is_swa_layer = True\\n            Q = Q * 0.0  # FORCE_ZERO_DBG\\n")\n'
if "FORCE_ZERO_DBG" in s:
    print("already")
else:
    anchor="open(ab,'w').write(a)\n"
    s=s.replace(anchor, add+anchor, 1)
    open(P,"w").write(s); print("added FORCE_ZERO_DBG")
'''
remote = ("python3 - <<'PYEOF'\n" + host_py + "\nPYEOF\n"
          "docker exec vllm-server bash -lc 'rm -rf /tmp/nxd_model/* /var/tmp/neuron-compile-cache/* /root/.cache/neuron/* /workspace/vllm/local-models/google/gemma-4-E2B-it/neuron-compiled-artifacts/* 2>/dev/null; echo nuked; find / -name \"*.neff\" 2>/dev/null | wc -l'\n"
          "rm -rf /home/ubuntu/.cache/neuron/* 2>/dev/null && echo host-cleared\n"
          "docker restart vllm-server\n"
          "sleep 2; docker ps --filter name=vllm-server --format 'STATE={{.Status}}'\n")
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(50):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
