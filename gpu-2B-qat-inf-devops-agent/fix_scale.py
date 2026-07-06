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

# Add the softmax_scale fix into the existing rope_revert block in patch_and_run.sh.
host_py = r'''
P="/home/ubuntu/patch_and_run.sh"
s=open(P).read()
addline = "s=s.replace(\"softmax_scale=1.0\", \"softmax_scale=256 ** 0.5\")\n"
if "softmax_scale=256 ** 0.5" in s:
    print("scale fix already present")
else:
    anchor = "open(mg,'w').write(s)\n"
    assert anchor in s, "block anchor not found"
    s = s.replace(anchor, addline + anchor, 1)
    open(P,"w").write(s)
    print("inserted softmax_scale fix into fixup block")
for i,l in enumerate(open(P).read().split(chr(10)),1):
    if 52<=i<=70: print(i,l)
'''
remote = ("python3 - <<'PYEOF'\n" + host_py + "\nPYEOF\n"
          "echo '=== clear compile cache (scaling changes graph) ==='\n"
          "rm -rf /home/ubuntu/.cache/neuron/* 2>/dev/null && echo cache-cleared\n"
          "echo '=== restart ==='\n"
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
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:2000])
