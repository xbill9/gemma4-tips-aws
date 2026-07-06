import boto3, os, time, subprocess, base64
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

script_path="/tmp/claude-1000/-home-xbill-gemma4-tips-aws-gpu-2B-qat-inf-devops-agent/ee679ab6-3506-4ef4-8d78-fe0c183c59a3/scratchpad/hf_ref.py"
b64=base64.b64encode(open(script_path,"rb").read()).decode()
remote = ("echo '%s' | base64 -d > /home/ubuntu/hf_ref.py\n" % b64 +
          "docker cp /home/ubuntu/hf_ref.py vllm-server:/tmp/hf_ref.py\n"
          "docker exec -d vllm-server bash -lc 'cd /tmp && python3 hf_ref.py > /tmp/hf_ref.out 2>&1'\n"
          "sleep 3; echo '=== launched, initial out ==='; docker exec vllm-server bash -lc 'cat /tmp/hf_ref.out 2>/dev/null | head'\n")
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
