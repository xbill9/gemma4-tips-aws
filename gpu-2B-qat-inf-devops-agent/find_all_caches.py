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
remote = r'''
echo "===== NEURON env vars (cache locations) ====="
docker exec vllm-server bash -lc 'env | grep -iE "NEURON|CACHE|COMPIL" | sort'
echo "===== all *.neff on whole container FS ====="
docker exec vllm-server bash -lc 'find / -name "*.neff" 2>/dev/null | head -40'
echo "===== all neuron-compiled-artifacts dirs + their subdir mtimes ====="
docker exec vllm-server bash -lc 'find / -type d -name "neuron-compiled-artifacts" 2>/dev/null | while read d; do echo "$d:"; ls -la "$d" 2>/dev/null; done'
echo "===== search for MODULE_71ddc8b9 anywhere ====="
docker exec vllm-server bash -lc 'find / -path "*71ddc8b9*" 2>/dev/null | head'
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(50):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", "")[:4000])
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1000])
