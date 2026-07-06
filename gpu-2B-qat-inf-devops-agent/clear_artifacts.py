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
echo "=== clearing BOTH caches: compiled-artifacts + neuronxcc ==="
docker exec vllm-server bash -lc 'rm -rf /workspace/vllm/local-models/google/gemma-4-E2B-it/neuron-compiled-artifacts/* 2>/dev/null; echo artifacts-cleared; ls /workspace/vllm/local-models/google/gemma-4-E2B-it/neuron-compiled-artifacts 2>/dev/null | wc -l'
rm -rf /home/ubuntu/.cache/neuron/* 2>/dev/null && echo host-neuron-cache-cleared
echo "=== restart to force true recompile with softmax_scale=16 ==="
docker restart vllm-server
sleep 2; docker ps --filter name=vllm-server --format 'STATE={{.Status}}'
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(50):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
