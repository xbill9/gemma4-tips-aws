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
echo "===== running-container source: softmax_scale value ====="
docker exec vllm-server grep -n "softmax_scale=" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py 2>&1 | head
echo "===== find neuron-compiled-artifacts (config-hash keyed cache) INSIDE container ====="
docker exec vllm-server bash -lc 'find / -type d -name "neuron-compiled-artifacts" 2>/dev/null | head; echo "---"; find / -type d -name "*neuron-compiled*" 2>/dev/null | head' 2>&1 | head -20
echo "===== loader log: did it LOAD precompiled or RECOMPILE? ====="
docker logs vllm-server 2>&1 | grep -iE "pre-compiled|precompiled|Unable to find precompiled|Recompiling|Successfully loaded pre-compiled|compiled_model_path|neuron-compiled-artifacts" | tail -12
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
