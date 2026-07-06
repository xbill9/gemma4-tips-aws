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
AB=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py
echo "===== prep_qkv_tensors (550-600): where move_heads_front + layernorm + rope happen ====="
docker exec vllm-server sed -n '550,600p' $AB
echo "===== apply_rotary_embedding head (478-500): does it ALSO apply q/k layernorm? ====="
docker exec vllm-server sed -n '478,500p' $AB
echo "===== move_heads_front definition (does layernorm arg get applied?) ====="
docker exec vllm-server bash -lc 'grep -rn "def move_heads_front" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/ 2>/dev/null; F=$(grep -rln "def move_heads_front" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/ | head -1); echo "--- $F ---"; sed -n "/def move_heads_front/,/return/p" $F | head -40'
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
