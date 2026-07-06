import boto3, os, time

region = open("active_deployment_region.txt").read().strip()
for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()

ec2 = boto3.client("ec2", region_name=region)
ssm = boto3.client("ssm", region_name=region)
iid = ec2.describe_instances(Filters=[
    {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
    {"Name": "instance-state-name", "Values": ["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]

remote = r'''
AB=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py
echo "########## apply_rotary_embedding FULL (478-545) ##########"
docker exec vllm-server sed -n '478,545p' $AB
echo "########## RotaryEmbedding def location ##########"
docker exec vllm-server bash -lc "grep -rln 'class RotaryEmbedding' /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/ 2>/dev/null | head -3"
echo "########## RotaryEmbedding __init__ + forward ##########"
RE=$(docker exec vllm-server bash -lc "grep -rln 'class RotaryEmbedding' /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/ 2>/dev/null | head -1")
docker exec vllm-server bash -lc "awk '/class RotaryEmbedding/{f=1} f{print NR\": \"\$0} f&&/def forward/{c++} c&&/return/{print; exit}' $RE" 2>&1 | head -80
'''
cmd = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cmd, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent", "").strip():
    print("STDERR:\n", o["StandardErrorContent"][:2000])
