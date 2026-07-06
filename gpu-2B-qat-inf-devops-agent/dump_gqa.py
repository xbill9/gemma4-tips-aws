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
GQA=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/gqa.py
docker cp vllm-server:$GQA /tmp/gqa.py 2>/dev/null
echo "########## gqa.py: class GroupQueryAttention_QKV __init__ ##########"
awk '/class GroupQueryAttention_QKV/{f=1} f{print NR": "$0} f&&/def forward/{exit}' /tmp/gqa.py | sed -n '1,140p'
echo "########## gqa.py: sharding_strategy / ColumnParallel / replicate / _init_ definitions ##########"
grep -nE "sharding_strategy|ColumnParallel|BaseParallelLinear|replicate|GQA.QKV_|class GQA|def get_num|preshard|fused_qkv|_shard|Linear\(|per_partition" /tmp/gqa.py | head -50
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
