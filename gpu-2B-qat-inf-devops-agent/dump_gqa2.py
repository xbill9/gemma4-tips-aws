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
echo "########## gqa.py 395-412 (the ColumnParallel-vs-plain condition) ##########"
sed -n '395,412p' /tmp/gqa.py | nl -ba -v395
echo "########## gqa.py determine_sharding_strategy + get_shardable_head_counts (62-115) ##########"
sed -n '62,115p' /tmp/gqa.py | nl -ba -v62
echo "########## gqa.py preshard_hook (332-360) ##########"
sed -n '332,360p' /tmp/gqa.py | nl -ba -v332
echo "########## modeling_gemma3: how NeuronGemma3Attention passes num_kv_heads / desired_sharding_strategy ##########"
grep -nE "desired_sharding_strategy|REPLICATE|CONVERT_TO_MHA|num_key_value_heads|GroupQueryAttention_QKV|get_num_attention_heads|self.num_heads|divide\(" /tmp/ab.py | head -30
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
