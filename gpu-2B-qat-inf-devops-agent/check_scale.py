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
MG=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py
echo "===== every use of query_pre_attn_scalar across both files (is scaling applied anywhere else?) ====="
docker exec vllm-server grep -nE "query_pre_attn_scalar" $AB $MG 2>&1
echo "===== NeuronGemma3Attention super().__init__ softmax_scale line (MG) ====="
docker exec vllm-server grep -nE "softmax_scale" $MG 2>&1
echo "===== any other Q scaling (multiply/normalizer/scaling) in AB forward ====="
docker exec vllm-server grep -nE "Q \*|Q = Q \*|self.scaling|normalizer|\* math.sqrt|scalar \*\* |\*\* -0.5|\*\*-0.5" $AB $MG 2>&1 | head
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
