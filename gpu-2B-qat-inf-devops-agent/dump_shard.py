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
set -e
TRACE=/opt/conda/lib/python3.12/site-packages/neuronx_distributed/trace/trace.py
MG=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py
docker cp vllm-server:$TRACE /tmp/trace.py 2>/dev/null
docker cp vllm-server:$MG /tmp/mg.py 2>/dev/null
echo "########## trace.py shard_children (find + 120 lines) ##########"
awk 'NR>=1{if(/def shard_children/){p=NR}} p&&NR>=p&&NR<p+120{print NR": "$0}' /tmp/trace.py | sed -n '1,140p'
echo "########## modeling_gemma3.py: convert_hf_to_neuron_state_dict / preshard / q_proj / qkv ##########"
grep -nE "def convert_hf_to_neuron_state_dict|def preshard|q_proj|k_proj|v_proj|qkv|partition|preshard_hook|def get_state_dict|padding|pad_|weight\[" /tmp/mg.py | head -60
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
