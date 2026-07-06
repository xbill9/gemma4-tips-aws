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
echo "########## modeling_gemma3.py convert_hf_to_neuron_state_dict (549-690) ##########"
sed -n '549,690p' /tmp/mg.py | nl -ba -v549
echo "########## attention_base.py qkv / init_gqa / ColumnParallel (module side) ##########"
AB=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py
docker cp vllm-server:$AB /tmp/ab.py 2>/dev/null
grep -nE "qkv_proj|Wqkv|q_proj|ColumnParallel|GroupQueryAttention|self.fused_qkv|init_gqa|tensor_model_parallel|def __init__|BaseGroupQuery|preprocess_qkv|def initialize" /tmp/ab.py | head -50
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
