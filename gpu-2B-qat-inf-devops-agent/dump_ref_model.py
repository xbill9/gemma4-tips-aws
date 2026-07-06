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
D=/tmp/refenv/lib/python3.12/site-packages/transformers/models
echo "=== gemma4 model dirs ==="
docker exec vllm-server bash -lc "ls $D | grep -i gemma"
echo "=== gemma4_text modeling: attention class + rope + scaling + qk_norm + head_dim ==="
docker exec vllm-server bash -lc "F=\$(ls $D/gemma4_text/modeling_gemma4_text.py 2>/dev/null || ls $D/gemma4/modeling_gemma4.py 2>/dev/null); echo FILE=\$F; grep -nE 'class .*Attention|def forward|scaling|head_dim|q_norm|k_norm|apply_rotary|partial_rotary|rope|self.scaling|attention_type|sliding|softcap|repeat_kv|query_pre_attn' \$F | head -70"
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"]); print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
