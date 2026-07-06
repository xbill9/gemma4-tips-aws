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

# Patch the HOST patcher so restarts re-apply the corrected values (idempotent, durable).
remote = r'''
F=/home/ubuntu/patch_transformers.py
echo "host patcher exists?"; ls -l $F 2>&1
echo "--- occurrences before ---"
grep -c '"global_head_dim", 512' $F 2>/dev/null
grep -c 'int((256 if is_swa_layer else 512) \* partial_factor)' $F 2>/dev/null
cp $F ${F}.bak.rope 2>/dev/null
python3 - <<PYEOF
F="/home/ubuntu/patch_transformers.py"
s=open(F).read()
a=s.count('"global_head_dim", 512')
s=s.replace('"global_head_dim", 512', '"global_head_dim", 256')
b=s.count('int((256 if is_swa_layer else 512) * partial_factor)')
s=s.replace('int((256 if is_swa_layer else 512) * partial_factor)', 'int(self.head_dim * partial_factor)')
open(F,"w").write(s)
print("patched host patcher: global_head_dim 512->256 x", a, "| rotary hardcode x", b)
PYEOF
echo "--- occurrences after ---"
grep -c '"global_head_dim", 512' $F 2>/dev/null
grep -c 'int((256 if is_swa_layer else 512) \* partial_factor)' $F 2>/dev/null
grep -n '"global_head_dim", 256' $F 2>/dev/null | head
grep -n 'int(self.head_dim \* partial_factor)' $F 2>/dev/null | head
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
    print("STDERR:\n", o["StandardErrorContent"][:3000])
