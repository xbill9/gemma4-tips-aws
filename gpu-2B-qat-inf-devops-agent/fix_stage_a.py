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

# In-container python that applies the two edits idempotently and reports before/after.
py = r'''
import re
MG="/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py"
AB="/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py"

mg=open(MG).read()
n1=mg.count('"global_head_dim", 512')
mg2=mg.replace('"global_head_dim", 512', '"global_head_dim", 256')
open(MG,"w").write(mg2)

ab=open(AB).read()
old='int((256 if is_swa_layer else 512) * partial_factor)'
new='int(self.head_dim * partial_factor)'
n2=ab.count(old)
ab2=ab.replace(old,new)
open(AB,"w").write(ab2)

print("modeling_gemma3: replaced global_head_dim 512->256 occurrences:", n1)
print("attention_base : replaced rotary_dim hardcode occurrences:", n2)
print("--- verify modeling_gemma3 global_head_dim lines ---")
for i,l in enumerate(open(MG),1):
    if "global_head_dim" in l: print(i, l.rstrip())
print("--- verify attention_base rotary_dim line ---")
for i,l in enumerate(open(AB),1):
    if "rotary_dim = int(" in l: print(i, l.rstrip())
'''
remote = "docker exec -i vllm-server python3 - <<'PYEOF'\n" + py + "\nPYEOF\n"
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
