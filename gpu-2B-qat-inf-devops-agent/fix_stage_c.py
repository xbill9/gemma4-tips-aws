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

pybody = r'''
import re
def patch(path, label):
    s=open(path).read()
    # Remove the global-layer fused_qkv override so global shards identically to sliding.
    target="            updated_config.neuron_config.fused_qkv = False\n"
    n=s.count(target)
    if n:
        s=s.replace(target, "")
        open(path,"w").write(s)
    print(f"{label}: removed fused_qkv=False x{n}")
    # show the global branch context
    for i,l in enumerate(open(path),1):
        if "swa_layer" in l or "fused_qkv" in l or "global_head_dim" in l or "num_global_key_value_heads" in l:
            print("  ", i, l.rstrip())

patch("/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py", "CONTAINER modeling_gemma3")
'''
host_py = pybody.replace(
    '"/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py", "CONTAINER modeling_gemma3"',
    '"/home/ubuntu/patch_transformers.py", "HOST patcher"')

remote = ("echo '===== CONTAINER ====='\n"
          "docker exec -i vllm-server python3 - <<'PYEOF'\n" + pybody + "\nPYEOF\n"
          "echo '===== HOST ====='\n"
          "python3 - <<'PYEOF2'\n" + host_py + "\nPYEOF2\n")
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:2000])
