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
docker exec vllm-server python3 -c "
import glob
from safetensors import safe_open
fs=sorted(glob.glob('/root/.cache/huggingface/**/*.safetensors',recursive=True))
fs=[f for f in fs if 'gemma-4-E2B' in f]
print('num shards:', len(fs))
allk=[]
for f in fs:
    with safe_open(f,framework='pt') as h:
        for k in h.keys():
            allk.append((k,f))
# exact keys for layer 0 that contain norm
print('=== all layer-0 norm-ish keys (exact) ===')
for k,f in allk:
    if '.0.' in k and 'norm' in k.lower() and 'audio' not in k and 'vision' not in k:
        with safe_open(f,framework='pt') as h:
            t=h.get_tensor(k).float()
        print(repr(k),'mean=%.5f'%t.mean().item(),'shape',tuple(t.shape))
print('=== does model.language_model prefix exist? sample 5 text keys ===')
for k,f in allk[:0]:
    pass
cnt=0
for k,f in allk:
    if 'language_model' in k and 'layers.0.' in k and cnt<8:
        print(repr(k)); cnt+=1
" 2>&1 | head -40
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(50):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"]); print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
