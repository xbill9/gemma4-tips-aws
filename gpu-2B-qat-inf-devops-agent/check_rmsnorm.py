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
echo "===== which RMSNorm class does gemma3 use + its forward (1+weight or plain?) ====="
docker exec vllm-server bash -lc 'grep -rn "get_rmsnorm_cls\|class.*RMSNorm\|1.0 + \|1 + self.weight\|self.weight)\|def forward" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/custom_calls.py 2>/dev/null | head -20; echo "---gemma3 rmsnorm import---"; grep -n "rmsnorm\|RMSNorm\|get_rmsnorm" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py | head'
echo "===== actual norm weight magnitudes from checkpoint (delta~0 or full~1?) ====="
docker exec vllm-server python3 -c "
import glob,os
from safetensors import safe_open
fs=sorted(glob.glob('/root/.cache/huggingface/**/*.safetensors',recursive=True))
fs=[f for f in fs if 'gemma-4-E2B' in f]
import statistics
shown=0
for f in fs:
    with safe_open(f,framework='pt') as h:
        for k in h.keys():
            if k.endswith('norm.weight') and shown<6:
                t=h.get_tensor(k).float()
                print(k, 'mean=%.4f'%t.mean().item(), 'min=%.3f'%t.min().item(), 'max=%.3f'%t.max().item())
                shown+=1
    if shown>=6: break
" 2>&1 | head -20
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(50):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:2000])
