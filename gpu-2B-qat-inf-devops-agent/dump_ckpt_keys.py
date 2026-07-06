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
echo "===== EXACT checkpoint keys for layer-0 norms (input_layernorm, q_norm, post_per_layer_input_norm) ====="
docker exec vllm-server python3 -c "
import glob
from safetensors import safe_open
fs=sorted(glob.glob('/root/.cache/huggingface/**/*.safetensors',recursive=True))
fs=[f for f in fs if 'gemma-4-E2B' in f]
want=['layers.0.input_layernorm','layers.0.self_attn.q_norm','layers.0.post_per_layer_input_norm','layers.0.post_attention_layernorm']
for f in fs:
    with safe_open(f,framework='pt') as h:
        for k in h.keys():
            if any(w in k for w in want):
                t=h.get_tensor(k).float()
                print(repr(k),'mean=%.4f'%t.mean().item(),'shape',tuple(t.shape))
" 2>&1 | head -20
echo "===== convert per-layer norm handling (the pass placeholders + rename block, lines 605-666) ====="
docker exec vllm-server sed -n '605,666p' /tmp/mg_n.py 2>/dev/null || docker cp vllm-server:/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py /tmp/mg2.py && sed -n '605,668p' /tmp/mg2.py
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"]); print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
