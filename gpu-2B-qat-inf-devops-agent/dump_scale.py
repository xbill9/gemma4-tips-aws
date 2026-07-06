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
echo "===== FULL config.json (all keys) ====="
docker exec vllm-server python3 -c "
import json,glob
p=[x for x in glob.glob('/root/.cache/huggingface/**/config.json',recursive=True) if 'gemma-4-E2B' in x][0]
c=json.load(open(p)); t=c.get('text_config',c)
import pprint
for k in sorted(t.keys()):
    v=t[k]
    if k in ('layer_types',): continue
    print(k,'=',v)
" 2>&1
echo "===== attention_base.py: softmax_scale / scaling / query_pre_attn / softcap usage ====="
docker exec vllm-server grep -nE "softmax_scale|query_pre_attn|scaling|/ self\.|sqrt|softcap|logit_soft|q_layernorm|k_layernorm|gemma4_shared_kv|gemma4_store_kv|source_idx|move_heads_front" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py 2>&1 | head -40
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
