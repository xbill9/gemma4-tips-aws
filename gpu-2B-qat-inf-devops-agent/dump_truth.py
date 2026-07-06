import boto3, os, time

region = open("active_deployment_region.txt").read().strip() if os.path.exists("active_deployment_region.txt") else "us-east-2"
creds_path = "/home/xbill/gemma4-tips-aws/.aws_creds"
for line in open(creds_path):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ[k.strip()] = v.strip()

ec2 = boto3.client("ec2", region_name=region)
ssm = boto3.client("ssm", region_name=region)
iid = ec2.describe_instances(Filters=[
    {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
    {"Name": "instance-state-name", "Values": ["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]
print("Instance:", iid)

# Dump: (1) real HF config.json rope/attention fields  (2) on-box attention_base rope block
remote = r'''
echo "===== HF config.json (gemma-4-E2B-it) ====="
CFG=$(docker exec vllm-server bash -lc 'find /root/.cache/huggingface -name config.json -path "*gemma-4-E2B*" 2>/dev/null | head -1')
echo "config path: $CFG"
docker exec vllm-server python3 -c "
import json,glob
p=glob.glob('/root/.cache/huggingface/**/config.json',recursive=True)
p=[x for x in p if 'gemma-4-E2B' in x] or p
c=json.load(open(p[0]))
t=c.get('text_config',c)
for k in ['model_type','num_hidden_layers','head_dim','hidden_size','num_attention_heads','num_key_value_heads','sliding_window','sliding_window_pattern','layer_types','rope_theta','rope_local_base_freq','rope_scaling','rope_parameters','partial_rotary_factor','query_pre_attn_scalar','final_logit_softcapping','attn_logit_softcapping']:
    if k in t or k in c:
        print(k,'=',t.get(k,c.get(k)))
" 2>&1
echo "===== on-box attention_base.py rope block ====="
docker exec vllm-server grep -nE "partial_factor|rotary_dim|is_swa|min_dim|def apply_rotary_embedding|def compute_for_token_gen" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py 2>&1 | head -40
echo "===== gemma3 modeling rope selection ====="
docker exec vllm-server grep -nE "% 6|global_rotary_emb|local_rotary_emb|global_head_dim|global_rope_theta|global_dim" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py 2>&1 | head -40
'''
cmd = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cmd, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"):
        break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
err = o.get("StandardErrorContent", "")
if err.strip():
    print("STDERR:\n", err)
