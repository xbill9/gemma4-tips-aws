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
set -e
AB=/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py
docker cp vllm-server:$AB /tmp/ab_i.py
python3 - <<'PYEOF'
T="/tmp/ab_i.py"; s=open(T).read()
def ins(anchor, tag):
    global s
    if anchor in s and ("EXEC "+tag) not in s:
        pr='        import sys as _e; _e.stderr.write("[EXEC %s]\\n"); _e.stderr.flush()\n' % tag
        # insert the print line BEFORE the anchor line, matching its indentation by using anchor's leading spaces
        idx=s.index(anchor)
        line_start=s.rfind("\n",0,idx)+1
        indent=s[line_start:idx]
        s=s[:line_start]+indent+('import sys as _e; _e.stderr.write("[EXEC %s]\\n"); _e.stderr.flush()\n' % tag)+indent+anchor+s[idx+len(anchor):] if False else s
    return
# simpler: line-based insertion
lines=s.split("\n"); out=[]
targets={
 'is_qkv_cte_fuse_rope_nki_kernel_enabled = self.neuron_config.is_prefill_stage':'PREP_QKV',
 'QK = torch.matmul(Q, K.transpose(2, 3)) / self.softmax_scale':'SCALED_QK',
 'prior_scores = torch.matmul(Q, K_prior) / self.softmax_scale':'TOKEN_GEN_NATIVE',
 'flash_attn_strategy = self.get_flash_attention_strategy(q_len, attention_mask is not None)':'PERFORM_PREFILL',
 'from .utils import _rotate_half':'APPLY_ROTARY',
}
done=set()
for l in lines:
    for anch,tag in targets.items():
        if anch in l and tag not in done:
            indent=l[:len(l)-len(l.lstrip())]
            out.append(indent+'import sys as _e; _e.stderr.write("[EXEC %s]\\n"); _e.stderr.flush()' % tag)
            done.add(tag)
    out.append(l)
open(T,"w").write("\n".join(out))
print("instrumented:", sorted(done))
PYEOF
docker cp /tmp/ab_i.py vllm-server:$AB
echo "=== restart (prints fire at trace, before long compile) ==="
docker restart vllm-server
sleep 2; docker ps --filter name=vllm-server --format 'STATE={{.Status}}'
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
