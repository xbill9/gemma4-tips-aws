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

# Rewrite the rope_fixup block in patch_and_run.sh to RESTORE the correct 512 baseline.
host_py = r'''
P="/home/ubuntu/patch_and_run.sh"
s=open(P).read()
start=s.find('echo "Applying gemma4 rope/head-dim fixup')
end=s.find('echo "Starting vLLM Server')
assert start!=-1 and end!=-1 and start<end, "anchors not found"

newblock = "".join([
 'echo "Reverting to correct 512 global head-dim baseline (rope_revert_v2)..."\n',
 "python3 - <<'FIXEOF'\n",
 'mg="/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py"\n',
 'ab="/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py"\n',
 "s=open(mg).read()\n",
 's=s.replace(chr(34)+"global_head_dim"+chr(34)+", 256", chr(34)+"global_head_dim"+chr(34)+", 512")\n',
 'anc="updated_config.num_key_value_heads = getattr(config, "+chr(34)+"num_global_key_value_heads"+chr(34)+", 1)"\n',
 'if anc in s and "neuron_config.fused_qkv = False" not in s:\n',
 '    s=s.replace(anc, anc+chr(10)+"            updated_config.neuron_config.fused_qkv = False")\n',
 "open(mg,'w').write(s)\n",
 "a=open(ab).read()\n",
 'a=a.replace("int(self.head_dim * partial_factor)", "int((256 if is_swa_layer else 512) * partial_factor)")\n',
 "open(ab,'w').write(a)\n",
 "print('[rope_revert] global_head_dim->512, rotary->512-expr, fused_qkv=False restored')\n",
 "FIXEOF\n\n",
])
s2 = s[:start] + newblock + s[end:]
open(P,"w").write(s2)
print("rewrote fixup block")
for i,l in enumerate(open(P).read().split(chr(10)),1):
    if i>=50 and i<=90: print(i,l)
'''
remote = ("python3 - <<'PYEOF'\n" + host_py + "\nPYEOF\n"
          "echo '=== clear compile cache (head_dim 256->512 changes graph) ==='\n"
          "rm -rf /home/ubuntu/.cache/neuron/* 2>/dev/null && echo cache-cleared\n"
          "echo '=== restart ==='\n"
          "docker restart vllm-server\n"
          "sleep 2; docker ps --filter name=vllm-server --format 'STATE={{.Status}}'\n")
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(50):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:2000])
