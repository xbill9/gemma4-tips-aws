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

# Host-side python: insert an idempotent fixup block into patch_and_run.sh before the vLLM launch.
host_py = r'''
P="/home/ubuntu/patch_and_run.sh"
s=open(P).read()
MARK="rope_fixup_v1"
if MARK in s:
    print("fixup already present, skipping insert")
else:
    block = (
        'echo "Applying gemma4 rope/head-dim fixup ('+MARK+')..."\n'
        "python3 - <<'FIXEOF'\n"
        'mg="/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py"\n'
        'ab="/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py"\n'
        "s=open(mg).read()\n"
        "s=s.replace('\\\"global_head_dim\\\", 512', '\\\"global_head_dim\\\", 256')\n"
        's="\\n".join(l for l in s.split("\\n") if "neuron_config.fused_qkv = False" not in l)\n'
        "open(mg,'w').write(s)\n"
        "a=open(ab).read()\n"
        "a=a.replace('int((256 if is_swa_layer else 512) * partial_factor)', 'int(self.head_dim * partial_factor)')\n"
        "open(ab,'w').write(a)\n"
        "print('[rope_fixup] head_dim->256 ; rope->self.head_dim ; dropped global fused_qkv=False')\n"
        "FIXEOF\n\n"
    )
    anchor='echo "Starting vLLM Server'
    idx=s.find(anchor)
    if idx==-1:
        # fallback: before the python launch line
        anchor="python3 -m vllm.entrypoints"
        idx=s.find(anchor)
    assert idx!=-1, "anchor not found"
    s2=s[:idx]+block+s[idx:]
    open(P,"w").write(s2)
    print("inserted fixup block before:", anchor)

print("----- resulting patch_and_run.sh (tail) -----")
for i,l in enumerate(open(P).read().split("\n"),1):
    if i>=38: print(i, l)
'''
remote = "python3 - <<'PYEOF'\n" + host_py + "\nPYEOF\n"
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(40):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:2000])
