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

OLD='V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)'
NEW=('V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)\n'
     '        if getattr(self, "gemma4_v_norm", False):\n'
     '            V = (V.float() * torch.rsqrt(V.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(V.dtype)')

host_py = "P='/home/ubuntu/patch_and_run.sh'\ns=open(P).read()\n"
host_py += "add='a=a.replace(%r, %r)\\n'\n" % (OLD, NEW)
host_py += ("if 'gemma4_v_norm' in s and 'a.replace' in s and 'rsqrt' in s:\n"
            "    print('vnorm fix already present')\n"
            "else:\n"
            "    anchor=\"a=open(ab).read()\\n\"\n"
            "    assert anchor in s\n"
            "    s=s.replace(anchor, anchor+'# vnorm-fix\\n'+add, 1)\n"
            "    open(P,'w').write(s); print('inserted v-norm fix')\n")
remote = ("python3 - <<'PYEOF'\n" + host_py + "\nPYEOF\n"
          "docker exec vllm-server bash -lc 'rm -rf /tmp/nxd_model/* /var/tmp/neuron-compile-cache/* /root/.cache/neuron/* /workspace/vllm/local-models/google/gemma-4-E2B-it/neuron-compiled-artifacts/* 2>/dev/null; echo nuked'\n"
          "rm -rf /home/ubuntu/.cache/neuron/* 2>/dev/null && echo host-cleared\n"
          "docker restart vllm-server; sleep 2; docker ps --filter name=vllm-server --format 'S={{.Status}}'\n")
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(50):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"]); print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent","").strip(): print("STDERR:", o["StandardErrorContent"][:1500])
