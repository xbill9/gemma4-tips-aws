import boto3, os, time, subprocess, base64
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

OLD='self.qkv_cte_nki_kernel_fuse_rope = self.neuron_config.qkv_cte_nki_kernel_fuse_rope'
NEW='self.qkv_cte_nki_kernel_fuse_rope = False  # force eager rope/qk-norm/v-norm path'
inject = ("import base64 as _b64\n"
          "a=a.replace(_b64.b64decode('%s').decode(), _b64.b64decode('%s').decode())\n"
          % (base64.b64encode(OLD.encode()).decode(), base64.b64encode(NEW.encode()).decode()))
ib=base64.b64encode(inject.encode()).decode()
host_py = ("import base64\nP='/home/ubuntu/patch_and_run.sh'\ns=open(P).read()\n"
           "inj=base64.b64decode('%s').decode()\n" % ib +
           "if 'force eager rope' in s:\n    print('eager already')\nelse:\n"
           "    anchor='a=open(ab).read()\\n'\n    assert anchor in s\n"
           "    s=s.replace(anchor, anchor+'# force-eager\\n'+inj, 1)\n"
           "    open(P,'w').write(s); print('inserted force-eager fix')\n")
remote = ("python3 - <<'PYEOF'\n" + host_py + "PYEOF\n"
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
