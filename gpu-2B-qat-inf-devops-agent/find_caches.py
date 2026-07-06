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
echo "===== host /home/ubuntu/.cache/neuron tree (depth 3, dirs+sizes) ====="
find /home/ubuntu/.cache/neuron -maxdepth 3 2>/dev/null | head -60
echo "--- host neuron cache disk usage ---"
du -sh /home/ubuntu/.cache/neuron/* 2>/dev/null | head
echo "===== search host for serialized weight shards / compiled artifacts ====="
find /home/ubuntu /tmp /var/tmp -maxdepth 5 \( -name "*.pt" -o -name "weights*" -o -name "*sharded*" -o -name "*.neff" -o -name "*_ckpt*" -o -name "model.pt*" \) 2>/dev/null | grep -viE "huggingface/hub" | head -40
echo "===== loader: where does it put compiled/sharded artifacts? ====="
grep -nE "serialize_path|compiled_path|CACHE|traced_model|checkpoint_cache|artifacts|BASE_COMPILE|compiled_model_path|save|\.pt" /tmp/../tmp/loader.py 2>/dev/null | head
docker cp vllm-server:/opt/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py /tmp/loader.py 2>/dev/null
grep -nE "serialize_path|compiled_path|traced_model|checkpoint.*cache|artifacts|compiled_model_path|persist|\.load\(|save_pretrained|get_builder|shard_checkpoint|CACHE_DIR|cache_dir" /tmp/loader.py | head -40
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
