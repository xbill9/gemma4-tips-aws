import boto3, os, time

region = open("active_deployment_region.txt").read().strip()
for line in open("/home/xbill/gemma4-tips-aws/.aws_creds"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k.strip()] = v.strip()

ec2 = boto3.client("ec2", region_name=region)
ssm = boto3.client("ssm", region_name=region)
iid = ec2.describe_instances(Filters=[
    {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
    {"Name": "instance-state-name", "Values": ["running"]}])["Reservations"][0]["Instances"][0]["InstanceId"]

remote = r'''
echo "=== clearing neuron compile caches ==="
docker exec vllm-server bash -lc 'rm -rf /root/.cache/neuron/* /var/tmp/neuron-compile-cache/* 2>/dev/null; echo cleared-in-container'
rm -rf /home/ubuntu/.cache/neuron/* 2>/dev/null && echo cleared-on-host
echo "=== restarting container ==="
docker restart vllm-server
echo "=== container state ==="
docker ps --filter name=vllm-server --format '{{.Names}} {{.Status}}'
echo "=== re-verify patched values took effect (in-container source) ==="
docker exec vllm-server bash -lc 'grep -n "global_head_dim\", 256" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py | head; grep -n "rotary_dim = int(self.head_dim" /opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py'
'''
cmd = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(60):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cmd, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent", "").strip():
    print("STDERR:\n", o["StandardErrorContent"][:3000])
