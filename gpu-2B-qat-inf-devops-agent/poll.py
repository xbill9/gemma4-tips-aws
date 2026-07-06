import boto3, os, time, sys

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
echo "=== health ==="
curl -s -m 4 -o /dev/null -w "http_code=%{http_code}\n" http://localhost:8080/health || echo "health: no response"
echo "=== last meaningful log lines ==="
docker logs --tail 400 vllm-server 2>&1 | grep -viE "Will convert to torch.bfloat16|warnings.warn|UserWarning: Found torch.float32" | grep -iE "error|traceback|exception|compil|trace|percent|%|Application|Starting vLLM|Uvicorn|Route|running on|api_server|loaded|generat|Neff|latency|Engine|worker_error|Init" | tail -30
echo "=== container status ==="
docker ps -a --filter name=vllm-server --format '{{.Status}}'
'''
cmd = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]})["Command"]["CommandId"]
for _ in range(60):
    time.sleep(3)
    o = ssm.get_command_invocation(CommandId=cmd, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print(o.get("StandardOutputContent", ""))
if o.get("StandardErrorContent", "").strip():
    print("STDERR:\n", o["StandardErrorContent"][:1500])
