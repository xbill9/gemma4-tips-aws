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
echo "=== mounts (Binds) ==="
docker inspect -f '{{range .HostConfig.Binds}}{{println .}}{{end}}' vllm-server
echo "=== Mounts (verbose) ==="
docker inspect -f '{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}} (rw={{.RW}}){{println}}{{end}}' vllm-server
echo "=== Devices ==="
docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} -> {{.PathInContainer}}{{println}}{{end}}' vllm-server
echo "=== ShmSize / Privileged / CapAdd ==="
docker inspect -f 'shm={{.HostConfig.ShmSize}} priv={{.HostConfig.Privileged}} caps={{.HostConfig.CapAdd}} netmode={{.HostConfig.NetworkMode}}' vllm-server
echo "=== Ports ==="
docker inspect -f '{{json .HostConfig.PortBindings}}' vllm-server
echo "=== Env (neuron/hf/vllm relevant) ==="
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' vllm-server | grep -iE 'NEURON|HF_|HUGGING|VLLM|TENSOR|MODEL|PATH=' | head -40
echo "=== Entrypoint / Cmd ==="
docker inspect -f 'entrypoint={{json .Config.Entrypoint}} cmd={{json .Config.Cmd}}' vllm-server
echo "=== tail of patch_and_run.sh (how is the server launched? exec?) ==="
docker exec vllm-server bash -lc 'tail -25 /home/ubuntu/patch_and_run.sh 2>/dev/null || echo "no patch_and_run.sh at that path"; echo "---find it---"; ls -l /home/ubuntu/patch_and_run.sh 2>/dev/null'
echo "=== who is PID1 in container ==="
docker exec vllm-server bash -lc 'ps -o pid,ppid,cmd -1 2>/dev/null | head; echo ---; ps aux | grep -iE "vllm|python|patch_and_run" | grep -v grep | head'
'''
cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                       Parameters={"commands": [remote]}, TimeoutSeconds=600)["Command"]["CommandId"]
for _ in range(60):
    time.sleep(4)
    o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
    if o["Status"] not in ("Pending", "InProgress"): break
print("STATUS:", o["Status"])
print(o.get("StandardOutputContent", "")[-6000:])
if o.get("StandardErrorContent","").strip():
    print("STDERR:", o["StandardErrorContent"][-2000:])
