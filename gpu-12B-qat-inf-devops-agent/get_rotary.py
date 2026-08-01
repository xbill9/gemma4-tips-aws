import base64
import time
import boto3

ssm = boto3.client('ssm', region_name='us-east-1')
instance_id = "i-0af2ceb15e7807e96"

def run_ssm_command(commands):
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands}
    )
    command_id = response["Command"]["CommandId"]
    while True:
        time.sleep(1)
        result = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        status = result["Status"]
        if status in ["Success", "Failed", "Cancelled", "TimedOut"]:
            return status, result.get("StandardOutputContent", ""), result.get("StandardErrorContent", "")

cmd = f"docker exec vllm-server python3 -c \"import base64; f = open('/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/utils.py', 'rb'); print(base64.b64encode(f.read()).decode()); f.close()\""
status, stdout, stderr = run_ssm_command([cmd])
if status == "Success":
    with open("utils_remote.py", "wb") as f:
        f.write(base64.b64decode(stdout.strip()))
    print("Saved utils_remote.py")
else:
    print(stderr)
