import boto3
import os
import time

region = "us-east-2"
if os.path.exists("active_deployment_region.txt"):
    with open("active_deployment_region.txt", "r") as f:
        region = f.read().strip()

creds = {}
if os.path.exists(".aws_creds"):
    with open(".aws_creds", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v
for k, v in creds.items():
    os.environ[k] = v

ssm = boto3.client('ssm', region_name=region)
instance_id = "i-04a3612cbe1d14209"
command_id = "66e4afa3-905c-4560-84ce-a8c450e1f7e4"

print("Waiting for deployment command to complete...")
for poll in range(60):
    try:
        out = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        status = out.get('Status')
        if status in ['Success', 'Failed', 'Cancelled', 'TimedOut']:
            print(f"FINAL STATUS: {status}")
            print("STDOUT:")
            print(out.get('StandardOutputContent'))
            print("STDERR:")
            print(out.get('StandardErrorContent'))
            break
    except Exception as e:
        print(f"Poll {poll}: exception {e}")
    time.sleep(5)
else:
    print("Deployment command timed out polling")
