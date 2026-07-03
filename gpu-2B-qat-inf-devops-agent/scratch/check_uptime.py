import boto3
import os
import sys
import time

def main():
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v
    for k, v in creds.items():
        os.environ[k] = v

    ssm = boto3.client('ssm', region_name='us-east-1')
    instance_id = "i-08dc36bcfb8241ee5"
    
    print("Checking docker uptime...")
    res = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': ['docker ps -a --filter name=vllm-server']}
    )
    command_id = res['Command']['CommandId']
    
    for _ in range(15):
        time.sleep(1)
        try:
            result = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
            if result["Status"] in ["Success", "Failed", "TimedOut", "Cancelled"]:
                if result["Status"] == "Success":
                    print(result["StandardOutputContent"])
                break
        except Exception:
            pass

if __name__ == "__main__":
    main()
