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
    
    print("Sending SSM command to fetch docker logs & container status...")
    commands = [
        "docker ps -a --filter name=vllm-server",
        "docker logs --tail 100 vllm-server 2>&1"
    ]
    res = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': commands}
    )
    command_id = res['Command']['CommandId']
    print(f"Command ID: {command_id}. Waiting for completion...")
    
    for _ in range(30):
        time.sleep(1)
        try:
            result = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
            if result["Status"] in ["Success", "Failed", "TimedOut", "Cancelled"]:
                print(f"Status: {result['Status']}")
                if result["Status"] == "Success":
                    content = result["StandardOutputContent"]
                    print("--- OUTPUT ---")
                    print(content)
                    print("--------------")
                else:
                    print("--- ERROR ---")
                    print(result.get("StandardErrorContent"))
                    print("-------------")
                break
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
