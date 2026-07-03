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
    
    print("Inspecting container full info...")
    commands = [
        "docker inspect vllm-server"
    ]
    res = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': commands}
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
                    import json
                    info_json = result["StandardOutputContent"].strip()
                    try:
                        parsed = json.loads(info_json)
                        container_info = parsed[0]
                        print(f"Status: {container_info['State']['Status']}")
                        print(f"RestartCount: {container_info['RestartCount']}")
                        print("State details:")
                        print(json.dumps(container_info['State'], indent=2))
                    except Exception as e:
                        print(info_json[:1000])
                break
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
