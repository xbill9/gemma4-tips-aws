import boto3
import os
import sys
import time

def run_command(commands):
    # Load credentials
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v
    for k, v in creds.items():
        os.environ[k] = v

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    ec2 = boto3.client('ec2', region_name=region)
    ssm = boto3.client('ssm', region_name=region)

    # Find the active instance
    service_name = "inferentia-12b-devops-agent"
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [service_name]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )
    instances = []
    for reservation in response.get("Reservations", []):
        instances.extend(reservation.get("Instances", []))
    
    if not instances:
        print("No running EC2 instance found.")
        sys.exit(1)

    instance_id = instances[0]["InstanceId"]
    print(f"Running command on {instance_id}: {commands}")

    res = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': commands}
    )
    command_id = res['Command']['CommandId']

    for _ in range(60):
        time.sleep(1)
        try:
            result = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
            if result["Status"] in ["Success", "Failed", "TimedOut", "Cancelled"]:
                print(f"SSM Command Status: {result['Status']}")
                if result["Status"] == "Success":
                    print("--- OUTPUT ---")
                    print(result["StandardOutputContent"])
                    print("--------------")
                else:
                    print("--- ERROR ---")
                    print(result.get("StandardErrorContent"))
                    print("-------------")
                return result["Status"] == "Success", result.get("StandardOutputContent", "") + "\n" + result.get("StandardErrorContent", "")
        except Exception as e:
            pass
    print("Command timed out.")
    return False, "Timed out"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 execute_ssm.py 'command1' 'command2' ...")
        sys.exit(1)
    run_command(sys.argv[1:])
