import boto3
import os
import sys
import time

def main():
    # Load credentials
    creds = {}
    creds_path = "/home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/.aws_creds"
    if os.path.exists(creds_path):
        with open(creds_path, "r") as f:
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
    print(f"Fetching /home/ubuntu/vllm_logs.txt from {instance_id}...")

    res = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': ["cat /home/ubuntu/vllm_logs_clean.txt"]}
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
                    content = result["StandardOutputContent"]
                    output_path = "/home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/scratch/real_instance_logs.txt"
                    with open(output_path, "w", errors="ignore") as f:
                        f.write(content)
                    print(f"Successfully wrote logs to {output_path}")
                else:
                    print(f"Error standard output: {result.get('StandardOutputContent')}")
                    print(f"Error standard error: {result.get('StandardErrorContent')}")
                return
        except Exception as e:
            pass
    print("Command timed out.")

if __name__ == "__main__":
    main()
