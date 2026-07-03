import boto3
import os
import sys
import time

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_host_ssm.py <shell_command_or_script_file>")
        sys.exit(1)

    arg = sys.argv[1]
    if os.path.exists(arg):
        with open(arg, "r") as f:
            script = f.read()
    else:
        script = arg

    # Resolve Region
    region = "us-east-1"
    if os.path.exists("active_deployment_region.txt"):
        with open("active_deployment_region.txt", "r") as f:
            region = f.read().strip()

    # Read from .aws_creds if present
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
    ec2 = boto3.client('ec2', region_name=region)

    # Query all running instances under our service tag
    instances_resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": ["inferentia-2b-devops-agent"]},
            {"Name": "instance-state-name", "Values": ["running"]}
        ]
    )
    instance_ids = []
    for reservation in instances_resp.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            instance_ids.append(instance["InstanceId"])

    if not instance_ids:
        print("No running instances found!")
        sys.exit(1)

    instance_id = instance_ids[0]

    resp = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [script]}
    )
    command_id = resp['Command']['CommandId']
    print(f"Sent SSM Command to {instance_id}. ID: {command_id}. Waiting for completion...")

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
            pass
        time.sleep(2)

if __name__ == "__main__":
    main()
