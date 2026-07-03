import boto3
import os
import sys
import time
import json

def main():
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
    public_ip = instances_resp["Reservations"][0]["Instances"][0].get("PublicIpAddress")
    print(f"Discovered active instance: {instance_id} with Public IP: {public_ip}")

    print("Beginning polling of vLLM server...")
    
    # We will poll for up to 15 minutes (90 * 10 seconds)
    max_polls = 90
    for poll in range(1, max_polls + 1):
        print(f"\n--- Poll {poll}/{max_polls} ---")
        
        # Check logs using SSM
        script = "docker logs --tail 20 vllm-server 2>&1"
        try:
            resp = ssm.send_command(
                InstanceIds=[instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": [script]}
            )
            cmd_id = resp['Command']['CommandId']
            
            # Wait for SSM command to finish (typically 2-4 seconds)
            for ssm_poll in range(10):
                time.sleep(1)
                cmd_out = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
                if cmd_out.get('Status') in ['Success', 'Failed']:
                    break
            
            status = cmd_out.get('Status')
            stdout = cmd_out.get('StandardOutputContent', '').strip()
            
            if status == 'Success':
                print("LATEST DOCKER LOGS:")
                print(stdout)
                
                # Check for ready message
                if "Uvicorn running on" in stdout:
                    print("\n[SUCCESS] vLLM server has started and is listening!")
                    break
                elif "Traceback" in stdout or "Error" in stdout:
                    print("\n[WARNING] Potential crash/error detected in logs!")
            else:
                print(f"SSM command failed with status {status}")
                
        except Exception as e:
            print(f"Error checking logs via SSM: {e}")
            
        # Also perform local curl check inside the host
        curl_script = "curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/v1/models"
        try:
            resp = ssm.send_command(
                InstanceIds=[instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": [curl_script]}
            )
            cmd_id = resp['Command']['CommandId']
            for ssm_poll in range(10):
                time.sleep(1)
                cmd_out = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
                if cmd_out.get('Status') in ['Success', 'Failed']:
                    break
            
            if cmd_out.get('Status') == 'Success':
                http_code = cmd_out.get('StandardOutputContent', '').strip()
                print(f"HTTP Status check (http://localhost:8080/v1/models): {http_code}")
                if http_code == "200":
                    print("\n[SUCCESS] Model endpoint responded with HTTP 200!")
                    break
        except Exception as e:
            pass

        time.sleep(10)
    else:
        print("\n[TIMEOUT] Reached max polling duration. Server may still be compiling.")
        sys.exit(1)

    # If we broke out, the server is ready! Let's perform a test inference.
    print("\nRunning a test Chat Completion inference...")
    inference_script = """
curl -s -X POST http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [
      {"role": "user", "content": "Tell me a short joke about neural networks."}
    ],
    "max_tokens": 100
  }'
"""
    try:
        resp = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [inference_script]}
        )
        cmd_id = resp['Command']['CommandId']
        print("Sent inference test command. Waiting for completion...")
        for ssm_poll in range(30):
            time.sleep(2)
            cmd_out = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
            if cmd_out.get('Status') in ['Success', 'Failed']:
                break
        
        if cmd_out.get('Status') == 'Success':
            print("\nINFERENCE RESPONSE:")
            response_text = cmd_out.get('StandardOutputContent', '').strip()
            print(response_text)
            try:
                data = json.loads(response_text)
                joke = data['choices'][0]['message']['content']
                print("\nEXTRACTED ANSWER:")
                print(joke)
            except Exception:
                pass
        else:
            print("Inference test command failed!")
            print(cmd_out.get('StandardErrorContent'))
    except Exception as e:
        print(f"Error running inference check: {e}")

if __name__ == "__main__":
    main()
