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
    
    print("Fetching last 1000 lines of docker logs...")
    res = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': ['docker logs --tail 1000 vllm-server 2>&1']}
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
                    content = result["StandardOutputContent"]
                    output_file = "/home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/scratch/crash_reason_output.txt"
                    with open(output_file, "w") as f:
                        f.write(content)
                    print(f"Full logs saved to {output_file}")
                    
                    lines = content.splitlines()
                    # Find indices where new runs start
                    restart_indices = []
                    for i, line in enumerate(lines):
                        if "Starting vLLM Server with optimized parameters" in line:
                            restart_indices.append(i)
                    
                    print(f"Found {len(restart_indices)} restart boundaries.")
                    if restart_indices:
                        last_idx = restart_indices[-1]
                        print(f"Last restart boundary is at line {last_idx}")
                        # Print 40 lines before the last restart to see the traceback of the crash
                        print("=== BACKTRACE ===")
                        for idx in range(max(0, last_idx - 60), last_idx):
                            print(lines[idx])
                        print("=================")
                    else:
                        print("No restart boundary found in the last 1000 lines.")
                break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
