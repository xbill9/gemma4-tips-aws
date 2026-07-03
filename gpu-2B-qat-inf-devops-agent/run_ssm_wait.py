import boto3
import time
import sys

ssm = boto3.client('ssm', region_name='us-east-2')

# Send the command
resp = ssm.send_command(
    InstanceIds=['i-04a3612cbe1d14209'],
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': ["docker exec vllm-server python3 /load_gemma3n_pretrained.py"]}
)
cmd_id = resp['Command']['CommandId']
print(f"Sent SSM Command ID: {cmd_id}")

# Wait for completion
while True:
    time.sleep(2)
    invocation = ssm.get_command_invocation(CommandId=cmd_id, InstanceId='i-04a3612cbe1d14209')
    status = invocation['Status']
    print(f"Status: {status}...")
    if status not in ['Pending', 'InProgress']:
        print("\n--- STDOUT ---")
        print(invocation.get('StandardOutputContent', ''))
        print("\n--- STDERR ---")
        print(invocation.get('StandardErrorContent', ''))
        break
