import boto3
import time

ssm = boto3.client('ssm', region_name='us-east-1')
status = "InProgress"
while status == "InProgress":
    resp = ssm.get_command_invocation(CommandId='6ce14801-15ca-4921-915f-de14d60531d4', InstanceId='i-0af2ceb15e7807e96')
    status = resp.get('Status')
    if status != "InProgress":
        print("Status:", status)
        print("OUT:", resp.get('StandardOutputContent'))
        print("ERR:", resp.get('StandardErrorContent'))
        break
    time.sleep(10)
