import boto3
ssm = boto3.client('ssm', region_name='us-east-1')
script = '''#!/bin/bash
docker logs --tail 200 vllm-server 2>&1 | grep -v 'weight' | tail -n 20
'''
resp = ssm.send_command(
    InstanceIds=["i-0af2ceb15e7807e96"],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
import time
time.sleep(5)
out = ssm.get_command_invocation(CommandId=resp['Command']['CommandId'], InstanceId='i-0af2ceb15e7807e96')
print(out.get('StandardOutputContent'))
