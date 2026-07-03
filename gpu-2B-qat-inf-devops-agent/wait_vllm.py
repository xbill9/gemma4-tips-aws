import boto3
import time

ssm = boto3.client('ssm', region_name='us-east-1')
script = '''#!/bin/bash
for i in {1..30}; do
    logs=$(docker logs --tail 200 vllm-server 2>&1)
    if echo "$logs" | grep -q "Uvicorn running on http://0.0.0.0:8080"; then
        echo "vLLM is READY!"
        exit 0
    fi
    if echo "$logs" | grep -q "Traceback"; then
        echo "Crash detected!"
        docker logs --tail 100 vllm-server
        exit 1
    fi
    sleep 10
done
echo "Timeout waiting for vLLM"
exit 1
'''

resp = ssm.send_command(
    InstanceIds=["i-0af2ceb15e7807e96"],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [script]}
)
print("Command sent! ID:", resp['Command']['CommandId'])
