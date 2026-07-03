import boto3
import time

ssm = boto3.client('ssm', region_name='us-east-2')

# 1. Copy from container to host
cmd1 = ssm.send_command(
    InstanceIds=['i-04a3612cbe1d14209'],
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': ["docker cp vllm-server:/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/models/gemma3/modeling_gemma3.py /home/ubuntu/modeling_gemma3_neuron.py"]}
)
time.sleep(3)

# 2. Read in chunks using head/tail/sed
for start_line in range(1, 2000, 300):
    cmd2 = ssm.send_command(
        InstanceIds=['i-04a3612cbe1d14209'],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': [f"sed -n '{start_line},{start_line+299}p' /home/ubuntu/modeling_gemma3_neuron.py"]}
    )
    time.sleep(2)
    output = ssm.get_command_invocation(CommandId=cmd2['Command']['CommandId'], InstanceId='i-04a3612cbe1d14209')['StandardOutputContent']
    if not output.strip():
        break
    with open('scratch/modeling_gemma3_neuron.py', 'a' if start_line > 1 else 'w') as f:
        f.write(output)

print("Successfully retrieved modeling_gemma3_neuron.py locally!")
