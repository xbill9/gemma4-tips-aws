import os
f = '/opt/conda/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/serving.py'
with open(f, 'r') as file:
    c = file.read()
target = """        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]"""
repl = """        for i, token_id in enumerate(token_ids):
            if i >= len(top_logprobs):
                continue
            step_top_logprobs = top_logprobs[i]"""
c = c.replace(target, repl)
with open(f, 'w') as file:
    file.write(c)
