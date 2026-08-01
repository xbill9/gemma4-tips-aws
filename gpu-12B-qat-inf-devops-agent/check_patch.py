target_token_gen = '''        if getattr(self, "head_dim", 0) == 256:
            if Q.shape[-1] == 512:
                Q = Q[..., :256]
            if K.shape[-1] == 512:
                K = K[..., :256]
            if V.shape[-1] == 512:
                V = V[..., :256]'''
with open('attention_base_remote.py', 'r') as f:
    content = f.read()
if target_token_gen in content:
    print("Found!")
else:
    print("Not found!")
