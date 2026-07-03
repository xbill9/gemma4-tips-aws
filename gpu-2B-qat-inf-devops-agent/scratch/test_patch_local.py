import os

def patch_file(filepath, target, replacement):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    with open(filepath, 'r') as f:
        content = f.read()
    if replacement in content:
        print(f"  Result: ALREADY APPLIED")
        return True
    if target in content:
        print(f"  Result: TARGET MATCHED (READY TO APPLY)")
        return True
    else:
        print(f"  Result: TARGET NOT FOUND")
        # Print a small debug snippet around where we expect it to be
        return False

attention_base_file = 'attention_base_remote.py'

print("1. Testing get_flash_attention_strategy_cp:")
target_cp = '    def get_flash_attention_strategy_cp(self, q_len):'
repl_cp = '''    def get_flash_attention_strategy_cp(self, q_len):
        if getattr(self, "head_dim", 0) > 128:
            return FlashAttentionStrategy.NONE'''
patch_file(attention_base_file, target_cp, repl_cp)

print("\n2. Testing get_flash_attention_strategy:")
target_strategy = '    def get_flash_attention_strategy(self, q_len, has_attention_mask) -> FlashAttentionStrategy:'
repl_strategy = '''    def get_flash_attention_strategy(self, q_len, has_attention_mask) -> FlashAttentionStrategy:
        if getattr(self, "head_dim", 0) > 128:
            return FlashAttentionStrategy.NONE'''
patch_file(attention_base_file, target_strategy, repl_strategy)

print("\n3. Testing apply_rotary_embedding:")
target_rope = '''    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)'''
repl_rope = '''    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            max_dim = max(Q.shape[-1], K.shape[-1])
            if cos_cache is None or sin_cache is None or cos_cache.shape[-1] < max_dim:
                if Q.shape[-1] == max_dim:
                    cos_cache, sin_cache = self.rotary_emb(Q, position_ids)
                else:
                    cos_cache, sin_cache = self.rotary_emb(K, position_ids)
            from .utils import _rotate_half
            q_cos = cos_cache[..., :Q.shape[-1]]
            q_sin = sin_cache[..., :Q.shape[-1]]
            k_cos = cos_cache[..., :K.shape[-1]]
            k_sin = sin_cache[..., :K.shape[-1]]
            cos_q = q_cos.unsqueeze(1)
            sin_q = q_sin.unsqueeze(1)
            Q = (Q * cos_q) + (_rotate_half(Q) * sin_q)
            cos_k = k_cos.unsqueeze(1)
            sin_k = k_sin.unsqueeze(1)
            K = (K * cos_k) + (_rotate_half(K) * sin_k)'''
patch_file(attention_base_file, target_rope, repl_rope)

print("\n4. Testing prep_qkv_tensors:")
target_prep = '        return Q, K, V, cos_cache, sin_cache, residual'
repl_prep = '''        if Q.shape[-1] < 512:
            import torch.nn.functional as F
            Q = F.pad(Q, (0, 512 - Q.shape[-1]))
            K = F.pad(K, (0, 512 - K.shape[-1]))
            V = F.pad(V, (0, 512 - V.shape[-1]))
        return Q, K, V, cos_cache, sin_cache, residual'''
patch_file(attention_base_file, target_prep, repl_prep)

print("\n5. Testing attn_output merge multi head:")
target_merge = '''        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)'''
repl_merge = '''        if attn_output.shape[-1] == 512 and self.head_dim == 256:
            attn_output = attn_output[..., :256]
        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)'''
patch_file(attention_base_file, target_merge, repl_merge)

print("\n6. Testing compute_for_token_gen definitions:")
target_tokengen = '''    def compute_for_token_gen(
        self,
        Q,
        K,
        V,
        position_ids,
        past_key_value,
        attention_mask,
        active_mask,
        is_prefix_caching=False,
    ) -> Tensor:'''
repl_tokengen = '''    def compute_for_token_gen(
        self,
        Q,
        K,
        V,
        position_ids,
        past_key_value,
        attention_mask,
        active_mask,
        is_prefix_caching=False,
    ) -> Tensor:
        if getattr(self, "head_dim", 0) == 256:
            if Q.shape[-1] == 512:
                Q = Q[..., :256]
            if K.shape[-1] == 512:
                K = K[..., :256]
            if V.shape[-1] == 512:
                V = V[..., :256]
        if past_key_value is not None and len(past_key_value) > 0 and past_key_value[0] is not None:
            k_prior_heads = past_key_value[0].shape[1]
            if Q.shape[1] % k_prior_heads == 0:
                self.num_key_value_groups = Q.shape[1] // k_prior_heads
        elif K is not None:
            if Q.shape[1] % K.shape[1] == 0:
                self.num_key_value_groups = Q.shape[1] // K.shape[1]'''
patch_file(attention_base_file, target_tokengen, repl_tokengen)

print("\n7. Testing K_prior and V_prior repetition:")
target_prior_repeat = '''        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)'''
repl_prior_repeat = '''        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        prior_repeat = Q.shape[1] // K_prior.shape[1] if (K_prior is not None and K_prior.shape[1] > 0) else getattr(self, "num_key_value_groups", 1)
        K_prior = repeat_kv(K_prior, prior_repeat)
        V_prior = repeat_kv(V_prior, prior_repeat)'''
patch_file(attention_base_file, target_prior_repeat, repl_prior_repeat)

print("\n8. Testing K_active and V_active repetition:")
target_active_repeat = '''        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)'''
repl_active_repeat = '''        # ii. active (current/new) KV
        active_repeat = Q.shape[1] // K.shape[1] if (K is not None and K.shape[1] > 0) else getattr(self, "num_key_value_groups", 1)
        K_active = repeat_kv(K, active_repeat)
        V_active = repeat_kv(V, active_repeat)'''
patch_file(attention_base_file, target_active_repeat, repl_active_repeat)
