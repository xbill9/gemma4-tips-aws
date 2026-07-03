import re

def apply_patch():
    # File 1: gpt_oss_kv_cache_manager.py
    file1 = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/gpt_oss_kv_cache_manager.py'
    with open(file1, 'r') as f:
        content = f.read()

    new_method = """    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int):
        is_swa_layer = (layer_idx + 1) % 6 != 0
        if hasattr(self, "k_shapes"):
            k_shape = self.k_shapes[layer_idx]
        else:
            k_shape = getattr(self, "k_shape", full_k.shape)
        seq_dim_size = k_shape[-1] if transposed else k_shape[-2]
        if is_swa_layer:
            limit = min(self.sliding_window, seq_dim_size)
            position_ids = position_ids % limit
        else:
            position_ids = position_ids % seq_dim_size
        index = position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)"""

    # Using re to replace the whole method block
    content = re.sub(
        r'    def _get_index_to_update_new_position\(self, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int\):.*?return index\.view\(\*view_shape\)\.expand_as\(full_k\)',
        new_method,
        content,
        flags=re.DOTALL
    )

    with open(file1, 'w') as f:
        f.write(content)

    # File 2: kv_cache_manager.py
    file2 = '/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py'
    with open(file2, 'r') as f:
        content2 = f.read()

    new_method2 = """    def _get_index_to_update_new_position(self, seq_ids, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int):
        if getattr(self, "attention_chunk_size", None):
            if hasattr(self, "v_shapes"):
                cache_len = self.v_shapes[layer_idx][2]
            else:
                cache_len = self.attention_chunk_size
            if getattr(self.neuron_config, "apply_seq_ids_mask", False):
                if cache_len == self.attention_chunk_size + KV_CACHE_PAD_FOR_SEQ_IDS_MASKING:
                    position_ids = apply_seq_id_mask(
                        position_ids, seq_ids,
                        self.attention_chunk_size + KV_CACHE_PAD_FOR_SEQ_IDS_MASKING - 1, chunk_size=self.attention_chunk_size)
            else:
                position_ids = position_ids % cache_len
        elif getattr(self, "sliding_window", None):
            is_swa_layer = (layer_idx + 1) % 6 != 0
            if hasattr(self, "k_shapes"):
                k_shape = self.k_shapes[layer_idx]
            else:
                k_shape = getattr(self, "k_shape", full_k.shape)
            seq_dim_size = k_shape[-1] if transposed else k_shape[-2]
            if is_swa_layer:
                limit = min(self.sliding_window, seq_dim_size)
                position_ids = position_ids % limit
            else:
                position_ids = position_ids % seq_dim_size
        else:
            if getattr(self.config.neuron_config, "apply_seq_ids_mask", False):
                position_ids = apply_seq_id_mask(
                    position_ids, seq_ids,
                    self.neuron_config.max_length, chunk_size=self.attention_chunk_size)
        index = scatter_index if getattr(self, "is_medusa", False) else position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)"""

    content2 = re.sub(
        r'    def _get_index_to_update_new_position\(self, seq_ids, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int\):.*?return index\.view\(\*view_shape\)\.expand_as\(full_k\)',
        new_method2,
        content2,
        flags=re.DOTALL
    )

    with open(file2, 'w') as f:
        f.write(content2)

if __name__ == '__main__':
    apply_patch()
