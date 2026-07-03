#!/bin/bash
set -e
echo "Ensuring correct transformers version..."
pip install "transformers==4.57.6"

echo "Running python patcher for transformers..."
python3 /home/ubuntu/patch_transformers.py

echo "Registering neuron_quant quantization method in vLLM..."
cat << 'INNER_EOF' >> /opt/conda/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__init__.py

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import register_quantization_config
import torch

@register_quantization_config("neuron_quant")
class NeuronQuantConfig(QuantizationConfig):
    def get_name(self) -> str:
        return "neuron_quant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "NeuronQuantConfig":
        return cls()

    def get_quant_method(self, layer, prefix):
        return None
INNER_EOF

echo "Starting vLLM Server with memory optimizations..."
if [ -z "$tensor_parallel_size" ]; then
  tensor_parallel_size=2
fi
python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-12B-it \
  --max-model-len 1024 \
  --tensor-parallel-size $tensor_parallel_size \
  --max-num-seqs 2 \
  --swap-space 0 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 512 \
  --block-size 16 \
  --enable-auto-tool-choice \
  --tool-call-parser functiongemma \
  --host 0.0.0.0 \
  --port 8080


