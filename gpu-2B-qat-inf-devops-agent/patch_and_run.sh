#!/bin/bash
set -e
echo "Preserving pre-installed transformers 4.57.6..."

echo "Running comprehensive python patcher..."
python3 /apply_all_patches.py

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

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "NeuronQuantConfig":
        return cls()

    def get_quant_method(self, layer, prefix):
        return None
INNER_EOF

# Dynamic TP detection inside container based on exposed neuron devices
device_count=0
for dev in /dev/neuron*; do
    if [ -e "$dev" ]; then
        device_count=$((device_count + 1))
    fi
done
if [ $device_count -eq 0 ]; then
    device_count=1
fi
TP_SIZE=$((device_count * 2))
echo "Detected $device_count Neuron device(s). Using --tensor-parallel-size $TP_SIZE"

echo "Starting vLLM Server with optimized parameters..."
python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E2B-it \
  --max-model-len 1024 \
  --tensor-parallel-size $TP_SIZE \
  --max-num-seqs 2 \
  --num-gpu-blocks-override 128 \
  --swap-space 0 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 512 \
  --block-size 16 \
  --kv-cache-dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser functiongemma \
  --async-scheduling \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --host 0.0.0.0 \
  --port 8080
