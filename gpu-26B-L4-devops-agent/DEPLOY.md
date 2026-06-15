# Deployment Guide: Self-Hosted vLLM on Cloud Run (Gemma 4 26B-it)

This document summarizes the deployment state and configuration for the vLLM inference server used by the DevOps Agent.

## 📦 Model Artifacts
The model was extracted from `gemma-4-26B-it.tar.gz` and uploaded to Google Cloud Storage.


*   **Bucket:** `gs://aisprint-491218-bucket/`
*   **Path:** `gemma-4-26B-it/`
*   **Format:** Hugging Face Transformers (Safetensors)

## 🚀 Inference Stack (vLLM)
The inference server is deployed on Cloud Run with GPU acceleration and GCS FUSE.

*   **Service Name:** `gpu-26b-l4-devops-agent`
*   **Service URL:** `https://gpu-26b-l4-devops-agent-289270257791.us-east4.run.app`
*   **Region:** `us-east4`
*   **Hardware:** 
    *   **GPU:** 1x NVIDIA L4
    *   **vCPU:** 4
    *   **Memory:** 16GiB
*   **Configuration:**
    *   **Container Port:** `8080`
    *   **Max Model Length:** `4096`
    *   **Storage:** GCS FUSE mounted at `/mnt/models`
    *   **Zonal Redundancy:** Disabled (`--no-gpu-zonal-redundancy`)

## 🛠 Usage
To connect the MCP Agent to this service, export the following environment variables:

```bash
export VLLM_BASE_URL="https://gpu-26b-l4-devops-agent-289270257791.us-east4.run.app"
export MODEL_NAME="/mnt/models/gemma-4-26B-it"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then run the agent:
```bash
make run
```

## 📜 Deployment Command (Reference)
```bash
gcloud beta run deploy gpu-26b-l4-devops-agent \
  --image=vllm/vllm-openai:latest \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --memory=32Gi \
  --cpu=8 \
  --execution-environment=gen2 \
  --add-volume=name=model-volume,type=cloud-storage,bucket=aisprint-491218-bucket,readonly=true \
  --add-volume-mount=volume=model-volume,mount-path=/mnt/models \
  --args=--model=/mnt/models/gemma-4-26B-it,--dtype=float16,--quantization=fp8,--safetensors-load-strategy=prefetch,--max-model-len=4096,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=16,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={},--trust-remote-code,--host=0.0.0.0,--port=8000 \
  --no-allow-unauthenticated \
  --region=us-east4 \
  --no-gpu-zonal-redundancy \
  --timeout=3600
```
