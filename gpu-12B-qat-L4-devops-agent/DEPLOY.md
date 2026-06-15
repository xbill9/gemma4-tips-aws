# Deployment Guide: Self-Hosted vLLM on Cloud Run (Gemma 4 12B-it)

This document summarizes the deployment state and configuration for the vLLM inference server used by the DevOps Agent.

## 📦 Model Artifacts
The model was extracted from `gemma-4-12B-it-qat-w4a16-ct.tar.gz` and uploaded to Google Cloud Storage.


*   **Bucket:** `gs://aisprint-491218-bucket/`
*   **Path:** `gemma-4-12B-it-qat-w4a16-ct/`
*   **Format:** Hugging Face Transformers (Safetensors)

## 🚀 Inference Stack (vLLM)
The inference server is deployed on Cloud Run with GPU acceleration and GCS FUSE.

*   **Service Name:** `gpu-12b-qat-l4-devops-agent`
*   **Service URL:** `https://gpu-12b-qat-l4-devops-agent-289270257791.us-east4.run.app`
*   **Region:** `us-east4`
*   **Hardware:** 
    *   **GPU:** 1x NVIDIA L4
    *   **vCPU:** 4
    *   **Memory:** 16GiB
*   **Configuration:**
    *   **Container Port:** `8080`
    *   **Max Model Length:** `32768`
    *   **Storage:** GCS FUSE mounted at `/mnt/models`
    *   **Zonal Redundancy:** Disabled (`--no-gpu-zonal-redundancy`)

## 🛠 Usage
To connect the MCP Agent to this service, export the following environment variables:

```bash
export VLLM_BASE_URL="https://gpu-12b-qat-l4-devops-agent-289270257791.us-east4.run.app"
export MODEL_NAME="/mnt/models/gemma-4-12B-it-qat-w4a16-ct"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then run the agent:
```bash
make run
```

## 📜 Deployment Command (Reference)
```bash
gcloud beta run deploy gpu-12b-qat-l4-devops-agent \
  --image=vllm/vllm-openai:nightly \
  --command=python3,-m,vllm.entrypoints.openai.api_server \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --no-cpu-throttling \
  --concurrency=4 \
  --timeout=3600 \
  --startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=10,initialDelaySeconds=180,httpGet.port=8080,httpGet.path=/health \
  --max-instances=1 \
  --min-instances=0 \
  --port=8080 \
  --memory=16Gi \
  --cpu=4 \
  --execution-environment=gen2 \
  --add-volume=name=model-volume,type=cloud-storage,bucket=aisprint-491218-bucket,readonly=true,mount-options=uid=1001;gid=1001 \
  --add-volume-mount=volume=model-volume,mount-path=/mnt/models \
  --args=--model=/mnt/models/gemma-4-12B-it-qat-w4a16-ct,--quantization=compressed-tensors,--dtype=bfloat16,--max-model-len=32768,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={},--host=0.0.0.0,--port=8080 \
  --no-allow-unauthenticated \
  --region=us-east4 \
  --set-env-vars=VLLM_ENABLE_CUDA_COMPATIBILITY=1,VLLM_USE_V1=0,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,VLLM_DISABLE_FLASHINFER=1,VLLM_USE_FLASHINFER_SAMPLER=0,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MKL_NUM_THREADS=1,OMP_NUM_THREADS=1,MALLOC_TRIM_THRESHOLD_=65536
```
