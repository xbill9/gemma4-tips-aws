# Deployment Guide: Self-Hosted vLLM on Cloud Run (Gemma 4)

This document summarizes the deployment state and configuration for the vLLM inference server used by the DevOps Agent.

## 📦 Model Artifacts
The model weights were uploaded to Google Cloud Storage.

*   **Bucket:** `gs://aisprint-491218-bucket/`
*   **Path:** `gemma-4-E4B-it/` (or `gemma-4-26B-A4B-it/`)
*   **Format:** Hugging Face Transformers (Safetensors)

## 🚀 Inference Stack (vLLM)
The inference server is deployed on Cloud Run with GPU acceleration and GCS FUSE.

*   **Service Name:** `gpu-4b-6000-devops-agent`
*   **Service URL:** `https://gpu-4b-6000-devops-agent-289270257791.us-central1.run.app`
*   **Active Route/Status URL:** `https://gpu-4b-6000-devops-agent-wgcq55zbfq-uc.a.run.app`
*   **Region:** `us-central1`
*   **Hardware:** 
    *   **GPU:** 1x NVIDIA RTX PRO 6000
    *   **vCPU:** 20
    *   **Memory:** 80GiB
*   **Configuration:**
    *   **Container Port:** `8080`
    *   **Max Model Length:** `32767`
    *   **Storage:** GCS FUSE mounted at `/mnt/models`
    *   **Zonal Redundancy:** Disabled (`--no-gpu-zonal-redundancy`)

## 🛠 Usage
To connect the MCP Agent to this service, export the following environment variables:

```bash
export VLLM_BASE_URL="https://gpu-4b-6000-devops-agent-289270257791.us-central1.run.app"
export MODEL_NAME="/mnt/models/gemma-4-E4B-it"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then run the agent:
```bash
make run
```

## 📜 Deployment Command (Reference)
```bash
gcloud beta run deploy gpu-4b-6000-devops-agent \
  --image="vllm/vllm-openai:latest" \
  --project="aisprint-491218" \
  --region="us-central1" \
  --execution-environment=gen2 \
  --no-allow-unauthenticated \
  --cpu=20 \
  --memory=80Gi \
  --gpu=1 \
  --gpu-type=nvidia-rtx-pro-6000 \
  --no-gpu-zonal-redundancy \
  --no-cpu-throttling \
  --max-instances=3 \
  --min-instances=1 \
  --concurrency=16 \
  --port=8080 \
  --timeout=3600 \
  --cpu-boost \
  --startup-probe="tcpSocket.port=8080,initialDelaySeconds=240,failureThreshold=40,timeoutSeconds=10,periodSeconds=15" \
  --command="bash" \
  --args="^;^-c;vllm serve /mnt/models/gemma-4-E4B-it --served-model-name gpu-4b-6000-devops-agent --enable-log-requests --enable-chunked-prefill --enable-prefix-caching --generation-config auto --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --dtype bfloat16 --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 8 --gpu-memory-utilization 0.95 --tensor-parallel-size 1 --load-format runai_streamer --port 8080 --host 0.0.0.0 --max-model-len 32767" \
  --add-volume=name=model-volume,type=cloud-storage,bucket=aisprint-491218-bucket,readonly=true,mount-options=uid=1001;gid=1001 \
  --add-volume-mount=volume=model-volume,mount-path=/mnt/models \
  --set-env-vars="MODEL_NAME=/mnt/models/gemma-4-E4B-it,GOOGLE_CLOUD_PROJECT=aisprint-491218,GOOGLE_CLOUD_REGION=us-central1"
```
