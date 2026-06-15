# Deployment Guide: Local Inference Stack (Gemma 4)

This document summarizes the deployment state and configuration for the local self-hosted inference server running on Docker.

## 📦 Model Artifacts
The model used is **Gemma 4 (E2B-it)**, served locally.

*   **Model ID:** `google/gemma-4-E2B-it`
*   **Local Image mapping:** `gemma4:e2b` (Ollama) or `google/gemma-4-E2B-it` (vLLM)

## 🚀 Local Inference Stack
The inference server is deployed locally using Docker.

*   **Hardware Requirements:** 
    - Standard CPU/GPU host machine.
*   **Software Requirements:**
    - Docker daemon.
    - Ollama container (`ollama/ollama:latest`) or vLLM container.
*   **Default Port:** `8000` (mapped to `11434` in Ollama).

## 🛠 Setup & Run

### 1. Launch local container
To start the container manually:
```bash
docker run --name gemma4 -d -p 8000:11434 -e OLLAMA_NUM_THREADS=4 -v ollama_local_volume:/root/.ollama ollama/ollama:latest
```

### 2. Pull the model
Download the Gemma 4 model inside the container:
```bash
docker exec -t gemma4 ollama pull gemma4:e2b
```

### 3. Verification
Verify the model is running and responding via curl:
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma4:e2b",
        "messages": [{"role": "user", "content": "Hello Gemma 4!"}]
    }'
```

## 📜 MCP Agent Integration
To connect the MCP Agent to this local service, export the following environment variables:

```bash
export LOCAL_VLLM_PORT="8000"
export MODEL_NAME="google/gemma-4-E2B-it"
```

Then run the agent:
```bash
make run
```
