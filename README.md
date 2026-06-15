# 🚀 Gemma 4 DevOps Agents

Welcome to the **Gemma-4 DevOps Agents** workspace. This repository contains nine specialized, self-hosted AI-driven DevOps/SRE agents powered by Google's **Gemma 4** model. These agents are packaged as Model Context Protocol (MCP) servers to analyze, monitor, and troubleshoot infrastructure components.

---

## 📂 Project Structure

This workspace is organized into nine distinct sub-agents, each tailored to a specific environment, model configuration, and serving stack:

| Sub-Agent | Purpose | Serving Engine | Target Infrastructure |
| :--- | :--- | :--- | :--- |
| [Local DevOps Agent](file:///home/xbill/gemma4-tips/local-devops-agent) | CPU/GPU local analysis & prototyping | Ollama / vLLM | Local Docker / Workstations |
| [GPU DevOps Agent (4B L4)](file:///home/xbill/gemma4-tips/gpu-4B-L4-devops-agent) | Serverless cloud SRE (4B model on L4 GPU) | vLLM | Google Cloud Run (us-east4) |
| [GPU DevOps Agent (4B 6000)](file:///home/xbill/gemma4-tips/gpu-4B-6000-devops-agent) | Serverless cloud SRE (4B model on RTX 6000 GPU) | vLLM | Google Cloud Run (us-central1) |
| [GPU DevOps Agent (26B 6000)](file:///home/xbill/gemma4-tips/gpu-26B-6000-devops-agent) | Serverless cloud SRE (26B model on RTX 6000 GPU) | vLLM | Google Cloud Run (us-central1) |
| [GPU DevOps Agent (31B 6000)](file:///home/xbill/gemma4-tips/gpu-31B-6000-devops-agent) | Serverless cloud SRE (31B model on RTX 6000 GPU) | vLLM | Google Cloud Run (us-central1) |
| [GPU DevOps Agent (6000)](file:///home/xbill/gemma4-tips/gpu-6000-devops-agent) | Serverless cloud SRE (RTX 6000 GPU configuration) | vLLM | Google Cloud Run (us-central1) |
| [GPU DevOps Agent (vLLM)](file:///home/xbill/gemma4-tips/gpu-vllm-devops-agent) | Serverless cloud SRE (L4 GPU configuration) | vLLM | Google Cloud Run (us-east4) |
| [GPU DevOps Agent (31B QAT L4)](file:///home/xbill/gemma4-tips/gpu-31B-qat-L4-devops-agent) | Serverless cloud SRE (31B QAT model on L4 GPU) | vLLM | Google Cloud Run (us-east4) |
| [TPU DevOps Agent (26B)](file:///home/xbill/gemma4-tips/tpu-26B-devops-agent) | Ultra-high performance TPU SRE (26B configuration) | vLLM | Google Cloud TPUs (v6e Trillium) |
| [TPU DevOps Agent (31B)](file:///home/xbill/gemma4-tips/tpu-31B-devops-agent) | Ultra-high performance TPU SRE (31B configuration) | vLLM | Google Cloud TPUs (v6e Trillium) |
| [TPU DevOps Agent (12B v6e-1)](file:///home/xbill/gemma4-tips/tpu-12B-v6e1-devops-agent) | Ultra-high performance TPU SRE (12B configuration) | vLLM | Google Cloud TPUs (v6e Trillium) |

---

## 🛠 Features & Capabilities

- **Automated SRE Diagnostics:** Fetches and reviews system, container, and Cloud Logging entries using Gemma 4 to identify root causes and generate 3-step remediation plans.
- **Serving Stack Control:** Built-in tools to provision, start, stop, restart, and scale your vLLM and Ollama containers or Cloud TPU Queued Resources.
- **Observability Dashboards:** Real-time dashboards monitoring HBM usage, Tensor Core pressure, Prometheus metrics, and service latencies.
- **Model Benchmarking:** Tools to run load tests and vLLM's internal benchmark suites, returning performance metrics (TTFT, throughput, P95 latency).
- **Gemini CLI Integration:** Custom setup instructions using a LiteLLM Proxy to route standard Gemini CLI commands directly to your private, self-hosted Gemma 4 instance.

---

## 🏗 Global Makefile Usage

A root [Makefile](file:///home/xbill/gemma4-tips/Makefile) is provided to manage the sub-agents collectively:

- **Help / Display commands:**
  ```bash
  make all
  ```
- **Install dependencies in all subdirectories:**
  ```bash
  make install
  ```
- **Run tests across all agents:**
  ```bash
  make test
  ```
- **Lint all Python directories:**
  ```bash
  make lint
  ```
- **Clean build/cache folders:**
  ```bash
  make clean
  ```

---

## 🚀 Sub-Agent Overviews

### 1. [Local DevOps Agent](file:///home/xbill/gemma4-tips/local-devops-agent)
- **Role:** Specialized SRE for local containerized workloads.
- **Inference Stack:** Runs `gemma4:e2b` or `google/gemma-4-E2B-it` via local Docker (`ollama/ollama` or CPU/GPU vLLM).
- **Documentation:** See [local-devops-agent/README.md](file:///home/xbill/gemma4-tips/local-devops-agent/README.md) and [local-devops-agent/GEMINI.md](file:///home/xbill/gemma4-tips/local-devops-agent/GEMINI.md).

### 2. [GPU DevOps Agent (4B L4)](file:///home/xbill/gemma4-tips/gpu-4B-L4-devops-agent)
- **Role:** SRE for serverless GPU-accelerated Cloud Run endpoints running the 4B configuration on L4 GPU.
- **Inference Stack:** Runs `google/gemma-4-E4B-it` via vLLM on Cloud Run.
- **Documentation:** See [gpu-4B-L4-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-4B-L4-devops-agent/README.md).

### 3. [GPU DevOps Agent (4B 6000)](file:///home/xbill/gemma4-tips/gpu-4B-6000-devops-agent)
- **Role:** SRE for serverless GPU-accelerated Cloud Run endpoints running the 4B configuration on RTX 6000 GPU.
- **Inference Stack:** Runs `google/gemma-4-E4B-it` via vLLM on Cloud Run.
- **Documentation:** See [gpu-4B-6000-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-4B-6000-devops-agent/README.md).

### 4. [GPU DevOps Agent (26B 6000)](file:///home/xbill/gemma4-tips/gpu-26B-6000-devops-agent)
- **Role:** SRE for serverless GPU-accelerated Cloud Run endpoints running the 26B configuration on RTX 6000 GPU.
- **Inference Stack:** Runs `google/gemma-4-26B-it` via vLLM on Cloud Run.
- **Documentation:** See [gpu-26B-6000-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-26B-6000-devops-agent/README.md).

### 5. [GPU DevOps Agent (31B 6000)](file:///home/xbill/gemma4-tips/gpu-31B-6000-devops-agent)
- **Role:** SRE for serverless GPU-accelerated Cloud Run endpoints running the 31B configuration on RTX 6000 GPU.
- **Inference Stack:** Runs `google/gemma-4-26B-A4B-it` via vLLM on Cloud Run.
- **Documentation:** See [gpu-31B-6000-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-31B-6000-devops-agent/README.md).

### 6. [GPU DevOps Agent (6000)](file:///home/xbill/gemma4-tips/gpu-6000-devops-agent)
- **Role:** Cloud-based SRE managing GPU-accelerated serverless endpoints (RTX 6000 GPU configuration).
- **Inference Stack:** Runs `google/gemma-4-E4B-it` via vLLM on Cloud Run.
- **Documentation:** See [gpu-6000-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-6000-devops-agent/README.md).

### 7. [GPU DevOps Agent (vLLM)](file:///home/xbill/gemma4-tips/gpu-vllm-devops-agent)
- **Role:** Cloud-based SRE managing GPU-accelerated serverless endpoints (L4 GPU configuration).
- **Inference Stack:** Runs `google/gemma-4-E4B-it` via vLLM on Cloud Run.
- **Documentation:** See [gpu-vllm-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-vllm-devops-agent/README.md).

### 8. [TPU DevOps Agent (26B)](file:///home/xbill/gemma4-tips/tpu-26B-devops-agent)
- **Role:** High-performance TPU SRE/DevOps managing large-scale private clusters (26B configuration).
- **Inference Stack:** Runs `google/gemma-4-31B-it` via vLLM on Google Cloud TPUs (v6e Trillium).
- **Documentation:** See [tpu-26B-devops-agent/README.md](file:///home/xbill/gemma4-tips/tpu-26B-devops-agent/README.md).

### 9. [TPU DevOps Agent (31B)](file:///home/xbill/gemma4-tips/tpu-31B-devops-agent)
- **Role:** High-performance TPU SRE/DevOps managing large-scale private clusters (31B configuration).
- **Inference Stack:** Runs `google/gemma-4-31B-it` via vLLM on Google Cloud TPUs (v6e Trillium).
- **Documentation:** See [tpu-31B-devops-agent/README.md](file:///home/xbill/gemma4-tips/tpu-31B-devops-agent/README.md) and [tpu-31B-devops-agent/GEMINI.md](file:///home/xbill/gemma4-tips/tpu-31B-devops-agent/GEMINI.md).

### 10. [GPU DevOps Agent (31B QAT L4)](file:///home/xbill/gemma4-tips/gpu-31B-qat-L4-devops-agent)
- **Role:** Serverless cloud SRE leveraging the 31B QAT configuration on L4 GPU.
- **Inference Stack:** Runs `google/gemma-4-31B-it-qat-w4a16-ct` via vLLM on Cloud Run.
- **Documentation:** See [gpu-31B-qat-L4-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-31B-qat-L4-devops-agent/README.md).

### 11. [TPU DevOps Agent (12B v6e-1)](file:///home/xbill/gemma4-tips/tpu-12B-v6e1-devops-agent)
- **Role:** High-performance TPU SRE/DevOps managing clusters (12B configuration).
- **Inference Stack:** Runs `google/gemma-4-12B-it` via vLLM on Google Cloud TPUs (v6e Trillium).
- **Documentation:** See [tpu-12B-v6e1-devops-agent/README.md](file:///home/xbill/gemma4-tips/tpu-12B-v6e1-devops-agent/README.md) and [tpu-12B-v6e1-devops-agent/GEMINI.md](file:///home/xbill/gemma4-tips/tpu-12B-v6e1-devops-agent/GEMINI.md).

---

## 🔒 Security & Credentials
When deploying to Google Cloud or Hugging Face, secure credentials using:
- **Hugging Face Access Token:** Saved locally or to Google Secret Manager.
- **Application Default Credentials (ADC):** Set up using GCP credentials helper scripts.

## Credits
Google Cloud credits are provided for this project.

#AgenticArchitect #GoogleAntigravity
