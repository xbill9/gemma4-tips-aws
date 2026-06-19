# Self-Hosted vLLM DevOps Agent (MCP Server)

This project provides an automated DevOps/SRE assistant that leverages **Gemma models self-hosted via vLLM on AWS Inferentia (Inf2)**. It bridges CloudWatch Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on EC2 Inferentia) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your EC2 deployment for the model needs:
*   **Hardware Platform:** AWS Inferentia (Inf2 instances).
*   **Neuron Accelerator:** 1x AWS Inferentia2 Device (2 Neuron Cores) e.g., `inf2.xlarge`.
*   **Storage:** An S3 Bucket containing the Gemma model weights (e.g., `s3://vllm-models-bucket/google/gemma-4-12B-it/`).
*   **Image:** `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04`

### 2. Software & API Dependencies
The agent relies on AWS services and Python libraries:
*   **Libraries:** `mcp`, `fastmcp`, `boto3`, `httpx`, `openai`, and `huggingface_hub`.
*   **Permissions:** The IAM credentials running the agent need:
    *   `cloudwatch:GetLogEvents` / `logs:FilterLogEvents` (to read logs).
    *   Access to read/write from the target S3 bucket.
    *   Access to the vLLM endpoint (port 8080 on the EC2 instance).

### 3. Environment Variables
You can configure the following variables for the MCP server:
*   `AWS_DEFAULT_REGION`: Your AWS region (defaults to `us-east-1`).
*   `AWS_BUCKET_NAME`: S3 bucket name (defaults to `vllm-models-bucket`).
*   `VLLM_BASE_URL`: The URL of your EC2 vLLM service. **If omitted, the agent will attempt to auto-discover it using EC2 tags.**
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-12B-it`).

## 🛠 Usage & Setup

### Step 1: Prepare Model Weights
Use the built-in tool `get_huggingface_model_copy_instructions` to download Gemma weights and upload to your S3 bucket.

### Step 2: Deploy vLLM to AWS EC2 (Inferentia)
Run the `get_vllm_deployment_config` tool within the MCP server to generate the exact deployment commands, or deploy via the provided [Makefile](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/Makefile):
```bash
make deploy
```

### Step 3: Run the MCP Server
Install dependencies and run the server:
```bash
make install
# Optional: export VLLM_BASE_URL="your-vllm-url"
make run
```

## 🛠 Available Tools

The following tools are available via the MCP server:

### 🐳 Infrastructure & Deployment
*   **[deploy_vllm](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L459)**: Deploys vLLM to Cloud Run GPU (NVIDIA L4 in us-east4).
*   **[destroy_vllm](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L524)**: Deletes the Cloud Run vLLM service.
*   **[status_vllm](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L550)**: Checks the status of the Cloud Run vLLM service.
*   **[update_vllm_scaling](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L576)**: Updates min/max instances for scaling.
*   **[get_vllm_deployment_config](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L393)**: Generates the `gcloud` deployment command.
*   **[get_vllm_gpu_deployment_config](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L606)**: Generates a GKE manifest for GPU (NVIDIA L4).
*   **[check_gpu_quotas](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L783)**: Checks L4 and other GPU quotas for a region.
*   **[get_vllm_endpoint](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L237)**: Returns the current active vLLM endpoint URL.

### 📦 Model Management
*   **[list_vertex_models](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L251)**: Lists models in the Vertex AI Registry.
*   **[list_bucket_models](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L267)**: Lists model weights in GCS bucket.
*   **[save_hf_token](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L49)**: Securely saves a Hugging Face API token to Secret Manager.
*   **[get_vertex_ai_model_copy_instructions](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L693)**: Guide to transfer Gemma models from Vertex AI Model Garden to GCS.
*   **[get_huggingface_model_copy_instructions](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L737)**: Guide to transfer Gemma models from Hugging Face and upload to GCS.
*   **[get_huggingfacehub_download_path](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L718)**: Resolves local cache path using huggingface_hub.

### 📊 Monitoring & Status
*   **[get_system_status](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L974)**: Provides a high-level status dashboard of the Cloud Run service and health.
*   **[get_endpoint](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L1042)**: Verifies connectivity and returns the active service URL.
*   **[get_model_details](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L935)**: Retrieves detailed model metadata and engine state from `/v1/models`.
*   **[verify_model_health](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L824)**: Deep health check by querying the model with a simple prompt and measuring latency.

### 📈 Performance & Benchmarking
*   **[run_benchmark](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L1066)**: Runs performance/concurrency benchmark sweeps against the Cloud Run vLLM GPU endpoint.

### 💬 Interaction & Diagnostics
*   **[query_gemma4](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L853)**: Primary tool to query the self-hosted model with standard chat message format.
*   **[query_gemma4_with_stats](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L872)**: Queries the model and returns streaming performance statistics (TTFT, throughput).
*   **[query_vllm](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L368)**: Direct text completions querying tool.
*   **[analyze_cloud_logging](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L297)**: Fetches logs from GCP Logging and analyzes them using the model.
*   **[analyze_gpu_logs](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L1215)**: Fetches Cloud Run logs and uses Gemma 4 to analyze them for SRE/DevOps errors.
*   **[suggest_sre_remediation](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L343)**: Suggests remediation plans for SRE errors using the model.
*   **[get_help](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py#L1228)**: Provides help text and summarizes the configuration options and all available SRE/DevOps tools.

## 📦 Resources
The server exposes the following MCP resources:
*   **`config://vllm-deployment-template`**: A YAML template for Cloud Run GPU deployment.

## 📊 Performance Benchmarks (Standard vs. QAT)

The self-hosted **Gemma 4 12B QAT** model has been benchmarked on a single **NVIDIA L4 GPU** (Cloud Run Gen2) to measure concurrency limits:
* **High Concurrency Stability**: The QAT INT4 model maintains a **100% request success rate** up to **512 concurrent users** (with context windows up to 2048 tokens).
* **The QAT Advantage**: The standard 12B model (bfloat16) leaves 0 GB of free VRAM for the KV cache on a single L4 GPU, failing at concurrencies above 8. The QAT model (w4a16) frees up **~18 GB of VRAM** for the KV cache, representing a **~64x improvement in concurrency capacity**.
* Detailed matrix results and SRE insights are available in [benchmark_report_summary.md](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/benchmark_report_summary.md).

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```
This script simulates log analysis, remediation suggestions, and infrastructure configuration generation.

## 🛠 Makefile Helpers
The included [Makefile](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/Makefile) provides several shortcuts:
*   `make install`: Installs Python dependencies listed in [requirements.txt](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/requirements.txt).
*   `make run`: Starts the MCP server via [server.py](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/server.py).
*   `make deploy`: Deploys vLLM to Cloud Run with GPU.
*   `make destroy`: Removes the vLLM Cloud Run service.
*   `make status`: Checks the status of the vLLM service.
*   `make query PROMPT="your prompt"`: Queries the vLLM model directly via `curl`.
*   `make test`: Runs the test suite in [test_agent.py](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/test_agent.py).

## 🧪 Testing
Run the included test suite in [test_agent.py](file:///home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/test_agent.py) to verify the tool registration and basic functionality:
```bash
make test
```
