# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the project and assist in development. It contains information relevant to the Gemini CLI's operation and interaction with this project, including specific setup instructions for integrating with the CLI.

## Role
This project functions as an expert DevOps and SRE Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the local self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted locally via Docker and Ollama/vLLM**. It analyzes local infrastructure issues and logs to suggest remediations.

## 🟢 Current Status: ONLINE
The Gemma 4 inference stack is currently deployed and active locally.
*   **Active Endpoint:** `http://localhost:8000`
*   **Model:** `google/gemma-4-E2B-it` (running locally as `gemma4:e2b`)

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (Ollama/vLLM on Docker) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running local OpenAI-compatible inference instance. Your local Docker deployment needs:
*   **Hardware:** Local machine or VM with sufficient CPU/GPU resources.
*   **Software:** Docker with `ollama/ollama:latest` or `vllm/vllm-tpu` (run in CPU/GPU mode).
*   **Model:** `gemma4:e2b` (Ollama ID) or `google/gemma-4-E2B-it`.
*   **Networking:** Host port 8000 mapped to the container's serving port.

### 2. Environment Variables
You can configure the following variables for the MCP server:
*   `MODEL_NAME`: The model identifier (defaults to `google/gemma-4-E2B-it`).
*   `LOCAL_DOCKER_IMAGE`: The local Docker image to use (defaults to `ollama/ollama:latest`).
*   `LOCAL_VLLM_PORT`: Port number for local API server (defaults to `8000`).

## 🛠 Usage & Setup

### Step 1: Run the Local Inference Stack
Start the container and pull the model:
```bash
make run
```
You can also use the [manage_docker](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L127) tool with action `start` to automatically run and pull the model.

### Step 2: Run the MCP Server
Install dependencies and run the server locally:
```bash
make install
make run
```

### Step 3: LiteLLM Proxy Setup
To enable seamless integration with the Gemini CLI, you can set up a LiteLLM proxy to route requests to your self-hosted local endpoint.

#### 1. Install LiteLLM Proxy
You need the [proxy] version of LiteLLM:
```bash
pip install 'litellm[proxy]'
```

#### 2. Create a configuration file ([litellm_config.yaml](file:///home/xbill/gemma4-tips/local-devops-agent/agents/litellm_config.yaml))
Create this file to map the Gemini model names to your local endpoint:
```yaml
model_list:
  - model_name: "gemma4-local"
    litellm_params:
      model: "openai/gemma4:e2b" # Tell LiteLLM it's an OpenAI-style endpoint
      api_base: "http://localhost:8000/v1" # Your local Ollama/vLLM IP
      api_key: "none"
    router_settings:
      model_group_alias:
        "gemini-2.0-flash": "gemma4-local"
        "gemini-2.0-flash-lite": "gemma4-local"
        "gemini-1.5-flash": "gemma4-local"
        "gemini-1.5-pro": "gemma4-local"
```

#### 3. Start the LiteLLM Proxy
```bash
litellm --config [litellm_config.yaml](file:///home/xbill/gemma4-tips/local-devops-agent/agents/litellm_config.yaml) --port 4000
```

#### 4. Configure Gemini CLI to use the Proxy
```bash
export GOOGLE_GEMINI_BASE_URL="http://localhost:4000"
export GEMINI_MODEL="google/gemma-4-E2B-it"
export GEMINI_API_KEY="local-proxy-token"
```

Now, every time you run `gemini`, it will be powered by your local self-hosted Gemma 4 stack.

## 🛠 Available Tools

The following tools are available via the MCP server:

### Deployment & Configuration
*   **[manage_docker](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L127)**: Manages the local container (`start`, `stop`, `restart`, `status`, `log`, and `rm` actions).
*   **[save_hf_token](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L113)**: Securely saves a Hugging Face API token locally.

### Monitoring & Status
*   **[get_system_status](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L162)**: Provides a high-level status dashboard of the local Docker container and vLLM service.
*   **[get_endpoint](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L194)**: Verifies local endpoint connectivity and returns the active service URL.
*   **[get_active_models](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L456)**: Gets active resource usage (actively loaded models, sizes, CPU/GPU status, context size) via `ollama ps` (Ollama backend only).
*   **[get_help](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L418)**: Provides help text and summarizes the configuration options and all available SRE/DevOps tools.

### Performance & Benchmarking
*   **[run_benchmark](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L286)**: Runs vLLM's internal serving benchmark tool inside the local container, or falls back to local benchmarking suite.
*   **[get_docker_logs](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L359)**: Streams startup and execution logs from the local Docker container.
*   **[analyze_local_logs](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L371)**: Fetches the local container logs and uses Gemma 4 to analyze them for SRE issues.

### AI & Diagnostics
*   **[query_gemma4](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L205)**: Primary tool for interacting with the self-hosted local model.
*   **[query_gemma4_with_stats](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L225)**: Provides streaming responses with TTFT and total latency metrics.
*   **[verify_model_health](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L84)**: Performs a health check by querying the model with a simple prompt and measuring response latency.
*   **[get_system_details](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L390)**: Retrieves detailed information about the running local model, engine, and versions.
*   **[get_model_show_details](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L469)**: Gets deep model parameters, architecture, license, and config details via `ollama show <model_name>` (Ollama backend only).
