# 🤖 Gemini Workspace Context: Gemma 4 DevOps Agents

This workspace context file is designed to help **Gemini Code Assistant** (and developer tools) quickly understand the layout, goals, tools, and integration methods of the **Gemma-4 DevOps Agents** project.

---

## 🎯 Project Overview & Role

This repository provides a set of **Model Context Protocol (MCP) servers** representing specialized AI DevOps/SRE agents. They serve two main purposes:
1. **Infrastructure Operations:** Starting, stopping, configuring, scaling, and benchmarking Gemma 4 serving stacks (Ollama or vLLM) on Local, GPU, and TPU environments.
2. **Log & SRE Diagnostics:** Utilizing the self-hosted Gemma 4 models to analyze system/cloud logs and generate remediation suggestions.

---

## 📂 Quick Navigation

Here are the key entrypoints in the codebase:
- **Root Makefile:** [Makefile](file:///home/xbill/gemma4-tips/Makefile) (manages actions across all agents)
- **Local Agent:**
  - Server source: [local-devops-agent/server.py](file:///home/xbill/gemma4-tips/local-devops-agent/server.py)
  - Details: [local-devops-agent/GEMINI.md](file:///home/xbill/gemma4-tips/local-devops-agent/GEMINI.md) & [local-devops-agent/README.md](file:///home/xbill/gemma4-tips/local-devops-agent/README.md)
- **GPU Agent (4B L4):**
  - Server source: [gpu-4B-L4-devops-agent/server.py](file:///home/xbill/gemma4-tips/gpu-4B-L4-devops-agent/server.py)
  - Details: [gpu-4B-L4-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-4B-L4-devops-agent/README.md)
- **GPU Agent (4B 6000):**
  - Server source: [gpu-4B-6000-devops-agent/server.py](file:///home/xbill/gemma4-tips/gpu-4B-6000-devops-agent/server.py)
  - Details: [gpu-4B-6000-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-4B-6000-devops-agent/README.md)
- **GPU Agent (26B 6000):**
  - Server source: [gpu-26B-6000-devops-agent/server.py](file:///home/xbill/gemma4-tips/gpu-26B-6000-devops-agent/server.py)
  - Details: [gpu-26B-6000-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-26B-6000-devops-agent/README.md)
- **GPU Agent (31B 6000):**
  - Server source: [gpu-31B-6000-devops-agent/server.py](file:///home/xbill/gemma4-tips/gpu-31B-6000-devops-agent/server.py)
  - Details: [gpu-31B-6000-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-31B-6000-devops-agent/README.md)
- **GPU Agent (6000):**
  - Server source: [gpu-6000-devops-agent/server.py](file:///home/xbill/gemma4-tips/gpu-6000-devops-agent/server.py)
  - Details: [gpu-6000-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-6000-devops-agent/README.md)
- **GPU Agent (vLLM):**
  - Server source: [gpu-vllm-devops-agent/server.py](file:///home/xbill/gemma4-tips/gpu-vllm-devops-agent/server.py)
  - Details: [gpu-vllm-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-vllm-devops-agent/README.md)
- **TPU Agent (26B):**
  - Server source: [tpu-26B-devops-agent/server.py](file:///home/xbill/gemma4-tips/tpu-26B-devops-agent/server.py)
  - Details: [tpu-26B-devops-agent/README.md](file:///home/xbill/gemma4-tips/tpu-26B-devops-agent/README.md)
- **TPU Agent (31B):**
  - Server source: [tpu-31B-devops-agent/server.py](file:///home/xbill/gemma4-tips/tpu-31B-devops-agent/server.py)
  - Details: [tpu-31B-devops-agent/GEMINI.md](file:///home/xbill/gemma4-tips/tpu-31B-devops-agent/GEMINI.md) & [tpu-31B-devops-agent/README.md](file:///home/xbill/gemma4-tips/tpu-31B-devops-agent/README.md)
- **GPU Agent (31B QAT L4):**
  - Server source: [gpu-31B-qat-L4-devops-agent/server.py](file:///home/xbill/gemma4-tips/gpu-31B-qat-L4-devops-agent/server.py)
  - Details: [gpu-31B-qat-L4-devops-agent/README.md](file:///home/xbill/gemma4-tips/gpu-31B-qat-L4-devops-agent/README.md)
- **TPU Agent (12B v6e-1):**
  - Server source: [tpu-12B-v6e1-devops-agent/server.py](file:///home/xbill/gemma4-tips/tpu-12B-v6e1-devops-agent/server.py)
  - Details: [tpu-12B-v6e1-devops-agent/GEMINI.md](file:///home/xbill/gemma4-tips/tpu-12B-v6e1-devops-agent/GEMINI.md) & [tpu-12B-v6e1-devops-agent/README.md](file:///home/xbill/gemma4-tips/tpu-12B-v6e1-devops-agent/README.md)

---

## 🛠 Development Workflow & Makefile Tasks

For developer convenience, the root [Makefile](file:///home/xbill/gemma4-tips/Makefile) aggregates tasks across all sub-agents:

```bash
# Run 'make clean', 'make test', 'make lint', or 'make install' to invoke it on all agents
make install   # Prepares dependencies for all servers
make lint      # Standardizes code formatting
make test      # Validates server initializations and mock tests
```

---

## 🔗 Integration with Gemini CLI via LiteLLM Proxy

You can redirect your standard Gemini CLI commands to run against the private self-hosted Gemma 4 models in this repository. This allows developers to use their own self-hosted inference engines under the hood.

> [!NOTE]
> Below are the configurations to route local or cloud endpoints via a LiteLLM Proxy.

### 1. Install LiteLLM Proxy
```bash
pip install 'litellm[proxy]'
```

### 2. Configure LiteLLM
Choose the configuration based on which agent endpoint you wish to target.

#### Option A: Target Local Agent (Ollama/vLLM)
Create a `litellm_config.yaml`:
```yaml
model_list:
  - model_name: "gemma4-local"
    litellm_params:
      model: "openai/gemma4:e2b"
      api_base: "http://localhost:8000/v1"
      api_key: "none"
    router_settings:
      model_group_alias:
        "gemini-2.0-flash": "gemma4-local"
        "gemini-2.0-flash-lite": "gemma4-local"
        "gemini-1.5-flash": "gemma4-local"
        "gemini-1.5-pro": "gemma4-local"
```

#### Option B1: Target Cloud Run GPU Agent (RTX 6000 Config)
Create a `litellm_config.yaml`:
```yaml
model_list:
  - model_name: "gemma4-gpu-6000"
    litellm_params:
      model: "openai/google/gemma-4-26B-it"
      api_base: "https://your-cloud-run-url/v1"
      api_key: "none"
    router_settings:
      model_group_alias:
        "gemini-2.0-flash": "gemma4-gpu-6000"
        "gemini-2.0-flash-lite": "gemma4-gpu-6000"
        "gemini-1.5-flash": "gemma4-gpu-6000"
        "gemini-1.5-pro": "gemma4-gpu-6000"
```

#### Option B2: Target Cloud Run GPU Agent (L4 Config)
Create a `litellm_config.yaml`:
```yaml
model_list:
  - model_name: "gemma4-gpu-l4"
    litellm_params:
      model: "openai/google/gemma-4-E4B-it"
      api_base: "https://your-cloud-run-url/v1"
      api_key: "none"
    router_settings:
      model_group_alias:
        "gemini-2.0-flash": "gemma4-gpu-l4"
        "gemini-2.0-flash-lite": "gemma4-gpu-l4"
        "gemini-1.5-flash": "gemma4-gpu-l4"
        "gemini-1.5-pro": "gemma4-gpu-l4"
```

#### Option C: Target Cloud TPU Agent
Create a `litellm_config.yaml`:
```yaml
model_list:
  - model_name: "gemma4-tpu"
    litellm_params:
      model: "openai/google/gemma-4-31B-it"
      api_base: "http://YOUR_TPU_IP_ADDRESS:8000/v1"
      api_key: "none"
    router_settings:
      model_group_alias:
        "gemini-2.0-flash": "gemma4-tpu"
        "gemini-2.0-flash-lite": "gemma4-tpu"
        "gemini-1.5-flash": "gemma4-tpu"
        "gemini-1.5-pro": "gemma4-tpu"
```

### 3. Run Proxy & Export Variables
Run the proxy locally:
```bash
litellm --config litellm_config.yaml --port 4000
```
Then configure your shell environment:
```bash
export GOOGLE_GEMINI_BASE_URL="http://localhost:4000"
export GEMINI_API_KEY="local-proxy-token"
# Select model target corresponding to option chosen
export GEMINI_MODEL="google/gemma-4-31B-it" # Or google/gemma-4-E2B-it / google/gemma-4-26B-it / google/gemma-4-E4B-it
```

---

## 🔧 Technical Standards for vLLM & Gemma 4 Tool Calling
When managing TPU/GPU deployments or customizing vLLM serving, ensure the following vLLM serving parameters are applied for stable Gemma 4 tool integration:
- **Optimization flags:** `--tensor-parallel-size 8` (TPU v6e-8), `--disable_chunked_mm_input`, `--max-model-len 16384`.
- **Tool Parsing:** `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4` to enable native function calling compatibility.
- **Multimodal configuration:** `--limit-mm-per-prompt '{"image":4,"audio":1}'` and `--max_num_batched_tokens 4096`.
- **Universal SRE Help:** All agents expose a standardized `get_help` tool providing details on active configuration environment variables and all exposed tools.
