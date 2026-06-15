# Gemma 4 Local DevOps Agent Tools

This document summarizes the MCP tools available in [server.py](file:///home/xbill/gemma4-tips/local-devops-agent/server.py) for the Local Gemma 4 SRE Agent.

## Deployment & Configuration

-   **[manage_docker](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L127)**: Manages the local vLLM/Ollama Docker container (actions: `start`, `stop`, `restart`, `status`, `log`, `rm`).
-   **[save_hf_token](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L113)**: Securely saves a Hugging Face API token locally in environment variables and cache.

## Monitoring & Status

-   **[get_system_status](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L162)**: Provides a high-level status dashboard of the local Docker container and vLLM service.
-   **[get_endpoint](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L194)**: Verifies connectivity and returns the active local vLLM service URL.
-   **[get_active_models](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L456)**: Gets active resource usage (actively loaded models, sizes, CPU/GPU status, context size) via `ollama ps` (Ollama backend only).
-   **[get_help](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L418)**: Provides help text and summarizes the configuration options and available tools.

## Performance & Benchmarking

-   **[run_benchmark](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L286)**: Runs vLLM's internal serving benchmark tool inside the local container.
-   **[get_docker_logs](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L359)**: Retrieves startup and execution logs from the local Docker container.
-   **[analyze_local_logs](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L371)**: Fetches the local container logs and uses Gemma 4 to analyze them for SRE/DevOps errors.

## Interaction & Diagnostics

-   **[query_gemma4](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L205)**: Queries the self-hosted local model.
-   **[query_gemma4_with_stats](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L225)**: Queries the local model and provides streaming-based performance metrics (TTFT, throughput, latency).
-   **[verify_model_health](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L84)**: Performs a health check by querying the model with a simple prompt and measuring response latency.
-   **[get_system_details](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L390)**: Retrieves detailed information about the running local model, engine, and versions.
-   **[get_model_show_details](file:///home/xbill/gemma4-tips/local-devops-agent/server.py#L469)**: Gets deep model parameters, architecture, license, and config details via `ollama show <model_name>` (Ollama backend only).
