import asyncio
import json
import logging
import os
import shlex
import sys
import time
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

# Setup logging
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm-devops-agent")

# Initialize FastMCP server
mcp = FastMCP("Local Gemma 4 SRE Agent")

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "gemma4:e2b")
LOCAL_DOCKER_IMAGE = os.getenv("LOCAL_DOCKER_IMAGE", "ollama/ollama:latest")
LOCAL_VLLM_PORT = int(os.getenv("LOCAL_VLLM_PORT", "8000"))


def get_current_model_name() -> str:
    """Returns the correct model identifier based on local mapping."""
    name_map = {
        "google/gemma-4-E2B-it": "gemma4:e2b",
        "google/gemma-4-E4B-it": "gemma4:e4b",
        "google/gemma-4-26B-A4B-it": "gemma4:26b",
        "google/gemma-4-31B-it": "gemma4:31b",
    }
    if MODEL_NAME in name_map:
        return name_map[MODEL_NAME]
    if ":" in MODEL_NAME:
        return MODEL_NAME
    return "gemma4:e2b"


# --- Helper Functions ---


async def run_command(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Runs a shell command asynchronously."""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return process.returncode or 0, stdout.decode().strip(), stderr.decode().strip()
    except asyncio.TimeoutError:
        try:
            process.kill()
        except Exception:
            pass
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


VLLM_URL = f"http://localhost:{LOCAL_VLLM_PORT}"


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the local vLLM service."""
    return AsyncOpenAI(base_url=f"{VLLM_URL}/v1", api_key="not-needed")


async def check_service_health(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Helper to check local endpoint health status."""
    for path, name in [("health", "health"), ("v1/models", "v1/models")]:
        try:
            res = await client.get(f"{VLLM_URL}/{path}", timeout=2)
            if res.status_code == 200:
                return True, name
        except Exception:
            pass
    return False, ""


@mcp.tool()
async def verify_model_health() -> str:
    """Runs a deep health check with latency reporting on the local model."""
    try:
        client = await get_vllm_client()
        start_time = time.monotonic()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, is the model working?"}],
            model=get_current_model_name(),
            max_tokens=200,
        )
        end_time = time.monotonic()
        latency = end_time - start_time
        if not chat_completion.choices:
            return "❌ Model health check FAILED: No choices returned."
        response_content = chat_completion.choices[0].message.content

        if response_content:
            return (
                f"✅ Model health check PASSED.\n"
                f"Response: '{response_content[:50]}...'\n"
                f"Latency: {latency:.2f} seconds."
            )
        else:
            return "❌ Model health check FAILED: Empty response."
    except Exception as e:
        return f"❌ Model health check FAILED: {e}"


@mcp.tool()
async def save_hf_token(token: str) -> str:
    """Securely saves a Hugging Face API token locally."""
    os.environ["HF_TOKEN"] = token
    try:
        hf_cache_token_path = os.path.expanduser("~/.cache/huggingface/token")
        os.makedirs(os.path.dirname(hf_cache_token_path), exist_ok=True)
        with open(hf_cache_token_path, "w") as f:
            f.write(token)
        return "✅ Token saved locally to environment and ~/.cache/huggingface/token"
    except Exception as e:
        return f"✅ Token saved locally to environment (failed to write to file: {e})"


@mcp.tool()
async def manage_docker(action: str = "status") -> str:
    """Manages the local vLLM/Ollama Docker container (actions: start, stop, restart, status, log, rm)."""
    logger.info(f"Local deployment mode: Managing Docker with action '{action}'")
    ollama_model = get_current_model_name()
    docker_run_cmd = os.getenv(
        "LOCAL_DOCKER_RUN_CMD",
        f'docker run --name gemma4 -d -p {LOCAL_VLLM_PORT}:11434 --cpuset-cpus="0-7" '
        f"-e OLLAMA_KV_CACHE_TYPE=q4_0 -e OLLAMA_NUM_PARALLEL=1 -e OLLAMA_NUM_THREADS=4 "
        f"-v ollama_local_volume:/root/.ollama {LOCAL_DOCKER_IMAGE}",
    )
    commands = {
        "start": f"(docker start gemma4 || {docker_run_cmd}) && sleep 2 && docker exec -d gemma4 ollama pull {ollama_model}",
        "stop": "docker stop gemma4",
        "restart": "docker restart gemma4",
        "status": "docker ps -a --filter name=gemma4",
        "log": "docker logs --tail 100 gemma4",
        "rm": "docker rm -f gemma4",
    }
    cmd_str = commands.get(action, commands["status"])
    process = await asyncio.create_subprocess_shell(
        cmd_str,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
        rc = process.returncode or 0
        if rc != 0:
            return f"❌ Local Docker {action} failed: {stderr.decode().strip() or stdout.decode().strip()}"
        return f"✅ Local Docker {action} command executed:\n{stdout.decode().strip()}"
    except Exception as e:
        return f"❌ Local Docker {action} execution error: {e}"


@mcp.tool()
async def get_system_status() -> str:
    """Provides a high-level dashboard of local system status."""
    health = "🔴 Offline"
    async with httpx.AsyncClient() as client:
        is_up, _ = await check_service_health(client)
        if is_up:
            health = f"🟢 Online ({VLLM_URL})"

    docker_status = "🔴 Unknown (Container check failed)"
    try:
        rc, out, _ = await run_command(["docker", "ps", "-a", "--filter", "name=gemma4", "--format", "{{.Status}}"])
        if rc == 0 and out:
            docker_status = f"🟢 Running ({out})" if "Up" in out else f"🔴 Stopped ({out})"
        elif rc == 0:
            docker_status = "🔴 Not Created (Container 'gemma4' does not exist)"
    except Exception as e:
        docker_status = f"🔴 Error checking container: {e}"

    if "🟢" in health:
        next_step = "Use `query_gemma4` to interact with the model."
    else:
        next_step = "Call `manage_docker` with action='start' to start the local Docker container."

    return (
        f"### 🌀 Local System Status\n"
        f"- **vLLM Health:** {health}\n"
        f"- **Docker Container Status:** {docker_status}\n"
        f"**👉 Next Step:** {next_step}"
    )


@mcp.tool()
async def get_endpoint() -> str:
    """Returns the active local vLLM service URL if available."""
    async with httpx.AsyncClient() as client:
        is_up, endpoint_type = await check_service_health(client)
        if is_up:
            label = "vLLM/Ollama" if endpoint_type == "v1/models" else "vLLM"
            return f"🟢 Local {label} is Online at: {VLLM_URL}"
        return f"🔴 Local vLLM endpoint configured at {VLLM_URL} but is Unreachable. Start it with `manage_docker`."


@mcp.tool()
async def query_gemma4(prompt: str) -> str:
    """Queries the self-hosted local Gemma 4 model."""
    logger.info(f"Querying local model with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=get_current_model_name(),
        )
        if not chat_completion.choices:
            return "❌ Query failed: No choices returned from model."
        response = chat_completion.choices[0].message.content or "No response from model."
        logger.info(f"Model response: '{response[:100]}...'")
        return response
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        return f"❌ An error occurred while querying the model: {e}"


@mcp.tool()
async def query_gemma4_with_stats(prompt: str) -> str:
    """
    Queries the self-hosted local Gemma 4 model and returns detailed performance statistics.

    This tool provides:
    - The full model response.
    - Time to First Token (TTFT).
    - Total generation time.
    - Tokens per second.
    """
    logger.info(f"Querying model with stats with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()

        start_time = time.monotonic()
        ttft = None
        response_content = ""
        total_tokens = 0

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=get_current_model_name(),
            stream=True,
        )

        async for chunk in stream:
            if ttft is None:
                ttft = time.monotonic() - start_time

            content = chunk.choices[0].delta.content
            if content:
                response_content += content
                total_tokens += 1  # Rough token count

        end_time = time.monotonic()
        total_time = end_time - start_time

        if not response_content:
            return "❌ Model returned an empty response."

        tokens_per_second = total_tokens / (total_time - ttft) if ttft and total_time > ttft else 0

        stats_report = (
            f"### 📊 Performance Stats\n"
            f"- **Time to First Token (TTFT):** `{ttft:.3f}s`\n"
            f"- **Total Generation Time:** `{total_time:.3f}s`\n"
            f"- **Tokens per Second:** `{tokens_per_second:.2f} tokens/s`\n"
            f"- **Total Tokens (approx.):** `{total_tokens}`\n"
            f"\n### 💬 Model Response\n"
            f"{response_content}"
        )

        logger.info(f"Model response with stats: TTFT={ttft:.3f}s, TotalTime={total_time:.3f}s")
        return stats_report

    except Exception as e:
        logger.error(f"Error querying model with stats: {e}")
        return f"❌ An error occurred while querying the model with stats: {e}"


@mcp.tool()
async def run_benchmark(
    backend: str = "vllm",
    model: str = "google/gemma-4-E2B-it",
    dataset_name: str = "random",
    num_prompts: int = 100,
    random_input_len: int = 1024,
    random_output_len: int = 128,
    max_concurrency: Optional[int] = None,
) -> str:
    """Runs vLLM's internal benchmark tool inside the local container or falls back to the local benchmarking suite if Ollama is running."""
    if "ollama" in LOCAL_DOCKER_IMAGE.lower():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        suite_path = os.path.join(current_dir, "benchmarking_suite.py")
        if not os.path.exists(suite_path):
            return "❌ Local benchmark failed: benchmarking_suite.py not found."

        cmd = [
            sys.executable,
            suite_path,
            "--url",
            VLLM_URL,
            "--model",
            get_current_model_name(),
            "--requests",
            str(num_prompts),
            "--tokens",
            str(random_output_len),
            "--output",
            os.path.join(current_dir, "benchmark_results.csv"),
        ]
        if max_concurrency:
            concs = [c for c in [1, 2, 4, 8, 16, 32, 64] if c <= max_concurrency]
            if max_concurrency not in concs:
                concs.append(max_concurrency)
            cmd.extend(["--concurrencies", ",".join(map(str, concs))])
        rc, out, err = await run_command(cmd, timeout=600)
        if rc != 0:
            return f"⚠️ Local benchmark failed.\nError: {err}\nOutput: {out}"
        return f"✅ Local benchmark completed:\n{out}"

    benchmark_cmd = (
        "vllm bench serve "
        f"--backend {backend} "
        f"--model {model} "
        f"--dataset-name {dataset_name} "
        f"--num-prompts {num_prompts} "
        f"--random-input-len {random_input_len} "
        f"--random-output-len {random_output_len}"
    )
    if max_concurrency:
        benchmark_cmd += f" --max-concurrency {max_concurrency}"

    docker_cmd = (
        f"docker run --rm -v /dev/shm:/dev/shm --shm-size 10gb "
        f"-e HF_TOKEN=$(docker exec gemma4 env | grep HF_TOKEN | cut -d= -f2 || echo '') "
        f"{LOCAL_DOCKER_IMAGE} {benchmark_cmd}"
    )
    process = await asyncio.create_subprocess_shell(
        docker_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
        rc = process.returncode or 0
        if rc != 0:
            return f"⚠️ Local benchmark failed.\nError: {stderr.decode()}\nOutput: {stdout.decode()}"
        return f"✅ Local benchmark completed:\n{stdout.decode()}"
    except Exception as e:
        return f"❌ Local benchmark execution error: {e}"


@mcp.tool()
async def get_docker_logs(tail: Optional[int] = None) -> str:
    """Retrieves logs from the local vLLM/Ollama Docker container."""
    log_cmd = "docker logs gemma4"
    if tail:
        log_cmd += f" --tail {tail}"
    rc, out, err = await run_command(shlex.split(log_cmd))
    if rc != 0:
        return f"⚠️ Failed to get local Docker logs.\nError: {err}"
    return f"✅ Local Docker logs:\n{out}"


@mcp.tool()
async def analyze_local_logs(limit: int = 15) -> str:
    """Fetches local Docker container logs and uses Gemma 4 to analyze them for errors."""
    logs_result = await get_docker_logs(tail=limit)

    prompt = (
        f"You are a DevOps and SRE expert. Analyze the following local vLLM/Ollama container logs and summarize any errors, "
        f"their potential root causes, and recommended remediation steps:\n\n"
        f"{logs_result}\n\n"
        f"Provide a concise summary."
    )

    try:
        analysis = await query_gemma4(prompt)
        return analysis
    except Exception as e:
        return f"❌ Analysis failed because model is unreachable: {e}"


@mcp.tool()
async def get_system_details() -> str:
    """Retrieves detailed information about the running local model, engine, and versions."""
    report = f"### 🧩 Model Details ({VLLM_URL})\n\n"

    async with httpx.AsyncClient(timeout=10) as client:
        # 1. Get Model Details from /v1/models
        try:
            models_res = await client.get(f"{VLLM_URL}/v1/models")
            if models_res.status_code == 200:
                models_data = models_res.json()
                report += "**Model Information (`/v1/models`):**\n"
                report += f"```json\n{json.dumps(models_data, indent=2)}\n```\n"
            else:
                report += f"⚠️ Could not fetch model details. Status: {models_res.status_code}\n\n"
        except Exception as e:
            report += f"❌ Error fetching model details: {e}\n\n"

        # 2. Get Health Status
        is_up, endpoint_type = await check_service_health(client)
        if is_up:
            report += f"**Health Status (`/{endpoint_type}`):**\n- Status: `Healthy` ✅\n\n"
        else:
            report += "**Health Status:**\n- Status: `Unhealthy` ❌\n\n"

    return report


@mcp.tool()
async def get_help() -> str:
    """Provides help text and summarizes the configuration options and all available SRE/DevOps tools for this local MCP server."""
    return (
        "### 🛠️ Local Gemma 4 SRE Agent Help & Configuration\n\n"
        "You can configure this MCP server using the following environment variables:\n\n"
        f"- **`MODEL_NAME`**: Hugging Face model ID for local Gemma 4 mapping.\n"
        f"  - *Current Value:* `{MODEL_NAME}` (runs as `{get_current_model_name()}` locally)\n"
        f"- **`LOCAL_DOCKER_IMAGE`**: Local Docker image name (e.g. `ollama/ollama:latest`).\n"
        f"  - *Current Value:* `{LOCAL_DOCKER_IMAGE}`\n"
        f"- **`LOCAL_VLLM_PORT`**: Port number for local vLLM/Ollama API server.\n"
        f"  - *Current Value:* `{LOCAL_VLLM_PORT}`\n\n"
        "### ℹ️ Active Mode Summary\n"
        "The server is running in **LOCAL** mode.\n\n"
        "---\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "Below is a summary of the tools exposed by this SRE/DevOps agent:\n\n"
        "#### 🐳 Deployment & Configuration\n"
        "- **`manage_docker`**: Manages the local container (actions: `start`, `stop`, `restart`, `status`, `log`, `rm`).\n"
        "- **`save_hf_token`**: Securely saves a Hugging Face API token locally in environment variables and cache.\n\n"
        "#### 📊 Monitoring & Status\n"
        "- **`get_metrics`**: Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.\n"
        "- **`get_system_status`**: Provides a high-level status dashboard of the local Docker container and vLLM service.\n"
        "- **`get_endpoint`**: Verifies connectivity and returns the active local vLLM service URL.\n"
        "- **`get_active_models`**: Gets the active resource usage (actively loaded models, sizes, CPU/GPU status, context size) via ollama ps.\n"
        "- **`get_model_show_details`**: Gets deep model parameters, architecture, license, and config details via ollama show.\n"
        "- **`get_help`**: Provides this help text and summarizes configuration/tools.\n\n"
        "#### 📈 Performance & Benchmarking\n"
        "- **`run_benchmark`**: Runs vLLM's internal serving benchmark tool inside the local container.\n"
        "- **`get_docker_logs`**: Retrieves startup and execution logs from the local Docker container.\n"
        "- **`analyze_local_logs`**: Fetches the local container logs and uses Gemma 4 to analyze them for SRE/DevOps errors.\n\n"
        "#### 💬 Interaction & Diagnostics\n"
        "- **`query_gemma4`**: Primary tool to query the self-hosted local model.\n"
        "- **`query_gemma4_with_stats`**: Queries the local model and provides streaming-based performance metrics (TTFT, throughput, latency).\n"
        "- **`verify_model_health`**: Performs a deep health check by querying the model with a simple prompt and measuring response latency.\n"
        "- **`get_system_details`**: Retrieves detailed information about the running local model, engine, and versions.\n"
    )


@mcp.tool()
async def get_active_models() -> str:
    """Gets the active resource usage (actively loaded models, sizes, CPU/GPU status, context size) via ollama ps."""
    if "ollama" not in LOCAL_DOCKER_IMAGE.lower():
        return "❌ Active resource usage (ollama ps) is only supported on Ollama backend."

    cmd = ["docker", "exec", "gemma4", "ollama", "ps"]
    rc, out, err = await run_command(cmd, timeout=30)
    if rc != 0:
        return f"⚠️ Failed to check active models.\nError: {err}\nOutput: {out}"
    return f"### 📊 Active Loaded Models:\n\n```\n{out}\n```"


@mcp.tool()
async def get_model_show_details(model_name: str) -> str:
    """Gets deep model parameters, architecture, license, and config details via ollama show <model_name>."""
    if "ollama" not in LOCAL_DOCKER_IMAGE.lower():
        return "❌ Deep model details (ollama show) are only supported on Ollama backend."

    cmd = ["docker", "exec", "gemma4", "ollama", "show", model_name]
    rc, out, err = await run_command(cmd, timeout=30)
    if rc != 0:
        return f"⚠️ Failed to get model details for {model_name}.\nError: {err}\nOutput: {out}"
    return f"### 🧩 Model Details for `{model_name}`:\n\n```\n{out}\n```"


@mcp.tool()
async def get_metrics() -> str:
    """
    Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{VLLM_URL}/metrics")
            if res.status_code == 200:
                return res.text
            else:
                return f"❌ Failed to fetch metrics. Status code: {res.status_code}\nResponse: {res.text}"
    except Exception as e:
        return f"❌ Error connecting to vLLM metrics endpoint: {e}"


if __name__ == "__main__":
    mcp.run()
