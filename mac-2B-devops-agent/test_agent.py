import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Mocking FastMCP and other dependencies before importing server
mock_mcp = MagicMock()
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()
sys.modules["mcp.server.fastmcp"].FastMCP = MagicMock(return_value=mock_mcp)


# Mock decorative tools
def mock_decorator(*args, **kwargs):
    def wrapper(func):
        return func

    return wrapper


mock_mcp.tool = mock_decorator
mock_mcp.resource = mock_decorator

from server import (  # noqa: E402
    MODEL_NAME,
    get_help,
    get_system_details,
    query_gemma4,
    query_gemma4_with_stats,
    save_hf_token,
    verify_model_health,
)


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    def test_model_name_default(self):
        """Verify the default model is Gemma 4."""
        self.assertEqual(MODEL_NAME, "gemma4:e2b")

    @patch("server.get_vllm_client", new_callable=AsyncMock)
    async def test_verify_model_health_success(self, mock_get_client):
        """Test successful model health check."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "READY"

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await verify_model_health()
        self.assertIn("✅ Model health check PASSED.", result)
        self.assertIn("READY", result)

    @patch("server.get_vllm_client", new_callable=AsyncMock)
    async def test_verify_model_health_empty_choices(self, mock_get_client):
        """Test model health check when choices list is empty."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = []

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await verify_model_health()
        self.assertIn("❌ Model health check FAILED: No choices returned.", result)

    @patch("server.get_vllm_client", new_callable=AsyncMock)
    async def test_query_gemma4_empty_choices(self, mock_get_client):
        """Test query_gemma4 when choices list is empty."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = []

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await query_gemma4("Hello")
        self.assertIn("❌ Query failed: No choices returned from model.", result)

    @patch("server.get_vllm_client", new_callable=AsyncMock)
    async def test_query_gemma4_with_stats_success(self, mock_get_client):
        """Test query with performance metrics."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_stream = AsyncMock()

        # Create chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        mock_stream.__aiter__.return_value = [chunk1, chunk2]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        result = await query_gemma4_with_stats("Hi")
        self.assertIn("Hello world", result)
        self.assertIn("Performance Stats", result)
        self.assertIn("TTFT", result)

    @patch("server.VLLM_URL", "http://test-url:8000")
    @patch("httpx.AsyncClient", autospec=True)
    async def test_get_system_details_success(self, mock_async_client):
        """Test retrieving model stats."""

        # Mock httpx.AsyncClient and its get method
        mock_client_instance = mock_async_client.return_value.__aenter__.return_value

        mock_models_response = MagicMock()
        mock_models_response.json.return_value = {"data": [{"id": "test-model", "max_model_len": 4096}]}
        mock_models_response.status_code = 200

        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_client_instance.get.side_effect = [
            mock_models_response,
            mock_health_response,
        ]

        result = await get_system_details()
        self.assertIn("### 🧩 Model Details", result)
        self.assertIn("test-model", result)
        self.assertIn("Healthy", result)

    @patch("os.makedirs")
    @patch("builtins.open", create=True)
    async def test_save_hf_token(self, mock_open, mock_makedirs):
        """Test saving HF token locally."""
        result = await save_hf_token("test-token")
        self.assertIn("✅ Token saved locally", result)
        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()

    async def test_get_help(self):
        """Test retrieving the help summary of options."""
        result = await get_help()
        self.assertIn("Local Gemma 4 SRE Agent Help & Configuration", result)
        self.assertIn("MODEL_NAME", result)
        self.assertIn("LOCAL_DOCKER_IMAGE", result)

    @patch("server.run_command", new_callable=AsyncMock)
    @patch("os.path.exists", return_value=True)
    async def test_run_benchmark_ollama(self, mock_exists, mock_run_command):
        """Test run_benchmark falls back to local script when Ollama is running."""
        import server
        from server import run_benchmark

        original_image = server.LOCAL_DOCKER_IMAGE
        server.LOCAL_DOCKER_IMAGE = "ollama/ollama:latest"
        try:
            mock_run_command.return_value = (0, "Mock benchmark output", "")

            result = await run_benchmark(num_prompts=5, random_output_len=10)
            self.assertIn("✅ Local benchmark completed", result)
            self.assertIn("Mock benchmark output", result)
            mock_run_command.assert_called_once()
            # Verify the args passed to run_command
            called_args = mock_run_command.call_args[0][0]
            self.assertIn("benchmarking_suite.py", called_args[1])
            self.assertIn("--requests", called_args)
            self.assertIn("5", called_args)
            self.assertIn("--tokens", called_args)
            self.assertIn("10", called_args)
        finally:
            server.LOCAL_DOCKER_IMAGE = original_image

    @patch("server.run_command", new_callable=AsyncMock)
    @patch("os.path.exists", return_value=True)
    async def test_run_benchmark_ollama_with_concurrency(self, mock_exists, mock_run_command):
        """Test run_benchmark passes --concurrencies when max_concurrency is specified."""
        import server
        from server import run_benchmark

        original_image = server.LOCAL_DOCKER_IMAGE
        server.LOCAL_DOCKER_IMAGE = "ollama/ollama:latest"
        try:
            mock_run_command.return_value = (0, "Mock benchmark output", "")

            result = await run_benchmark(num_prompts=5, random_output_len=10, max_concurrency=4)
            self.assertIn("✅ Local benchmark completed", result)
            mock_run_command.assert_called_once()
            # Verify the args passed to run_command
            called_args = mock_run_command.call_args[0][0]
            self.assertIn("--concurrencies", called_args)
            self.assertIn("1,2,4", called_args)
        finally:
            server.LOCAL_DOCKER_IMAGE = original_image

    @patch("asyncio.create_subprocess_shell", new_callable=AsyncMock)
    async def test_run_benchmark_vllm(self, mock_subprocess):
        """Test run_benchmark uses docker run with vllm bench serve for vLLM images."""
        import server
        from server import run_benchmark

        original_image = server.LOCAL_DOCKER_IMAGE
        server.LOCAL_DOCKER_IMAGE = "vllm/vllm-openai:latest"
        try:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"vllm output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await run_benchmark(num_prompts=5, random_output_len=10)
            self.assertIn("✅ Local benchmark completed", result)
            self.assertIn("vllm output", result)
            mock_subprocess.assert_called_once()
            called_cmd = mock_subprocess.call_args[0][0]
            self.assertIn("vllm bench serve", called_cmd)
        finally:
            server.LOCAL_DOCKER_IMAGE = original_image

    @patch("server.is_docker_available", new_callable=AsyncMock, return_value=True)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_get_active_models_ollama(self, mock_run_command, mock_is_docker):
        """Test get_active_models runs docker exec ollama ps on Ollama backend."""
        import server
        from server import get_active_models

        original_image = server.LOCAL_DOCKER_IMAGE
        server.LOCAL_DOCKER_IMAGE = "ollama/ollama:latest"
        try:
            mock_run_command.return_value = (0, "gemma4:e2b   7.7 GB   100% CPU", "")
            result = await get_active_models()
            self.assertIn("gemma4:e2b", result)
            mock_run_command.assert_called_once()
            called_cmd = mock_run_command.call_args[0][0]
            self.assertEqual(called_cmd, ["docker", "exec", "gemma4", "ollama", "ps"])
        finally:
            server.LOCAL_DOCKER_IMAGE = original_image

    async def test_get_active_models_vllm(self):
        """Test get_active_models returns not supported on vLLM backend."""
        import server
        from server import get_active_models

        original_image = server.LOCAL_DOCKER_IMAGE
        server.LOCAL_DOCKER_IMAGE = "vllm/vllm-openai:latest"
        try:
            result = await get_active_models()
            self.assertIn("only supported on Ollama backend", result)
        finally:
            server.LOCAL_DOCKER_IMAGE = original_image

    @patch("server.is_docker_available", new_callable=AsyncMock, return_value=True)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_get_model_show_details_ollama(self, mock_run_command, mock_is_docker):
        """Test get_model_show_details runs docker exec ollama show on Ollama backend."""
        import server
        from server import get_model_show_details

        original_image = server.LOCAL_DOCKER_IMAGE
        server.LOCAL_DOCKER_IMAGE = "ollama/ollama:latest"
        try:
            mock_run_command.return_value = (0, "architecture gemma4\nparameters 5.1B", "")
            result = await get_model_show_details("gemma4:e2b")
            self.assertIn("gemma4:e2b", result)
            self.assertIn("architecture gemma4", result)
            mock_run_command.assert_called_once()
            called_cmd = mock_run_command.call_args[0][0]
            self.assertEqual(called_cmd, ["docker", "exec", "gemma4", "ollama", "show", "gemma4:e2b"])
        finally:
            server.LOCAL_DOCKER_IMAGE = original_image

    async def test_get_model_show_details_vllm(self):
        """Test get_model_show_details returns not supported on vLLM backend."""
        import server
        from server import get_model_show_details

        original_image = server.LOCAL_DOCKER_IMAGE
        server.LOCAL_DOCKER_IMAGE = "vllm/vllm-openai:latest"
        try:
            result = await get_model_show_details("gemma4:e2b")
            self.assertIn("only supported on Ollama backend", result)
        finally:
            server.LOCAL_DOCKER_IMAGE = original_image


if __name__ == "__main__":
    unittest.main()
