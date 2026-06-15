import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# --- Example Agent using a Gemma 4 model hosted on a vLLM endpoint ---

# Endpoint URL provided by your local deployment
api_base_url = "http://localhost:8000/v1"

# Model name as recognized by *your* local endpoint configuration
model_name_at_endpoint = os.getenv("MODEL_NAME", "openai/gemma4:e4b")

# Authentication (Example: using gcloud identity token for a Cloud Run deployment)
# Adapt this based on your endpoint's security
auth_headers = None  # Authentication handled externally or not required for this setup

root_agent = LlmAgent(
    model=LiteLlm(
        model=model_name_at_endpoint,
        api_base=api_base_url,
        # This extra_body values specific to Gemma 4.
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True  # Enable thinking
            },
            "skip_special_tokens": False,  # Should be set to False
        },
        # Pass authentication headers if needed
        extra_headers=auth_headers,
        # Alternatively, if endpoint uses an API key:
        api_key="none",
    ),
    name="vllm_agent",
    instruction="You are a helpful assistant running on a self-hosted vLLM endpoint.",
    # ... other agent parameters
)
