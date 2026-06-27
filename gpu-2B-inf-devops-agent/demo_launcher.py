# Grand Demo: Self-Hosted vLLM DevOps Agent

import asyncio

from server import (
    analyze_cloud_logging,
    get_deployment_template,
    get_huggingface_model_copy_instructions,
    get_vllm_deployment_config,
    suggest_sre_remediation,
)


async def devops_demo():
    print("🚀 AWS Inferentia Demo: Self-Hosted vLLM DevOps Agent")
    print("=" * 60)

    # Step 1: Log Analysis
    print("\n[Step 1] Analyzing Cloud Logging errors (severity=ERROR)...")
    # Simulate a call where some logs are found
    analysis = await analyze_cloud_logging(filter_query="severity=ERROR", limit=2)
    print(f"  ANALYSIS: {analysis[:200]}...")  # Truncate for display

    # Step 2: SRE Remediation
    print("\n[Step 2] Proposing remediation for 'MemoryLimitExceeded'...")
    remediation = await suggest_sre_remediation(error_message="Pod 'vllm-gemma' terminated with Reason: OOMKilled")
    print(f"  REMEDIATION: {remediation[:200]}...")

    # Step 3: Deployment Config & Hugging Face instructions
    print("\n[Step 3] Hugging Face Model Copy Instructions...")
    instructions = get_huggingface_model_copy_instructions(repo_id="google/gemma-4-E2B-it")
    print(instructions[:300] + "...")

    print("\n[Step 4] Generating EC2 AWS Inferentia Deployment Config...")
    config = get_vllm_deployment_config(
        service_name="inferentia-2b-devops-agent",
        model_path="google/gemma-4-E2B-it",
    )
    print(f"  COMMAND: {config[:300]}...")

    # Step 5: MCP Resources
    print("\n[Step 5] Reading MCP Resource (vLLM Deployment Template)...")
    template = get_deployment_template()
    print(f"  TEMPLATE (first 100 chars): {template.strip()[:100]}...")

    print("\n" + "=" * 60)
    print("✅ DevOps Agent Demo Complete: Self-hosted SRE intelligence ready!")


if __name__ == "__main__":
    asyncio.run(devops_demo())
