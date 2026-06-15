# Grand Demo: Local vLLM DevOps Agent (Gemma 4)

import asyncio

from server import (
    analyze_local_logs,
    get_endpoint,
)


async def devops_demo():
    print("🚀 Local Sprint Demo: Self-Hosted vLLM DevOps Agent (Gemma 4)")
    print("=" * 60)

    # Step 1: Discovery
    print("\n[Step 1] Discovering Local vLLM Endpoint...")
    try:
        endpoint = await get_endpoint()
        print(f"  FOUND: {endpoint}")
    except Exception as e:
        print(f"  NOTICE: {e} (Discovery failed, using simulation mode)")

    # Step 2: Log Analysis
    print("\n[Step 2] Analyzing local container logs...")
    # Queries the self-hosted Gemma 4
    analysis = await analyze_local_logs(limit=15)
    print(f"  ANALYSIS:\n{analysis}")

    print("\n" + "=" * 60)
    print("✅ DevOps Agent Demo Complete: Gemma 4 ready!")


if __name__ == "__main__":
    asyncio.run(devops_demo())
