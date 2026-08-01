import asyncio
import os
import sys

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import server

async def main():
    # Load AWS credentials if present locally
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v
    for k, v in creds.items():
        os.environ[k] = v

    # 1. Try SPOT deployment
    print("Attempting to deploy inf2.xlarge as SPOT instance...")
    try:
        res = await server.deploy_vllm(
            service_name="inferentia-12b-devops-agent",
            model_path="google/gemma-4-12B-it",
            key_name="alinux",
            instance_type="inf2.xlarge",
            market_type="spot"
        )
        print(res)
        if "Failed to deploy" in res:
            raise Exception("Deployment returned failure message.")
        print("\n🎉 Spot deployment request succeeded!")
        return
    except Exception as e:
        print(f"\n❌ Spot deployment failed or returned error: {e}")
        print("Falling back to ON-DEMAND deployment...")

    # 2. Try ON-DEMAND deployment (Fallback)
    try:
        res = await server.deploy_vllm(
            service_name="inferentia-12b-devops-agent",
            model_path="google/gemma-4-12B-it",
            key_name="alinux",
            instance_type="inf2.xlarge",
            market_type="on-demand"
        )
        print(res)
        if "Failed to deploy" in res:
            sys.exit(1)
        print("\n🎉 On-Demand deployment request succeeded!")
    except Exception as e:
        print(f"\n❌ On-Demand deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
