import asyncio
import os
import sys
import boto3

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import server

async def try_deployment(region, instance_type, market_type):
    print(f"\n⚡ Attempting deployment: Region={region}, Type={instance_type}, Market={market_type.upper()}")
    # Dynamically configure region settings
    os.environ["AWS_DEFAULT_REGION"] = region
    os.environ["AWS_REGION"] = region
    server.AWS_REGION = region
    boto3.setup_default_session(region_name=region)
    
    try:
        res = await server.deploy_vllm(
            service_name="inferentia-2b-devops-agent",
            model_path="google/gemma-4-E2B-it",
            key_name="alinux",
            instance_type=instance_type,
            market_type=market_type
        )
        print(res)
        if "Failed to deploy" not in res:
            return True, res
    except Exception as e:
        print(f"❌ Execution failed for {region}/{instance_type}/{market_type}: {e}")
    return False, None

async def main():
    # Load local AWS credentials if present
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v
    for k, v in creds.items():
        os.environ[k] = v

    regions = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]
    instance_types = ["inf2.xlarge", "inf2.8xlarge"]

    # Cascade 1: SPOT Instances across all regions first
    print("=== STARTING SPOT CASCADE DEPLOYMENT ===")
    for region in regions:
        for inst_type in instance_types:
            success, msg = await try_deployment(region, inst_type, "spot")
            if success:
                print(f"\n🎉 SUCCESS: Deployed Spot instance in {region}!")
                # Save deployment info
                with open("active_deployment_region.txt", "w") as f:
                    f.write(region)
                return

    # Cascade 2: ON-DEMAND Instances across all regions as fallback
    print("\n=== SPOT DEPLOYMENT FAILED EVERYWHERE, STARTING ON-DEMAND CASCADE ===")
    for region in regions:
        for inst_type in instance_types:
            success, msg = await try_deployment(region, inst_type, "on-demand")
            if success:
                print(f"\n🎉 SUCCESS: Deployed On-Demand instance in {region}!")
                # Save deployment info
                with open("active_deployment_region.txt", "w") as f:
                    f.write(region)
                return

    print("\n❌ Fatal: All Spot and On-Demand deployments failed across all regions.")
    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
