import asyncio
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from typing import Optional

import boto3
import httpx
from google.cloud import aiplatform, secretmanager, storage
from google.cloud import logging as cloud_logging
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

# Setup logging to stderr ONLY to avoid interfering with MCP stdio communication
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm-devops-agent")
logger.info("Initializing DevOps Agent MCP Server...")

# Initialize FastMCP server
mcp = FastMCP("Self-Hosted vLLM DevOps Agent")

# Load AWS credentials if .aws_creds exists
aws_creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".aws_creds")
if os.path.exists(aws_creds_path):
    with open(aws_creds_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east4")
BUCKET_NAME = f"{PROJECT_ID}-bucket"

# AWS Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "vllm-models-bucket")

# The URL of the self-hosted vLLM service on Cloud Run or AWS EC2
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-12B-it-qat-w4a16-ct")
HF_SECRET_ID = "hf-token"


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from AWS Secrets Manager, GCP Secret Manager, or environment variables."""
    # 1. Check environment variable
    val = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if val:
        return val

    # 2. Check AWS Secrets Manager
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        response = client.get_secret_value(SecretId=secret_id)
        if "SecretString" in response:
            return response["SecretString"]
    except Exception as e:
        logger.debug(f"AWS Secrets Manager failed: {e}")

    # 3. Fallback to GCP Secret Manager
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = await asyncio.to_thread(client.access_secret_version, request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.debug(f"GCP Secret Manager failed: {e}")

    return None


@mcp.tool()
async def save_hf_token(token: str) -> str:
    """Securely saves a Hugging Face API token to AWS Secrets Manager or GCP Secret Manager."""
    saved_aws = False
    saved_gcp = False

    try:
        import boto3
        from botocore.exceptions import ClientError

        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        try:
            client.create_secret(Name=HF_SECRET_ID, SecretString=token)
            saved_aws = True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceExistsException":
                client.put_secret_value(SecretId=HF_SECRET_ID, SecretString=token)
                saved_aws = True
            else:
                logger.warning(f"AWS Secrets Manager create/update failed: {e}")
    except Exception as e:
        logger.warning(f"AWS Secrets Manager connection/call failed: {e}")

    try:
        client = secretmanager.SecretManagerServiceClient()
        secret_parent = f"projects/{PROJECT_ID}/secrets/{HF_SECRET_ID}"
        try:
            await asyncio.to_thread(client.get_secret, request={"name": secret_parent})
        except Exception:
            await asyncio.to_thread(
                client.create_secret,
                request={
                    "parent": f"projects/{PROJECT_ID}",
                    "secret_id": HF_SECRET_ID,
                    "secret": {"replication": {"automatic": {}}},
                },
            )
        await asyncio.to_thread(
            client.add_secret_version,
            request={"parent": secret_parent, "payload": {"data": token.encode("UTF-8")}},
        )
        saved_gcp = True
    except Exception as e:
        logger.warning(f"GCP Secret Manager failed: {e}")

    if saved_aws and saved_gcp:
        return "✅ Token saved to both AWS Secrets Manager and GCP Secret Manager."
    elif saved_aws:
        return "✅ Token saved to AWS Secrets Manager."
    elif saved_gcp:
        return "✅ Token saved to GCP Secret Manager."
    else:
        return "❌ Failed to save token to Secret Manager (both AWS and GCP failed)."


DEFAULT_SERVICE_NAME = "gpu-12b-qat-l4-devops-agent"


def discover_vllm_url(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """Attempts to automatically discover the AWS EC2 instance public IP/DNS or Cloud Run service URL."""
    if VLLM_BASE_URL:
        logger.info(f"Using provided VLLM_BASE_URL: {VLLM_BASE_URL}")
        return VLLM_BASE_URL

    # 1. AWS EC2 Discovery
    if os.getenv("AWS_ACCESS_KEY_ID"):
        logger.info(f"Attempting to discover AWS EC2 vLLM URL for: {service_name}")
        try:
            import boto3

            ec2 = boto3.client("ec2", region_name=AWS_REGION)
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [service_name]},
                    {"Name": "instance-state-name", "Values": ["running"]},
                ]
            )
            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    ip = instance.get("PublicIpAddress")
                    if ip:
                        url = f"http://{ip}:8080"
                        logger.info(f"📡 Automatically discovered AWS vLLM at: {url}")
                        return url
        except Exception as e:
            logger.warning(f"⚠️ Error during AWS vLLM discovery: {str(e)}")

    # 2. GCP Cloud Run Discovery
    logger.info(f"Attempting to discover GCP Cloud Run URL for service: {service_name}")
    try:
        cmd = [
            "gcloud",
            "run",
            "services",
            "describe",
            service_name,
            f"--project={PROJECT_ID}",
            "--region",
            LOCATION,
            "--format",
            "value(status.url)",
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if process.returncode == 0:
            url = process.stdout.strip()
            if url:
                logger.info(f"📡 Automatically discovered GCP vLLM at: {url}")
                return url
            else:
                logger.warning("⚠️ gcloud returned empty URL for service.")
        else:
            logger.warning(
                f"⚠️ gcloud failed to discover Cloud Run URL (code {process.returncode}): {process.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ Discovery timed out after 15 seconds.")
    except Exception as e:
        logger.warning(f"⚠️ Error during vLLM discovery: {str(e)}")

    logger.error("❌ Failed to discover service URL.")
    return None


# Resolve base URL at runtime
_ACTIVE_VLLM_URL = None


def get_vllm_url() -> str:
    """Returns the cached vLLM URL or discovers it if needed."""
    global _ACTIVE_VLLM_URL
    if not _ACTIVE_VLLM_URL:
        _ACTIVE_VLLM_URL = discover_vllm_url()

    if not _ACTIVE_VLLM_URL:
        raise Exception(
            "Could not determine vLLM service URL. Ensure you are authenticated and the service/instance exists."
        )

    return _ACTIVE_VLLM_URL


def get_auth_token() -> str:
    """For AWS, returns empty string. For GCP, gets Google Cloud Identity Token."""
    if os.getenv("AWS_ACCESS_KEY_ID"):
        return ""
    try:
        return (
            subprocess.check_output(
                ["gcloud", "auth", "print-identity-token"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        logger.warning(f"⚠️ Could not obtain identity token: {str(e)}")
        return ""


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the Cloud Run vLLM service."""
    vllm_url = get_vllm_url()
    token = get_auth_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return AsyncOpenAI(
        base_url=f"{vllm_url}/v1",
        api_key=token or "not-needed",
        default_headers=headers,
    )


async def get_active_model_name(client: AsyncOpenAI) -> str:
    """Queries the vLLM endpoint to find the active model name, or falls back to configuration."""
    try:
        models_response = await client.models.list()
        if models_response.data:
            return models_response.data[0].id
    except Exception as e:
        logger.warning(f"⚠️ Failed to dynamically query active model from vLLM: {e}")

    # Fallback
    if "/" not in MODEL_NAME:
        return f"/mnt/models/{MODEL_NAME}"
    return MODEL_NAME


# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)


@mcp.resource("config://vllm-deployment-template")
def get_deployment_template() -> str:
    """Returns a base template for AWS EC2 L4 GPU vLLM deployment."""
    return """
# AWS EC2 vLLM Deployment Template
# Required Instance: g6.2xlarge (1x NVIDIA L4 GPU, 24GB VRAM)
# Recommended AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)

InstanceType: g6.2xlarge
ImageId: ami-012ba162b9cd2729c (us-east-1)
Ports:
  - Container Port: 8080
  - Host Port: 8080

Docker Run Command:
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN=$HF_TOKEN \\
  vllm/vllm-openai:nightly \\
  --model google/gemma-4-12B-it-qat-w4a16-ct \\
  --quantization compressed-tensors \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization 0.95 \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{}' \\
  --host 0.0.0.0 \\
  --port 8080
"""


@mcp.tool()
def get_vllm_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """
    Returns the current active vLLM endpoint URL.

    Args:
        service_name: The service name or instance Name tag to describe (defaults to 'gpu-12b-qat-l4-devops-agent').
    """
    if service_name == DEFAULT_SERVICE_NAME:
        return get_vllm_url()
    return discover_vllm_url(service_name)


@mcp.tool()
def list_vertex_models() -> str:
    """
    Uses the Vertex AI SDK (part of ADK ecosystem) to list models in the project registry.
    """
    try:
        models = aiplatform.Model.list()
        if not models:
            return "No models found in Vertex AI Model Registry."

        model_list = [f"- {m.display_name} (ID: {m.name})" for m in models]
        return "### Vertex AI Model Registry\n" + "\n".join(model_list)
    except Exception as e:
        return f"Error listing models from Vertex AI: {str(e)}"


@mcp.tool()
def list_bucket_models(bucket_name: Optional[str] = None) -> str:
    """
    Lists the contents of an S3 bucket or GCS bucket to check for uploaded model files.

    Args:
        bucket_name: The S3 or GCS bucket name. Defaults to AWS_BUCKET_NAME or BUCKET_NAME depending on provider.
    """
    if not bucket_name:
        bucket_name = os.getenv("AWS_BUCKET_NAME", "vllm-models-bucket")

    # If it starts with s3:// or AWS credentials exist, try S3
    is_s3 = bucket_name.startswith("s3://") or bool(os.getenv("AWS_ACCESS_KEY_ID"))
    clean_bucket = bucket_name.replace("s3://", "").replace("gs://", "")

    if is_s3:
        try:
            import boto3

            s3 = boto3.client("s3", region_name=AWS_REGION)
            response = s3.list_objects_v2(Bucket=clean_bucket, MaxKeys=100)
            contents = response.get("Contents", [])
            if not contents:
                return f"The S3 bucket '{clean_bucket}' is empty or does not exist."

            file_list = [
                f"- s3://{clean_bucket}/{obj['Key']} ({obj['Size'] / 1024 / 1024:.2f} MB)" for obj in contents[:50]
            ]
            summary = f"### Contents of S3 Bucket: {clean_bucket}\n"
            summary += "\n".join(file_list)

            if len(contents) > 50:
                summary += f"\n\n(Showing 50 of {len(contents)} items)"

            return summary
        except Exception as e:
            logger.warning(f"Error listing S3 bucket '{clean_bucket}': {e}")
            # Fall through to GCS if S3 failed

    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(clean_bucket)
        blobs = list(bucket.list_blobs(max_results=100))

        if not blobs:
            return f"The GCS bucket '{clean_bucket}' is empty."

        file_list = [f"- gs://{clean_bucket}/{b.name} ({b.size / 1024 / 1024:.2f} MB)" for b in blobs[:50]]
        summary = f"### Contents of GCS Bucket: {clean_bucket}\n"
        summary += "\n".join(file_list)

        if len(blobs) > 50:
            summary += f"\n\n(Showing 50 of {len(blobs)} items)"

        return summary
    except Exception as e:
        return f"Error listing objects in bucket '{clean_bucket}' (tried S3 and GCS): {str(e)}"


@mcp.tool()
async def analyze_cloud_logging(filter_query: str, limit: int = 5) -> str:
    """
    Fetches and summarizes error logs from AWS CloudWatch or Google Cloud Logging.

    Args:
        filter_query: Query filter. For CloudWatch, this is the log group name. For GCS, a standard log query.
        limit: Number of recent logs to fetch.
    """
    combined_logs = ""

    # 1. Try AWS CloudWatch if AWS credentials exist
    if os.getenv("AWS_ACCESS_KEY_ID"):
        try:
            import boto3

            logs_client = boto3.client("logs", region_name=AWS_REGION)
            log_group_name = filter_query

            try:
                groups = logs_client.describe_log_groups(logGroupNamePrefix=log_group_name)
                group_list = groups.get("logGroups", [])
                if group_list:
                    log_group_name = group_list[0]["logGroupName"]
            except Exception:
                pass

            streams = logs_client.describe_log_streams(
                logGroupName=log_group_name, orderBy="LastEventTime", descending=True, limit=1
            )
            stream_list = streams.get("logStreams", [])
            if stream_list:
                stream_name = stream_list[0]["logStreamName"]
                events = logs_client.get_log_events(logGroupName=log_group_name, logStreamName=stream_name, limit=limit)
                log_texts = [
                    f"Timestamp: {ev.get('timestamp')} | Message: {ev.get('message')}"
                    for ev in events.get("events", [])
                ]
                combined_logs = "\n---\n".join(log_texts)
        except Exception as e:
            logger.warning(f"Failed to fetch AWS CloudWatch logs: {e}")

    # 2. GCP Fallback if no logs found
    if not combined_logs:
        try:
            logging_client = cloud_logging.Client(project=PROJECT_ID)
            entries = list(
                logging_client.list_entries(filter_=filter_query, order_by=cloud_logging.DESCENDING, page_size=limit)
            )
            if entries:
                log_texts = [
                    f"Timestamp: {e.timestamp} | Severity: {e.severity} | Message: {str(e.payload)[:1000] if isinstance(e.payload, str) else json.dumps(e.payload)[:1000]}"
                    for e in entries
                ]
                combined_logs = "\n---\n".join(log_texts)
        except Exception as e:
            logger.warning(f"Failed to fetch Google Cloud logs: {e}")

    if not combined_logs:
        return "No matching logs found."

    try:
        if len(combined_logs) > 12000:
            combined_logs = combined_logs[:12000] + "... (truncated)"

        prompt = f"Analyze the following DevOps logs and provide a high-level summary of the critical issues and potential root causes:\n\n{combined_logs}\n\nSummary:"

        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### Log Analysis (Self-Hosted vLLM)\n\n{response_text}"

    except Exception as e:
        return f"Error analyzing logs via self-hosted vLLM: {str(e)}"


@mcp.tool()
async def suggest_sre_remediation(error_message: str) -> str:
    """
    Proposes remediation steps for a specific SRE incident using self-hosted vLLM.

    Args:
        error_message: The error or incident description to remediate.
    """
    prompt = f"As an expert SRE, suggest a 3-step remediation plan for the following error:\n\nError: {error_message}\n\nRemediation Plan:"

    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### Remediation Plan\n\n{response_text}"
    except Exception as e:
        return f"Error fetching remediation plan: {str(e)}"


@mcp.tool()
async def query_vllm(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Directly queries the self-hosted vLLM model with a custom prompt.

    Args:
        prompt: The text prompt to send to the model.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for deterministic).
    """
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### vLLM Response\n\n{response_text}"
    except Exception as e:
        return f"Error querying vLLM: {str(e)}"


@mcp.tool()
def get_vllm_deployment_config(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-12B-it-qat-w4a16-ct",
    key_name: str = "alinux",
    gpu_memory_utilization: float = 0.95,
) -> str:
    """
    Generates the AWS CLI command and UserData script to deploy vLLM to an AWS EC2 g6.2xlarge instance (NVIDIA L4).

    Args:
        service_name: The Name tag for the EC2 instance.
        model_path: Hugging Face repo ID or S3 URI of the model.
        key_name: Key Pair name for SSH access (default: 'alinux').
        gpu_memory_utilization: The fraction of GPU VRAM to use for KV cache (default: 0.95).
    """
    quant_arg = (
        "--quantization compressed-tensors" if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]) else ""
    )

    user_data = f"""#!/bin/bash
if ! command -v docker &> /dev/null; then
    apt-get update -y
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
fi
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN="$(aws ssm get-parameter --name /vllm/HF_TOKEN --with-decryption --query Parameter.Value --output text 2>/dev/null || echo '')" \\
  vllm/vllm-openai:nightly \\
  --model {model_path} \\
  {quant_arg} \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization {gpu_memory_utilization} \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{{}}' \\
  --host 0.0.0.0 \\
  --port 8080
"""

    aws_cmd = (
        f"aws ec2 run-instances \\\n"
        f"  --image-id ami-012ba162b9cd2729c \\\n"
        f"  --instance-type g6.2xlarge \\\n"
        f"  --key-name {key_name} \\\n"
        f"  --tag-specifications 'ResourceType=instance,Tags=[{{Key=Name,Value={service_name}}}]' \\\n"
        f'  --instance-market-options \'{{"MarketType":"spot","SpotOptions":{{"SpotInstanceType":"one-time"}}}}\' \\\n'
        f"  --user-data file://user_data.sh"
    )

    return (
        f"### 🚀 AWS EC2 g6.2xlarge (NVIDIA L4) Spot Instance vLLM Deployment Config\n\n"
        f"#### 1. UserData Script (`user_data.sh`):\n"
        f"```bash\n{user_data}\n```\n\n"
        f"#### 2. Run Instance CLI Command:\n"
        f"```bash\n{aws_cmd}\n```\n\n"
        f"#### 3. Prerequisites:\n"
        f'- Save your HF Token in AWS SSM Parameter Store: `aws ssm put-parameter --name /vllm/HF_TOKEN --value "your-token" --type SecureString`\n'
        f"- Ensure the security group allows inbound TCP traffic on port `8080`."
    )


@mcp.tool()
async def deploy_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-12B-it-qat-w4a16-ct",
    key_name: str = "alinux",
    subnet_id: Optional[str] = None,
) -> str:
    """
    Deploys vLLM to AWS EC2 g6.2xlarge (NVIDIA L4) instance.

    Args:
        service_name: Tag Name for the EC2 instance.
        model_path: Hugging Face repo ID or S3 URI.
        key_name: AWS EC2 Key Pair name (default: 'alinux').
        subnet_id: Subnet ID to launch in (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # 1. Resolve or create security group vllm-devops-sg
    sg_name = "vllm-devops-sg"
    sg_id = None
    try:
        sgs = ec2.describe_security_groups(GroupNames=[sg_name])
        sg_id = sgs["SecurityGroups"][0]["GroupId"]
    except Exception:
        try:
            vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
            vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None
            if not vpc_id:
                vpcs = ec2.describe_vpcs()
                vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None

            sg_res = ec2.create_security_group(
                GroupName=sg_name, Description="Security Group for vLLM DevOps Agent", VpcId=vpc_id
            )
            sg_id = sg_res["GroupId"]
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Allow SSH"}],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8080,
                        "ToPort": 8080,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Allow vLLM API"}],
                    },
                ],
            )
        except Exception as e:
            return f"Failed to create/configure security group: {str(e)}"

    # 2. Resolve Subnet
    if not subnet_id:
        try:
            subnets = ec2.describe_subnets(
                Filters=[
                    {"Name": "map-public-ip-on-launch", "Values": ["true"]},
                    {"Name": "availability-zone", "Values": [f"{AWS_REGION}f", f"{AWS_REGION}d", f"{AWS_REGION}a"]},
                ]
            )
            if subnets["Subnets"]:
                subnet_id = subnets["Subnets"][0]["SubnetId"]
            else:
                subnets = ec2.describe_subnets()
                subnet_id = subnets["Subnets"][0]["SubnetId"] if subnets["Subnets"] else None
        except Exception as e:
            logger.warning(f"Failed to discover subnet: {e}")

    # 3. Retrieve HF Token from Secret Manager/environment to inject
    hf_token = await get_secret() or ""

    # 4. UserData script
    quant_arg = (
        "--quantization compressed-tensors" if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]) else ""
    )
    user_data = f"""#!/bin/bash
if ! command -v docker &> /dev/null; then
    apt-get update -y
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
fi
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN="{hf_token}" \\
  vllm/vllm-openai:nightly \\
  --model {model_path} \\
  {quant_arg} \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization 0.95 \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{{}}' \\
  --host 0.0.0.0 \\
  --port 8080
"""

    # 5. Launch EC2 Instance
    try:
        run_args = {
            "ImageId": "ami-012ba162b9cd2729c",  # DLAMI Ubuntu 22.04
            "InstanceType": "g6.2xlarge",
            "MinCount": 1,
            "MaxCount": 1,
            "KeyName": key_name,
            "SecurityGroupIds": [sg_id],
            "UserData": user_data,
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": service_name}]}],
            "IamInstanceProfile": {"Name": "aws-elasticbeanstalk-ec2-role"},
            "InstanceMarketOptions": {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}},
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 150,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    }
                }
            ],
        }
        if subnet_id:
            run_args["SubnetId"] = subnet_id

        instance = ec2.run_instances(**run_args)
        inst_id = instance["Instances"][0]["InstanceId"]

        return (
            f"🚀 Successfully requested AWS EC2 g6.2xlarge Spot Instance deployment for service '{service_name}'.\n"
            f"Instance ID: `{inst_id}`\n"
            f"Key Pair: `{key_name}`\n"
            f"Subnet ID: `{subnet_id}`\n"
            f"Please wait a few minutes for the instance to initialize and pull the vLLM docker image."
        )
    except Exception as e:
        return f"Failed to deploy AWS EC2 instance:\nError: {str(e)}"


@mcp.tool()
async def start_ec2(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-12B-it-qat-w4a16-ct",
    key_name: str = "alinux",
    subnet_id: Optional[str] = None,
    instance_type: str = "g6.2xlarge",
    market_type: str = "on-demand",
    instance_id: Optional[str] = None,
) -> str:
    """
    Starts an existing stopped EC2 instance, or provisions a new one with NVIDIA L4 GPU if none exists.

    Args:
        service_name: Tag Name for the EC2 instance.
        model_path: Hugging Face repo ID or S3 URI (used if launching a new instance).
        key_name: AWS EC2 Key Pair name (default: 'alinux').
        subnet_id: Subnet ID to launch in (optional).
        instance_type: EC2 instance type (default: 'g6.2xlarge').
        market_type: Market type for the instance ('spot' or 'on-demand', default: 'on-demand').
        instance_id: Direct Instance ID to start if it already exists (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # Check if instance already exists (either by instance_id or service_name)
    existing_instance_ids = []
    try:
        if instance_id:
            res = ec2.describe_instances(InstanceIds=[instance_id])
            for reservation in res.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    if inst["State"]["Name"] in ["stopped", "stopping"]:
                        existing_instance_ids.append(inst["InstanceId"])
        else:
            res = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [service_name]},
                    {"Name": "instance-state-name", "Values": ["stopped", "stopping"]},
                ]
            )
            for reservation in res.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    existing_instance_ids.append(inst["InstanceId"])
    except Exception as e:
        logger.info(f"Checking existing instances returned: {e}")

    # If stopped instance exists, start it!
    if existing_instance_ids:
        try:
            ec2.start_instances(InstanceIds=existing_instance_ids)
            return f"🚀 Successfully requested start for existing stopped EC2 Instance(s): {', '.join(existing_instance_ids)}"
        except Exception as e:
            return f"Failed to start existing EC2 instance(s) {existing_instance_ids}:\nError: {str(e)}"

    # Otherwise, provision a new one!
    # 1. Resolve or create security group vllm-devops-sg
    sg_name = "vllm-devops-sg"
    sg_id = None
    try:
        sgs = ec2.describe_security_groups(GroupNames=[sg_name])
        sg_id = sgs["SecurityGroups"][0]["GroupId"]
    except Exception:
        try:
            vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
            vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None
            if not vpc_id:
                vpcs = ec2.describe_vpcs()
                vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None

            sg_res = ec2.create_security_group(
                GroupName=sg_name, Description="Security Group for vLLM DevOps Agent", VpcId=vpc_id
            )
            sg_id = sg_res["GroupId"]
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Allow SSH"}],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8080,
                        "ToPort": 8080,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Allow vLLM API"}],
                    },
                ],
            )
        except Exception as e:
            return f"Failed to create/configure security group: {str(e)}"

    # 2. Resolve Subnet
    if not subnet_id:
        try:
            subnets = ec2.describe_subnets(
                Filters=[
                    {"Name": "map-public-ip-on-launch", "Values": ["true"]},
                    {"Name": "availability-zone", "Values": [f"{AWS_REGION}f", f"{AWS_REGION}d", f"{AWS_REGION}a"]},
                ]
            )
            if subnets["Subnets"]:
                subnet_id = subnets["Subnets"][0]["SubnetId"]
            else:
                subnets = ec2.describe_subnets()
                subnet_id = subnets["Subnets"][0]["SubnetId"] if subnets["Subnets"] else None
        except Exception as e:
            logger.warning(f"Failed to discover subnet: {e}")

    # 3. Retrieve HF Token
    try:
        hf_token = await get_secret() or ""
    except Exception as e:
        logger.warning(f"Failed to retrieve HF token: {e}. Defaulting to empty.")
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or ""

    # 4. UserData script
    quant_arg = (
        "--quantization compressed-tensors" if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]) else ""
    )
    user_data = f"""#!/bin/bash
if ! command -v docker &> /dev/null; then
    apt-get update -y
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
fi
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN="{hf_token}" \\
  vllm/vllm-openai:nightly \\
  --model {model_path} \\
  {quant_arg} \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization 0.95 \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{{}}' \\
  --host 0.0.0.0 \\
  --port 8080
"""

    # 5. Launch EC2 Instance
    try:
        run_args = {
            "ImageId": "ami-012ba162b9cd2729c",  # DLAMI Ubuntu 22.04
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "KeyName": key_name,
            "SecurityGroupIds": [sg_id],
            "UserData": user_data,
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": service_name}]}],
            "IamInstanceProfile": {"Name": "aws-elasticbeanstalk-ec2-role"},
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 150,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    }
                }
            ],
        }
        if subnet_id:
            run_args["SubnetId"] = subnet_id

        if market_type.lower() == "spot":
            run_args["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}}

        instance = ec2.run_instances(**run_args)
        inst_id = instance["Instances"][0]["InstanceId"]

        return (
            f"🚀 Successfully requested AWS EC2 {instance_type} {market_type} Instance deployment for service '{service_name}'.\n"
            f"Instance ID: `{inst_id}`\n"
            f"Key Pair: `{key_name}`\n"
            f"Subnet ID: `{subnet_id}`\n"
            f"Please wait a few minutes for the instance to initialize and pull the vLLM docker image."
        )
    except Exception as e:
        return f"Failed to deploy AWS EC2 instance:\nError: {str(e)}"


@mcp.tool()
async def destroy_vllm(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Cleans up the vLLM Docker container on the AWS EC2 instance(s) matching the specified service Name tag,
    without terminating the EC2 instance(s).

    Args:
        service_name: Name tag of the instance(s) to clean up.
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    ssm = boto3.client("ssm", region_name=AWS_REGION)

    try:
        response = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [service_name]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
            ]
        )
        instance_ids = []
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instance_ids.append(instance["InstanceId"])

        if not instance_ids:
            return f"No active EC2 instances found matching service name tag '{service_name}' to clean up."

        # Send SSM command to stop and remove container
        cmd_response = ssm.send_command(
            InstanceIds=instance_ids,
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": ["docker stop vllm-server || true", "docker rm vllm-server || true"]},
        )
        command_id = cmd_response["Command"]["CommandId"]

        return (
            f"🧹 Successfully requested cleanup of the 'vllm-server' Docker container on EC2 Instance(s): {', '.join(instance_ids)}.\n"
            f"SSM Command ID: `{command_id}` (EC2 instance(s) remain running)."
        )
    except Exception as e:
        return f"Failed to clean up container for service '{service_name}':\nError: {str(e)}"


@mcp.tool()
def stop_ec2(
    service_name: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> str:
    """
    Stops AWS EC2 instance(s) by service name tag or instance ID.

    Args:
        service_name: Name tag of the instance(s) to stop (optional).
        instance_id: Direct Instance ID to stop (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    instance_ids = []

    if instance_id:
        instance_ids.append(instance_id)
    else:
        target_name = service_name or DEFAULT_SERVICE_NAME
        try:
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [target_name]},
                    {"Name": "instance-state-name", "Values": ["pending", "running"]},
                ]
            )
            for reservation in response.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    instance_ids.append(inst["InstanceId"])
        except Exception as e:
            return f"Failed to search for EC2 instances to stop:\nError: {str(e)}"

    if not instance_ids:
        target = f"Instance ID '{instance_id}'" if instance_id else f"service tag '{service_name or DEFAULT_SERVICE_NAME}'"
        return f"No active/pending EC2 instances found to stop matching {target}."

    try:
        ec2.stop_instances(InstanceIds=instance_ids)
        return f"🛑 Successfully requested stopping for EC2 Instance(s): {', '.join(instance_ids)}"
    except Exception as e:
        return f"Failed to stop EC2 instances {instance_ids}:\nError: {str(e)}"


@mcp.tool()
def status_vllm(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Checks the status of the AWS EC2 instance(s) matching the specified service Name tag.

    Args:
        service_name: Name tag of the instance(s) to check.
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    try:
        response = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [service_name]}])
        instances_info = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                info = (
                    f"- **Instance ID**: `{inst['InstanceId']}`\n"
                    f"  - **Type**: `{inst['InstanceType']}`\n"
                    f"  - **State**: `{inst['State']['Name']}`\n"
                    f"  - **Public IP**: `{inst.get('PublicIpAddress', 'None')}`\n"
                    f"  - **Public DNS**: `{inst.get('PublicDnsName', 'None')}`\n"
                    f"  - **Launch Time**: `{inst['LaunchTime']}`\n"
                )
                instances_info.append(info)

        if not instances_info:
            return f"No EC2 instances found matching service name tag '{service_name}'."

        return f"### AWS EC2 Status for service tag '{service_name}':\n\n" + "\n".join(instances_info)
    except Exception as e:
        return f"Failed to get status for service '{service_name}':\nError: {str(e)}"


@mcp.tool()
def status_ec2(
    service_name: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> str:
    """
    Checks the status of AWS EC2 instance(s) by service name tag or instance ID.

    Args:
        service_name: Name tag of the instance(s) to check (optional).
        instance_id: Direct Instance ID to check (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    filters = []
    instance_ids = []

    if instance_id:
        instance_ids.append(instance_id)
    elif service_name:
        filters.append({"Name": "tag:Name", "Values": [service_name]})
    else:
        filters.append({"Name": "tag:Name", "Values": [DEFAULT_SERVICE_NAME]})

    try:
        run_args = {}
        if filters:
            run_args["Filters"] = filters
        if instance_ids:
            run_args["InstanceIds"] = instance_ids

        response = ec2.describe_instances(**run_args)
        instances_info = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                info = (
                    f"- **Instance ID**: `{inst['InstanceId']}`\n"
                    f"  - **Type**: `{inst['InstanceType']}`\n"
                    f"  - **State**: `{inst['State']['Name']}`\n"
                    f"  - **Public IP**: `{inst.get('PublicIpAddress', 'None')}`\n"
                    f"  - **Public DNS**: `{inst.get('PublicDnsName', 'None')}`\n"
                    f"  - **Launch Time**: `{inst['LaunchTime']}`\n"
                )
                instances_info.append(info)

        if not instances_info:
            target = f"Instance ID '{instance_id}'" if instance_id else f"service tag '{service_name or DEFAULT_SERVICE_NAME}'"
            return f"No EC2 instances found matching {target}."

        target_desc = f"Instance ID '{instance_id}'" if instance_id else f"service tag '{service_name or DEFAULT_SERVICE_NAME}'"
        return f"### AWS EC2 Status for {target_desc}:\n\n" + "\n".join(instances_info)
    except Exception as e:
        return f"Failed to get status for EC2 target:\nError: {str(e)}"


@mcp.tool()
async def check_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    instance_id: Optional[str] = None,
) -> str:
    """
    Checks the status of the vLLM container and engine running on the EC2 instance(s).

    Args:
        service_name: Name tag of the instance(s) to check.
        instance_id: Direct Instance ID to check (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    ssm = boto3.client("ssm", region_name=AWS_REGION)

    # 1. Resolve instance
    filters = []
    instance_ids = []
    if instance_id:
        instance_ids.append(instance_id)
    else:
        filters.append({"Name": "tag:Name", "Values": [service_name]})
        filters.append({"Name": "instance-state-name", "Values": ["pending", "running"]})

    try:
        run_args = {}
        if filters:
            run_args["Filters"] = filters
        if instance_ids:
            run_args["InstanceIds"] = instance_ids

        response = ec2.describe_instances(**run_args)
        instances = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                instances.append(inst)
    except Exception as e:
        return f"Failed to describe EC2 instances:\nError: {str(e)}"

    if not instances:
        target = f"Instance ID '{instance_id}'" if instance_id else f"active service tag '{service_name}'"
        return f"No active EC2 instances found matching {target}."

    reports = []
    for inst in instances:
        inst_id = inst["InstanceId"]
        state = inst["State"]["Name"]
        ip = inst.get("PublicIpAddress")

        report = f"### 🖥️ Instance: `{inst_id}` ({state})\n"
        if state != "running":
            report += f"❌ Instance is not running (Current State: `{state}`). Skipping container checks.\n"
            reports.append(report)
            continue

        if not ip:
            report += "⚠️ No Public IP associated with this running instance.\n"
            reports.append(report)
            continue

        # 2. Check Docker Container status via SSM
        docker_status = "Unknown (SSM Unreachable)"
        try:
            cmd_res = ssm.send_command(
                InstanceIds=[inst_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": ["docker inspect -f '{{.State.Status}}' vllm-server 2>&1"]},
            )
            cmd_id = cmd_res["Command"]["CommandId"]

            # Poll for completion (up to 5 seconds)
            for _ in range(5):
                await asyncio.sleep(1)
                try:
                    result = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=inst_id)
                    if result["Status"] == "Success":
                        docker_status = result["StandardOutputContent"].strip()
                        break
                    elif result["Status"] in ["Failed", "TimedOut", "Cancelled"]:
                        docker_status = f"Failed (SSM Status: {result['Status']})"
                        break
                except Exception:
                    pass
        except Exception as e:
            docker_status = f"Error querying SSM: {str(e)}"

        report += f"- **Docker Container (`vllm-server`)**: `{docker_status}`\n"

        # 3. Check vLLM HTTP health endpoint
        http_status = "Unreachable"
        try:
            async with httpx.AsyncClient(timeout=3) as http_client:
                res = await http_client.get(f"http://{ip}:8080/health")
                if res.status_code == 200:
                    http_status = "Healthy ✅"
                else:
                    http_status = f"Unhealthy (HTTP Code: {res.status_code}) ❌"
        except Exception as e:
            http_status = f"Unreachable (Error: {e}) ❌"

        report += f"- **vLLM API Endpoint (`http://{ip}:8080/health`)**: `{http_status}`\n"
        reports.append(report)

    return "\n".join(reports)


@mcp.tool()
def update_vllm_scaling(instance_type: str, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Updates the EC2 instance type (scaling vertically) for the vLLM service instance.
    Note: The instance must be stopped to change its type.

    Args:
        instance_type: The new AWS EC2 instance type (e.g. 'g6.4xlarge', 'g6.2xlarge').
        service_name: The Name tag of the instance to scale.
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    try:
        response = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [service_name]},
                {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
            ]
        )
        instances = []
        for reservation in response.get("Reservations", []):
            instances.extend(reservation.get("Instances", []))

        if not instances:
            return f"No active EC2 instances found matching service name tag '{service_name}'."

        inst = instances[0]
        inst_id = inst["InstanceId"]
        current_type = inst["InstanceType"]
        state = inst["State"]["Name"]

        if state == "running":
            ec2.stop_instances(InstanceIds=[inst_id])
            return (
                f"⏸️ Instance `{inst_id}` is currently running. We have requested it to stop to perform vertical scaling.\n"
                f"Current Type: `{current_type}` -> Target Type: `{instance_type}`.\n"
                f"Please wait for the instance to stop, then run this tool again to apply the new instance type."
            )
        elif state == "stopped":
            ec2.modify_instance_attribute(InstanceId=inst_id, InstanceType={"Value": instance_type})
            ec2.start_instances(InstanceIds=[inst_id])
            return (
                f"⚡ Successfully scaled EC2 instance `{inst_id}` from `{current_type}` to `{instance_type}`.\n"
                f"Requested the instance to start back up."
            )
        else:
            return f"Instance `{inst_id}` is in state '{state}'. Vertical scaling is only supported when stopped."
    except Exception as e:
        return f"Failed to scale service '{service_name}':\nError: {str(e)}"


@mcp.tool()
def get_vllm_gpu_deployment_config(
    cluster_name: str = "eks-gpu-cluster", model_name: str = "google/gemma-4-12B-it-qat-w4a16-ct"
) -> str:
    """
    Generates an AWS EKS manifest and node group instructions for deploying vLLM on L4 GPU (g6.2xlarge).

    Args:
        cluster_name: The name of the EKS cluster.
        model_name: The model identifier (e.g., 'google/gemma-4-12B-it-qat-w4a16-ct').
    """
    manifest = f"""
### 🌀 vLLM on EKS GPU (AWS Deployment)

To deploy vLLM on EKS GPUs, use the following EKS Nodegroup config and Kubernetes manifest. This targets **NVIDIA L4 GPUs** via **g6.2xlarge** instances.

#### 1. Create a GPU Node Group via eksctl
```bash
eksctl create nodegroup \\
    --cluster={cluster_name} \\
    --region={AWS_REGION} \\
    --name=gpu-l4-nodes \\
    --node-type=g6.2xlarge \\
    --nodes=1 \\
    --nodes-min=1 \\
    --nodes-max=2 \\
    --node-volume-size=100
```

#### 2. Kubernetes Manifest (vllm-gpu.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-gpu
  template:
    metadata:
      labels:
        app: vllm-gpu
    spec:
      nodeSelector:
        k8s.amazonaws.com/accelerator: nvidia-l4
      containers:
      - name: vllm-gpu
        image: vllm/vllm-openai:nightly
        resources:
          limits:
            nvidia.com/gpu: "1"
          requests:
            nvidia.com/gpu: "1"
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - "--model={model_name}"
        - "--gpu-memory-utilization=0.95"
        - "--max-model-len=32768"
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-gpu-service
spec:
  selector:
    app: vllm-gpu
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

#### 3. Deployment Steps
1. Install NVIDIA Device Plugin for Kubernetes on your EKS cluster:
   `kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml`
2. Save the YAML above to `vllm-gpu.yaml`.
3. Apply it: `kubectl apply -f vllm-gpu.yaml`.
"""
    return manifest


@mcp.tool()
def get_vertex_ai_model_copy_instructions(model_name: str = "gemma-4-12B-it-qat-w4a16-ct") -> str:
    """
    Provides instructions and commands to transfer Gemma model artifacts from Vertex AI Model Garden to your GCS bucket.
    """
    instructions = f"""
### 🚀 Transferring {model_name} from Vertex AI Model Garden

To use vLLM with Cloud Storage FUSE without Hugging Face, follow these steps:

1. **Accept Terms:** Go to the Vertex AI Model Garden page for Gemma (https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335) and click 'Accept' on the license agreement.
2. **Download via Signed URL:** After accepting, the console provides a 'Download' button or a signed URL.
3. **Transfer to GCS:**
   If you have the artifacts locally after downloading from the console, use:
   `gcloud storage cp -r ./model_artifacts/* gs://{BUCKET_NAME}/{model_name}/`

4. **Alternative (Direct GCS Copy):**
   Google occasionally provides a managed GCS path for verified projects. If accessible, you can use:
   `gcloud storage cp -r gs://vertex-ai-models/gemma/{model_name}/* gs://{BUCKET_NAME}/{model_name}/`

Once the artifacts are in your bucket, use `get_vllm_deployment_config` to generate your Cloud Run deployment command.
"""
    return instructions


@mcp.tool()
async def get_huggingfacehub_download_path(
    repo_id: str = "google/gemma-4-12B-it-qat-w4a16-ct",
) -> str:
    """
    Returns the local cache path for a Hugging Face model using huggingface_hub.
    Note: This may trigger a download if the model is not already in the cache.
    """
    try:
        from huggingface_hub import snapshot_download

        token = await get_secret() or os.getenv("HF_TOKEN")
        # Run synchronous snapshot_download in a separate thread to avoid blocking the async event loop
        path = await asyncio.to_thread(snapshot_download, repo_id=repo_id, token=token)
        return f"Model '{repo_id}' is available at: {path}"
    except Exception as e:
        return f"Error resolving huggingface_hub path: {str(e)}"


@mcp.tool()
def get_huggingface_model_copy_instructions(
    repo_id: str = "google/gemma-4-12B-it-qat-w4a16-ct",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Provides instructions and commands to transfer Gemma model weights from Hugging Face to your S3 or GCS bucket.

    Args:
        repo_id: The Hugging Face repo ID (e.g., 'google/gemma-4-12B-it-qat-w4a16-ct').
        bucket_name: The target bucket name (defaults to AWS_BUCKET_NAME or BUCKET_NAME).
    """
    if not bucket_name:
        bucket_name = AWS_BUCKET_NAME if os.getenv("AWS_ACCESS_KEY_ID") else BUCKET_NAME

    model_name = repo_id.split("/")[-1]
    is_aws = os.getenv("AWS_ACCESS_KEY_ID") is not None

    if is_aws:
        instructions = f"""
### 📦 Transferring {model_name} from Hugging Face to AWS S3

To use Hugging Face weights with vLLM on EC2 via S3, follow these steps:

#### Option A: Download directly inside EC2 instance (Recommended)
You don't need to copy weights to S3 if you specify the Hugging Face Repo ID directly in `deploy_vllm`.
vLLM will download it automatically from Hugging Face when starting, using your `HF_TOKEN`.

#### Option B: Upload to S3 Bucket
If you prefer to host weights privately in S3:

1. **Download Model locally:**
   `python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('{repo_id}'))"`

2. **Upload to S3:**
   The command above outputs the local path. Use it to copy the artifacts:
   `aws s3 cp --recursive /path/to/downloaded/model/ s3://{bucket_name}/{model_name}/`

Once uploaded, you can deploy using:
`deploy_vllm(model_path="s3://{bucket_name}/{model_name}/")`
"""
    else:
        instructions = f"""
### 📦 Transferring {model_name} from Hugging Face to GCS

To use Hugging Face weights with vLLM on Cloud Run via GCS FUSE, follow these steps:

#### Option A: Using `huggingface_hub` Python Library (Recommended)
`huggingface_hub` simplifies the download process and can be run directly from python:

1. **Download Model:**
   `python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('{repo_id}'))"`

2. **Upload to GCS:**
   The command above outputs the local path. Use it to copy the artifacts:
   `gcloud storage cp -r /path/to/downloaded/model/* gs://{bucket_name}/{model_name}/`

#### Option B: Using `huggingface-cli`
1. **Setup Hugging Face CLI:**
   `pip install huggingface_hub`
   `huggingface-cli login`

2. **Download Model Artifacts:**
   `huggingface-cli download {repo_id} --local-dir ./{model_name}`

3. **Upload to GCS Bucket:**
   `gcloud storage cp -r ./{model_name}/* gs://{bucket_name}/{model_name}/`

Once uploaded, you can deploy using:
`get_vllm_deployment_config(model_path="{model_name}")`
"""
    return instructions


@mcp.tool()
def check_gpu_quotas(region: Optional[str] = None) -> str:
    """
    Checks GPU quotas for a specific AWS region (via Service Quotas) or Google Cloud region.

    Args:
        region: The AWS region (defaults to AWS_REGION) or GCP region.
    """
    if not region:
        region = AWS_REGION if os.getenv("AWS_ACCESS_KEY_ID") else LOCATION

    # If it looks like an AWS region or AWS credentials exist, try AWS Service Quotas
    is_aws = "-" in region and not region.startswith("us-east4")
    if is_aws or os.getenv("AWS_ACCESS_KEY_ID"):
        try:
            import boto3

            sq = boto3.client("service-quotas", region_name=region)
            quotas_to_check = [
                {"QuotaCode": "L-DB2E81BA", "Name": "Running On-Demand G and VT instances"},
                {"QuotaCode": "L-3819A6DF", "Name": "All G and VT Spot Instance Requests"},
                {"QuotaCode": "L-417A2355", "Name": "Running On-Demand P instances"},
            ]
            report = f"### 📊 AWS EC2 GPU Quotas for region `{region}`\n\n"
            for q in quotas_to_check:
                try:
                    res = sq.get_service_quota(ServiceCode="ec2", QuotaCode=q["QuotaCode"])
                    quota = res["Quota"]
                    report += f"- **{q['Name']}** ({quota['QuotaName']}):\n"
                    report += f"  - Limit: `{quota['Value']}` (vCPUs)\n"
                    report += f"  - Adjustable: `{quota['Adjustable']}`\n"
                except Exception as e:
                    report += f"- **{q['Name']}** ({q['QuotaCode']}): Could not fetch quota ({str(e)})\n"
            return report
        except Exception as e:
            logger.warning(f"Failed to check AWS EC2 quotas in `{region}`: {e}")

    # Fallback to GCP
    cmd = [
        "gcloud",
        "compute",
        "regions",
        "describe",
        region,
        f"--project={PROJECT_ID}",
        "--format=json(quotas)",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        quotas = data.get("quotas", [])

        # Filter for GPU quotas
        gpu_quotas = []
        for q in quotas:
            metric = q.get("metric", "")
            if "GPU" in metric or "ACCELERATOR" in metric:
                gpu_quotas.append(f"- **{metric}**:\n  - Limit: `{q.get('limit')}`\n  - Usage: `{q.get('usage')}`")

        if not gpu_quotas:
            return f"No GPU/Accelerator quotas found in GCP region `{region}`."

        return f"### 📊 GCP GPU Quotas for region `{region}`\n\n" + "\n".join(gpu_quotas)

    except subprocess.CalledProcessError as e:
        return f"Failed to retrieve GPU quotas for region `{region}`:\nError: {e.stderr}\nOutput: {e.stdout}"
    except Exception as e:
        return f"Error checking GPU quotas: {str(e)}"


@mcp.tool()
async def verify_model_health() -> str:
    """Runs a deep health check with latency reporting on the Cloud Run GPU-hosted model."""
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        start_time = time.monotonic()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, is the model working?"}],
            model=model_name,
            max_tokens=200,
        )
        end_time = time.monotonic()
        latency = end_time - start_time
        response_content = chat_completion.choices[0].message.content

        if response_content:
            return (
                f"✅ Model health check PASSED.\n"
                f"Model: {model_name}\n"
                f"Response: '{response_content[:50]}...'\n"
                f"Latency: {latency:.2f} seconds."
            )
        else:
            return "❌ Model health check FAILED: Empty response."
    except Exception as e:
        return f"❌ Model health check FAILED: {e}"


@mcp.tool()
async def query_gemma4(prompt: str) -> str:
    """Queries the self-hosted Gemma 4 model on Cloud Run."""
    logger.info(f"Querying Cloud Run model with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
        )
        response = chat_completion.choices[0].message.content or "No response from model."
        logger.info(f"Model response: '{response[:100]}...'")
        return response
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        return f"❌ An error occurred while querying the model: {e}"


@mcp.tool()
async def query_gemma4_with_stats(prompt: str) -> str:
    """
    Queries the self-hosted Gemma 4 model on Cloud Run and returns detailed performance statistics.

    This tool provides:
    - The full model response.
    - Time to First Token (TTFT).
    - Total generation time.
    - Tokens per second.
    """
    logger.info(f"Querying model with stats with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)

        start_time = time.monotonic()
        ttft = None
        response_content = ""
        total_tokens = 0

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True,
        )

        async for chunk in stream:
            if ttft is None:
                ttft = time.monotonic() - start_time

            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_content += content
                total_tokens += 1  # Rough token count

        end_time = time.monotonic()
        total_time = end_time - start_time

        if not response_content:
            return "❌ Model returned an empty response."

        tokens_per_second = total_tokens / (total_time - ttft) if ttft and total_time > ttft else 0

        stats_report = (
            f"### 📊 Performance Stats\n"
            f"- **Model:** `{model_name}`\n"
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
async def get_model_details() -> str:
    """Retrieves detailed information about the running Cloud Run model, engine, and versions."""
    report = ""
    try:
        vllm_url = get_vllm_url()
        report += f"### 🧩 Model Details ({vllm_url})\n\n"
        client = await get_vllm_client()

        # 1. Get Model Details from /v1/models
        try:
            models_res = await client.models.list()
            report += "**Model Information (`/v1/models`):**\n"
            models_list = [{"id": m.id, "object": m.object, "owned_by": m.owned_by} for m in models_res.data]
            report += f"```json\n{json.dumps(models_list, indent=2)}\n```\n"
        except Exception as e:
            report += f"❌ Error fetching model details via client: {e}\n\n"

        # 2. Get Health Status
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=10) as http_client:
            try:
                res = await http_client.get(f"{vllm_url}/health", headers=headers)
                if res.status_code == 200:
                    report += "**Health Status (`/health`):**\n- Status: `Healthy` ✅\n\n"
                else:
                    report += f"**Health Status (`/health`):**\n- Status: `Unhealthy` (Code: {res.status_code}) ❌\n\n"
            except Exception as e:
                report += f"**Health Status (`/health`):**\n- Status: `Unreachable` (Error: {e}) ❌\n\n"
    except Exception as e:
        report += f"❌ Error retrieving system URL or auth token: {e}"

    return report


@mcp.tool()
async def get_system_status(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Provides a high-level dashboard of AWS EC2 system status and vLLM health, or GCP Cloud Run.

    Args:
        service_name: The name tag of the EC2 service instance or Cloud Run service name.
    """
    health = "🔴 Offline"
    url = None
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                health = f"🟢 Online ({url})"
            else:
                health = f"🔴 Offline (Status {res.status_code}) ({url})"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    # Try AWS first
    ec2_status = "🔴 Unknown"
    is_aws = False
    if os.getenv("AWS_ACCESS_KEY_ID"):
        is_aws = True
        try:
            import boto3

            ec2 = boto3.client("ec2", region_name=AWS_REGION)
            response = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [service_name]}])
            instances = []
            for reservation in response.get("Reservations", []):
                instances.extend(reservation.get("Instances", []))

            if instances:
                # Prioritize active (running/pending) instances
                active_instances = [i for i in instances if i["State"]["Name"] in ["running", "pending"]]
                target_instance = active_instances[0] if active_instances else instances[0]
                state = target_instance["State"]["Name"]
                inst_id = target_instance["InstanceId"]
                if state == "running":
                    ec2_status = f"🟢 Running ({inst_id})"
                else:
                    ec2_status = f"🔴 {state.capitalize()} ({inst_id})"
            else:
                ec2_status = "🔴 Instance Not Found"
        except Exception as e:
            ec2_status = f"🔴 AWS Error: {str(e)}"

    if not is_aws or "Error" in ec2_status or "Not Found" in ec2_status:
        # GCP fallback
        try:
            cmd = [
                "gcloud",
                "run",
                "services",
                "describe",
                service_name,
                f"--project={PROJECT_ID}",
                f"--region={LOCATION}",
                "--format=json",
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if process.returncode == 0:
                import json

                data = json.loads(process.stdout)
                conditions = data.get("status", {}).get("conditions", [])
                ready_cond = next((c for c in conditions if c.get("type") == "Ready"), None)
                if ready_cond and ready_cond.get("status") == "True":
                    ec2_status = "🟢 GCP Ready"
                elif ready_cond:
                    ec2_status = f"🔴 GCP Not Ready ({ready_cond.get('status')})"
                else:
                    ec2_status = "🔴 GCP Not Ready (No Ready condition)"
            else:
                ec2_status = f"🔴 GCP Error checking service ({process.stderr.strip()})"
        except Exception as e:
            ec2_status = f"🔴 GCP Error: {str(e)}"

    if "🟢" in health:
        next_step = "Use `query_gemma4` to interact with the model."
    else:
        provider = "AWS EC2 instance" if is_aws else "Cloud Run service"
        next_step = f"Call `deploy_vllm` to provision/start the {provider} `{service_name}`."

    return (
        f"### 🌀 GPU vLLM System Status\n"
        f"- **vLLM Health:** {health}\n"
        f"- **Hosting Status:** {ec2_status}\n"
        f"**👉 Next Step:** {next_step}"
    )


@mcp.tool()
async def get_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Returns the active vLLM service URL if available.

    Args:
        service_name: The name of the service or instance Name tag to query.
    """
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                return f"🟢 vLLM is Online at: {url}"
            else:
                return f"🔴 vLLM is configured at {url} but returned status {res.status_code}."
    except Exception as e:
        return f"🔴 vLLM endpoint check failed: {e}. Try deploying/starting it with `deploy_vllm`."


@mcp.tool()
async def run_benchmark(
    model: Optional[str] = None,
    num_prompts: int = 20,
    random_output_len: int = 128,
    max_concurrency: int = 8,
) -> str:
    """
    Runs a performance/concurrency benchmark sweep against the Cloud Run vLLM GPU endpoint.

    Args:
        model: Model name to request (defaults to the active model from /v1/models).
        num_prompts: Number of requests to send per concurrency level.
        random_output_len: Max tokens to generate per request.
        max_concurrency: Maximum concurrency level to sweep up to (powers of 2, e.g. 1, 2, 4, 8).
    """
    from datetime import datetime

    try:
        url = get_vllm_url()
        token = get_auth_token()
    except Exception as e:
        return f"❌ Cannot run benchmark: {e}"

    # Get active model name if not provided
    client = await get_vllm_client()
    if not model:
        model = await get_active_model_name(client)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base_url = f"{url.rstrip('/')}/v1/completions"
    prompt = "Explain the importance of Site Reliability Engineering for large scale AI deployments."

    concurrencies = []
    c = 1
    while c <= max_concurrency:
        concurrencies.append(c)
        c *= 2
    if max_concurrency not in concurrencies:
        concurrencies.append(max_concurrency)

    results = []

    async def send_request(http_client: httpx.AsyncClient, sem: asyncio.Semaphore) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": random_output_len,
            "temperature": 0.0,
            "stream": False,
        }
        async with sem:
            start_time = time.perf_counter()
            try:
                response = await http_client.post(base_url, json=payload, headers=headers, timeout=120)
                end_time = time.perf_counter()
                if response.status_code == 200:
                    latency = end_time - start_time
                    data = response.json()
                    tokens = data.get("usage", {}).get("completion_tokens", random_output_len)
                    return {"success": True, "latency": latency, "tokens": tokens}
                else:
                    return {"success": False, "error": f"Status {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    # Warmup
    logger.info("Warming up model for benchmark...")
    async with httpx.AsyncClient() as http_client:
        await send_request(http_client, asyncio.Semaphore(1))

    logger.info(f"Starting GPU benchmark sweep against {url} with model {model}...")
    for concurrency in concurrencies:
        logger.info(f"Running sweep with concurrency={concurrency}...")
        sem = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient() as http_client:
            start_batch = time.perf_counter()
            tasks = [send_request(http_client, sem) for _ in range(num_prompts)]
            batch_results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_batch

        successes = [r for r in batch_results if r["success"]]
        latencies = [r["latency"] for r in successes]

        if not latencies:
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "concurrency": concurrency,
                    "total_requests": num_prompts,
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "p95_latency": 0.0,
                    "req_per_sec": 0.0,
                    "tokens_per_sec": 0.0,
                }
            )
            continue

        avg_lat = statistics.mean(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        throughput = len(successes) / total_time
        tokens_per_sec = sum(r["tokens"] for r in successes) / total_time

        results.append(
            {
                "timestamp": datetime.now().isoformat(),
                "concurrency": concurrency,
                "total_requests": num_prompts,
                "success_rate": len(successes) / num_prompts,
                "avg_latency": avg_lat,
                "p95_latency": p95_lat,
                "req_per_sec": throughput,
                "tokens_per_sec": tokens_per_sec,
            }
        )

    # Save to CSV
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "concurrency",
                "total_requests",
                "success_rate",
                "avg_latency",
                "p95_latency",
                "req_per_sec",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary_str = f"### 📊 GPU Benchmark Results (Model: `{model}`)\n\n"
    summary_str += "| Concurrency | Success Rate | Req/s | Tokens/s | Avg Latency | P95 Latency |\n"
    summary_str += "|---:|---:|---:|---:|---:|---:|\n"
    for r in results:
        summary_str += f"| {r['concurrency']} | {r['success_rate']:.1%} | {r['req_per_sec']:.2f} | {r['tokens_per_sec']:.2f} | {r['avg_latency']:.2f}s | {r['p95_latency']:.2f}s |\n"
    summary_str += f"\n\nResults saved to `{output_file}`"
    return summary_str


async def fetch_ec2_logs(instance_id: str, limit: int = 50) -> str:
    """Fetches docker logs from the running EC2 instance via SSM Run Command."""
    try:
        import boto3

        ssm = boto3.client("ssm", region_name=AWS_REGION)
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [f"docker logs --tail {limit} vllm-server 2>&1"]},
        )
        command_id = response["Command"]["CommandId"]

        # Poll for completion (up to 10 seconds)
        for _ in range(10):
            await asyncio.sleep(1)
            try:
                result = ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id,
                )
                if result["Status"] in ["Success", "Failed", "TimedOut", "Cancelled"]:
                    if result["Status"] == "Success":
                        return result["StandardOutputContent"]
                    else:
                        return f"SSM Command failed: {result.get('StandardErrorContent')}"
            except Exception:
                pass
        return "SSM Command timed out."
    except Exception as e:
        return f"Failed to fetch logs via SSM: {str(e)}"


@mcp.tool()
async def analyze_gpu_logs(limit: int = 15, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Fetches vLLM logs for the specified service and uses Gemma 4 to analyze them for errors.

    Args:
        limit: Number of log entries to fetch.
        service_name: Name of the Cloud Run service or AWS EC2 instance tag Name.
    """
    # Try AWS SSM logs first if AWS credentials exist
    if os.getenv("AWS_ACCESS_KEY_ID"):
        try:
            import boto3

            ec2 = boto3.client("ec2", region_name=AWS_REGION)
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [service_name]},
                    {"Name": "instance-state-name", "Values": ["running"]},
                ]
            )
            instances = []
            for reservation in response.get("Reservations", []):
                instances.extend(reservation.get("Instances", []))

            if instances:
                inst_id = instances[0]["InstanceId"]
                logger.info(f"Fetching EC2 logs for instance {inst_id} via SSM...")
                raw_logs = await fetch_ec2_logs(inst_id, limit)
                # Prepare prompt for Gemma
                prompt = f"Analyze the following vLLM docker container logs and provide a high-level summary of critical issues:\n\n{raw_logs}\n\nSummary:"
                client = await get_vllm_client()
                model_name = await get_active_model_name(client)
                chat_completion = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    max_tokens=512,
                    temperature=0.2,
                )
                response_text = chat_completion.choices[0].message.content or ""
                return f"### AWS EC2 Log Analysis (Self-Hosted vLLM)\n\n{response_text}"
        except Exception as e:
            logger.warning(f"Failed to fetch/analyze AWS EC2 logs: {e}")

    # Fallback to GCP
    filter_query = f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service_name}"'
    return await analyze_cloud_logging(filter_query, limit)


@mcp.tool()
async def get_help() -> str:
    """Provides help text and summarizes the configuration options and all available SRE/DevOps tools for this AWS/GCP MCP server."""
    return (
        "### 🛠️ AWS/GCP Gemma 4 SRE Agent Help & Configuration\n\n"
        "You can configure this MCP server using the following environment variables:\n\n"
        "**AWS Configuration:**\n"
        f"- **`AWS_REGION`**: The AWS Region for EC2/EKS deployment.\n"
        f"  - *Current Value:* `{AWS_REGION}`\n"
        f"- **`AWS_BUCKET_NAME`**: S3 Bucket used to store model weights.\n"
        f"  - *Current Value:* `{AWS_BUCKET_NAME}`\n\n"
        "**GCP Configuration:**\n"
        f"- **`GOOGLE_CLOUD_PROJECT`**: Your GCP Project ID.\n"
        f"  - *Current Value:* `{PROJECT_ID}`\n"
        f"- **`GOOGLE_CLOUD_LOCATION`**: The GCP Region for deployment.\n"
        f"  - *Current Value:* `{LOCATION}`\n"
        f"- **`BUCKET_NAME`**: GCS Bucket used to store model weights.\n"
        f"  - *Current Value:* `{BUCKET_NAME}`\n\n"
        "**General serving:**\n"
        f"- **`MODEL_NAME`**: Default Hugging Face repository or path.\n"
        f"  - *Current Value:* `{MODEL_NAME}`\n"
        f"- **`VLLM_BASE_URL`**: The explicit URL of your vLLM service. (If not set, it is auto-discovered via EC2 tags or Cloud Run)\n"
        f"  - *Current Value:* `{VLLM_BASE_URL or 'Not set (auto-discovering)'}`\n\n"
        "### ℹ️ Active Mode Summary\n"
        f"The server is running in **{'AWS' if os.getenv('AWS_ACCESS_KEY_ID') else 'CLOUD RUN'}** mode.\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "Below is a summary of the tools exposed by this SRE/DevOps agent:\n\n"
        "#### 🐳 Infrastructure & Deployment\n"
        "- **`start_ec2`**: Starts an existing stopped EC2 instance, or provisions a new one (with NVIDIA L4 GPU) if none exists.\n"
        "- **`status_ec2`**: Checks the state, type, public IP, DNS, and launch details of EC2 instances.\n"
        "- **`stop_ec2`**: Safely stops active EC2 instances without deleting the root EBS volumes.\n"
        "- **`check_vllm`**: Checks the status of the vLLM container and engine running on the EC2 instance(s).\n"
        "- **`deploy_vllm`**: Deploys vLLM to AWS EC2 g6.2xlarge or GCP Cloud Run GPU.\n"
        "- **`destroy_vllm`**: Cleans up the vLLM Docker container on the AWS EC2 instance without terminating it, or deletes the Cloud Run vLLM service.\n"
        "- **`status_vllm`**: Checks the status of the AWS EC2 instance or Cloud Run vLLM service.\n"
        "- **`update_vllm_scaling`**: Scales EC2 instance type vertically or updates Cloud Run min/max instances.\n"
        "- **`get_vllm_deployment_config`**: Generates the AWS EC2 / GCP deployment command and user data.\n"
        "- **`get_vllm_gpu_deployment_config`**: Generates an AWS EKS nodegroup config or GKE manifest for GPU (NVIDIA L4).\n"
        "- **`check_gpu_quotas`**: Checks GPU/Accelerator quotas for an AWS or GCP region.\n\n"
        "#### 📊 Model Management\n"
        "- **`list_vertex_models`**: Lists models in the Vertex AI Registry.\n"
        "- **`list_bucket_models`**: Lists model weights in S3 or GCS bucket.\n"
        "- **`save_hf_token`**: Securely saves a Hugging Face API token to AWS Secrets Manager or Secret Manager.\n"
        "- **`get_vertex_ai_model_copy_instructions`**: Instructions to copy model from Vertex AI Model Garden to GCS.\n"
        "- **`get_huggingface_model_copy_instructions`**: Instructions to download model from Hugging Face and upload to S3/GCS.\n"
        "- **`get_huggingfacehub_download_path`**: Resolves local cache path using huggingface_hub.\n\n"
        "#### 📊 Monitoring & Status\n"
        "- **`get_metrics`**: Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.\n"
        "- **`get_system_status`**: Provides a high-level status dashboard of the service and health.\n"
        "- **`get_endpoint`**: Verifies connectivity and returns the active service URL.\n"
        "- **`get_model_details`**: Retrieves detailed model metadata and engine state from `/v1/models`.\n"
        "- **`verify_model_health`**: Deep health check by querying the model with a simple prompt and measuring latency.\n\n"
        "#### 📈 Performance & Benchmarking\n"
        "- **`run_benchmark`**: Runs performance/concurrency benchmark sweeps against the vLLM GPU endpoint.\n\n"
        "#### 💬 Interaction & Diagnostics\n"
        "- **`query_gemma4`**: Primary tool to query the self-hosted model with standard chat message format.\n"
        "- **`query_gemma4_with_stats`**: Queries the model and returns streaming performance statistics (TTFT, throughput).\n"
        "- **`query_vllm`**: Direct text completions querying tool.\n"
        "- **`analyze_cloud_logging`**: Fetches logs from AWS CloudWatch or GCP Logging and analyzes them using the model.\n"
        "- **`analyze_gpu_logs`**: Fetches service logs and uses Gemma 4 to analyze them for SRE/DevOps errors.\n"
        "- **`suggest_sre_remediation`**: Suggests remediation plans for SRE errors using the model.\n"
    )


@mcp.tool()
async def get_metrics() -> str:
    """
    Fetches the Prometheus metrics from the active vLLM service.
    """
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{url}/metrics", headers=headers)
            if res.status_code == 200:
                return res.text
            else:
                return f"🔴 Failed to retrieve metrics. Status code: {res.status_code}\n\nResponse:\n{res.text[:1000]}"
    except Exception as e:
        return f"🔴 Error fetching metrics: {e}"


if __name__ == "__main__":
    mcp.run()
