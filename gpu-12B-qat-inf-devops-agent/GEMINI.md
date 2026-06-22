# 🤖 Gemini Workspace Context: AWS Inferentia 12B DevOps Agent

This context guide summarizes the configuration, optimal serving parameters, and capabilities of the self-hosted **Gemma 4 DevOps/SRE Agent** running on **[AWS Inferentia](https://aws.amazon.com/ai/machine-learning/inferentia/)** (`inf2` instances).

---

to authenticate to aws run the save-aws-creds.sh

## ⚙️ Active Environment Configuration

This agent targets AWS deployments utilizing:
- **Default Region**: `us-east-1` (configurable via `AWS_DEFAULT_REGION`)
- **Default Model**: `google/gemma-4-12B-it` (configurable via `MODEL_NAME`)
- **Default S3 Bucket**: `vllm-models-bucket` (configurable via `AWS_BUCKET_NAME`)
- **Default Service Name**: `inferentia-12b-devops-agent`

To serve `google/gemma-4-12B-it` using vLLM on AWS Inferentia, you must use the AWS Neuron SDK-compatible container image.

### 🚀 Serving Command

Run the server using the Neuron-optimized vLLM container:

```bash
docker run -d --name vllm-server \
  --device /dev/neuron0 \
  --ipc=host \
  --restart always \
  -p 8080:8080 \
  -e HF_TOKEN="<your-hf-token>" \
  -e NEURON_CC_FLAGS="--model-type transformer" \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  -v /home/ubuntu/.cache/neuron:/root/.cache/neuron \
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-12B-it \
  --quantization neuron_quant \
  --max-model-len 16384 \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 512 \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 8080
```

> [!NOTE]
> **Gemma 4 Tool Calling Options**: If tool calling features are needed and supported by the container, you can pass `--enable-auto-tool-choice`. However, avoid using `--tool-call-parser gemma4` in this container's version of vLLM (v0.16.0), as it causes a `KeyError`. Fallback to `--tool-call-parser functiongemma` if needed.

### ⚙️ Key Neuron Options & Flags

| Flag | Recommended Setting | Purpose |
| :--- | :--- | :--- |
| `--device` | `/dev/neuron0` | Exposes the AWS Inferentia2 hardware device to the Docker container. |
| `--quantization` | `neuron_quant` | Optimizes and quantizes the model weights for runtime execution on Neuron cores. |
| `--max-model-len` | `16384` | Context window size limit supported under typical neuron-compiled model configurations. |
| `--tensor-parallel-size` | `2` | Configured to map execution across both Neuron Cores inside a single Inferentia2 device. |

---

## 🧰 Key SRE & DevOps Capabilities

This agent exposes several tool categories via the Model Context Protocol (MCP):
- **Deployment & Scaling:** 
  - [deploy_vllm](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L459)
  - [destroy_vllm](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L524)
  - [status_vllm](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L550)
  - [update_vllm_scaling](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L576)
  - [get_vllm_deployment_config](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L393)
  - [get_vllm_gpu_deployment_config](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L606)
  - [check_gpu_quotas](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L783)
- **Model Transfer & Secret Management:** 
  - [list_bucket_models](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L267)
  - [save_hf_token](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L49)
  - [get_huggingface_model_copy_instructions](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L737)
  - [get_huggingfacehub_download_path](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L718)
- **System Monitoring & Health:** 
  - [get_system_status](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L974)
  - [get_endpoint](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L1042)
  - [get_model_details](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L935)
  - [verify_model_health](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L824)
- **Performance Benchmarking:** 
  - [run_benchmark](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L1066)
- **Diagnostics & SRE Remediation:** 
  - [query_gemma4](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L853)
  - [query_gemma4_with_stats](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L872)
  - [query_vllm](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L368)
  - [analyze_cloud_logging](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L297)
  - [analyze_gpu_logs](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L1215)
  - [suggest_sre_remediation](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L343)
  - [get_help](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py#L1228)

---

## 🛠 Command Line Setup

### Deploy/Run Quickstart
```bash
# 1. Install dependencies
make install

# 2. Deploy vLLM to EC2 (with AWS Inferentia)
make deploy

# 3. Check deployment status
make status

# 4. Start the MCP server locally
make run
```

---

## 📚 Key Source Code File Locations
- **MCP Server entrypoint**: [server.py](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/server.py)
- **Test Suite**: [test_agent.py](file:///home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/test_agent.py)

---

## 🔗 External Resources
- **[AWS Inferentia](https://aws.amazon.com/ai/machine-learning/inferentia/)**: AWS Inferentia deep learning hardware accelerator for high-performance and cost-effective inference.
- **[Gemma 4 on AWS Inferentia Cost Guide](https://lushbinary.com/blog/deploy-gemma-4-aws-ec2-sagemaker-inferentia-cost-guide/)**: Comprehensive deployment and cost guide for hosting Gemma 4 on AWS EC2 and SageMaker with Inferentia.
- **[vLLM AWS Neuron Installation Guide](https://docs.vllm.ai/en/v0.10.1/getting_started/installation/aws_neuron.html)**: Official installation and configuration guide for running vLLM on AWS Neuron devices.
- **[AWS Neuron Custom Quantization Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/custom-quantization.html)**: Developer guide for custom quantization under the AWS Neuron SDK.



