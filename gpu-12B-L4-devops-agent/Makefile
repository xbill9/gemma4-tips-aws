# Configuration - Update these or override via environment variables
PROJECT_ID ?= aisprint-491218
REGION ?= us-east4
BUCKET_NAME ?= $(PROJECT_ID)-bucket
SERVICE_NAME ?= gpu-12b-l4-devops-agent
MODEL_PATH ?= gemma-4-12B-it-text-fp8
QUANTIZATION ?= modelopt

install:
	pip install -r requirements.txt

run:
	python server.py

test:
	python test_agent.py

lint:
	ruff check .
	ruff format --check .
	mypy .

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Recommended deployment settings for vLLM on Cloud Run with NVIDIA L4 GPU
# Requires 'gcloud beta' component and 'gen2' execution environment
deploy-vllm:
	@echo "🚀 Deploying vLLM to Cloud Run ($(SERVICE_NAME)) in $(REGION)..."
	@if echo "$(MODEL_PATH)" | grep -q "/"; then \
		gcloud beta run deploy $(SERVICE_NAME) \
			--project=$(PROJECT_ID) \
			--image=vllm/vllm-openai:nightly \
			--gpu=1 \
			--gpu-type=nvidia-l4 \
			--no-gpu-zonal-redundancy \
			--no-cpu-throttling \
			--concurrency=4 \
			--timeout=3600 \
			--startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=25,initialDelaySeconds=180,httpGet.port=8000,httpGet.path=/health \
			--max-instances=1 \
			--min-instances=0 \
			--port=8000 \
			--memory=32Gi \
			--cpu=8 \
			--execution-environment=gen2 \
			--set-secrets=HF_TOKEN=hf-token:latest \
			--args='^|^--model=$(MODEL_PATH)|--dtype=bfloat16|--quantization=$(QUANTIZATION)|--max-model-len=16384|--disable-chunked-mm-input|--gpu-memory-utilization=0.90|--kv-cache-dtype=fp8|--tensor-parallel-size=1|--max-num-seqs=8|--enable-chunked-prefill|--max-num-batched-tokens=4096|--enable-auto-tool-choice|--tool-call-parser=gemma4|--reasoning-parser=gemma4|--async-scheduling|--enforce-eager|--limit-mm-per-prompt={"image":0,"audio":0,"video":0}|--host=0.0.0.0|--port=8000' \
			--no-allow-unauthenticated \
			--region=$(REGION) \
			--set-env-vars=VLLM_ENABLE_CUDA_COMPATIBILITY=1,VLLM_USE_V1=0,HF_HUB_OFFLINE=0,TRANSFORMERS_OFFLINE=0; \
	else \
		gcloud beta run deploy $(SERVICE_NAME) \
			--project=$(PROJECT_ID) \
			--image=vllm/vllm-openai:nightly \
			--command=sh,-c \
			--gpu=1 \
			--gpu-type=nvidia-l4 \
			--no-gpu-zonal-redundancy \
			--no-cpu-throttling \
			--concurrency=4 \
			--timeout=3600 \
			--startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=25,initialDelaySeconds=180,httpGet.port=8000,httpGet.path=/health \
			--max-instances=1 \
			--min-instances=0 \
			--port=8000 \
			--memory=32Gi \
			--cpu=8 \
			--execution-environment=gen2 \
			--add-volume=name=model-volume,type=cloud-storage,bucket=$(BUCKET_NAME),readonly=true \
			--add-volume-mount=volume=model-volume,mount-path=/mnt/models \
			--args='^|^mkdir -p /tmp/model && echo "⌛ Copying model weights from GCS FUSE to local tmpfs..." && cp -r /mnt/models/$(MODEL_PATH)/* /tmp/model/ && echo "✅ Copy complete. Starting vLLM..." && exec python3 -m vllm.entrypoints.openai.api_server --model=/tmp/model --dtype=bfloat16 --quantization=$(QUANTIZATION) --max-model-len=16384 --disable-chunked-mm-input --gpu-memory-utilization=0.90 --kv-cache-dtype=fp8 --tensor-parallel-size=1 --max-num-seqs=8 --enable-chunked-prefill --max-num-batched-tokens=4096 --enable-auto-tool-choice --tool-call-parser=gemma4 --reasoning-parser=gemma4 --async-scheduling --enforce-eager --limit-mm-per-prompt="{\"image\":0,\"audio\":0,\"video\":0}" --host=0.0.0.0 --port=8000' \
			--no-allow-unauthenticated \
			--region=$(REGION) \
			--set-env-vars=VLLM_ENABLE_CUDA_COMPATIBILITY=1,VLLM_USE_V1=0,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1; \
	fi

# Deploy the vLLM inference stack
deploy: deploy-vllm

# Destroy the vLLM inference stack
destroy-vllm:
	@echo "🗑️  Destroying Cloud Run service $(SERVICE_NAME) in $(REGION)..."
	gcloud run services delete $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--region=$(REGION) \
		--quiet

destroy: destroy-vllm

# Check status of the Cloud Run service
status:
	@echo "🔍 Checking status of Cloud Run service $(SERVICE_NAME) in $(REGION)..."
	gcloud run services describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--region=$(REGION)

# Get the endpoint URL of the deployed Cloud Run service
endpoint:
	@gcloud run services describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--region=$(REGION) \
		--format='value(status.url)'

PROMPT ?= "What is Site Reliability Engineering?"

# Query the vLLM model via the Cloud Run endpoint
query:
	@echo "🔍 Querying $(SERVICE_NAME) with prompt: \"$(PROMPT)\"..."
	@ENDPOINT=$$(gcloud run services describe $(SERVICE_NAME) --project=$(PROJECT_ID) --region=$(REGION) --format='value(status.url)'); \
	TOKEN=$$(gcloud auth print-identity-token); \
	curl -s -X POST "$$ENDPOINT/v1/completions" \
		-H "Authorization: Bearer $$TOKEN" \
		-H "Content-Type: application/json" \
		-d '{"model": "/mnt/models/$(MODEL_PATH)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test clean deploy-vllm deploy destroy-vllm destroy status endpoint query
