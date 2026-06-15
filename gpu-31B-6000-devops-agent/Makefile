# Configuration - Update these or override via environment variables
PROJECT_ID ?= aisprint-491218
REGION ?= us-central1
BUCKET_NAME ?= $(PROJECT_ID)-bucket
SERVICE_NAME ?= gpu-31b-6000-devops-agent
MODEL_PATH ?= gemma-4-31B-it


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

# Recommended deployment settings for vLLM on Cloud Run with NVIDIA RTX PRO 6000 GPU
# Requires 'gcloud beta' component and 'gen2' execution environment
deploy-vllm:
	@echo "🚀 Deploying vLLM to Cloud Run ($(SERVICE_NAME)) in $(REGION)..."
	gcloud beta run deploy $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--image=vllm/vllm-openai:latest \
		--gpu=1 \
		--gpu-type=nvidia-rtx-pro-6000 \
		--no-gpu-zonal-redundancy \
		--no-cpu-throttling \
		--concurrency=16 \
		--timeout=3600 \
		--startup-probe=tcpSocket.port=8080,initialDelaySeconds=240,failureThreshold=120,timeoutSeconds=10,periodSeconds=15 \
		--max-instances=3 \
		--min-instances=1 \
		--port=8080 \
		--memory=80Gi \
		--cpu=20 \
		--cpu-boost \
		--execution-environment=gen2 \
		--command=bash \
		--add-volume=name=model-volume,type=cloud-storage,bucket=$(BUCKET_NAME),readonly=true,mount-options='uid=1001;gid=1001' \
		--add-volume-mount=volume=model-volume,mount-path=/mnt/models \
		--set-env-vars=MODEL_NAME=/mnt/models/$(MODEL_PATH),GOOGLE_CLOUD_PROJECT=$(PROJECT_ID),GOOGLE_CLOUD_REGION=$(REGION) \
		--args='^;^-c;vllm serve /mnt/models/$(MODEL_PATH) --served-model-name $(SERVICE_NAME) --enable-log-requests --enable-chunked-prefill --enable-prefix-caching --generation-config auto --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --dtype bfloat16 --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 8 --gpu-memory-utilization 0.95 --tensor-parallel-size 1 --load-format runai_streamer --port 8080 --host 0.0.0.0 --max-model-len 32767' \
		--no-allow-unauthenticated \
		--region=$(REGION)

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
		-d '{"model": "$(SERVICE_NAME)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test clean deploy-vllm deploy destroy-vllm destroy status endpoint query
