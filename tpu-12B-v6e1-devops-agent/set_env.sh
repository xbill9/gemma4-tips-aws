#!/bin/bash

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Error: No active gcloud account found."
    echo "Please run 'gcloud auth login' and try again."
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "(unset)" ]; then
    if [ -f "$HOME/project_id.txt" ]; then
        PROJECT_ID=$(cat "$HOME/project_id.txt")
    fi
fi

if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "(unset)" ]; then
    echo "Warning: No gcloud project is currently set."
    echo "Run 'gcloud config set project [PROJECT_ID]' to configure it."
fi

gcloud config set project $PROJECT_ID


if [ -f "$HOME/gemini.key" ]; then
    GOOGLE_API_KEY=$(cat "$HOME/gemini.key")
else
    read -p "Enter Gemini KEY: " GOOGLE_API_KEY
    echo "$GOOGLE_API_KEY" > "$HOME/gemini.key"
fi

export MCP_SERVER_URL=https://mcp-https-python-wgcq55zbfq-rj.a.run.app/mcp

cat <<EOF > .env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=us-east5
GOOGLE_CLOUD_REGION=us-east5
GOOGLE_CLOUD_ZONE=us-east5-a
DEFAULT_RESOURCE_ID=node-1
MODEL=google/gemma-4-12B-it
MODEL_NAME=google/gemma-4-12B-it
GENAI_MODEL="gemini-2.5-flash"
GOOGLE_API_KEY=$GOOGLE_API_KEY
GEMINI_API_KEY=$GOOGLE_API_KEY
MCP_SERVER_URL=$MCP_SERVER_URL
ACCELERATOR_TYPE=v6e-1
TENSOR_PARALLEL_SIZE=1
EOF

echo "Sourcing Env"
source .env

echo "Current Environment"
cat .env

echo "Cloud Login"
gcloud auth list

echo "Config List"
gcloud config list

echo "ADK Version"
adk --version
