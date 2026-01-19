#!/usr/bin/env bash
set -e

echo "=== OpenVINO Model Server: setup + start ==="

# --------------------------------------------------
# Config
# --------------------------------------------------
OVMS_IMAGE="openvino/model_server:latest"
MODELS_DIR="$(pwd)/models"
UID_GID="$(id -u):$(id -g)"

LLM_MODEL="OpenVINO/Qwen3-8B-int4-ov"
VLM_MODEL="OpenVINO/Phi-3.5-vision-instruct-int4-ov"

LLM_CONTAINER="ovms-llm"
VLM_CONTAINER="ovms-vlm"

LLM_PORT=8001
VLM_PORT=8002

# --------------------------------------------------
# Helper: stop & remove container if exists
# --------------------------------------------------
cleanup_container () {
  local NAME=$1
  if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "Found existing container '${NAME}', removing it..."
    docker stop "${NAME}" >/dev/null 2>&1 || true
    docker rm "${NAME}" >/dev/null 2>&1 || true
  fi
}

# --------------------------------------------------
# Pull OVMS image
# --------------------------------------------------
echo "Pulling OpenVINO Model Server image..."
docker pull ${OVMS_IMAGE}

# --------------------------------------------------
# Create models directory
# --------------------------------------------------
echo "Preparing models directory: ${MODELS_DIR}"
sudo mkdir -p "${MODELS_DIR}"
sudo chown -R ${UID_GID} "${MODELS_DIR}"
chmod -R 755 "${MODELS_DIR}"

# --------------------------------------------------
# Download Agent LLM (idempotent)
# --------------------------------------------------
echo "Downloading LLM model: ${LLM_MODEL}"
docker run --rm \
  --user ${UID_GID} \
  -v "${MODELS_DIR}:/models" \
  ${OVMS_IMAGE} \
  --pull \
  --model_repository_path /models \
  --source_model ${LLM_MODEL} \
  --task text_generation \
  --tool_parser hermes3

# --------------------------------------------------
# Download Vision Language Model (idempotent)
# --------------------------------------------------
echo "Downloading VLM model: ${VLM_MODEL}"
docker run --rm \
  --user ${UID_GID} \
  -v "${MODELS_DIR}:/models:rw" \
  ${OVMS_IMAGE} \
  --pull \
  --model_repository_path /models \
  --source_model ${VLM_MODEL} \
  --task text_generation \
  --pipeline_type VLM

echo "Model download complete."

# --------------------------------------------------
# Cleanup existing containers (VALIDATION)
# --------------------------------------------------
cleanup_container "${LLM_CONTAINER}"
cleanup_container "${VLM_CONTAINER}"

# --------------------------------------------------
# Start LLM service
# --------------------------------------------------
echo "Starting LLM service on port ${LLM_PORT}..."
docker run -d \
  --name ${LLM_CONTAINER} \
  --user ${UID_GID} \
  -p ${LLM_PORT}:8000 \
  -v "${MODELS_DIR}:/models" \
  ${OVMS_IMAGE} \
  --rest_port 8000 \
  --model_repository_path /models \
  --source_model ${LLM_MODEL} \
  --tool_parser hermes3 \
  --cache_size 2 \
  --task text_generation \
  --enable_prefix_caching true

# --------------------------------------------------
# Start VLM service
# --------------------------------------------------
echo "Starting VLM service on port ${VLM_PORT}..."
docker run -d \
  --name ${VLM_CONTAINER} \
  -p ${VLM_PORT}:8000 \
  -v "${MODELS_DIR}:/models:ro" \
  ${OVMS_IMAGE} \
  --rest_port 8000 \
  --model_name ${VLM_MODEL} \
  --model_path "/models/${VLM_MODEL}"

echo ""
echo "=== OpenVINO Model Server is running ==="
echo "LLM endpoint : http://localhost:${LLM_PORT}"
echo "VLM endpoint : http://localhost:${VLM_PORT}"
