#!/usr/bin/env bash
set -e

MODELS_DIR="$(pwd)/models"
UID_GID="$(id -u):$(id -g)"

LLM_MODEL="OpenVINO/Qwen3-8B-int4-ov"
VLM_MODEL="OpenVINO/Phi-3.5-vision-instruct-int4-ov"

LLM_CONTAINER="ovms-llm"
VLM_CONTAINER="ovms-vlm"

LLM_PORT=8001
VLM_PORT=8002

# --------------------------------------------------
# Detect Intel GPU
# --------------------------------------------------
OVMS_IMAGE="openvino/model_server:latest"
GPU_ARGS=""
TARGET_DEVICE_ARG=""

for r in /dev/dri/render*; do
  [ -e "$r" ] || continue

  if [ -f "/sys/class/drm/$(basename "$r")/device/vendor" ] &&
     grep -q "0x8086" "/sys/class/drm/$(basename "$r")/device/vendor"; then

    OVMS_IMAGE="openvino/model_server:latest-gpu"
    GPU_ARGS="--device=/dev/dri --group-add=$(stat -c '%g' "$r")"
    TARGET_DEVICE_ARG="--target_device=GPU"
    break
  fi
done

# --------------------------------------------------
# Prep (NO sudo)
# --------------------------------------------------
docker pull ${OVMS_IMAGE}

mkdir -p "${MODELS_DIR}"
chown -R ${UID_GID} "${MODELS_DIR}" 2>/dev/null || true
chmod -R 755 "${MODELS_DIR}"

docker rm -f ${LLM_CONTAINER} ${VLM_CONTAINER} >/dev/null 2>&1 || true

# --------------------------------------------------
# Download models
# --------------------------------------------------
docker run --rm \
  --user ${UID_GID} \
  -v "${MODELS_DIR}:/models" \
  ${OVMS_IMAGE} \
  --pull \
  --model_repository_path /models \
  --source_model ${LLM_MODEL} \
  --task text_generation \
  --tool_parser hermes3

docker run --rm \
  --user ${UID_GID} \
  -v "${MODELS_DIR}:/models" \
  ${OVMS_IMAGE} \
  --pull \
  --model_repository_path /models \
  --source_model ${VLM_MODEL} \
  --task text_generation \
  --pipeline_type VLM

# --------------------------------------------------
# Run LLM
# --------------------------------------------------
docker run -d \
  ${GPU_ARGS} \
  --name ${LLM_CONTAINER} \
  --user ${UID_GID} \
  -p ${LLM_PORT}:8000 \
  -v "${MODELS_DIR}:/models:rw" \
  ${OVMS_IMAGE} \
  --rest_port 8000 \
  --model_repository_path /models \
  --source_model ${LLM_MODEL} \
  --task text_generation \
  --tool_parser hermes3 \
  --cache_size 2 \
  --enable_prefix_caching true \
  ${TARGET_DEVICE_ARG}

# --------------------------------------------------
# Run VLM
# --------------------------------------------------
docker run -d \
  ${GPU_ARGS} \
  --name ${VLM_CONTAINER} \
  --user ${UID_GID} \
  -p ${VLM_PORT}:8000 \
  -v "${MODELS_DIR}:/models:ro" \
  ${OVMS_IMAGE} \
  --rest_port 8000 \
  --model_name ${VLM_MODEL} \
  --model_path /models/${VLM_MODEL} \
  ${TARGET_DEVICE_ARG}

sleep 2

# --------------------------------------------------
# Validate containers
# --------------------------------------------------
if docker ps --format '{{.Names}}' | grep -qx "${LLM_CONTAINER}" && \
   docker ps --format '{{.Names}}' | grep -qx "${VLM_CONTAINER}"; then

  echo ""
  echo "=============================================="
  echo "✅ SUCCESS: OpenVINO Model Server is running"
  echo "----------------------------------------------"
  echo "LLM : http://localhost:${LLM_PORT}"
  echo "VLM : http://localhost:${VLM_PORT}"
  echo "=============================================="
  echo ""

else
  echo ""
  echo "##############################################"
  echo "❌❌❌  ERROR: OVMS FAILED  ❌❌❌"
  echo "----------------------------------------------"
  echo "One or more containers are NOT running:"
  echo ""
  echo "Running containers:"
  docker ps
  echo ""
  exit 1
fi

