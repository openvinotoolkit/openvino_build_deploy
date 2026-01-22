#!/usr/bin/env bash
set -e

# --------------------------------------------------
# Config
# --------------------------------------------------
MODELS_DIR="$(pwd)/models"
UID_GID="$(id -u):$(id -g)"

LLM_MODEL="OpenVINO/Qwen3-8B-int4-ov"
VLM_MODEL="OpenVINO/Phi-3.5-vision-instruct-int4-ov"

LLM_CONTAINER="ovms-llm"
VLM_CONTAINER="ovms-vlm"

LLM_PORT=8001
VLM_PORT=8002

TIMEOUT=1800

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
    TARGET_DEVICE_ARG="--target_device GPU"
    break
  fi
done

# --------------------------------------------------
# Helper: wait for OVMS readiness with progress
# --------------------------------------------------
wait_for_ready() {
  local container="$1"
  local start_ts
  start_ts=$(date +%s)
  local last_hash=""

  echo "Waiting for ${container}..."

  while true; do
    if ! docker ps -q -f name="^${container}$" >/dev/null; then
      echo "ERROR: ${container} exited before becoming ready"
      docker logs "${container}" || true
      exit 1
    fi

    logs="$(docker logs --tail 100 "${container}" 2>&1 || true)"
    hash="$(printf '%s' "${logs}" | sha1sum | awk '{print $1}')"

    if [[ "${hash}" != "${last_hash}" ]]; then
      printf '%s\n' "${logs}" \
        | grep -E "Downloading|Fetching|Cloning|Loading|Initializing|OpenVINO|HuggingFace" \
        | sed "s/^/[${container}] /" || true
      last_hash="${hash}"
    fi

    if grep -q "REST server listening on port" <<< "${logs}"; then
      echo "${container} is ready"
      return 0
    fi

    if (( $(date +%s) - start_ts > TIMEOUT )); then
      echo "ERROR: timeout waiting for ${container}"
      docker logs "${container}" || true
      exit 1
    fi

    sleep 2
  done
}

# --------------------------------------------------
# Prep
# --------------------------------------------------
docker pull "${OVMS_IMAGE}"

mkdir -p "${MODELS_DIR}"
chown -R "${UID_GID}" "${MODELS_DIR}" 2>/dev/null || true
chmod -R 755 "${MODELS_DIR}"

docker rm -f "${LLM_CONTAINER}" "${VLM_CONTAINER}" >/dev/null 2>&1 || true

# --------------------------------------------------
# Run containers
# --------------------------------------------------
docker run -d \
  ${GPU_ARGS} \
  --name "${LLM_CONTAINER}" \
  --user "${UID_GID}" \
  -e https_proxy="${https_proxy}" \
  -e http_proxy="${http_proxy}" \
  -p "${LLM_PORT}:8000" \
  -v "${MODELS_DIR}:/models:rw" \
  "${OVMS_IMAGE}" \
  --rest_port 8000 \
  --model_repository_path /models \
  --source_model "${LLM_MODEL}" \
  --task text_generation \
  --tool_parser hermes3 \
  ${TARGET_DEVICE_ARG}

docker run -d \
  ${GPU_ARGS} \
  --name "${VLM_CONTAINER}" \
  --user "${UID_GID}" \
  -e https_proxy="${https_proxy}" \
  -e http_proxy="${http_proxy}" \
  -p "${VLM_PORT}:8000" \
  -v "${MODELS_DIR}:/models:rw" \
  "${OVMS_IMAGE}" \
  --rest_port 8000 \
  --source_model "${VLM_MODEL}" \
  --model_repository_path /models \
  --model_name "${VLM_MODEL}" \
  --task text_generation \
  --pipeline_type VLM \
  ${TARGET_DEVICE_ARG} \
  --log_level DEBUG

# --------------------------------------------------
# Wait for readiness (PARALLEL)
# --------------------------------------------------
wait_for_ready "${LLM_CONTAINER}" &
PID_LLM=$!

wait_for_ready "${VLM_CONTAINER}" &
PID_VLM=$!

wait "${PID_LLM}"
wait "${PID_VLM}"

# --------------------------------------------------
# Final status
# --------------------------------------------------
echo ""
echo "=============================================="
echo "OpenVINO Model Server is running on:"
echo "----------------------------------------------"
echo "LLM : http://localhost:${LLM_PORT}"
echo "VLM : http://localhost:${VLM_PORT}"
echo "=============================================="
echo ""
