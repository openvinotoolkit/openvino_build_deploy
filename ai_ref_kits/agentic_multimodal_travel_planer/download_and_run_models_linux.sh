#!/usr/bin/env bash
set -e

# --------------------------------------------------
# Default Config
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
DEVICE=""  # Empty = auto-detect Intel GPU, "CPU" to force CPU, or "GPU", "GPU.0", "GPU.1" etc.

# --------------------------------------------------
# Parse command line arguments
# --------------------------------------------------
show_help() {
  cat << EOF
Usage: $0 [OPTIONS]

Options:
  -h, --help              Show this help message
  -s, --stop              Stop and remove running containers
  
Configuration:
  --llm-model MODEL       LLM model to use (default: ${LLM_MODEL})
  --vlm-model MODEL       VLM model to use (default: ${VLM_MODEL})
  --llm-container NAME    LLM container name (default: ${LLM_CONTAINER})
  --vlm-container NAME    VLM container name (default: ${VLM_CONTAINER})
  --llm-port PORT         LLM server port (default: ${LLM_PORT})
  --vlm-port PORT         VLM server port (default: ${VLM_PORT})
  --models-dir DIR        Models directory (default: ${MODELS_DIR})
  --timeout SECONDS       Timeout in seconds (default: ${TIMEOUT})
  --device DEVICE         Device: CPU, GPU, GPU.0, GPU.1, etc. (default: auto-detect)

Examples:
  # Start with defaults (auto-detect Intel GPU)
  $0

  # Force CPU
  $0 --device CPU

  # Force first GPU
  $0 --device GPU.0

  # Use second GPU
  $0 --device GPU.1

  # Use different models
  $0 --llm-model "OpenVINO/Llama-3.1-8B-int4-ov" --vlm-model "OpenVINO/LLaVA-NeXT-7B-int4-ov"

  # Combine options
  $0 --device GPU.0 --llm-port 9001

  # Stop containers
  $0 --stop

Without options, the script will start the model servers with default configuration.
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    -s|--stop)
      echo "Stopping OpenVINO Model Server containers..."
      docker stop "${LLM_CONTAINER}" "${VLM_CONTAINER}" 2>/dev/null || true
      docker rm "${LLM_CONTAINER}" "${VLM_CONTAINER}" 2>/dev/null || true
      echo "✓ Containers stopped and removed"
      exit 0
      ;;
    --llm-model)
      LLM_MODEL="$2"
      shift 2
      ;;
    --vlm-model)
      VLM_MODEL="$2"
      shift 2
      ;;
    --llm-container)
      LLM_CONTAINER="$2"
      shift 2
      ;;
    --vlm-container)
      VLM_CONTAINER="$2"
      shift 2
      ;;
    --llm-port)
      LLM_PORT="$2"
      shift 2
      ;;
    --vlm-port)
      VLM_PORT="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --device)
      DEVICE=$(echo "$2" | tr '[:lower:]' '[:upper:]')
      # Validate: CPU, GPU, GPU.0, GPU.1, etc.
      if [[ ! "${DEVICE}" =~ ^(CPU|GPU(\.[0-9]+)?)$ ]]; then
        echo "Error: Invalid device '${DEVICE}'. Must be CPU, GPU, GPU.0, GPU.1, etc."
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information."
      exit 1
      ;;
  esac
done

# --------------------------------------------------
# Detect Intel GPU
# --------------------------------------------------
OVMS_IMAGE="openvino/model_server:latest"
GPU_ARGS=""
TARGET_DEVICE_ARG=""
INTEL_GPUS=0

for r in /dev/dri/render*; do
  v="/sys/class/drm/$(basename "$r")/device/vendor"
  [ -f "$v" ] && grep -q 0x8086 "$v" || continue

  INTEL_GPUS=$((INTEL_GPUS+1))
  GID="$(stat -c '%g' "$r")"
done

# Configure device based on flag or auto-detect
if [[ -z "${DEVICE}" ]]; then
  # No flag: auto-detect (current behavior)
  if [ "$INTEL_GPUS" -gt 0 ]; then
    OVMS_IMAGE="openvino/model_server:latest-gpu"
    GPU_ARGS="--device=/dev/dri --group-add=${GID}"
    TARGET_DEVICE_ARG="--target_device GPU.$([ "$INTEL_GPUS" -gt 1 ] && echo 1 || echo 0)"
    ACTUAL_DEVICE="GPU (auto-detected)"
  else
    ACTUAL_DEVICE="CPU (no GPU detected)"
  fi
elif [[ "${DEVICE}" == "CPU" ]]; then
  # Force CPU
  TARGET_DEVICE_ARG="--target_device CPU"
  ACTUAL_DEVICE="CPU (forced)"
elif [[ "${DEVICE}" =~ ^GPU(\.[0-9]+)?$ ]]; then
  # Force GPU with optional index
  if [ "$INTEL_GPUS" -eq 0 ]; then
    echo "ERROR: GPU requested but no Intel GPU detected"
    exit 1
  fi
  OVMS_IMAGE="openvino/model_server:latest-gpu"
  GPU_ARGS="--device=/dev/dri --group-add=${GID}"
  
  # Extract GPU index if specified (e.g., GPU.1 -> 1)
  if [[ "${DEVICE}" == *"."* ]]; then
    GPU_INDEX="${DEVICE#*.}"
    TARGET_DEVICE_ARG="--target_device GPU.${GPU_INDEX}"
    ACTUAL_DEVICE="GPU.${GPU_INDEX} (forced)"
  else
    # Just "GPU" - use first available
    TARGET_DEVICE_ARG="--target_device GPU.0"
    ACTUAL_DEVICE="GPU.0 (forced)"
  fi
fi

# --------------------------------------------------
# Progress bar helpers
# --------------------------------------------------
TOTAL_STEPS=5
CURRENT_STEP=0

show_progress() {
  local step_name="$1"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  local percent=$((CURRENT_STEP * 100 / TOTAL_STEPS))
  local filled=$((percent / 2))
  local empty=$((50 - filled))
  
  printf "\r\033[K"  # Clear line
  printf "Progress: ["
  printf "%${filled}s" | tr ' ' '='
  printf "%${empty}s" | tr ' ' ' '
  printf "] %3d%% - %s" "$percent" "$step_name"
  
  if [ "$CURRENT_STEP" -eq "$TOTAL_STEPS" ]; then
    printf "\n"
  fi
}

# --------------------------------------------------
# Helper: wait for OVMS readiness with progress
# --------------------------------------------------
wait_for_ready() {
  local container="$1"
  local start_ts
  start_ts=$(date +%s)
  local last_hash=""
  local spinner=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
  local spin_idx=0

  while true; do
    if ! docker ps -q -f name="^${container}$" >/dev/null; then
      echo ""
      echo "ERROR: ${container} exited before becoming ready"
      docker logs "${container}" || true
      exit 1
    fi

    logs="$(docker logs --tail 100 "${container}" 2>&1 || true)"
    hash="$(printf '%s' "${logs}" | sha1sum | awk '{print $1}')"

    # Show spinner while waiting
    printf "\r  ${spinner[$spin_idx]} Waiting for ${container}..."
    spin_idx=$(( (spin_idx + 1) % 10 ))

    if [[ "${hash}" != "${last_hash}" ]]; then
      # Show relevant log lines
      local recent_logs
      recent_logs=$(printf '%s\n' "${logs}" \
        | grep -E "Downloading|Fetching|Cloning|Loading|Initializing|model loaded" \
        | tail -1 || true)
      if [ -n "$recent_logs" ]; then
        printf "\r\033[K  ℹ %s\n" "$recent_logs"
      fi
      last_hash="${hash}"
    fi

    if grep -q "REST server listening on port" <<< "${logs}"; then
      printf "\r\033[K  ✓ ${container} is ready\n"
      return 0
    fi

    if (( $(date +%s) - start_ts > TIMEOUT )); then
      echo ""
      echo "ERROR: timeout waiting for ${container}"
      docker logs "${container}" || true
      exit 1
    fi

    sleep 0.5
  done
}

# --------------------------------------------------
# Show Configuration
# --------------------------------------------------
echo ""
echo "=============================================="
echo "  OpenVINO Model Server Setup"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  LLM Model:     ${LLM_MODEL}"
echo "  VLM Model:     ${VLM_MODEL}"
echo "  LLM Container: ${LLM_CONTAINER}"
echo "  VLM Container: ${VLM_CONTAINER}"
echo "  LLM Port:      ${LLM_PORT}"
echo "  VLM Port:      ${VLM_PORT}"
echo "  Models Dir:    ${MODELS_DIR}"
echo "  Timeout:       ${TIMEOUT}s"
echo ""

# --------------------------------------------------
# Update agents_config.yaml
# --------------------------------------------------
CONFIG_FILE="$(dirname "$0")/config/agents_config.yaml"
if [ -f "${CONFIG_FILE}" ]; then
  # Update all api_base URLs with the current LLM port
  sed -i.bak "s|api_base: \"http://127.0.0.1:[0-9]*/v3\"|api_base: \"http://127.0.0.1:${LLM_PORT}/v3\"|g" "${CONFIG_FILE}"
  # Update model names
  sed -i.bak "s|model: \"openai:OpenVINO/[^\"]*\"|model: \"openai:${LLM_MODEL}\"|g" "${CONFIG_FILE}"
  echo "✓ Updated agents_config.yaml with current configuration"
else
  echo "⚠ config/agents_config.yaml not found"
fi

# --------------------------------------------------
# Prep
# --------------------------------------------------

show_progress "Pulling Docker image..."
docker pull "${OVMS_IMAGE}" >/dev/null 2>&1

mkdir -p "${MODELS_DIR}"
chown -R "${UID_GID}" "${MODELS_DIR}" 2>/dev/null || true
chmod -R 755 "${MODELS_DIR}"

docker rm -f "${LLM_CONTAINER}" "${VLM_CONTAINER}" >/dev/null 2>&1 || true

# --------------------------------------------------
# Run containers
# --------------------------------------------------
show_progress "Starting LLM container..."
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
  --log_level DEBUG \
  ${TARGET_DEVICE_ARG} \
  >/dev/null 2>&1

show_progress "Starting VLM container..."
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
  --log_level DEBUG \
  ${TARGET_DEVICE_ARG} \
  >/dev/null 2>&1

# --------------------------------------------------
# Wait for readiness (PARALLEL)
# --------------------------------------------------
show_progress "Waiting for containers to be ready..."
echo ""

wait_for_ready "${LLM_CONTAINER}" &
PID_LLM=$!

wait_for_ready "${VLM_CONTAINER}" &
PID_VLM=$!

wait "${PID_LLM}"
wait "${PID_VLM}"

show_progress "Setup complete!"

# --------------------------------------------------
# Final status
# --------------------------------------------------
echo ""
echo "=============================================="
echo "  ✓ OpenVINO Model Server Ready!"
echo "=============================================="
echo ""
echo "Services available at:"
echo "  • LLM Server: http://localhost:${LLM_PORT}"
echo "  • VLM Server: http://localhost:${VLM_PORT}"
echo ""
echo "To stop the servers, run:"
echo "  $0 --stop"
echo "=============================================="
echo ""
