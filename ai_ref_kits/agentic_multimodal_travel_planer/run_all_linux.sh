#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_SCRIPT="${SCRIPT_DIR}/download_and_run_models_linux.sh"
MCP_SCRIPT="${SCRIPT_DIR}/start_mcp_servers.py"
AGENTS_SCRIPT="${SCRIPT_DIR}/start_agents.py"
UI_SCRIPT="${SCRIPT_DIR}/start_ui.py"

STOP_MODE=0
SKIP_MODELS=0
SKIP_MCP=0
SKIP_AGENTS=0
SKIP_UI=0
MODEL_ARGS=()

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS] [[--] <model-script-args>...]

Unified launcher for:
  1) download_and_run_models_linux.sh
  2) start_mcp_servers.py
  3) start_agents.py
  4) start_ui.py

Options:
  -h, --help         Show this help message
  -s, --stop         Stop agents, MCP servers, and OVMS containers
  --skip-models      Skip OVMS model/container startup
  --skip-mcp         Skip MCP server startup
  --skip-agents      Skip agent startup
  --skip-ui          Skip Gradio UI startup

Model script arguments:
  Any token that is not a launcher option above is forwarded to
  download_and_run_models_linux.sh (same as run_all_windows.bat).
  Optional '--' ends launcher parsing; everything after it is forwarded as-is.

Examples:
  $0
  $0 --device CPU --llm-port 9001 --vlm-port 9002
  $0 -- --device CPU --llm-port 9001 --vlm-port 9002
  $0 --skip-ui
  $0 --stop
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: Required file not found: ${path}"
    exit 1
  fi
}

run_python() {
  if command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  elif command -v python >/dev/null 2>&1; then
    python "$@"
  else
    echo "ERROR: python3/python not found in PATH."
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -s|--stop)
      STOP_MODE=1
      shift
      ;;
    --skip-models)
      SKIP_MODELS=1
      shift
      ;;
    --skip-mcp)
      SKIP_MCP=1
      shift
      ;;
    --skip-agents)
      SKIP_AGENTS=1
      shift
      ;;
    --skip-ui)
      SKIP_UI=1
      shift
      ;;
    --)
      shift
      MODEL_ARGS+=("$@")
      break
      ;;
    *)
      MODEL_ARGS+=("$1")
      shift
      ;;
  esac
done

require_file "${MODELS_SCRIPT}"
require_file "${MCP_SCRIPT}"
require_file "${AGENTS_SCRIPT}"
require_file "${UI_SCRIPT}"

cd "${SCRIPT_DIR}"

if [[ "${STOP_MODE}" -eq 1 ]]; then
  echo "Stopping unified stack..."
  run_python "${AGENTS_SCRIPT}" --stop || true
  run_python "${MCP_SCRIPT}" --stop --kill || true
  bash "${MODELS_SCRIPT}" --stop || true
  echo "All stop commands sent."
  exit 0
fi

if [[ "${SKIP_MODELS}" -eq 0 ]]; then
  echo ""
  echo "=== Step 1/4: Starting OVMS models ==="
  bash "${MODELS_SCRIPT}" "${MODEL_ARGS[@]}"
else
  echo "Skipping OVMS model startup."
fi

if [[ "${SKIP_MCP}" -eq 0 ]]; then
  echo ""
  echo "=== Step 2/4: Starting MCP servers ==="
  run_python "${MCP_SCRIPT}"
else
  echo "Skipping MCP startup."
fi

if [[ "${SKIP_AGENTS}" -eq 0 ]]; then
  echo ""
  echo "=== Step 3/4: Starting agents ==="
  run_python "${AGENTS_SCRIPT}"
else
  echo "Skipping agents startup."
fi

if [[ "${SKIP_UI}" -eq 0 ]]; then
  echo ""
  echo "=== Step 4/4: Starting UI ==="
  echo "Launching Gradio UI in foreground (Ctrl+C to stop UI)."
  run_python "${UI_SCRIPT}"
else
  echo "Skipping UI startup."
fi
