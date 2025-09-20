#!/bin/bash

SOURCE="src"
CHART="chart"
if [ ! -f "${SOURCE}/secrets/browser.auth" ]; then
  bash ${SOURCE}/secrets/generate_secrets.sh
fi

if [ ! -f .env ]; then
  touch .env
fi

USER_UID=$(stat -c '%u' "${SOURCE}"/* | sort -rn | head -1)
USER_GID=$(stat -c '%g' "${SOURCE}"/* | sort -rn | head -1)

echo "UID=$USER_UID" > .env
echo "GID=$USER_GID" >> .env

if [ ! -d "${SOURCE}/dlstreamer-pipeline-server/videos" ] || [ -z "$(find "${SOURCE}/dlstreamer-pipeline-server/videos" -type f -name "*.ts" 2>/dev/null)" ]; then
  VIDEO_URL="https://github.com/intel/metro-ai-suite/raw/refs/heads/videos/videos"
  VIDEOS=("1122east.ts" "1122west.ts" "1122north.ts" "1122south.ts")
  VIDEO_DIR="${SOURCE}/dlstreamer-pipeline-server/videos"

  mkdir -p "${VIDEO_DIR}"
  for VIDEO in "${VIDEOS[@]}"; do
    echo "Downloading ${VIDEO}..."
    curl -L "${VIDEO_URL}/${VIDEO}" -o "${VIDEO_DIR}/${VIDEO}"
    if [ ! -f "${VIDEO_DIR}/${VIDEO}" ]; then
        echo "Error: Failed to download ${VIDEO}"
        exit 1
    fi
  done
fi

# Copy files to chart
mkdir -p ${CHART}/files
mkdir -p ${CHART}/files/dlstreamer-pipeline-server/user_scripts/gvapython/sscape
mkdir -p ${CHART}/files/webserver
cp -r \
  ${SOURCE}/controller \
  ${SOURCE}/grafana \
  ${SOURCE}/mosquitto \
  ${SOURCE}/node-red \
  ${SOURCE}/secrets \
  ${CHART}/files
cp -r \
  ${SOURCE}/dlstreamer-pipeline-server/user_scripts \
  ${CHART}/files/dlstreamer-pipeline-server
cp \
  ${SOURCE}/dlstreamer-pipeline-server/config.json \
  ${CHART}/files/dlstreamer-pipeline-server/config.json
cp \
  ${SOURCE}/webserver/user_access_config.json \
  ${CHART}/files/webserver/user_access_config.json
