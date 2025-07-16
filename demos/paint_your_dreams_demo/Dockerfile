# Paint-Your-Dreams demo container build
FROM openvino/ubuntu24_dev:latest
USER root

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# --- basic tools -----------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- Intel® NPU user-mode runtime--------
RUN set -ex && \
    mkdir /tmp/npu && \
    dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu \
          intel-level-zero-npu level-zero || true && \
    wget -q https://github.com/oneapi-src/level-zero/releases/download/v1.17.44/level-zero_1.17.44+u24.04_amd64.deb -P /tmp/npu && \
    wget -q --no-check-certificate -nH --accept-regex="ubuntu24" --cut-dirs=5 \
         -r https://github.com/intel/linux-npu-driver/releases/expanded_assets/v1.13.0 \
         -P /tmp/npu && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/npu/*.deb && \
    rm -rf /tmp/npu /var/lib/apt/lists/*

# --- clone the demo --------------------------------------------------------
RUN git clone --depth 1 https://github.com/openvinotoolkit/openvino_build_deploy.git && \
    mv openvino_build_deploy/demos/paint_your_dreams_demo ./demo && \
    mv openvino_build_deploy/demos/utils ./utils && \
    rm -rf openvino_build_deploy


# --- python dependencies ---------------------------------------------------
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r demo/requirements.txt && \
    python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Expose Gradio default port
EXPOSE 7860

WORKDIR /workspace/demo

# Bind to all interfaces so host can reach web ui
ENTRYPOINT ["python", "main.py", "--local_network"]
