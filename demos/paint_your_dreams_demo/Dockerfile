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

    
# Intel� NPU drivers
RUN set -ex && \
    mkdir debs && \
    dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu level-zero || true && \
    wget -q https://github.com/oneapi-src/level-zero/releases/download/v1.22.4/level-zero_1.22.4+u24.04_amd64.deb -P ./debs && \
    wget -q --no-check-certificate -nH --accept-regex="ubuntu24" --cut-dirs=5 \
         -r https://github.com/intel/linux-npu-driver/releases/expanded_assets/v1.13.0 \
         -P ./debs && \
    apt-get update && \
    apt-get install -y --no-install-recommends ./debs/*.deb && \
    rm -rf debs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/ssl/certs/Intel*


# --- clone the demo --------------------------------------------------------
RUN git clone --depth 1 https://github.com/openvinotoolkit/openvino_build_deploy.git && \
    mv openvino_build_deploy/demos/paint_your_dreams_demo ./demo && \
    mv openvino_build_deploy/demos/utils ./utils && \
    rm -rf openvino_build_deploy

# --- python dependencies ---------------------------------------------------
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r demo/requirements.txt && \
    rm -rf /opt/intel/openvino/extras/opencv/python/cv2* && \
    python -m pip install --no-cache-dir --force-reinstall "opencv-python-headless>=4.10" && \
    python -m pip uninstall -y openmodelzoo-modelapi


# Expose Gradio default port
EXPOSE 7860

WORKDIR /workspace/demo

# Bind to all interfaces so host can reach web ui
ENTRYPOINT ["python", "main.py", "--local_network"]
