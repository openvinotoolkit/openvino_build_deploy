ARG OV_VERSION=2025.2.0
FROM openvino/ubuntu24_dev:${OV_VERSION}

ARG DEMO
USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    XAUTHORITY=/home/openvino/.Xauthority

WORKDIR /workspace

# ── 1. OS libs ───────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev build-essential ffmpeg libgtk-3-0 libgl1 \
        # Qt/XCB runtime
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 \
        libxcb-xinerama0 libsm6 libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── 2. Python deps ───────────────────────────────────────────
COPY --chown=openvino:openvino ${DEMO}/requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf /opt/intel/openvino/extras/opencv/python/cv2* && \
    python -m pip install --force-reinstall --no-cache-dir "opencv-python>=4.10" && \
    python -m pip uninstall -y openmodelzoo-modelapi

# ── 3. Copy code ─────────────────────────────────────────────
COPY --chown=openvino:openvino utils   /workspace/utils
COPY --chown=openvino:openvino ${DEMO} /workspace/demo
WORKDIR /workspace/demo

USER openvino
EXPOSE 7860
ENTRYPOINT ["python", "main.py"]
