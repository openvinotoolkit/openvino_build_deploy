# demos/paint_your_dreams_demo/Dockerfile
# -------------------------------------------------------------
ARG OV_VERSION=2025.2.0
FROM openvino/ubuntu24_dev:${OV_VERSION}

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# ── 1. Minimal extra OS packages ────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev build-essential ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── 2. Copy demo requirements and install Python deps ───────────────────────
COPY demos/paint_your_dreams_demo/requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    \
    # Replace OpenVINO’s OpenCV wheel (built for NumPy 1.26) with one
    # compiled for NumPy ≥ 2.0
    rm -rf /opt/intel/openvino/extras/opencv/python/cv2* && \
    python -m pip install --no-cache-dir --force-reinstall "opencv-python-headless>=4.10" && \
    \
    # openmodelzoo‑modelapi pins NumPy<2; the demo doesn’t need it.
    python -m pip uninstall -y openmodelzoo-modelapi

# ── 3. Copy source code: demo itself + shared utils folder ──────────────────
COPY demos/utils                    /workspace/utils
COPY demos/paint_your_dreams_demo   /workspace/demo

RUN chown -R openvino:openvino /workspace

WORKDIR /workspace/demo

# ── 4. Expose Gradio default port & run the app  as user openvino─────────────────────────────
USER openvino

EXPOSE 7860
ENTRYPOINT ["python", "main.py", "--local_network"]
