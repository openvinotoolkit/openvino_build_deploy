ARG OV_VERSION=2025.2.0
FROM openvino/ubuntu24_dev:${OV_VERSION}

ARG DEMO
USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    XAUTHORITY=/home/openvino/.Xauthority

WORKDIR /workspace

#set proxy ENV if needed
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} no_proxy=${no_proxy} \
    HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy} NO_PROXY=${no_proxy}

# (optional, helps apt)
RUN if [ -n "$http_proxy" ]; then \
      printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "$http_proxy" "$https_proxy" \
        > /etc/apt/apt.conf.d/99proxy ; \
    fi


# 1) OS libs + tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates \
        python3-venv python3-dev build-essential \
        ffmpeg libgtk-3-0 libgl1 \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 \
        libxcb-xinerama0 libsm6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# 2) Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# 3) Create venv and install Python deps with uv
#    (No system-site installs; avoid Debian "externally managed" block)
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY --chown=openvino:openvino ${DEMO}/requirements.txt .

# Optional: remove OpenVINO’s bundled cv2 wheel so the venv’s opencv gets used when/if imported via system path
RUN rm -rf /opt/intel/openvino/extras/opencv/python/cv2* || true

# Install project deps into the venv
ENV UV_INDEX_URL=https://pypi.org/simple \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_INDEX_STRATEGY=unsafe-best-match
RUN uv pip install --no-cache -r requirements.txt

# 4) Copy app code
COPY --chown=openvino:openvino utils   /workspace/utils
COPY --chown=openvino:openvino ${DEMO} /workspace/demo
WORKDIR /workspace/demo

USER openvino
EXPOSE 7860
ENTRYPOINT ["python", "main.py"]
