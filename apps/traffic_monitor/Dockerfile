FROM python:3.10-slim

# Install system dependencies for OpenCV, PySide6, OpenVINO, etc.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    xvfb \
    x11-apps \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY qt_app_pyside/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files and folders actually used by the main app
COPY qt_app_pyside/ ./qt_app_pyside/
COPY main.py ./main.py
COPY config.json ./config.json
COPY detection_openvino.py ./detection_openvino.py
COPY utils.py ./utils.py
COPY yolo11n.pt ./yolo11n.pt
COPY yolo11x.bin ./yolo11x.bin
COPY yolo11x.pt ./yolo11x.pt
COPY yolo11x.xml ./yolo11x.xml

# Set the entrypoint to the main app
CMD ["python", "qt_app_pyside/main.py"]
