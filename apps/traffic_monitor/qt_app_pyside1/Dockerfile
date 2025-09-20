# Dockerfile for qt_app_pyside1 (optimized)
FROM python:3.10-slim

# Install system dependencies for OpenCV, PySide6, OpenVINO, X11 GUI, and supervisor
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

# Copy requirements and install dependencies first for caching
COPY requirements_enhanced.txt ./requirements_enhanced.txt
RUN pip install --no-cache-dir -r requirements_enhanced.txt

# Copy all source code and models
COPY . .

# Copy supervisor config
COPY supervisord.conf /etc/supervisord.conf

# Make start.sh executable
RUN chmod +x start.sh

# Expose display for X11 and logs
ENV DISPLAY=:99
VOLUME ["/app/logs"]

# Use supervisor to run Xvfb and app together, with logging
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
