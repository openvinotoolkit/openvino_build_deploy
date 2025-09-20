FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py utils.py ./

# Environment variables
ENV VLM_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
ENV LOCAL_EMBED_MODEL_ID="CLIP-ViT-H-14"
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 8399

# Run the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8399"]
