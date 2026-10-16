# Face Detection and Person Recognition System - Docker image
FROM python:3.11-slim

# System libraries required by OpenCV and video I/O (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application scripts and the Haar cascade classifier
COPY *.py haarcascade_frontalface_default.xml ./

# Default command runs the file-based photo face detection.
# Override to run a different script, for example:
#   docker compose run --rm face-detection python Detect-Faces-in-Video.py
#   docker compose run --rm face-detection python Person-Recognition-in-Photo-and-Video.py
# The Real-Time-*.py scripts use a webcam (VideoCapture(0)) and do not work in a container.
CMD ["python", "Detect-Faces-in-Photo.py"]
