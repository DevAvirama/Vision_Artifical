FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para OpenCV y decodificación de stream RTSP (V380 Pro)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir --no-deps opencv-python-headless

COPY best.pt .
COPY visionArtificial.py .
COPY main.py .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]