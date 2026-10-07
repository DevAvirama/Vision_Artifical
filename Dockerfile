FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY . .

EXPOSE 8001

CMD ["uvicorn", "visionArtificial:app", "--host", "0.0.0.0", "--port", "8001"]
