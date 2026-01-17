FROM python:3.10-slim

WORKDIR /app

# system deps required by OpenCV/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Respect Railway's PORT env var (defaults to 8000 locally)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
