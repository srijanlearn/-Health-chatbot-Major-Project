# HealthyPartner v2 — Dockerfile
#
# NOTE: Ollama must run on the HOST machine (not inside this container)
# because it needs direct GPU/CPU access for model inference.
#
# Usage:
#   # On host: start Ollama with models pulled
#   ollama serve &
#   ollama pull qwen3:4b
#   ollama pull qwen2.5:0.5b
#
#   # Build and run container (backend only)
#   docker build -t healthypartner .
#   docker run -p 8000:8000 --env OLLAMA_HOST=http://host.docker.internal:11434 healthypartner
#
#   # Frontend still runs on host (Streamlit + browser access)
#   streamlit run frontend.py

FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

# Data and model cache directories
RUN mkdir -p /app/db /app/downloaded_files /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
