# Dockerfile
# ArthaSathi LLM — Production Docker Container
# Build: docker build -t arthasathi:latest .
# Run:   docker run -p 8000:8000 --gpus all -v $(pwd)/models:/app/models arthasathi:latest

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
    ffmpeg \
    git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure model directories exist
RUN mkdir -p /app/checkpoints /app/rag_index /app/arthasathi_tokenizer

# Environment variables (override at runtime)
ENV MODEL_PATH=/app/checkpoints/finetune/final_ft.pt
ENV TOKENIZER_DIR=/app/arthasathi_tokenizer
ENV RAG_INDEX_PATH=/app/rag_index
ENV DB_PATH=/app/arthasathi_users.db
ENV DEVICE=cuda

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
