ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG INSTALL_TORCH_CUDA="0"
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
ARG TORCH_PACKAGES="torch torchaudio"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIENEU_MODEL_DIR=/models \
    HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models/hub \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgomp1 \
        libsndfile1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip \
    && if [ "$INSTALL_TORCH_CUDA" = "1" ]; then \
         pip install --no-cache-dir $TORCH_PACKAGES --index-url "$TORCH_INDEX_URL"; \
       fi \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir transformers==4.57.6

COPY app.py voices.json ./

EXPOSE 8004

CMD ["sh", "-c", "exec python3 -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8004}"]
