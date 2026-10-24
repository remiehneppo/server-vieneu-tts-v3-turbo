ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG CMAKE_ARGS=""
ARG FORCE_CMAKE=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIENEU_MODEL_DIR=/models \
    HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models/hub \
    CMAKE_ARGS="${CMAKE_ARGS}" \
    FORCE_CMAKE="${FORCE_CMAKE}" \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py voices.json ./

EXPOSE 8004

CMD ["sh", "-c", "exec python3 -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8004}"]
