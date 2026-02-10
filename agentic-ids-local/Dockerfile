FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt constraints.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    -r requirements.txt \
    -c constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    SENTENCE_TRANSFORMERS_HOME=/models

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src ./src

WORKDIR /app/src
CMD ["python", "main.py"]
