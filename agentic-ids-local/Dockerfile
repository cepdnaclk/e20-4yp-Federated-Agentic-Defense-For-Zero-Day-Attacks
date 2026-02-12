FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a shared builder stage for better caching
FROM base AS builder
WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt constraints.txt ./

# Install large ML packages first for better layer caching
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow-cpu==2.20.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torch==2.10.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -c constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

FROM base AS runtime

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
