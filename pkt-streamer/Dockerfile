# Stage 1: Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Optimized uv settings
ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy

WORKDIR /pkt-streamer

# Install dependencies first (better caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the source code
COPY . .

# Final sync to install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# Stage 2: Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /pkt-streamer

# --- 1. DEFINE CONFIGURATION ---
# These match your config.env. 
# You can override these when running: docker run -e STREAM_DELAY=0.5 ...
ENV CSV_PATH="/data/UNSW-NB15_1.csv"
ENV FEATURES_METADATA="/data/dataset_features.json"
ENV API_URL="http://localhost:5000/detect"
ENV API_TIMEOUT=0.5
ENV STREAM_DELAY=3
ENV DROP_COLUMNS="attack_cat,Label"

# --- 2. COPY ASSETS ---
# Copy the virtual environment and app code from the builder stage
COPY --from=builder /pkt-streamer /pkt-streamer

# --- 3. EXECUTION ---
# Ensure the virtual environment is used
ENV PATH="/pkt-streamer/.venv/bin:$PATH"

# Run the script
CMD ["python", "main.py"]