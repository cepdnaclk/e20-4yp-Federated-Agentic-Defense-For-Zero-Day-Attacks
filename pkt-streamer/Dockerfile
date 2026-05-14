# Stage 1: Build stage
FROM python:3.12-slim-bookworm AS builder

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /pkt-streamer

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /pkt-streamer

# --- 1. DEFINE CONFIGURATION ---
ENV CSV_PATH="/data/UNSW-NB15_1.csv"
ENV FEATURES_METADATA="/data/dataset_features.json"
# Note: This is overridden in docker-compose.yml to use the service name
ENV API_URL="http://agentic-ids-local:5000/detect"
ENV API_TIMEOUT=0.5
ENV STREAM_DELAY=3
ENV DROP_COLUMNS="attack_cat,Label"

# --- 2. COPY ASSETS ---
# Copy the virtual environment and the rest of the code
COPY --from=builder /opt/venv /opt/venv
COPY . .

# --- 3. EXECUTION ---
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "main.py"]



