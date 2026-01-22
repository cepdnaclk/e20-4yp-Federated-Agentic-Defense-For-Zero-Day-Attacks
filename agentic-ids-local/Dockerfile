# Stage 1: Build
FROM python:3.12-slim-bookworm AS builder

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /local-agentic-ids

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file to leverage Docker's layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /local-agentic-ids

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy your application source code
COPY . .

# Set the working directory to where your main script resides
WORKDIR /local-agentic-ids/src
CMD ["python", "main.py"]