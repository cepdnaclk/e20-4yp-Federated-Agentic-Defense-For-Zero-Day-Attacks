# Docker Compose Configuration Guide

This document provides details about the Docker Compose setup for the Federated Agentic Defense system.

## Services Overview

The system consists of three main services:

### 1. agentic-ids-local
- **Purpose**: Local Intrusion Detection System Agent that receives network flow data and performs anomaly detection
- **Container Port**: 5000
- **Host Port**: 5000
- **Network**: ids-network
- **Endpoints**:
  - `POST /detect` - Receives network flow features for anomaly detection
  - `GET /health` - Health check endpoint
- **Dependencies**: None
- **Health Check**: Uses curl to check `/health` endpoint

### 2. fl-server (Federated Learning Server)
- **Purpose**: Global aggregator server for Zero-Day attack intelligence sharing
- **Container Port**: 8000
- **Host Port**: 8000
- **Network**: ids-network
- **Endpoints**:
  - `GET /` - Root endpoint (health check)
  - `POST /upload_update` - Receives intelligence updates from local agents
  - `GET /get_global_model` - Retrieves aggregated model
  - `GET /statistics` - Views knowledge base statistics
- **Dependencies**: None
- **Health Check**: Uses curl to check `/` endpoint
- **Volumes**: `./federation_simulation/knowledge_base:/app/knowledge_base` (persists knowledge base)

### 3. pkt-streamer
- **Purpose**: Streams UNSW-NB15 dataset features to the IDS agent for testing
- **Network**: ids-network
- **Exposed Ports**: None (client-only service)
- **Dependencies**: 
  - Waits for `agentic-ids-local` to be healthy before starting
- **Environment Variables**:
  - `CSV_PATH=/data/UNSW-NB15_1.csv`
  - `FEATURES_METADATA=/data/dataset_features.json`
  - `API_URL=http://agentic-ids-local:5000/detect`
  - `API_TIMEOUT=0.5`
  - `STREAM_DELAY=3`
  - `DROP_COLUMNS=attack_cat,Label`
- **Volumes**: `./pkt-streamer/data:/data:ro` (read-only dataset mount)

## Network Configuration

All services are connected via a custom bridge network named `ids-network`. This allows:
- Service discovery by service name (e.g., `http://agentic-ids-local:5000`)
- Isolation from other Docker networks
- Inter-container communication

## Port Mappings

| Service | Container Port | Host Port | Protocol |
|---------|---------------|-----------|----------|
| agentic-ids-local | 5000 | 5000 | TCP |
| fl-server | 8000 | 8000 | TCP |
| pkt-streamer | N/A | N/A | N/A |

## Service Communication Flow

```
pkt-streamer --> http://agentic-ids-local:5000/detect
                      |
                      v
               agentic-ids-local (Anomaly Detection)
                      |
                      v (Zero-Day detected)
               http://fl-server:8000/upload_update
```

## Prerequisites

1. **Dataset**: Place the UNSW-NB15 dataset CSV file in `./pkt-streamer/data/` directory
   - File should be named: `UNSW-NB15_1.csv`
   - Feature metadata is already provided: `dataset_features.json`

2. **Docker**: Docker and Docker Compose must be installed
   - Docker version: 20.10+
   - Docker Compose version: 2.0+

## Usage

### Build all services
```bash
docker compose build
```

### Start all services
```bash
docker compose up
```

### Start services in detached mode
```bash
docker compose up -d
```

### View logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f agentic-ids-local
docker compose logs -f fl-server
docker compose logs -f pkt-streamer
```

### Stop services
```bash
docker compose down
```

### Check service health
```bash
# Check agentic-ids-local
curl http://localhost:5000/health

# Check fl-server
curl http://localhost:8000/
```

## Troubleshooting

### Service Won't Start
1. Check logs: `docker compose logs <service-name>`
2. Verify ports are not in use: `netstat -tulpn | grep -E '5000|8000'`
3. Ensure dataset files exist in `./pkt-streamer/data/`

### pkt-streamer Fails to Connect
- Verify `agentic-ids-local` is healthy: `docker compose ps`
- Check network connectivity: `docker compose exec pkt-streamer ping agentic-ids-local`

### fl-server Knowledge Base Not Persisting
- Ensure the directory exists: `mkdir -p ./federation_simulation/knowledge_base`
- Check volume mount: `docker compose config | grep knowledge_base`

## Configuration Updates

### Changing Ports
To change exposed ports, edit `docker-compose.yml`:
```yaml
services:
  agentic-ids-local:
    ports:
      - "NEW_HOST_PORT:5000"  # Keep container port as 5000
```

### Adjusting Stream Delay
Edit environment variables in `docker-compose.yml`:
```yaml
environment:
  - STREAM_DELAY=5  # Increase delay between packets
```

## Health Checks

All services have health checks configured:
- **Interval**: 10 seconds
- **Timeout**: 5 seconds
- **Retries**: 5

Services are considered healthy when their health check endpoints respond successfully.
