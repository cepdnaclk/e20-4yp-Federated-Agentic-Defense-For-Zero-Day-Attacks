# Docker Compose Configuration Testing

This document describes the testing performed to verify the docker-compose configuration and service connectivity.

## Services Overview

The docker-compose.yml defines three services:

1. **fl-server** (Federated Learning Server)
   - Port: 8000
   - Framework: FastAPI
   - Purpose: Global aggregator for attack intelligence sharing

2. **local-agentic-ids** (IDS Agent)
   - Port: 5000
   - Framework: Flask
   - Purpose: Local intrusion detection agent with ML/DL models

3. **pkt-streamer** (Packet Streamer)
   - No exposed port (client service)
   - Purpose: Streams network traffic data to the IDS agent

## Port Configuration

| Service | Internal Port | External Port | Protocol |
|---------|--------------|---------------|----------|
| fl-server | 8000 | 8000 | HTTP (FastAPI) |
| local-agentic-ids | 5000 | 5000 | HTTP (Flask) |
| pkt-streamer | - | - | HTTP Client |

## Service Dependencies

The services start in the following order:
1. **fl-server** starts first
2. **local-agentic-ids** starts after fl-server is running
3. **pkt-streamer** starts after local-agentic-ids is healthy

Dependencies are configured using `depends_on` with health checks:
- pkt-streamer waits for local-agentic-ids to be healthy (responds on /health endpoint)
- local-agentic-ids waits for fl-server to start

## Network Configuration

All services run on the default Docker Compose network, enabling them to communicate using service names:
- pkt-streamer connects to `http://local-agentic-ids:5000/detect`
- local-agentic-ids connects to `http://fl-server:8000/upload_update` and `/get_global_model`

## Build Testing

All services successfully build:

```bash
docker compose build
```

### Build Results:
- ✅ fl-server: Built successfully
- ✅ local-agentic-ids: Built successfully  
- ✅ pkt-streamer: Built successfully

## Configuration Validation

Docker Compose configuration validated:

```bash
docker compose config
```

Output shows:
- ✅ Valid YAML syntax
- ✅ Correct service definitions
- ✅ Proper volume mounts
- ✅ Correct environment variables
- ✅ Valid health check configurations

## Key Fixes Applied

1. **Directory Path Fix**: Changed `./local-agentic-ids` to `./agentic-ids-local` to match actual directory name
2. **Network Communication**: Changed pkt-streamer API_URL from `localhost` to `local-agentic-ids` for Docker networking
3. **FL Server Integration**: Added fl-server service with proper Dockerfile
4. **Port Alignment**: Fixed FL server port from 9090 to 8000 across all components
5. **API Endpoints**: Updated FL client to use correct server endpoints (`/upload_update`, `/get_global_model`)
6. **Health Checks**: Added curl to Dockerfiles for health check support
7. **Service Dependencies**: Enabled proper startup order with health check conditions

## Environment Variables

### fl-server
- `HOST=0.0.0.0`
- `PORT=8000`

### local-agentic-ids
- `PORT=5000`
- `FL_SERVER_URL=http://fl-server:8000`

### pkt-streamer
- `CSV_PATH=/data/UNSW-NB15_1.csv`
- `FEATURES_METADATA=/data/dataset_features.json`
- `API_URL=http://local-agentic-ids:5000/detect`
- `API_TIMEOUT=0.5`
- `STREAM_DELAY=3`
- `DROP_COLUMNS=attack_cat,Label`

## Running the Services

### Prerequisites
1. Download the UNSW-NB15 dataset from [UNSW Research Page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
2. Place `UNSW-NB15_1.csv` in the `./data/` directory at repository root
3. Ensure `dataset_features.json` is in the data directory

### Start Services
```bash
docker compose up
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f local-agentic-ids
docker compose logs -f fl-server
docker compose logs -f pkt-streamer
```

### Stop Services
```bash
docker compose down
```

## Testing Connectivity

### Test fl-server
```bash
curl http://localhost:8000/
# Expected: {"status": "online", "service": "Federated Learning Global Server", ...}

curl http://localhost:8000/statistics
# Expected: Statistics about the knowledge base
```

### Test local-agentic-ids
```bash
curl http://localhost:5000/health
# Expected: {"status": "ok"}

curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"flow_id": 1, "features": {...}}'
# Expected: {"status": "received"}
```

## Notes

- The pkt-streamer service will exit after processing all data from the CSV file
- Data files are mounted read-only to prevent accidental modification
- The fl-server persists knowledge base to `./federation_simulation/knowledge_base/`
- All services use multi-stage Docker builds for optimized image sizes

## Troubleshooting

### Port Already in Use
If you get port conflicts, change the external port mapping in docker-compose.yml:
```yaml
ports:
  - "5001:5000"  # Map external 5001 to internal 5000
```

### Data File Not Found
Ensure the data directory structure is correct:
```
./data/
  ├── UNSW-NB15_1.csv
  └── dataset_features.json
```

### Service Health Check Failing
Check logs for the failing service:
```bash
docker compose logs local-agentic-ids
```

Common issues:
- Missing dependencies
- Port binding errors
- Configuration errors
