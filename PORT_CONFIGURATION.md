# Port Configuration Documentation

This document provides a comprehensive overview of port configurations across all services in the Federated Agentic Defense system.

## Network Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Federated Network (Docker)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐                                          │
│  │   FL Server       │                                          │
│  │   Port: 9090      │◄─────────────┐                          │
│  └───────────────────┘               │                          │
│            ▲                          │                          │
│            │                          │                          │
│            │ /api/submit_update       │                          │
│            │ /api/broadcast/*         │                          │
│            │                          │                          │
│  ┌─────────┴─────────┐     ┌─────────┴─────────┐               │
│  │ IDS Agent Org A   │     │ IDS Agent Org B   │               │
│  │ Port: 5000        │     │ Port: 5000        │    ...        │
│  │                   │     │                   │               │
│  │ FL_SERVER_URL:    │     │ FL_SERVER_URL:    │               │
│  │ fl-server:9090    │     │ fl-server:9090    │               │
│  └───────────────────┘     └───────────────────┘               │
│            ▲                          ▲                          │
│            │                          │                          │
│            │ POST /detect             │ POST /detect            │
│            │                          │                          │
│  ┌─────────┴─────────┐     ┌─────────┴─────────┐               │
│  │ Pkt Stream Org A  │     │ Pkt Stream Org B  │               │
│  │ (Client only)     │     │ (Client only)     │    ...        │
│  │                   │     │                   │               │
│  │ API_URL:          │     │ API_URL:          │               │
│  │ ids-agent-org-a:  │     │ ids-agent-org-b:  │               │
│  │     5000/detect   │     │     5000/detect   │               │
│  └───────────────────┘     └───────────────────┘               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Service Port Mapping

### FL Server (Federated Learning Server)
- **Port**: 9090
- **Configuration**: 
  - Environment variable: `PORT` (default: 9090)
  - File: `fl-server/app.py` (line 68)
  - Docker Compose: `PORT=9090`
- **Endpoints**:
  - `/api/register` - Agent registration
  - `/api/submit_update` - Submit model updates
  - `/api/broadcast/model` - Fetch global model
  - `/api/broadcast/signatures` - Fetch signatures

### Agentic IDS Local
- **Port**: 5000
- **Configuration**:
  - Environment variable: `PORT` (default: 5000)
  - File: `agentic-ids-local/src/main.py` (line 13)
  - Docker Compose: `PORT=5000`, `AGENT_PORT=5000`
- **Endpoints**:
  - `/health` - Health check
  - `/detect` - Packet detection endpoint
- **FL Server Connection**:
  - Environment variable: `FL_SERVER_URL`
  - Default: `http://fl-server:9090`
  - File: `agentic-ids-local/src/agents/Orchestrator/orchestrator.py` (line 32)

### Packet Streamer
- **No listening port** (client only)
- **Configuration**:
  - Environment variable: `API_URL`
  - Docker Compose: `http://ids-agent-org-{a,b,c}:5000/detect`
  - File: `pkt-streamer/main.py` (line 15)
- **Connects to**: Agentic IDS Local `/detect` endpoint

## Docker Networking

When running in Docker Compose, services communicate using Docker's internal DNS:
- Services reference each other by service name (e.g., `fl-server`, `ids-agent-org-a`)
- Not by `localhost` or `127.0.0.1`

### Service Name Resolution
```
fl-server          → Available at http://fl-server:9090
ids-agent-org-a    → Available at http://ids-agent-org-a:5000
ids-agent-org-b    → Available at http://ids-agent-org-b:5000
ids-agent-org-c    → Available at http://ids-agent-org-c:5000
```

## Connection Flow

```
pkt-streamer-org-a  →  ids-agent-org-a:5000  →  fl-server:9090
pkt-streamer-org-b  →  ids-agent-org-b:5000  →  fl-server:9090
pkt-streamer-org-c  →  ids-agent-org-c:5000  →  fl-server:9090
```

## Common Issues

### Issue: "Connection refused" errors
**Cause**: Using `localhost` instead of Docker service names
**Solution**: 
- In Docker: Use service names from docker-compose.yml
- Local development: Use `localhost` or `127.0.0.1`

### Issue: Port conflicts
**Cause**: Multiple services trying to bind to the same port
**Solution**: Each service instance must use unique ports or run in isolated containers

## Environment Variables Summary

| Service | Variable | Default | Purpose |
|---------|----------|---------|---------|
| fl-server | `PORT` | 9090 | FL server listening port |
| agentic-ids-local | `PORT` | 5000 | IDS agent listening port |
| agentic-ids-local | `FL_SERVER_URL` | `http://fl-server:9090` | FL server endpoint |
| pkt-streamer | `API_URL` | N/A | IDS agent endpoint to send packets |
| pkt-streamer | `API_TIMEOUT` | 0.5 | Request timeout in seconds |

## Local Development vs Docker

### Local Development
When running services locally (outside Docker), use `localhost`:
```bash
# FL Server
PORT=9090 python fl-server/app.py

# Agentic IDS Local
PORT=5000 FL_SERVER_URL=http://localhost:9090 python agentic-ids-local/src/main.py

# Packet Streamer
API_URL=http://localhost:5000/detect python pkt-streamer/main.py
```

### Docker Compose
When using Docker Compose, service names are used automatically via environment variables defined in `docker-compose.yml`.

## Verifying Connectivity

To test connectivity between services:

1. **Check FL Server is running**:
   ```bash
   curl http://fl-server:9090/api/register
   # or locally: curl http://localhost:9090/api/register
   ```

2. **Check Agentic IDS Local is running**:
   ```bash
   curl http://ids-agent-org-a:5000/health
   # or locally: curl http://localhost:5000/health
   ```

3. **Test packet streaming**:
   Check pkt-streamer logs for successful POST requests to the IDS agent.

## Troubleshooting

### Quick Diagnostics

Use the provided tools to quickly diagnose issues:

```bash
# 1. Check port configuration
./check_ports.sh

# 2. Validate configuration consistency
python test_port_config.py

# 3. Test live connectivity (when services are running)
./test_connectivity.sh
```

### Detailed Troubleshooting

1. **Check if port is already in use**:
   ```bash
   lsof -i :9090  # Check FL server port
   lsof -i :5000  # Check IDS agent port
   ```

2. **Verify Docker network connectivity**:
   ```bash
   docker-compose exec ids-agent-org-a ping fl-server
   ```

3. **Check service logs**:
   ```bash
   docker-compose logs fl-server
   docker-compose logs ids-agent-org-a
   docker-compose logs pkt-streamer-org-a
   ```
