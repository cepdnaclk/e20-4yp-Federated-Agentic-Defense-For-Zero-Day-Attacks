# Federated Agentic Defense For Zero-Day Attacks

An intelligent, federated system for detecting and mitigating zero-day network attacks using agentic AI and federated learning.

## Architecture Overview

This project consists of three main components:

1. **FL Server** (Federated Learning Server) - Coordinates federated learning across multiple agents
2. **Agentic IDS Local** - Local Intrusion Detection System with AI agents
3. **Packet Streamer** - Streams network packet data for analysis

## Quick Start

### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

This will start all services:
- FL Server on port 9090
- Three IDS agents (org-a, org-b, org-c) each on internal port 5000
- Three packet streamers (org-a, org-b, org-c)

### Port Configuration

| Service | Port | Description |
|---------|------|-------------|
| FL Server | 9090 | Federated Learning coordination server |
| Agentic IDS Local | 5000 | Local IDS agent API endpoint |
| Packet Streamer | N/A | Client-only (sends data to IDS) |

For detailed port configuration information, see [PORT_CONFIGURATION.md](PORT_CONFIGURATION.md).

### Verifying Configuration

Run the port configuration checker:

```bash
./check_ports.sh
```

This script will:
- Check if expected ports are listening
- Verify Docker Compose configuration
- Scan configuration files
- Test connectivity (if services are running)

## Service Communication

```
Packet Streamer → Agentic IDS Local (port 5000) → FL Server (port 9090)
```

- **Packet Streamer** sends network flow data to the IDS agent
- **Agentic IDS Local** processes flows and sends signatures to FL server
- **FL Server** aggregates updates and broadcasts global model

## Components

### FL Server
- Implements FedAvg aggregation algorithm
- Drift detection for anomaly patterns
- Zero-day attack classification
- Signature knowledge base with versioning

### Agentic IDS Local
- Multi-agent orchestration system
- A1: Triage Agent - Initial classification
- A2: Suspicious Agent - Known attack mitigation
- A3: Federation Agent - Federated learning client
- Autoencoder-based anomaly detection

### Packet Streamer
- Streams network packet data from CSV files
- Configurable stream delay and API timeout
- Feature extraction and preprocessing

## Configuration

### Environment Variables

See [PORT_CONFIGURATION.md](PORT_CONFIGURATION.md) for complete environment variable documentation.

Key variables:
- `PORT` - Service listening port
- `FL_SERVER_URL` - URL of the FL server
- `API_URL` - IDS agent endpoint for packet streamer
- `ORG_ID` - Organization identifier

## Development

### Local Development

When running services locally (not in Docker):

1. **Start FL Server**:
   ```bash
   cd fl-server
   PORT=9090 python app.py
   ```

2. **Start Agentic IDS**:
   ```bash
   cd agentic-ids-local/src
   PORT=5000 FL_SERVER_URL=http://localhost:9090 python main.py
   ```

3. **Start Packet Streamer**:
   ```bash
   cd pkt-streamer
   API_URL=http://localhost:5000/detect python main.py
   ```

### Docker Development

```bash
# Rebuild specific service
docker-compose build fl-server

# View logs
docker-compose logs -f fl-server
docker-compose logs -f ids-agent-org-a

# Stop all services
docker-compose down
```

## Troubleshooting

### Connection Issues

If services can't connect to each other:

1. **Check port configuration**: Run `./check_ports.sh`
2. **Verify Docker network**: Ensure all services are on the `federated-net` network
3. **Check logs**: Use `docker-compose logs <service-name>`
4. **Test endpoints**:
   ```bash
   # FL Server health
   curl http://localhost:9090/api/register
   
   # IDS Agent health
   curl http://localhost:5000/health
   ```

### Port Conflicts

If you see "port already in use" errors:

```bash
# Find what's using the port
lsof -i :9090
lsof -i :5000

# Or use netstat
netstat -tuln | grep :9090
```

## Documentation

- [PORT_CONFIGURATION.md](PORT_CONFIGURATION.md) - Detailed port and connectivity documentation
- [architecture.md](architecture.md) - System architecture details

## Contributing

Please refer to the project guidelines at:
https://projects.ce.pdn.ac.lk/docs/how-to-add-a-project

