# Port Configuration Analysis Report

## Executive Summary

A comprehensive analysis was conducted on the port configurations for the three main components of the Federated Agentic Defense system:
1. **FL Server** (Federated Learning Server)
2. **Agentic IDS Local** (Intrusion Detection System)
3. **Packet Streamer** (Network Data Streamer)

**Result**: All port configurations are correct and consistent. No mismatches were found that would prevent connectivity.

## Analysis Details

### Components Scanned

#### 1. FL Server
- **Port**: 9090
- **Configuration Source**: `fl-server/app.py` (line 68)
- **Docker Configuration**: `docker-compose.yml` sets `PORT=9090`
- **Status**: ✓ Correct

#### 2. Agentic IDS Local
- **Port**: 5000
- **Configuration Source**: `agentic-ids-local/src/main.py` (line 13)
- **Docker Configuration**: `docker-compose.yml` sets `PORT=5000`
- **FL Server Connection**: Correctly configured to `http://fl-server:9090`
- **Status**: ✓ Correct

#### 3. Packet Streamer
- **Port**: N/A (client-only component)
- **Target URL**: Configured via `API_URL` environment variable
- **Docker Configuration**: Correctly set to `http://ids-agent-org-{a,b,c}:5000/detect`
- **Local Configuration**: `pkt-streamer/config.env` uses `localhost:5000` (for local development)
- **Status**: ✓ Correct

## Why Components Connect Successfully

### Docker Networking
All services run in the same Docker network (`federated-net`), which provides:
- Internal DNS resolution by service name
- Network isolation from the host
- Service-to-service communication without exposing ports to the host

### Connection Flow
```
Packet Streamer → POST /detect → IDS Agent (port 5000)
IDS Agent → POST /api/submit_update → FL Server (port 9090)
IDS Agent → GET /api/broadcast/* → FL Server (port 9090)
```

### Environment Variable Hierarchy
Docker Compose environment variables override any local configuration files:
- `pkt-streamer/config.env` uses `localhost:5000` (for local dev)
- `docker-compose.yml` overrides with `http://ids-agent-org-a:5000/detect` (for Docker)
- This dual configuration supports both local development and containerized deployment

## Findings

### No Port Mismatches Found
1. **FL Server**: Consistently uses port 9090 across all configurations
2. **IDS Agents**: All three instances consistently use port 5000
3. **Service Names**: Docker Compose correctly uses service names for inter-container communication
4. **FL Server URL**: IDS agents correctly reference `fl-server:9090`

### Configuration Best Practices Observed
1. ✓ Environment variables for configuration
2. ✓ Default values in code
3. ✓ Docker Compose overrides for container networking
4. ✓ Separate configuration for local vs Docker deployment

## Why There Might Be Connection Issues (If Any)

If connection issues are observed, they are NOT due to port mismatches. Possible causes:

### 1. Services Not Started
- **Check**: `docker-compose ps` to verify all services are running
- **Fix**: `docker-compose up --build`

### 2. Startup Order
- **Issue**: Packet streamer might start before IDS agent is ready
- **Mitigation**: Docker Compose `depends_on` is configured correctly
- **Note**: `depends_on` only waits for container start, not service readiness

### 3. Network Issues
- **Check**: `docker network ls` and `docker network inspect federated-net`
- **Fix**: `docker-compose down && docker-compose up --build`

### 4. Firewall/Security Groups
- **Issue**: Host firewall blocking Docker network traffic
- **Fix**: Check Docker daemon settings and host firewall rules

### 5. Resource Constraints
- **Issue**: Services failing due to insufficient memory/CPU
- **Check**: `docker stats` and service logs

## Recommendations

### 1. Health Checks (Future Enhancement)
Add health checks to docker-compose.yml:
```yaml
services:
  fl-server:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/api/register"]
      interval: 10s
      timeout: 5s
      retries: 3
```

### 2. Wait Scripts (Future Enhancement)
Implement wait-for-it.sh or similar to ensure services wait for dependencies to be truly ready:
```yaml
pkt-streamer-org-a:
  command: ["./wait-for-it.sh", "ids-agent-org-a:5000", "--", "python", "main.py"]
```

### 3. Monitoring
Consider adding:
- Service health endpoints
- Connectivity monitoring
- Log aggregation

## Deliverables

This analysis has produced:

1. **PORT_CONFIGURATION.md** - Comprehensive port documentation with:
   - Network diagram
   - Port mapping table
   - Environment variable reference
   - Troubleshooting guide

2. **check_ports.sh** - Automated port validation script that:
   - Checks if ports are listening
   - Validates configuration files
   - Scans source code for port references
   - Tests endpoint connectivity

3. **test_port_config.py** - Python test suite that:
   - Validates docker-compose.yml configurations
   - Verifies source code default ports
   - Ensures consistency across files
   - Can be run in CI/CD pipeline

4. **Updated README.md** - User-friendly documentation with:
   - Quick start guide
   - Port configuration summary
   - Troubleshooting section
   - Local vs Docker development instructions

## Conclusion

**No port mismatches exist in the current configuration.** All services are correctly configured to communicate with each other using appropriate ports and service names. The system architecture properly supports both local development (using localhost) and containerized deployment (using Docker service names).

If connectivity issues persist, they are likely due to:
- Services not being fully started/ready
- Network configuration issues
- Resource constraints
- Application-level errors (not port configuration)

Recommended next steps:
1. Run `./check_ports.sh` to verify current state
2. Run `python test_port_config.py` to validate configuration
3. Check service logs: `docker-compose logs -f`
4. Verify services are running: `docker-compose ps`
