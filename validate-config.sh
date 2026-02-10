#!/bin/bash
# Docker Compose Configuration Validation Script
# This script validates the Docker Compose setup for the Federated Agentic Defense system

set -e

echo "======================================"
echo "Docker Compose Configuration Validator"
echo "======================================"
echo ""

# Check if docker-compose.yml exists
echo "[✓] Checking if docker-compose.yml exists..."
if [ ! -f "docker-compose.yml" ]; then
    echo "[✗] ERROR: docker-compose.yml not found!"
    exit 1
fi
echo "    Found: docker-compose.yml"
echo ""

# Validate docker-compose syntax
echo "[✓] Validating docker-compose.yml syntax..."
if docker compose config > /dev/null 2>&1; then
    echo "    Syntax is valid"
else
    echo "[✗] ERROR: Invalid docker-compose.yml syntax"
    exit 1
fi
echo ""

# Check for required directories
echo "[✓] Checking required directories..."
DIRS=("agentic-ids-local" "pkt-streamer" "federation_simulation")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "    Found: $dir/"
    else
        echo "[✗] ERROR: Directory $dir/ not found!"
        exit 1
    fi
done
echo ""

# Check for Dockerfiles
echo "[✓] Checking Dockerfiles..."
DOCKERFILES=(
    "agentic-ids-local/Dockerfile"
    "pkt-streamer/Dockerfile"
    "federation_simulation/Dockerfile"
)
for dockerfile in "${DOCKERFILES[@]}"; do
    if [ -f "$dockerfile" ]; then
        echo "    Found: $dockerfile"
    else
        echo "[✗] ERROR: $dockerfile not found!"
        exit 1
    fi
done
echo ""

# Check for data directory and dataset_features.json
echo "[✓] Checking data directory..."
if [ -d "pkt-streamer/data" ]; then
    echo "    Found: pkt-streamer/data/"
    if [ -f "pkt-streamer/data/dataset_features.json" ]; then
        echo "    Found: pkt-streamer/data/dataset_features.json"
    else
        echo "[!] WARNING: pkt-streamer/data/dataset_features.json not found"
    fi
    if [ -f "pkt-streamer/data/UNSW-NB15_1.csv" ]; then
        echo "    Found: pkt-streamer/data/UNSW-NB15_1.csv"
    else
        echo "[!] WARNING: pkt-streamer/data/UNSW-NB15_1.csv not found"
        echo "    NOTE: You'll need to download the UNSW-NB15 dataset to run the system"
    fi
else
    echo "[✗] ERROR: pkt-streamer/data/ directory not found!"
    exit 1
fi
echo ""

# Validate service names and ports
echo "[✓] Validating service configuration..."
SERVICES=("agentic-ids-local" "fl-server" "pkt-streamer")
for service in "${SERVICES[@]}"; do
    if docker compose config --services | grep -q "^${service}$"; then
        echo "    Service defined: $service"
    else
        echo "[✗] ERROR: Service $service not defined in docker-compose.yml!"
        exit 1
    fi
done
echo ""

# Check port configurations
echo "[✓] Checking port configurations..."
if docker compose config | grep -q "published.*5000" || docker compose config | grep -q '"5000"'; then
    echo "    agentic-ids-local port 5000 mapped correctly"
else
    echo "[✗] ERROR: agentic-ids-local port 5000 not mapped correctly!"
    exit 1
fi

if docker compose config | grep -q "published.*8000" || docker compose config | grep -q '"8000"'; then
    echo "    fl-server port 8000 mapped correctly"
else
    echo "[✗] ERROR: fl-server port 8000 not mapped correctly!"
    exit 1
fi
echo ""

# Check network configuration
echo "[✓] Checking network configuration..."
if docker compose config | grep -q "ids-network"; then
    echo "    Network 'ids-network' is configured"
else
    echo "[✗] ERROR: Network 'ids-network' not configured!"
    exit 1
fi
echo ""

# Check environment variables for pkt-streamer
echo "[✓] Checking environment variables for pkt-streamer..."
PKT_CONFIG=$(docker compose config | grep -A 15 "pkt-streamer:")
ENV_VARS=("API_URL" "CSV_PATH" "FEATURES_METADATA")
for var in "${ENV_VARS[@]}"; do
    if echo "$PKT_CONFIG" | grep -q "$var"; then
        echo "    $var is configured"
    else
        echo "[✗] ERROR: $var not configured for pkt-streamer!"
        exit 1
    fi
done
echo ""

# Check API_URL points to correct service
echo "[✓] Checking API_URL configuration..."
if echo "$PKT_CONFIG" | grep "API_URL" | grep -q "agentic-ids-local:5000"; then
    echo "    API_URL correctly points to agentic-ids-local:5000"
else
    echo "[✗] ERROR: API_URL does not point to agentic-ids-local:5000!"
    exit 1
fi
echo ""

# Check depends_on configuration
echo "[✓] Checking service dependencies..."
if docker compose config | grep -A 5 "pkt-streamer" | grep -q "depends_on"; then
    echo "    pkt-streamer has dependency configuration"
    if docker compose config | grep -A 5 "pkt-streamer" | grep -A 3 "depends_on" | grep -q "agentic-ids-local"; then
        echo "    pkt-streamer depends on agentic-ids-local"
    else
        echo "[!] WARNING: pkt-streamer should depend on agentic-ids-local"
    fi
else
    echo "[!] WARNING: pkt-streamer has no dependency configuration"
fi
echo ""

# Summary
echo "======================================"
echo "Validation Complete!"
echo "======================================"
echo ""
echo "Summary:"
echo "  ✓ Docker Compose configuration is valid"
echo "  ✓ All required files and directories exist"
echo "  ✓ Services are properly configured"
echo "  ✓ Port mappings are correct"
echo "  ✓ Network configuration is correct"
echo "  ✓ Environment variables are set"
echo "  ✓ Service dependencies are configured"
echo ""
echo "Next steps:"
echo "  1. Ensure UNSW-NB15 dataset is placed in pkt-streamer/data/"
echo "  2. Build services: docker compose build"
echo "  3. Start services: docker compose up"
echo ""
