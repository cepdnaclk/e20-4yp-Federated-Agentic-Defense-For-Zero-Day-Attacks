#!/bin/bash
# Docker Compose Validation Script
# This script validates the docker-compose configuration and checks service connectivity

set -e

echo "================================"
echo "Docker Compose Validation Script"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker compose is available
echo -n "Checking Docker Compose availability... "
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Error: Docker Compose is not available. Please install Docker and Docker Compose."
    exit 1
fi

# Validate docker-compose.yml syntax
echo -n "Validating docker-compose.yml syntax... "
if docker compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Error: docker-compose.yml has syntax errors"
    exit 1
fi

# Check if required directories exist
echo -n "Checking required directories... "
MISSING_DIRS=()
if [ ! -d "./agentic-ids-local" ]; then
    MISSING_DIRS+=("agentic-ids-local")
fi
if [ ! -d "./pkt-streamer" ]; then
    MISSING_DIRS+=("pkt-streamer")
fi
if [ ! -d "./federation_simulation" ]; then
    MISSING_DIRS+=("federation_simulation")
fi

if [ ${#MISSING_DIRS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Missing directories: ${MISSING_DIRS[*]}"
    exit 1
fi

# Check if Dockerfiles exist
echo -n "Checking Dockerfiles... "
MISSING_DOCKERFILES=()
if [ ! -f "./agentic-ids-local/Dockerfile" ]; then
    MISSING_DOCKERFILES+=("agentic-ids-local/Dockerfile")
fi
if [ ! -f "./pkt-streamer/Dockerfile" ]; then
    MISSING_DOCKERFILES+=("pkt-streamer/Dockerfile")
fi
if [ ! -f "./federation_simulation/Dockerfile" ]; then
    MISSING_DOCKERFILES+=("federation_simulation/Dockerfile")
fi

if [ ${#MISSING_DOCKERFILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Missing Dockerfiles: ${MISSING_DOCKERFILES[*]}"
    exit 1
fi

# Check for data directory (warning only)
echo -n "Checking data directory... "
if [ -d "./data" ] && [ -f "./data/UNSW-NB15_1.csv" ]; then
    echo -e "${GREEN}✓${NC}"
elif [ -d "./pkt-streamer/data" ]; then
    echo -e "${YELLOW}!${NC}"
    echo "  Warning: ./data/UNSW-NB15_1.csv not found"
    echo "  pkt-streamer will fail to run without the dataset"
    echo "  Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
else
    echo -e "${YELLOW}!${NC}"
    echo "  Warning: Data directory not found"
fi

# Test building each service
echo ""
echo "Testing service builds (this may take a few minutes)..."
echo "-------------------------------------------------------"

SERVICES=("fl-server" "local-agentic-ids" "pkt-streamer")
BUILD_FAILED=()

for service in "${SERVICES[@]}"; do
    echo -n "Building ${service}... "
    if docker compose build "${service}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        BUILD_FAILED+=("${service}")
    fi
done

if [ ${#BUILD_FAILED[@]} -ne 0 ]; then
    echo ""
    echo -e "${RED}Build failed for: ${BUILD_FAILED[*]}${NC}"
    exit 1
fi

# Summary
echo ""
echo "================================"
echo "Validation Summary"
echo "================================"
echo -e "${GREEN}✓ Docker Compose configuration is valid${NC}"
echo -e "${GREEN}✓ All required directories exist${NC}"
echo -e "${GREEN}✓ All Dockerfiles exist${NC}"
echo -e "${GREEN}✓ All services build successfully${NC}"
echo ""
echo "Configuration Details:"
echo "  - fl-server: Port 8000 (FastAPI)"
echo "  - local-agentic-ids: Port 5000 (Flask)"
echo "  - pkt-streamer: Client (sends to local-agentic-ids)"
echo ""
echo "Service Dependencies:"
echo "  1. fl-server starts first"
echo "  2. local-agentic-ids starts after fl-server"
echo "  3. pkt-streamer starts after local-agentic-ids is healthy"
echo ""
echo "Network Configuration:"
echo "  - pkt-streamer -> http://local-agentic-ids:5000/detect"
echo "  - local-agentic-ids -> http://fl-server:8000"
echo ""
if [ ! -f "./data/UNSW-NB15_1.csv" ]; then
    echo -e "${YELLOW}Note: Download dataset before running services${NC}"
    echo "      See DOCKER_TESTING.md for details"
    echo ""
fi
echo "To start services: docker compose up"
echo "To view logs: docker compose logs -f"
echo "To stop services: docker compose down"
echo ""
