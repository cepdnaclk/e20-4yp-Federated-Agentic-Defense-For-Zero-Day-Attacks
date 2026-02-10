#!/bin/bash

# Live Connectivity Test
# Tests actual connectivity between running services

set -e

echo "=========================================="
echo "Live Service Connectivity Test"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test endpoint with detailed output
test_endpoint() {
    local url=$1
    local description=$2
    local expected_status=${3:-200}
    
    echo -n "Testing $description... "
    
    if ! command -v curl &> /dev/null; then
        echo -e "${YELLOW}SKIP (curl not installed)${NC}"
        return 2
    fi
    
    response=$(curl -s -w "\n%{http_code}" -o /tmp/response_body.txt "$url" 2>&1 || echo "000")
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" == "$expected_status" ]; then
        echo -e "${GREEN}OK (HTTP $http_code)${NC}"
        return 0
    elif [ "$http_code" == "000" ]; then
        echo -e "${RED}FAILED (Connection refused or timeout)${NC}"
        return 1
    else
        echo -e "${YELLOW}HTTP $http_code (expected $expected_status)${NC}"
        return 1
    fi
}

# Function to test POST endpoint
test_post_endpoint() {
    local url=$1
    local description=$2
    local data=$3
    
    echo -n "Testing $description (POST)... "
    
    if ! command -v curl &> /dev/null; then
        echo -e "${YELLOW}SKIP (curl not installed)${NC}"
        return 2
    fi
    
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$data" \
        -o /tmp/response_body.txt \
        "$url" 2>&1 || echo "000")
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        echo -e "${GREEN}OK (HTTP $http_code)${NC}"
        return 0
    elif [ "$http_code" == "000" ]; then
        echo -e "${RED}FAILED (Connection refused or timeout)${NC}"
        return 1
    else
        echo -e "${YELLOW}HTTP $http_code${NC}"
        return 1
    fi
}

echo -e "${BLUE}1. Testing FL Server...${NC}"
echo ""

# FL Server endpoints
test_post_endpoint \
    "http://localhost:9090/api/register" \
    "FL Server Registration" \
    '{"agent_id": "test-agent", "capabilities": {}}'

test_endpoint \
    "http://localhost:9090/api/broadcast/signatures" \
    "FL Server Signature Broadcast" \
    200

test_endpoint \
    "http://localhost:9090/api/broadcast/model" \
    "FL Server Model Broadcast" \
    200

echo ""
echo -e "${BLUE}2. Testing Agentic IDS Local...${NC}"
echo ""

# IDS Agent health check
test_endpoint \
    "http://localhost:5000/health" \
    "IDS Agent Health Check" \
    200

# IDS Agent detection endpoint
test_post_endpoint \
    "http://localhost:5000/detect" \
    "IDS Agent Detection" \
    '{
        "flow_id": 999,
        "features": {
            "srcip": "192.168.1.1",
            "sport": 12345,
            "dstip": "192.168.1.2",
            "dsport": 80,
            "proto": "tcp",
            "state": "FIN",
            "dur": 0.1,
            "sbytes": 1000,
            "dbytes": 500,
            "sttl": 64,
            "dttl": 64,
            "service": "http",
            "Spkts": 10,
            "Dpkts": 5,
            "tcprtt": 0.01,
            "ct_srv_src": 1
        }
    }'

echo ""
echo -e "${BLUE}3. Testing Docker Internal Connectivity...${NC}"
echo ""

if command -v docker &> /dev/null && docker ps | grep -q fl-server; then
    echo "Testing from inside IDS agent container..."
    
    # Test FL Server from IDS agent
    docker exec $(docker ps -q -f name=ids-agent-org-a) \
        sh -c "command -v curl > /dev/null && curl -s -f http://fl-server:9090/api/broadcast/signatures > /dev/null && echo -e '${GREEN}✓ IDS Agent → FL Server connectivity OK${NC}'" \
        2>/dev/null || echo -e "${RED}✗ IDS Agent → FL Server connectivity FAILED${NC}"
    
    # Test IDS agent from packet streamer
    docker exec $(docker ps -q -f name=pkt-streamer-org-a) \
        sh -c "command -v curl > /dev/null && curl -s -f http://ids-agent-org-a:5000/health > /dev/null && echo -e '${GREEN}✓ Packet Streamer → IDS Agent connectivity OK${NC}'" \
        2>/dev/null || echo -e "${RED}✗ Packet Streamer → IDS Agent connectivity FAILED${NC}"
else
    echo -e "${YELLOW}Docker containers not running. Skipping internal connectivity tests.${NC}"
    echo "To test Docker connectivity, run: docker-compose up -d"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "If all tests passed, the services are running correctly and can communicate."
echo ""
echo "If tests failed:"
echo "  - Check if services are running: docker-compose ps"
echo "  - Check service logs: docker-compose logs -f [service-name]"
echo "  - Verify port configurations: ./check_ports.sh"
echo "  - Run validation tests: python test_port_config.py"
echo ""
echo "For more information, see:"
echo "  - PORT_CONFIGURATION.md"
echo "  - PORT_ANALYSIS_REPORT.md"
echo ""
