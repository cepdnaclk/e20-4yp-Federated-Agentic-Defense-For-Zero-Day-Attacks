#!/bin/bash

# Port Configuration Checker Script
# This script verifies port configurations and connectivity across all services

echo "======================================"
echo "Port Configuration Checker"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    local port=$1
    local service=$2
    echo -n "Checking port $port ($service)... "
    if command -v lsof &> /dev/null; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo -e "${GREEN}LISTENING${NC}"
            return 0
        else
            echo -e "${RED}NOT LISTENING${NC}"
            return 1
        fi
    elif command -v netstat &> /dev/null; then
        if netstat -tuln | grep -q ":$port "; then
            echo -e "${GREEN}LISTENING${NC}"
            return 0
        else
            echo -e "${RED}NOT LISTENING${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}SKIP (no lsof/netstat)${NC}"
        return 2
    fi
}

# Function to check HTTP endpoint
check_endpoint() {
    local url=$1
    local description=$2
    echo -n "Checking $description at $url... "
    if command -v curl &> /dev/null; then
        if curl -s -f -m 2 "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
            return 0
        else
            # Try to see if connection is refused or timeout
            response=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$url" 2>&1 || echo "failed")
            if [[ "$response" == "000" ]] || [[ "$response" == "failed" ]]; then
                echo -e "${RED}UNREACHABLE${NC}"
            else
                echo -e "${YELLOW}HTTP $response${NC}"
            fi
            return 1
        fi
    else
        echo -e "${YELLOW}SKIP (no curl)${NC}"
        return 2
    fi
}

# Function to scan configuration files
scan_config_file() {
    local file=$1
    echo ""
    echo "Configuration in $file:"
    if [ -f "$file" ]; then
        grep -E "(PORT|URL)" "$file" | grep -v "^#" || echo "  No port/URL configuration found"
    else
        echo -e "  ${RED}File not found${NC}"
    fi
}

echo "1. Checking expected ports..."
echo ""

# Check FL Server port
check_port 9090 "FL Server"

# Check Agentic IDS Local port
check_port 5000 "Agentic IDS Local"

echo ""
echo "2. Checking Docker Compose configuration..."
echo ""

if [ -f "docker-compose.yml" ]; then
    echo "FL Server configuration:"
    grep -A 2 "fl-server:" docker-compose.yml | grep -E "(PORT|environment)" || echo "  Not found"
    
    echo ""
    echo "IDS Agent configuration:"
    grep -A 5 "ids-agent-org-a:" docker-compose.yml | grep -E "(PORT|FL_SERVER_URL)" || echo "  Not found"
    
    echo ""
    echo "Packet Streamer configuration:"
    grep -A 5 "pkt-streamer-org-a:" docker-compose.yml | grep -E "API_URL" || echo "  Not found"
else
    echo -e "${RED}docker-compose.yml not found${NC}"
fi

echo ""
echo "3. Checking configuration files..."

# Check pkt-streamer config.env
scan_config_file "pkt-streamer/config.env"

# Check for any .env files
for env_file in .env */.env */config.env; do
    if [ -f "$env_file" ] && [ "$env_file" != "pkt-streamer/config.env" ]; then
        scan_config_file "$env_file"
    fi
done

echo ""
echo "4. Scanning source code for port references..."
echo ""

echo "FL Server (fl-server/app.py):"
grep -n "PORT\|port.*=" fl-server/app.py 2>/dev/null | head -5 || echo "  Not found"

echo ""
echo "Agentic IDS Local (agentic-ids-local/src/main.py):"
grep -n "PORT\|port.*=" agentic-ids-local/src/main.py 2>/dev/null | head -5 || echo "  Not found"

echo ""
echo "Orchestrator FL_SERVER_URL (agentic-ids-local/src/agents/Orchestrator/orchestrator.py):"
grep -n "FL_SERVER_URL" agentic-ids-local/src/agents/Orchestrator/orchestrator.py 2>/dev/null | head -3 || echo "  Not found"

echo ""
echo "5. Testing connectivity (if services are running)..."
echo ""

# Try localhost endpoints
check_endpoint "http://localhost:9090/api/register" "FL Server (localhost)"
check_endpoint "http://localhost:5000/health" "Agentic IDS Local (localhost)"

echo ""
echo "======================================"
echo "Summary"
echo "======================================"
echo ""
echo "Expected Port Configuration:"
echo "  - FL Server: 9090"
echo "  - Agentic IDS Local: 5000"
echo "  - Packet Streamer: N/A (client only)"
echo ""
echo "Expected Service URLs (Docker):"
echo "  - FL Server: http://fl-server:9090"
echo "  - IDS Agent Org A: http://ids-agent-org-a:5000"
echo "  - IDS Agent Org B: http://ids-agent-org-b:5000"
echo "  - IDS Agent Org C: http://ids-agent-org-c:5000"
echo ""
echo "For more details, see PORT_CONFIGURATION.md"
echo ""
