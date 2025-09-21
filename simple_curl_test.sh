#!/bin/bash

# Simple cURL-based API Test
# This script tests basic API connectivity using only cURL
# Perfect for customers who want to quickly verify API access

# Configuration
SERVER_URL="${1:-http://209.51.170.185:8000}"
API_KEY="${2:-ak_0f345df6f8af9ea80140bf434fdba478_sk_03ddc015b97aca737a1cb690d4f41fa86d3877c957eef8362713290055a7964a}"
BASE_URL="${SERVER_URL}/simapp-api"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🔌 Simple Monte Carlo API Connectivity Test"
echo "============================================"
echo "Server: $SERVER_URL"
echo "API Key: ${API_KEY:0:20}..."
echo ""

# Test 1: Basic Health Check
echo -e "${BLUE}Testing API Health...${NC}"
health_response=$(curl -s -w "%{http_code}" -o /dev/null \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/health")

if [ $health_response -eq 200 ]; then
    echo -e "${GREEN}✅ API Health Check: PASSED${NC}"
    
    # Get detailed health info
    curl -s -H "Authorization: Bearer $API_KEY" "$BASE_URL/health" | \
        python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'   Status: {data.get(\"status\", \"unknown\")}')
    print(f'   Version: {data.get(\"version\", \"unknown\")}')
    print(f'   GPU Available: {data.get(\"gpu_available\", False)}')
except:
    print('   (Could not parse health details)')
"
else
    echo -e "${RED}❌ API Health Check: FAILED (HTTP $health_response)${NC}"
fi
echo ""

# Test 2: Authentication Check
echo -e "${BLUE}Testing Authentication...${NC}"
models_response=$(curl -s -w "%{http_code}" -o /dev/null \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/models")

if [ $models_response -eq 200 ]; then
    echo -e "${GREEN}✅ Authentication: PASSED${NC}"
    echo "   API key is valid and authorized"
elif [ $models_response -eq 401 ]; then
    echo -e "${RED}❌ Authentication: FAILED${NC}"
    echo "   API key is invalid or unauthorized"
else
    echo -e "${RED}❌ Authentication: UNKNOWN (HTTP $models_response)${NC}"
    echo "   Unexpected response code"
fi
echo ""

# Test 3: API Endpoints Availability
echo -e "${BLUE}Testing Endpoint Availability...${NC}"

endpoints=(
    "health:GET"
    "models:GET"
)

for endpoint_info in "${endpoints[@]}"; do
    IFS=':' read -r endpoint method <<< "$endpoint_info"
    
    if [ "$method" = "GET" ]; then
        response_code=$(curl -s -w "%{http_code}" -o /dev/null \
            -H "Authorization: Bearer $API_KEY" \
            "$BASE_URL/$endpoint")
    fi
    
    if [ $response_code -eq 200 ]; then
        echo -e "   ${GREEN}✅ $method /$endpoint${NC}"
    else
        echo -e "   ${RED}❌ $method /$endpoint (HTTP $response_code)${NC}"
    fi
done
echo ""

# Test 4: Network Connectivity Details
echo -e "${BLUE}Network Connectivity Details...${NC}"

# Test basic server reachability
if curl -s --connect-timeout 5 "$SERVER_URL" > /dev/null; then
    echo -e "${GREEN}✅ Server Reachable${NC}"
else
    echo -e "${RED}❌ Server Unreachable${NC}"
fi

# Test SSL/TLS if HTTPS
if [[ $SERVER_URL == https* ]]; then
    ssl_info=$(curl -s -w "%{ssl_verify_result}:%{time_connect}" -o /dev/null "$SERVER_URL")
    IFS=':' read -r ssl_result connect_time <<< "$ssl_info"
    
    if [ "$ssl_result" = "0" ]; then
        echo -e "${GREEN}✅ SSL Certificate Valid${NC}"
    else
        echo -e "${RED}❌ SSL Certificate Issues${NC}"
    fi
    echo "   Connection time: ${connect_time}s"
fi

# DNS resolution check
domain=$(echo $SERVER_URL | sed 's|https\?://||' | sed 's|/.*||')
if nslookup $domain > /dev/null 2>&1; then
    echo -e "${GREEN}✅ DNS Resolution Working${NC}"
else
    echo -e "${RED}❌ DNS Resolution Failed${NC}"
fi
echo ""

# Summary
echo "📊 CONNECTIVITY SUMMARY"
echo "======================="

if [ $health_response -eq 200 ] && [ $models_response -eq 200 ]; then
    echo -e "${GREEN}🎉 SUCCESS: API is fully accessible and working${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Upload your Excel model using the file upload endpoint"
    echo "2. Configure your Monte Carlo variables"
    echo "3. Run simulations and retrieve results"
    echo ""
    echo "For complete testing, run:"
    echo "   python3 api_test_client.py --server $SERVER_URL --api-key $API_KEY"
elif [ $health_response -eq 200 ]; then
    echo -e "${RED}⚠️  PARTIAL: API is reachable but authentication failed${NC}"
    echo ""
    echo "Issues:"
    echo "- Check your API key format and validity"
    echo "- Ensure your subscription is active"
    echo "- Contact support if the key should be working"
else
    echo -e "${RED}❌ FAILURE: Cannot reach the API${NC}"
    echo ""
    echo "Issues:"
    echo "- Check server URL and network connectivity"
    echo "- Verify firewall settings"
    echo "- Ensure the service is running"
fi

echo ""
echo "🔧 Troubleshooting:"
echo "- Server URL: $SERVER_URL"
echo "- Full Base URL: $BASE_URL"
echo "- Your IP: $(curl -s ifconfig.me 2>/dev/null || echo 'Unknown')"
echo "- Test time: $(date)"

# Check if this was run from Mac via SSH
if [ ! -z "$SSH_CLIENT" ]; then
    echo "- SSH connection: $SSH_CLIENT"
fi
