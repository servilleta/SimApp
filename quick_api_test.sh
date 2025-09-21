#!/bin/bash

# Quick Monte Carlo API Test Script
# Usage: ./quick_api_test.sh [SERVER_URL] [API_KEY]

# Default values
SERVER_URL="${1:-http://209.51.170.185:8000}"
API_KEY="${2:-ak_2830891165dcd30d35782f89d96b0fdf_sk_622697748904bc7e058c525378b4397a8b04021f1b953e596e65d22a814f5f1d}"
BASE_URL="${SERVER_URL}/simapp-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🚀 Quick Monte Carlo API Test"
echo "================================"
echo "Server: $SERVER_URL"
echo "API Key: ${API_KEY:0:20}..."
echo "Base URL: $BASE_URL"
echo ""

# Function to print test step
print_step() {
    echo -e "${BLUE}🔸 $1${NC}"
    echo "----------------------------------------"
}

# Function to check response
check_response() {
    local response_code=$1
    local test_name="$2"
    
    if [ $response_code -eq 200 ]; then
        echo -e "${GREEN}✅ $test_name PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $test_name FAILED (HTTP $response_code)${NC}"
        return 1
    fi
}

# Test 1: Health Check
print_step "Step 1: API Health Check"
health_response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/health")

check_response $health_response "Health Check"
if [ $health_response -eq 200 ]; then
    echo "Health Status:"
    cat /tmp/health_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/health_response.json
fi
echo ""

# Test 2: List Models
print_step "Step 2: List Existing Models"
models_response=$(curl -s -w "%{http_code}" -o /tmp/models_response.json \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/models")

check_response $models_response "List Models"
if [ $models_response -eq 200 ]; then
    echo "Models:"
    cat /tmp/models_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/models_response.json
fi
echo ""

# Test 3: Create Sample Excel File and Upload
print_step "Step 3: File Upload Test"

# Create a simple CSV that we'll upload as Excel-like data
cat > /tmp/sample_model.csv << 'EOF'
Description,Value,Formula
Market Volatility,0.15,
Expected Return,0.08,
Risk Free Rate,0.03,
Initial Investment,1000000,
Adjusted Return,,=B2-B3
Portfolio Value,,=B4*(1+B5)
EOF

echo "Created sample CSV file for upload test"

# Test upload endpoint
upload_response=$(curl -s -w "%{http_code}" -o /tmp/upload_response.json \
    -H "Authorization: Bearer $API_KEY" \
    -F "file=@/tmp/sample_model.csv" \
    "$BASE_URL/models")

check_response $upload_response "File Upload"
if [ $upload_response -eq 200 ]; then
    echo "Upload Response:"
    cat /tmp/upload_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/upload_response.json
    
    # Extract model_id for next test
    MODEL_ID=$(cat /tmp/upload_response.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('model_id', ''))" 2>/dev/null)
    echo "Model ID: $MODEL_ID"
else
    echo "⚠️ Upload failed, will skip simulation test"
    MODEL_ID=""
fi
echo ""

# Test 4: Start Simulation (if upload was successful)
if [ ! -z "$MODEL_ID" ]; then
    print_step "Step 4: Start Simulation"
    
    # Create simulation request
    cat > /tmp/simulation_request.json << EOF
{
    "model_id": "$MODEL_ID",
    "iterations": 1000,
    "confidence_levels": [0.95, 0.99],
    "variables": [
        {
            "cell": "B1",
            "name": "Market_Volatility",
            "distribution": {
                "type": "triangular",
                "min": 0.05,
                "mode": 0.15,
                "max": 0.35
            }
        }
    ],
    "output_cells": ["B6"]
}
EOF

    simulation_response=$(curl -s -w "%{http_code}" -o /tmp/simulation_response.json \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d @/tmp/simulation_request.json \
        "$BASE_URL/simulations")
    
    check_response $simulation_response "Start Simulation"
    if [ $simulation_response -eq 200 ]; then
        echo "Simulation Response:"
        cat /tmp/simulation_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/simulation_response.json
        
        # Extract simulation_id for progress tracking
        SIM_ID=$(cat /tmp/simulation_response.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('simulation_id', ''))" 2>/dev/null)
        echo "Simulation ID: $SIM_ID"
    else
        SIM_ID=""
    fi
else
    echo "⚠️ Skipping simulation test (no model ID)"
    SIM_ID=""
fi
echo ""

# Test 5: Check Simulation Progress (if simulation was started)
if [ ! -z "$SIM_ID" ]; then
    print_step "Step 5: Check Simulation Progress"
    
    # Check progress a few times
    for i in {1..5}; do
        echo "Progress check $i/5..."
        progress_response=$(curl -s -w "%{http_code}" -o /tmp/progress_response.json \
            -H "Authorization: Bearer $API_KEY" \
            "$BASE_URL/simulations/$SIM_ID/progress")
        
        if [ $progress_response -eq 200 ]; then
            status=$(cat /tmp/progress_response.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null)
            progress=$(cat /tmp/progress_response.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('progress', 0))" 2>/dev/null)
            echo "Status: $status, Progress: $progress%"
            
            if [ "$status" = "completed" ]; then
                echo -e "${GREEN}✅ Simulation completed!${NC}"
                break
            elif [ "$status" = "failed" ]; then
                echo -e "${RED}❌ Simulation failed!${NC}"
                break
            fi
        else
            echo -e "${RED}❌ Progress check failed (HTTP $progress_response)${NC}"
            break
        fi
        
        if [ $i -lt 5 ]; then
            sleep 2
        fi
    done
else
    echo "⚠️ Skipping progress check (no simulation ID)"
fi
echo ""

# Summary
echo "🎯 TEST SUMMARY"
echo "==============="
echo "Health Check: $([ $health_response -eq 200 ] && echo "✅ PASS" || echo "❌ FAIL")"
echo "List Models: $([ $models_response -eq 200 ] && echo "✅ PASS" || echo "❌ FAIL")"
echo "File Upload: $([ $upload_response -eq 200 ] && echo "✅ PASS" || echo "❌ FAIL")"
echo "Simulation: $([ ! -z "$SIM_ID" ] && echo "✅ STARTED" || echo "❌ SKIPPED")"

# Cleanup
rm -f /tmp/health_response.json /tmp/models_response.json /tmp/upload_response.json
rm -f /tmp/simulation_response.json /tmp/progress_response.json /tmp/simulation_request.json
rm -f /tmp/sample_model.csv

echo ""
echo "🔧 For detailed testing, run:"
echo "python3 api_test_client.py --server $SERVER_URL --api-key $API_KEY"
