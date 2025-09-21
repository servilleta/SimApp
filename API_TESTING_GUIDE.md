# Monte Carlo API Testing Guide

This guide provides comprehensive tools to test the Monte Carlo simulation API from your Mac via SSH, simulating real customer usage patterns.

## 🚀 Quick Start

### Prerequisites on Your Mac

```bash
# Install required Python packages
pip3 install requests openpyxl

# Or if you prefer using a virtual environment
python3 -m venv api_test_env
source api_test_env/bin/activate
pip install requests openpyxl
```

### SSH to Server and Run Tests

```bash
# SSH to your server
ssh paperspace@209.51.170.185

# Navigate to project directory
cd PROJECT

# Option 1: Quick bash test (basic API endpoints)
./quick_api_test.sh

# Option 2: Comprehensive Python test (full workflow)
python3 api_test_client.py

# Option 3: Test with custom server/API key
python3 api_test_client.py --server https://your-server.com --api-key your_api_key_here
```

## 🧪 Test Tools Overview

### 1. Quick Bash Test (`quick_api_test.sh`)

**Purpose**: Fast validation of basic API functionality  
**Duration**: ~30 seconds  
**Best for**: Quick health checks and basic connectivity testing

**Features**:
- ✅ API health check
- ✅ List existing models
- ✅ File upload test with sample CSV
- ✅ Basic simulation request
- ✅ Progress monitoring (5 checks)

**Usage**:
```bash
# Default settings (demo server + demo API key)
./quick_api_test.sh

# Custom server and API key
./quick_api_test.sh https://your-server.com your_api_key_here
```

### 2. Comprehensive Python Test (`api_test_client.py`)

**Purpose**: Complete end-to-end API workflow testing  
**Duration**: 2-5 minutes  
**Best for**: Full validation before customer deployment

**Features**:
- ✅ API health and system status
- ✅ Excel file generation and upload
- ✅ Complete simulation configuration
- ✅ Real-time progress monitoring
- ✅ Results retrieval and analysis
- ✅ Report downloads (PDF, Excel, JSON)
- ✅ Detailed test reporting

**Usage**:
```bash
# Run all tests
python3 api_test_client.py

# Run specific test
python3 api_test_client.py --test health
python3 api_test_client.py --test upload
python3 api_test_client.py --test simulation

# Custom configuration
python3 api_test_client.py \
  --server http://209.51.170.185:8000 \
  --api-key mc_test_demo123456789012345678901234
```

## 📊 Sample Excel Model

The test scripts automatically create a sample portfolio risk model with:

### Input Variables (Monte Carlo Variables)
| Cell | Description | Base Value | Distribution |
|------|-------------|------------|--------------|
| B4 | Market Volatility | 15% | Triangular(5%, 15%, 35%) |
| B5 | Expected Return | 8% | Normal(8%, 2%) |
| B6 | Risk-Free Rate | 3% | Triangular(1%, 3%, 5%) |
| B7 | Initial Investment | $1,000,000 | Fixed |

### Target Output Cells
| Cell | Description | Formula |
|------|-------------|---------|
| B16 | Portfolio Value | =B7*(1+B10) |
| B17 | Total Return % | =(B12-B7)/B7*100 |
| B18 | Risk-Adjusted Performance | =B11 |

## 🔑 API Authentication

### Demo API Keys (for testing)
```bash
# Test/Demo Key (Starter tier)
mc_test_demo123456789012345678901234

# Production Key (Professional tier) 
mc_live_prod123456789012345678901234
```

### Authentication Format
```bash
# Headers required
Authorization: Bearer mc_test_demo123456789012345678901234
Content-Type: application/json  # For JSON requests
```

## 🌐 API Endpoints Tested

### Base URL
```
http://209.51.170.185:8000/monte-carlo-api
```

### Endpoints Covered
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | System health and status |
| GET | `/models` | List uploaded models |
| POST | `/models` | Upload Excel file |
| POST | `/simulations` | Start simulation |
| GET | `/simulations/{id}/progress` | Monitor progress |
| GET | `/simulations/{id}/results` | Get results |
| POST | `/simulations/{id}/generate-download-token` | Get download token |
| GET | `/download/{token}/{format}` | Download reports |

## 📝 Expected Test Results

### Successful Test Output

```bash
🧪 Step 1: API Health Check
----------------------------------------
✅ Health Check PASSED
{
  "status": "healthy",
  "gpu_available": true,
  "version": "1.0.0"
}

🧪 Step 2: List Existing Models  
----------------------------------------
✅ List Models PASSED
{
  "models": [...]
}

🧪 Step 3: File Upload Test
----------------------------------------
✅ File Upload PASSED
{
  "model_id": "mdl_abc123...",
  "filename": "portfolio_risk_model.xlsx",
  "formulas_count": 8,
  "status": "uploaded"
}

🧪 Step 4: Start Simulation
----------------------------------------
✅ Start Simulation PASSED
{
  "simulation_id": "sim_xyz789...",
  "status": "queued",
  "estimated_completion": "2024-01-15T10:45:00Z"
}

🧪 Step 5: Check Simulation Progress
----------------------------------------
Status: running, Progress: 25%
Status: running, Progress: 75%
Status: completed, Progress: 100%
✅ Simulation completed!

🎯 OVERALL RESULT: 6/6 tests passed
🎉 ALL TESTS PASSED! The API is working correctly.
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Connection Refused
```bash
❌ Connection error: Connection refused
```
**Solution**: Check if the server is running and accessible
```bash
# Test basic connectivity
curl -I http://209.51.170.185:8000
```

#### 2. Invalid API Key
```bash
❌ Health check failed: 401 - Invalid API key
```
**Solution**: Verify your API key format and validity
```bash
# Check API key format (should start with mc_test_ or mc_live_)
echo $API_KEY | head -c 20
```

#### 3. File Upload Fails
```bash
❌ Upload failed: 413 - File size exceeds limit
```
**Solution**: Check file size limits for your subscription tier
- Starter: 10MB
- Professional: 50MB  
- Enterprise: 500MB

#### 4. Simulation Timeout
```bash
⏰ Timeout: Simulation did not complete within 60 seconds
```
**Solution**: Normal for complex models. Check progress manually:
```bash
curl -H "Authorization: Bearer $API_KEY" \
  http://209.51.170.185:8000/monte-carlo-api/simulations/{sim_id}/progress
```

### Debug Mode

For detailed debugging, enable verbose output:

```bash
# Bash script with debug
bash -x quick_api_test.sh

# Python script with debug
python3 api_test_client.py --test all 2>&1 | tee debug.log
```

## 📊 Performance Benchmarks

### Expected Response Times
| Endpoint | Expected Time | Acceptable Limit |
|----------|---------------|------------------|
| Health Check | < 500ms | 2s |
| File Upload (10MB) | < 5s | 30s |
| Start Simulation | < 2s | 10s |
| Progress Check | < 500ms | 2s |
| Results (10K iterations) | < 30s | 300s |

### GPU vs CPU Performance
- **GPU (Ultra Engine)**: 10,000 iterations in ~15-30 seconds
- **CPU (Standard)**: 10,000 iterations in ~2-5 minutes

## 🔄 Automated Testing

### Continuous Testing Script
```bash
#!/bin/bash
# Run tests every hour and log results
while true; do
    echo "$(date): Starting API tests..."
    python3 api_test_client.py >> api_test.log 2>&1
    sleep 3600
done
```

### Load Testing
```bash
# Test multiple concurrent requests
for i in {1..5}; do
    python3 api_test_client.py --test health &
done
wait
```

## 📈 Integration Examples

### Customer Integration Template
```python
import requests

class MonteCarloClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://209.51.170.185:8000/monte-carlo-api"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def upload_model(self, file_path):
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/models",
                files={"file": f},
                headers=self.headers
            )
        return response.json()
    
    def run_simulation(self, model_id, variables, output_cells):
        config = {
            "model_id": model_id,
            "variables": variables,
            "output_cells": output_cells,
            "iterations": 10000
        }
        response = requests.post(
            f"{self.base_url}/simulations",
            json=config,
            headers=self.headers
        )
        return response.json()

# Usage
client = MonteCarloClient("your_api_key_here")
model = client.upload_model("my_model.xlsx")
results = client.run_simulation(model["model_id"], variables, outputs)
```

## 📞 Support

If tests fail consistently:

1. **Check server status**: Is the backend service running?
2. **Verify API key**: Are you using the correct authentication?
3. **Test network**: Can you reach the server from your Mac?
4. **Review logs**: Check both test output and server logs
5. **Contact support**: Provide test results file for debugging

## 📁 Test Artifacts

After running tests, you'll find:
- `api_test_results_[timestamp].json` - Detailed test results
- `test_results_[simulation_id].pdf` - Downloaded PDF report
- `test_results_[simulation_id].xlsx` - Downloaded Excel report  
- `test_results_[simulation_id].json` - Downloaded JSON results

These files can be used to verify the API is working correctly and producing expected outputs.
