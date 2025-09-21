# Monte Carlo Simulation Test Suite

## Overview

This comprehensive test suite validates that the Monte Carlo simulation platform is working correctly with the FULL_EVALUATION approach, processing all formulas in the Excel model.

## Test Files Created

### 1. Unit Tests
- **`test_monte_carlo_validation.py`** - Core functionality tests
  - Monte Carlo variable variation
  - Dependency tracking
  - Formula evaluation chain
  - Statistical results
  - Ultra Engine performance
  - Edge cases

### 2. Integration Tests  
- **`test_simulation_integration.py`** - Service integration tests
  - Full simulation workflow
  - Formula evaluation completeness
  - Monte Carlo variation
  - Constants management
  - Backend logs validation

### 3. End-to-End Tests
- **`test_monte_carlo_e2e.py`** - API testing
  - Backend health check
  - Simulation creation
  - Simulation execution
  - Results validation
  - Multiple targets

### 4. Statistical Tests
- **`test_statistical_validation.py`** - Mathematical correctness
  - Uniform distribution
  - Normal distribution
  - Histogram accuracy
  - Percentile calculations
  - Sensitivity analysis
  - Convergence properties

### 5. Utility Tests
- **`test_quick_validation.py`** - Quick validation checklist
- **`test_backend_logs.py`** - Backend log analyzer
- **`run_all_tests.py`** - Master test runner

## Running the Tests

### Quick Backend Check
```bash
cd backend
python3 test_backend_logs.py
```

### Statistical Validation
```bash
python3 test_statistical_validation.py
```

### Run All Tests
```bash
python3 run_all_tests.py
```

## What to Look For

### ✅ Success Indicators
1. **Backend Logs show:**
   - `[FULL_EVALUATION] Processing complete Excel model with 1990 formulas`
   - `[CONSTANTS] Using X Excel constants for non-calculated cells`
   - Different target values for each iteration

2. **Results show:**
   - Non-zero standard deviation
   - Values changing between iterations  
   - Reasonable value ranges (no extremes like 1e+25)

3. **No occurrences of:**
   - `ULTRA-SELECTIVE`
   - Same value repeated for all iterations

### ❌ Failure Indicators
1. Zero standard deviation
2. Same value for all iterations
3. Extreme values (exponential explosion)
4. ULTRA-SELECTIVE approach in logs

## Rebuilding and Testing

After making changes:

```bash
# Stop containers
docker-compose down

# Rebuild backend
docker-compose build --no-cache backend

# Start containers
docker-compose up -d

# Wait for startup
sleep 10

# Check logs
python3 backend/test_backend_logs.py
```

## Test Results Location

Test results are saved with timestamps:
- `monte_carlo_validation_report_*.json`
- `simulation_integration_report_*.json`
- `monte_carlo_e2e_report_*.json`
- `statistical_validation_report_*.json`
- `monte_carlo_test_report_*.json` (comprehensive)

## Troubleshooting

### Backend Not Responding
```bash
docker-compose ps
docker-compose logs backend | tail -50
```

### Old Code Still Running
```bash
docker-compose down
docker system prune -f
docker-compose build --no-cache backend
docker-compose up -d
```

### Missing Dependencies
```bash
pip install numpy scipy aiohttp asyncio
```

## Key Test Validations

1. **Variable Variation**: F4, F5, F6 properly vary between [min, max]
2. **Formula Chain**: All 1,990 formulas are evaluated
3. **No Double Calculation**: Constants exclude calculated cells
4. **Statistical Accuracy**: Results match expected distributions
5. **Performance**: Ultra Engine handles iterations efficiently

## Success Criteria

The Monte Carlo simulation is working correctly when:
- ✅ All formulas are evaluated (FULL_EVALUATION)
- ✅ Target values show proper variation
- ✅ Statistical properties are mathematically sound
- ✅ No exponential value explosion
- ✅ Platform works with any Excel file structure 