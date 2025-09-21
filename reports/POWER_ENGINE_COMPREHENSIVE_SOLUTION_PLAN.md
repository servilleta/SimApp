# Power Engine - Comprehensive Solution Plan

## Executive Summary

The Power Engine is **functionally complete** but has a deployment/routing issue causing the wrong engine to be selected. This document provides a comprehensive plan to ensure Power Engine works reliably.

## Current State Analysis

### ✅ What's Working
1. **Power Engine Core Functionality**
   - Monte Carlo simulation with real variance
   - GPU acceleration active
   - VLOOKUP support
   - Results generation (mean ~14M)
   
2. **Progress Callbacks** [[memory:2150576]]
   - All callbacks executing successfully
   - Proper dictionary format
   - Stage names aligned with frontend

3. **Backend Processing**
   - Completes in ~85 seconds
   - Handles 1000 iterations
   - Memory management working

### ❌ The Issue
1. **Engine Selection Mismatch**
   - Some code paths use WorldClassMonteCarloEngine instead of PowerMonteCarloEngine
   - Two different service.py files exist:
     - `/app/simulation/service.py` (correct, uses PowerMonteCarloEngine)
     - `/app/modules/simulation/engines/service.py` (incorrect constructor)

## Root Cause Analysis

### 1. Multiple Code Paths
```
backend/
├── simulation/
│   └── service.py          # ✅ Correct implementation
└── modules/
    └── simulation/
        └── engines/
            └── service.py  # ❌ Wrong constructor signature
```

### 2. Constructor Mismatch
```python
# Correct (simulation/service.py):
engine = PowerMonteCarloEngine(
    file_id=file_id,
    target_cell=target_cell_name,
    target_sheet=target_sheet_name,
    mc_inputs=mc_inputs,
    iterations=iterations,
    progress_callback=progress_callback,
    simulation_id=sim_id
)

# Wrong (modules/simulation/engines/service.py):
power_engine = PowerMonteCarloEngine(iterations=iterations)
```

## Comprehensive Solution Plan

### Phase 1: Immediate Fix (15 minutes)

#### Step 1: Verify Current Routing
```bash
# Check which service is being used
docker exec project-backend-1 grep -r "from.*service import" /app/simulation/
```

#### Step 2: Fix Constructor Mismatch
If modules service is used, update `/app/modules/simulation/engines/service.py`:
```python
# Fix the _run_power_simulation function
power_engine = PowerMonteCarloEngine(
    file_id=file_path.replace('uploads/', ''),
    target_cell=target_cell.split('!')[-1],
    target_sheet=target_cell.split('!')[0],
    mc_inputs=mc_inputs,
    iterations=iterations,
    progress_callback=progress_cb,
    simulation_id=sim_id,
    constants={c.name: c.value for c in constants}
)
```

#### Step 3: Rebuild and Deploy
```bash
docker-compose down
docker-compose build --no-cache backend
docker-compose up -d
```

### Phase 2: Testing Protocol (30 minutes)

#### Test Case 1: Basic Power Engine Test
1. Upload simple Excel file
2. Select Power Engine
3. Configure 1 variable
4. Run 100 iterations
5. Verify:
   - Progress updates 0-100%
   - Results display
   - No 18% stuck issue

#### Test Case 2: Complex File Test
1. Upload file with VLOOKUP
2. Select Power Engine
3. Configure 3 variables
4. Run 1000 iterations
5. Verify:
   - VLOOKUP works
   - Variance in results
   - Performance < 3 minutes

#### Test Case 3: Progress Tracking Test
1. Start simulation
2. Monitor backend logs:
```bash
docker logs -f project-backend-1 | grep -E "(Progress callback|Stage|Power Engine)"
```
3. Verify all stages report

### Phase 3: Long-term Improvements (1-2 days)

#### 1. Consolidate Code Paths
- Remove duplicate service implementations
- Single source of truth for engine selection
- Clear import paths

#### 2. Add Engine Validation
```python
def validate_engine_selection(engine_type: str, engine_instance):
    """Ensure correct engine is instantiated"""
    expected_map = {
        "power": PowerMonteCarloEngine,
        "enhanced": WorldClassMonteCarloEngine,
        "big": BigFilesEngine,
        "standard": MonteCarloSimulation
    }
    assert isinstance(engine_instance, expected_map[engine_type])
```

#### 3. Improve Logging
```python
logger.info(f"🎯 Engine Selection: {engine_type} -> {type(engine).__name__}")
```

#### 4. Add Health Checks
```python
@router.get("/health/engines")
async def check_engines():
    return {
        "power": test_power_engine(),
        "enhanced": test_enhanced_engine(),
        "big": test_big_engine()
    }
```

## Monitoring & Diagnostics

### Real-time Monitoring Commands
```bash
# Watch for engine selection
docker logs -f project-backend-1 | grep -E "(Engine selected|PowerMonteCarloEngine|WorldClass)"

# Monitor progress updates
docker logs -f project-backend-1 | grep -E "(Progress callback|Stage \d)"

# Check Redis progress
docker exec project-redis-1 redis-cli --scan --pattern "simulation:progress:*"
```

### Debug Checklist
- [ ] Correct engine selected in logs?
- [ ] Progress callbacks executing?
- [ ] Redis progress data updating?
- [ ] Frontend receiving WebSocket updates?
- [ ] No constructor errors in logs?

## Prevention Strategy

### 1. Code Review Process
- All engine changes require testing
- Verify constructor signatures match
- Check progress callback format

### 2. Automated Tests
```python
def test_power_engine_constructor():
    """Ensure Power Engine can be instantiated correctly"""
    engine = PowerMonteCarloEngine(
        file_id="test",
        target_cell="A1",
        target_sheet="Sheet1",
        mc_inputs=[],
        iterations=10,
        simulation_id="test-123"
    )
    assert engine is not None
```

### 3. Documentation
- Clear engine selection flow
- Constructor requirements
- Progress callback format

## Success Metrics

### Immediate Success (Today)
- [ ] Power Engine runs without 18% stuck issue
- [ ] Progress updates flow to frontend
- [ ] Results display correctly

### Short-term Success (This Week)
- [ ] All test cases pass
- [ ] No engine selection errors
- [ ] Consistent performance

### Long-term Success (This Month)
- [ ] Zero engine-related bugs
- [ ] Clear monitoring dashboard
- [ ] Automated testing in place

## Conclusion

The Power Engine is fundamentally sound. The issues are deployment/routing related and can be fixed quickly. Following this plan will ensure:

1. **Immediate Relief**: Fix stuck progress issue
2. **Reliable Operation**: Correct engine selection
3. **Future Prevention**: Better testing and monitoring

The key is ensuring the correct service.py file is used and that constructor signatures match between the engine definition and its usage. 