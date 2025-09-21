# Engine Selection Critical Fix

## Issue Description

**Problem**: Users could select the "Arrow Monte Carlo Engine" in the engine selection modal, but the simulation would always run with the "Enhanced" engine instead.

**Impact**: 
- User engine selection was completely ignored
- Arrow engine never used despite being selected
- Frontend showed "ENGINE: Enhanced" even when Arrow was chosen
- Backend always fell back to Enhanced engine regardless of user choice

## Root Cause Analysis

The issue was a **missing field in the backend schema**:

1. **Frontend**: Correctly sent `engine_type: "arrow"` in the simulation request
2. **Backend Schema**: `SimulationRequest` class was missing the `engine_type` field 
3. **Backend Service**: Used `getattr(request, 'engine_type', 'enhanced')` which always returned 'enhanced' because the field didn't exist
4. **Result**: Arrow engine selection was silently ignored, Enhanced engine always used

### Technical Details

**File**: `backend/simulation/schemas.py`  
**Line**: 22 (SimulationRequest class)

**Problem Code**:
```python
class SimulationRequest(BaseModel):
    simulation_id: str = Field(default_factory=lambda: str(uuid4()))
    file_id: str
    result_cell_coordinate: str
    result_cell_sheet_name: str
    variables: List[VariableConfig]
    iterations: int = 1000
    # ❌ MISSING: engine_type field
```

**File**: `backend/simulation/service.py`  
**Line**: 303

**Problem Code**:
```python
engine_type = getattr(request, 'engine_type', 'enhanced')  # Always returned 'enhanced'
```

## Fix Applied

### 1. Added Missing Schema Field

**File**: `backend/simulation/schemas.py`  
**Change**: Added `engine_type` field to `SimulationRequest`

```python
class SimulationRequest(BaseModel):
    simulation_id: str = Field(default_factory=lambda: str(uuid4()))
    file_id: str
    result_cell_coordinate: str
    result_cell_sheet_name: str
    variables: List[VariableConfig]
    iterations: int = 1000
    engine_type: str = "enhanced"  # ✅ ADDED: Engine type field with default
```

### 2. Fixed Service Access Pattern

**File**: `backend/simulation/service.py`  
**Change**: Replaced `getattr` with direct field access

```python
# Before (always returned 'enhanced'):
engine_type = getattr(request, 'engine_type', 'enhanced')

# After (correctly uses selected engine):
engine_type = request.engine_type  # Now properly defined in SimulationRequest schema
```

## Validation

After applying this fix:

1. **Frontend → Backend**: Engine selection properly passed in request
2. **Backend Processing**: Correct engine type accessed from request object
3. **Engine Execution**: Arrow engine runs when selected
4. **Progress Display**: Shows correct engine type (Arrow vs Enhanced)
5. **Correlation Fix**: Arrow engine includes the first variable correlation fix

## Expected Results

- **User selects Arrow**: Simulation runs with Arrow engine, shows "ENGINE: Arrow"
- **User selects Enhanced**: Simulation runs with Enhanced engine, shows "ENGINE: Enhanced"  
- **User selects Standard**: Simulation runs with Standard engine, shows "ENGINE: Standard"
- **Default behavior**: Enhanced engine (backwards compatible)

## Testing

To test the fix:
1. Navigate to simulation setup
2. Select "Arrow Monte Carlo Engine" in engine selection modal
3. Run simulation
4. Verify progress tracker shows "ENGINE: Arrow" 
5. Verify backend logs show Arrow engine execution
6. Verify first variable shows proper variation (not perfect correlation)

## Files Modified

- `backend/simulation/schemas.py` (line 22): Added `engine_type` field
- `backend/simulation/service.py` (line 303): Fixed field access pattern

## Related Issues

- **First Variable Perfect Correlation**: Fixed in Arrow engine (ARROW_FIRST_VARIABLE_CORRELATION_FIX.md)
- **Engine Selection UI**: Working correctly, issue was backend-only
- **Frontend State Management**: Working correctly, no changes needed

---

**Status**: ✅ **FIXED** - Engine selection now properly respected  
**Priority**: Critical - Core functionality  
**Deployment**: Applied in Docker rebuild

**Before**: All simulations used Enhanced engine regardless of selection  
**After**: Simulation uses the engine selected by user in modal 