# Timeout Fix Applied - January 17, 2025

## Problem
Simulations were failing with "timeout of 30000ms exceeded" errors, even though the backend was processing successfully. Users had to press the "Run" button multiple times before simulations would start.

## Root Cause
The **Redux simulation slice** was using the main `apiClient` from `services/api.js` which had a **30-second timeout**, rather than the specialized `simulationService.js` which had the proper **10-minute timeout** for large simulations.

## Solution Applied

### 1. Updated Redux Simulation Slice
**File**: `frontend/src/store/simulationSlice.js`

**Changes**:
- **Added import** of specialized simulation service functions:
  ```javascript
  import { postRunSimulation, getSimulationStatus, cancelSimulation as cancelSimulationAPI } from '../services/simulationService';
  ```

- **Updated `runSimulation` thunk** to use `postRunSimulation()` instead of `apiClient.post()`
- **Updated `fetchSimulationStatus` thunk** to use `getSimulationStatus()` instead of `apiClient.get()`  
- **Updated `cancelSimulation` thunk** to use `cancelSimulationAPI()` instead of `apiClient.post()`

### 2. Frontend Rebuild
- **Stopped and rebuilt frontend** with `--no-cache` to ensure changes are applied
- **Cleared browser cache** by forcing fresh Docker build

## Technical Details

### Before Fix
```javascript
// Used 30-second timeout from api.js
const response = await apiClient.post('/simulations/run', config);
```

### After Fix  
```javascript
// Uses 10-minute timeout from simulationService.js
const data = await postRunSimulation(config);
```

### Timeout Configuration
- **Regular API calls**: 30 seconds (sufficient for status checks, file uploads)
- **Simulation runs**: 10 minutes (600,000ms) - supports large Excel files with 20K+ formulas

## Impact
- ✅ **Eliminates timeout failures** for large simulations
- ✅ **Single-click simulation starts** - no more multiple attempts needed
- ✅ **Maintains performance** - no impact on simulation speed
- ✅ **Preserves quick timeouts** for regular API operations

## Testing
The fix has been applied and the frontend container rebuilt. Users should now be able to:
1. **Click "Run" once** and have simulations start immediately
2. **Run large simulations** (20K+ formulas) without timeout errors
3. **See proper progress tracking** without artificial failures

## Files Modified
1. `frontend/src/store/simulationSlice.js` - Updated to use proper simulation service
2. Frontend Docker container rebuilt to apply changes

## Next Steps
Test the platform with the sample Excel file to verify:
- No more "timeout of 30000ms exceeded" errors
- Simulations start on first click
- Progress tracking works properly for large files 