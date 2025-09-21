import { createSlice, createAsyncThunk, createSelector } from '@reduxjs/toolkit';
import { postRunSimulation, getSimulationStatus, getSimulationProgress, cancelSimulation as cancelSimulationAPI, createSimulationId } from '../services/simulationService';
import apiClient from '../services/api';
import { v4 as uuidv4 } from 'uuid';
import simulationLogger from '../services/simulationLogger';
import {
  normalizeSimulationId,
  getParentId,
  generateChildSimulationId,
  validateSimulationId,
  logIdValidationWarning
} from '../utils/simulationIdUtils';

// Thunk for running a new simulation
export const runSimulation = createAsyncThunk(
  'simulation/run',
  async (simulationConfig, { getState, rejectWithValue, dispatch }) => {
    const { 
      variables, 
      resultCells,
      iterations, 
      tempId, 
      engine_type,
      fileId,
      batch_id 
    } = simulationConfig;
    
    // 🚀 NEW: Get real simulation ID from backend first (eliminates temp ID system)
    let simulationId;
    try {
      const idResponse = await createSimulationId();
      simulationId = idResponse.simulation_id;
      console.log('🚀 [ID_CREATION] Got real simulation ID from backend:', simulationId);
    } catch (error) {
      console.error('🚨 [ID_CREATION] Failed to get real ID, generating fallback ID:', error);
      simulationId = `fallback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    // Ensure we always have a valid simulation ID
    if (!simulationId) {
      simulationId = `emergency_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    simulationLogger.initializeSimulation(simulationId, simulationConfig);
    
    const excelState = getState().excel;
    const { filename, original_filename } = excelState.fileInfo || {};
    // 🔥 FIX: Use current filename first, fallback to original_filename only if no current filename
    // This ensures new uploads override old filenames from loaded simulations
    const actualFilename = filename || original_filename;
    
    // Log validation stage
    simulationLogger.logStage(simulationId, simulationLogger.STAGES.REQUEST_VALIDATION, 'in_progress', {
      message: 'Validating simulation request'
    });
    
    if (!fileId) {
        simulationLogger.logError(simulationId, simulationLogger.STAGES.REQUEST_VALIDATION, 
          new Error('No file has been uploaded or processed.'));
        return rejectWithValue('No file has been uploaded or processed.');
    }
    if (!resultCells || resultCells.length === 0) {
        simulationLogger.logError(simulationId, simulationLogger.STAGES.REQUEST_VALIDATION,
          new Error('No result cells specified for the simulation.'));
        return rejectWithValue('No result cells specified for the simulation.');
    }
    
    simulationLogger.logStage(simulationId, simulationLogger.STAGES.REQUEST_VALIDATION, 'success', {
      message: 'Request validation completed',
      fileId,
      variableCount: variables?.length || 0,
      resultCellCount: resultCells?.length || 0
    });
    
    // For now, we'll just use the first result cell. 
    // This can be expanded later to handle multiple result cells.
    const resultCell = resultCells[0];
    const sheetName = resultCell.sheetName || resultCell.sheet_name || 'Sheet1';
    
    try {
      // Log API call preparation
      simulationLogger.logStage(simulationId, simulationLogger.STAGES.API_CALL, 'in_progress', {
        message: 'Preparing API call to backend'
      });
      
      // ✅ FIX: Transform variables to match the backend's `VariableConfig` schema
      const transformedVariables = variables.map(v => {
        // Variables can have either direct properties or params object
        const hasParams = v.params && typeof v.params === 'object';
        const hasDirectProps = v.min_value !== undefined && v.most_likely !== undefined && v.max_value !== undefined;
        
        if (!hasParams && !hasDirectProps) {
          throw new Error(`Variable ${v.name} is missing distribution parameters.`);
        }
        
        return {
          name: v.name, // Cell reference for backend processing
          display_name: v.display_name || v.variableName || v.name, // User-defined name for display
          sheet_name: v.sheetName || v.sheet_name || sheetName,
          min_value: hasParams ? v.params.min : v.min_value,
          most_likely: hasParams ? v.params.likely : v.most_likely,
          max_value: hasParams ? v.params.max : v.max_value,
        };
      });

      // ✅ FIX: Transform target cells to include display names
      const transformedTargetCells = resultCells.map(cell => ({
        name: cell.name,
        display_name: cell.display_name || cell.variableName || cell.name,
        sheet_name: cell.sheetName || cell.sheet_name || sheetName,
        format: cell.format || 'decimal',
        decimal_places: cell.decimalPlaces || 2
      }));

      const payload = {
        file_id: fileId,
        variables: transformedVariables,
        result_cell_coordinate: resultCell.name,
        result_cell_sheet_name: sheetName,
        target_cells: simulationConfig.targetCells || null,  // Add target_cells array
        target_cells_info: transformedTargetCells,  // ✅ NEW: Add target cell display names
        iterations: iterations,
        engine_type: engine_type || 'ultra',
        original_filename: actualFilename, // ✅ Fixed: Use actualFilename with fallback
        batch_id: batch_id || null,  // Include batch_id if provided
        simulation_id: simulationId  // 🚀 NEW: Use real simulation ID instead of temp_id
      };

      simulationLogger.logStage(simulationId, simulationLogger.STAGES.API_CALL, 'in_progress', {
        message: 'Sending request to backend',
        endpoint: '/api/simulations/run',
        payload: {
          ...payload,
          variables: `${transformedVariables.length} variables`,
          // Don't log full payload for security
        }
      });

      // 🚨 CRITICAL DEBUG: Validate payload before API call
      console.log('🚨 [DEBUG] About to call postRunSimulation with payload:', {
        fileId: payload.file_id,
        variablesCount: payload.variables?.length,
        targetCellsCount: payload.target_cells?.length,
        iterations: payload.iterations,
        engineType: payload.engine_type
      });
      
      // 🚀 EARLY CONNECTION: Connect WebSocket BEFORE API call to catch all progress updates
      console.log('🚀 [EARLY_CONNECTION] Connecting WebSocket before simulation starts');
      
      // Import WebSocket connection function dynamically to avoid circular imports
      const { connectWebSocket } = await import('../services/simulationService');
      
      // Connect WebSocket early using the simulation ID we already have
      const wsConnection = connectWebSocket(simulationId, dispatch);
      console.log('🚀 [EARLY_CONNECTION] WebSocket connected, ready to receive updates');

      const startTime = Date.now();
      
      console.log('🚨 [DEBUG] Calling postRunSimulation NOW (WebSocket already connected)...');
      const data = await postRunSimulation(payload);
      console.log('🚨 [DEBUG] postRunSimulation returned:', data);
      const responseTime = Date.now() - startTime;
      
      // Log successful API response
      simulationLogger.logApiResponse(simulationId, '/api/simulations/run', 200, {
        ...data,
        responseTime: `${responseTime}ms`
      });

      // Note: Progress tracking is handled automatically through Redux state
      // The simulation will be added to multipleResults in the fulfilled case

      // Log backend response processing
      simulationLogger.logStage(simulationId, simulationLogger.STAGES.BACKEND_RECEIVED, 'success', {
        message: 'Backend accepted simulation request',
        backendSimulationId: data.simulation_id,
        status: data.status,
        isBatch: !!(data.batch_simulation_ids && data.batch_simulation_ids.length > 0)
      });

      // Check if this is a batch response with multiple simulation IDs
      if (data.batch_simulation_ids && data.batch_simulation_ids.length > 0) {
        simulationLogger.logStage(simulationId, simulationLogger.STAGES.BACKGROUND_TASK, 'success', {
          message: 'Batch simulation queued',
          batchId: data.simulation_id,
          individualSimulations: data.batch_simulation_ids.length
        });
        
        // Copy logs from temp ID to each individual simulation ID for backend tracking
        const tempLogs = simulationLogger.getSimulationLogs(simulationId);
        if (tempLogs && data.batch_simulation_ids) {
          data.batch_simulation_ids.forEach((individualId, index) => {
            const realIdLogs = {
              ...tempLogs,
              simulationId: individualId
            };
            simulationLogger.logs.set(individualId, realIdLogs);
            
            simulationLogger.logStage(individualId, simulationLogger.STAGES.BACKGROUND_TASK, 'success', {
              message: `Batch simulation ${index + 1} ID mapping completed`,
              tempId: simulationId,
              realId: individualId,
              batchIndex: index + 1
            });
          });
        }
        
        // Return batch response with individual simulation IDs
        return {
          isBatch: true,
          batch_id: data.simulation_id,
          batch_simulation_ids: data.batch_simulation_ids,
          targetCells: simulationConfig.targetCells,
          requested_engine_type: engine_type || 'ultra',
          status: data.status,
          tempId: simulationId
        };
      }

      simulationLogger.logStage(simulationId, simulationLogger.STAGES.BACKGROUND_TASK, 'success', {
        message: 'Single simulation queued',
        backendSimulationId: data.simulation_id
      });

      // Copy logs from temp ID to real ID for backend tracking
      const tempLogs = simulationLogger.getSimulationLogs(simulationId);
      console.log(`[DEBUG] Copying logs from ${simulationId} to ${data.simulation_id}`, {
        tempLogsExists: !!tempLogs,
        tempLogsContent: tempLogs,
        realId: data.simulation_id
      });
      
      if (tempLogs && data.simulation_id) {
        // Clone the logs with the new simulation ID
        const realIdLogs = {
          ...tempLogs,
          simulationId: data.simulation_id
        };
        simulationLogger.logs.set(data.simulation_id, realIdLogs);
        
        // Also keep the temp logs for now to avoid timing issues
        simulationLogger.logs.set(simulationId, tempLogs);
        
        simulationLogger.logStage(data.simulation_id, simulationLogger.STAGES.BACKGROUND_TASK, 'success', {
          message: 'Simulation ID mapping completed',
          tempId: simulationId,
          realId: data.simulation_id
        });
        
        console.log(`[DEBUG] Logs copied successfully. Total logs now:`, simulationLogger.logs.size);
      } else {
        console.warn(`[DEBUG] Failed to copy logs - tempLogs: ${!!tempLogs}, realId: ${data.simulation_id}`);
      }

      // 🚀 FIXED: Use the backend's actual simulation ID, not frontend-generated ID
      return {
        realId: data.simulation_id, // Use the backend's actual simulation ID
        status: data.status,
        targetName: resultCell.name,
        resultCellCoordinate: resultCell.name,
        requested_engine_type: engine_type || 'ultra',
        batch_simulation_ids: data.batch_simulation_ids // Pass through for frontend
      };
      
    } catch (error) {
      console.error('🚨 [simulationSlice] Simulation run failed:', error);
      console.error('🚨 [DEBUG] Error details:', {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        stack: error.stack
      });
      
      // Log the error
      simulationLogger.logError(simulationId, simulationLogger.STAGES.API_CALL, error, {
        httpStatus: error.response?.status,
        responseData: error.response?.data
      });
      
      let errorMessage = 'Failed to initiate simulation processing.';
      if (error.response) {
        // Log API error response
        simulationLogger.logApiResponse(simulationId, '/api/simulations/run', 
          error.response.status, error.response.data, error);
          
        // Backend error with a response
        const detail = error.response.data?.detail;
        if (typeof detail === 'string') {
          if (detail.includes("is busy. Queuing simulation")) {
            errorMessage = "A simulation is already running. Your new simulation is queued.";
          } else {
            errorMessage = detail;
          }
        } else if (detail) {
          errorMessage = JSON.stringify(detail);
        }
      } else if (error.message) {
        // Generic client-side error
        errorMessage = error.message;
      }
      
      simulationLogger.logStage(simulationId, simulationLogger.STAGES.ERROR_HANDLING, 'failure', {
        message: 'Simulation failed to start',
        error: errorMessage
      });
      
      return rejectWithValue(errorMessage);
    }
  }
);

// Thunk for fetching simulation status and results
export const fetchSimulationStatus = createAsyncThunk(
  'simulation/fetchStatus',
  async (simulationId, { rejectWithValue }) => {
    try {
      const data = await getSimulationStatus(simulationId);
      return data;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// REMOVED: Helper function moved to utils/simulationIdUtils.js for consistency

// Thunk for fetching real-time simulation progress
export const fetchSimulationProgress = createAsyncThunk(
  'simulation/fetchProgress',
  async (simulationId, { rejectWithValue }) => {
    try {
      // Normalize simulation ID before making API call
      const normalizedId = normalizeSimulationId(simulationId);
      const data = await getSimulationProgress(normalizedId);
      
      if (data) {
        return {
          simulationId: normalizedId, // Return normalized ID for consistency
          normalizedId: normalizedId,
          ...data
        };
      }
      
      return null;
    } catch (error) {
      // Don't reject for progress failures to avoid breaking polling
      console.warn(`Progress fetch failed for ${simulationId}:`, error.message);
      return null;
    }
  }
);

// Thunk for fetching batch simulation results using parent-first flow
export const fetchBatchSimulationResults = createAsyncThunk(
  'simulation/fetchBatchResults',
  async (batchSimulationIds, { dispatch, getState }) => {
    const results = [];
    
    // First, try to get the parent simulation ID
    const firstChildId = batchSimulationIds[0];
    if (!firstChildId) {
      console.warn('[fetchBatchResults] No child IDs provided');
      return results;
    }
    
    const parentSimulationId = normalizeSimulationId(firstChildId);
    console.log('[fetchBatchResults] 🎯 Fetching parent simulation for batch:', parentSimulationId);
    
    try {
      // Fetch parent simulation data once
      const parentData = await getSimulationStatus(parentSimulationId);
      console.log('[fetchBatchResults] ✅ Parent simulation data received:', parentData.status);
      
      // Check if parent has multi_target_result
      if (parentData.multi_target_result && parentData.multi_target_result.statistics) {
        console.log('[fetchBatchResults] 🎯 Using parent multi_target_result for child objects');
        
        // Derive child objects from multi_target_result.statistics
        const targetStats = parentData.multi_target_result.statistics;
        batchSimulationIds.forEach((childId, index) => {
          const targetName = Object.keys(targetStats)[index];
          if (targetName && targetStats[targetName]) {
            const stats = targetStats[targetName];
            const childData = {
              simulation_id: childId,
              status: 'completed',
              target_name: targetName,
              mean: stats.mean,
              median: stats.median,
              std: stats.std,
              min: stats.min,
              max: stats.max,
              iterations_run: parentData.multi_target_result.total_iterations,
              message: `Target ${targetName} completed successfully`,
              results: {
                mean: stats.mean,
                median: stats.median,
                std_dev: stats.std,
                min_value: stats.min,
                max_value: stats.max,
                percentiles: stats.percentiles,
                histogram: stats.histogram || { bins: [], values: [] },
                raw_values: parentData.multi_target_result.target_results?.[targetName] || [],
                iterations_run: parentData.multi_target_result.total_iterations,
                errors: [],
                sensitivity_analysis: parentData.multi_target_result.sensitivity_data?.[targetName] || [],
                target_display_name: targetName
              }
            };
            
            results.push({ childId, data: childData });
            console.log('[fetchBatchResults] ✅ Derived child data for:', targetName);
          } else {
            console.warn('[fetchBatchResults] ⚠️ No target data found for index:', index);
            results.push({ childId, error: 'no_target_data' });
          }
        });
        
        return results;
      }
    } catch (error) {
      console.error('[fetchBatchResults] ❌ Failed to fetch parent simulation:', parentSimulationId, error);
    }
    
    // Fallback: Try to fetch individual child simulations (legacy behavior)
    console.log('[fetchBatchResults] 🔄 Falling back to individual child fetching');
    for (const childId of batchSimulationIds) {
      try {
        console.log('[fetchBatchResults] 🔍 Fetching results for child simulation:', childId);
        
        // Normalize child ID before making API call
        const normalizedChildId = normalizeSimulationId(childId);
        console.log('[fetchBatchResults] 🔧 Normalized child ID:', childId, '->', normalizedChildId);
        
        const childData = await getSimulationStatus(normalizedChildId);
        console.log('[fetchBatchResults] ✅ Child simulation data received:', normalizedChildId, 'status:', childData.status);
        results.push({ childId, data: childData });
      } catch (error) {
        // Handle 404s gracefully - child simulations may not exist individually
        if (error.message && error.message.includes('404')) {
          console.warn('[fetchBatchResults] ⚠️ Child simulation not found (404):', childId, '- This is expected for batch simulations');
          results.push({ childId, error: '404_not_found' });
        } else {
          console.error('[fetchBatchResults] ❌ Failed to fetch child simulation results:', childId, error);
          results.push({ childId, error: error.message });
        }
      }
    }
    
    return results;
  }
);

// Thunk for cancelling a simulation
export const cancelSimulation = createAsyncThunk(
  'simulation/cancelSimulation',
  async (simulationId, { rejectWithValue }) => {
    try {
      const data = await cancelSimulationAPI(simulationId);
      return {
        simulationId,
        ...data
      };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);


const initialState = {
  currentSimulationId: null,
  currentParentSimulationId: null, // Canonical parent ID for progress tracking
  status: 'idle', // idle, pending, running, completed, failed
  results: null,
  multipleResults: [], // Array to store results from multiple simulations
  error: null,
  iterations: 1000,
  // Progress tracking state
  progressData: {
    progressPercentage: 0,
    currentStage: 'idle',
    currentIteration: 0,
    totalIterations: 0,
    stageDescription: '',
    timestamp: null,
    hasCompleted: false // Track completion to stop polling
  },
  // Batch fetch tracking - used to trigger fetchBatchSimulationResults
  _pendingBatchFetch: null // Array of simulation IDs pending batch fetch
};

const simulationSlice = createSlice({
  name: 'simulation',
  initialState,
  reducers: {
    clearSimulation: (state) => {
      console.log('[simulationSlice] 🧹 Clearing all simulation data');
      state.currentSimulationId = null;
      state.currentParentSimulationId = null;
      state.status = 'idle';
      state.results = null;
      state.multipleResults = [];
      state.error = null;
      // Reset progress data
      state.progressData = {
        progressPercentage: 0,
        currentStage: 'idle',
        currentIteration: 0,
        totalIterations: 0,
        stageDescription: '',
        timestamp: null,
        hasCompleted: false
      };
    },
    setIterations: (state, action) => {
      state.iterations = action.payload;
    },
    updateSimulationId: (state, action) => {
      const { tempId, realId, status } = action.payload;
      console.log('🔧 [Redux] updateSimulationId called:', { tempId, realId, status });
      
      // 🚀 CRITICAL FIX: Update currentSimulationId for progress polling
      if (state.currentSimulationId === tempId) {
        console.log('🔧 [Redux] 🚀 CRITICAL: Updating currentSimulationId from temp to real ID:', tempId, '->', realId);
        state.currentSimulationId = realId;
        state.currentParentSimulationId = getParentId(realId);
      }
      
      const index = state.multipleResults.findIndex(sim => sim.temp_id === tempId);
      if (index !== -1) {
        console.log('🔧 [Redux] Found simulation at index:', index);
        state.multipleResults[index].simulation_id = realId;
        state.multipleResults[index].status = status || 'running';
        console.log('🔧 [Redux] Updated simulation:', state.multipleResults[index]);
      } else {
        console.log('🔧 [Redux] Simulation not found with tempId:', tempId);
        console.log('🔧 [Redux] Available simulations:', state.multipleResults.map(s => ({ temp_id: s.temp_id, simulation_id: s.simulation_id })));
      }
    },
    // ENHANCED: Add action to remove specific simulation
    removeSimulation: (state, action) => {
      const simulationId = action.payload;
      console.log('[simulationSlice] 🗑️ Removing specific simulation:', simulationId);
      
      // Remove from multipleResults
      state.multipleResults = state.multipleResults.filter(sim => 
        sim.simulation_id !== simulationId && sim.temp_id !== simulationId
      );
      
      // Clear current simulation if it matches
      if (state.currentSimulationId === simulationId) {
        state.currentSimulationId = null;
        state.status = 'idle';
        state.results = null;
      }
    },
    // ENHANCED: Add action to cancel all running simulations
    cancelAllRunningSimulations: (state) => {
      console.log('[simulationSlice] 🛑 Cancelling all running simulations');
      
      // Mark all running/pending simulations as cancelled
      state.multipleResults.forEach(sim => {
        if (sim.status === 'running' || sim.status === 'pending') {
          sim.status = 'cancelled';
          sim.error = 'Cancelled by user';
        }
      });
      
      // Update main status if current simulation is running
      if (state.status === 'running' || state.status === 'pending') {
        state.status = 'cancelled';
      }
    },
    // ENHANCED: Add action to restore saved simulation results
    restoreSimulationResults: (state, action) => {
      const savedResults = action.payload;
      console.log('[simulationSlice] 📊 Restoring saved simulation results:', savedResults);
      
      // Restore the multiple results array with new IDs to prevent backend calls
      if (savedResults.multipleResults) {
        state.multipleResults = savedResults.multipleResults.map((result, index) => ({
          ...result,
          // Generate new temporary IDs to prevent backend API calls
          simulation_id: `restored_${Date.now()}_${index}`,
          temp_id: `restored_temp_${Date.now()}_${index}`,
          // Ensure status is completed so no polling occurs
          status: 'completed',
          // Mark as restored for identification
          isRestored: true,
          // Preserve slider state if available
          sliderState: result.sliderState
        }));
      }
      
      // Restore current results if available
      if (savedResults.currentResults) {
        state.results = savedResults.currentResults;
        state.status = 'completed';
      }
      
      // Set status to completed if we have results
      if (state.multipleResults.length > 0) {
        state.status = 'completed';
      }
      
      // CRITICAL FIX: Ensure pending status is corrected when restoring results
      if (state.status === 'pending' && state.multipleResults.length > 0) {
        const hasCompletedWithResults = state.multipleResults.some(sim => 
          sim.status === 'completed' && (
            sim.results || 
            sim.mean !== undefined ||
            sim.histogram ||
            (sim.bin_edges && sim.counts) ||
            sim.iterations_run
          )
        );
        
        if (hasCompletedWithResults) {
          console.log('[simulationSlice] 🔧 RESTORE-FIX: Correcting pending status to completed');
          state.status = 'completed';
        }
      }
      
      // Restore slider states to global context if available
      if (savedResults.sliderStates && typeof window !== 'undefined') {
        console.log('[simulationSlice] 🎛️ Restoring slider states to global context:', savedResults.sliderStates);
        window.simulationSliderStates = savedResults.sliderStates;
      }
      
      console.log('[simulationSlice] 📊 Restored results with new IDs:', state.multipleResults);
    },
    updateSimulationProgress: (state, action) => {
        const { simulationId, progress, stage, iteration, total } = action.payload;
        console.log(`[simulationSlice] 📊 Updating progress for ${simulationId}: ${progress}% (${stage})`);
        
        // Find and update the matching simulation in multipleResults
        const simIndex = state.multipleResults.findIndex(sim => 
            sim.simulation_id === simulationId || sim.temp_id === simulationId
        );
        
        if (simIndex !== -1) {
            state.multipleResults[simIndex] = {
                ...state.multipleResults[simIndex],
                progress_percentage: progress,
                stage: stage,
                current_iteration: iteration,
                total_iterations: total,
                status: progress >= 100 ? 'completed' : 'running'
            };
            
            // If this is the current simulation, update main state
            if (state.currentSimulationId === simulationId) {
                state.progressData.progressPercentage = progress;
                state.progressData.currentStage = stage;
                state.progressData.currentIteration = iteration;
                state.progressData.totalIterations = total;
                state.status = progress >= 100 ? 'completed' : 'running';
            }
        }
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(runSimulation.pending, (state, action) => {
        state.status = 'pending';
        
        // CRITICAL FIX: Set currentSimulationId immediately using tempId for instant progress tracking
        const simulationConfig = action.meta.arg;
        if (simulationConfig && simulationConfig.tempId) {
          state.currentSimulationId = simulationConfig.tempId;
          state.currentParentSimulationId = getParentId(simulationConfig.tempId);
          console.log('[simulationSlice] 🚀 IMMEDIATE: Set currentSimulationId for instant tracking:', simulationConfig.tempId);
        }
        
        // CRITICAL FIX: Reset progress data when starting a new simulation
        state.progressData = {
          progressPercentage: 0,
          currentStage: 'initializing',
          currentIteration: 0,
          totalIterations: 0,
          stageDescription: 'Starting new simulation...',
          timestamp: Date.now(),
          hasCompleted: false
        };
        console.log('[simulationSlice] 🔄 Reset progress data for new simulation');
      })
      .addCase(runSimulation.fulfilled, (state, action) => {
        const payload = action.payload;
        
        // 🔥 PHASE 25 CRITICAL DEBUG: Track if this reducer is even being called
        console.log('🔥🔥🔥 [PHASE25] runSimulation.fulfilled EXECUTED! Timestamp:', new Date().toISOString());
        console.log('🔥🔥🔥 [PHASE25] Action payload received:', payload);
        console.log('🔥🔥🔥 [PHASE25] Current state:', { 
          status: state.status, 
          currentSimulationId: state.currentSimulationId,
          multipleResultsCount: state.multipleResults.length 
        });
        
        // Check if this is a batch response
        if (payload.isBatch && payload.batch_simulation_ids) {
          console.log('[simulationSlice] 🔧 PHASE22_DEBUG: Batch response received');
          console.log('[simulationSlice] 🔧 PHASE22_DEBUG: payload.batch_id:', payload.batch_id);
          console.log('[simulationSlice] 🔧 PHASE22_DEBUG: payload.batch_simulation_ids:', payload.batch_simulation_ids);
          console.log('[simulationSlice] 🔧 PHASE22_DEBUG: payload.targetCells:', payload.targetCells);
          
          state.status = 'running';
          // 🚨 CRITICAL FIX: Use actual simulation ID instead of batch ID for polling
          state.currentSimulationId = payload.batch_simulation_ids[0];
          // CRITICAL FIX: Use utility function for parent ID and validate IDs
          state.currentParentSimulationId = getParentId(payload.batch_id);
          console.log('[simulationSlice] 🚨 BATCH_FIX: Set currentSimulationId to first simulation ID:', payload.batch_simulation_ids[0]);
          console.log('[simulationSlice] 🎯 Set currentParentSimulationId:', state.currentParentSimulationId);
          
          // Validate batch simulation IDs
          payload.batch_simulation_ids.forEach(id => {
            logIdValidationWarning(id, 'batch simulation ID in runSimulation.fulfilled');
          });
          
          // Create entries for each simulation in the batch
          payload.batch_simulation_ids.forEach((simId, index) => {
            const targetCell = payload.targetCells[index];
            state.multipleResults.push({
              simulation_id: simId,
              temp_id: `${payload.batch_id}_${index}`,
              status: 'running',
              target_name: targetCell,
              result_cell_coordinate: targetCell,
              results: null,
              error: null,
              requested_engine_type: payload.requested_engine_type
            });
          });
          
          // 🎯 POLLING-ONLY: Batch simulation queued, progress will be tracked via HTTP polling
          console.log('🎯 [POLLING] Batch simulation queued:', payload.batch_id);
        } else {
          // Single simulation response (legacy)
          const { realId, status, targetName, resultCellCoordinate, tempId, requested_engine_type } = payload;
          
          state.status = 'running';
          state.currentSimulationId = realId;
          // CRITICAL FIX: Use utility function for parent ID and validate ID
          logIdValidationWarning(realId, 'single simulation ID in runSimulation.fulfilled');
          state.currentParentSimulationId = getParentId(realId);

          state.multipleResults.push({
            simulation_id: realId,
            temp_id: tempId,
            status: status,
            target_name: targetName,
            result_cell_coordinate: resultCellCoordinate,
            results: null,
            error: null,
            requested_engine_type: requested_engine_type // Store it here
          });
          
          // 🚀 PHASE 27 FINAL: IMMEDIATE WebSocket Connection with Backend Simulation ID
          performance.mark('phase27-websocket-single-start');
          console.log('🚀 [PHASE27] SINGLE - Connecting to backend simulation ID:', realId);
          
          // 🎯 POLLING-ONLY: Single simulation will be tracked via HTTP polling
          console.log('🎯 [POLLING] Single simulation queued:', realId);
        }
        
        // Dispatch simulation started event for sidebar update
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('simulation-completed', {
            detail: {
              simulation_id: payload.realId,
              status: 'started',
              timestamp: new Date().toISOString()
            }
          }));
          console.log('🚀 Simulation started event dispatched:', payload.realId);
        }
      })
      .addCase(runSimulation.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      })
      .addCase(fetchSimulationStatus.fulfilled, (state, action) => {
        const response = action.payload;
        const simulationId = action.meta.arg;
        
        console.log('[simulationSlice] fetchSimulationStatus.fulfilled - response:', response);
        console.log('[simulationSlice] fetchSimulationStatus.fulfilled - actual simulationId:', simulationId);
        console.log('[simulationSlice] fetchSimulationStatus.fulfilled - currentSimulationId:', state.currentSimulationId);
        console.log('[simulationSlice] fetchSimulationStatus.fulfilled - current status:', state.status);
        
        // 🚨 CRITICAL FIX: Set currentSimulationId if not already tracked
        if (!state.currentSimulationId && response.status !== 'not_found') {
          console.log('[simulationSlice] 🚨 AUTO-TRACKING: Setting currentSimulationId for external simulation:', simulationId);
          state.currentSimulationId = simulationId;
          state.status = response.status || 'running';
        }
        
        // SIMPLIFIED: Handle multi-target simulation completion directly
        if (response.status === 'completed' && response.multi_target_result && response.multi_target_result.statistics) {
          console.log('[simulationSlice] 🎯 MULTI-TARGET COMPLETION DETECTED - Processing direct results');
          console.log('[simulationSlice] Multi-target statistics available for targets:', Object.keys(response.multi_target_result.statistics));
          
          // Update parent simulation status
          if (state.currentSimulationId === simulationId) {
            state.status = 'completed';
            state.results = response;
          }
          
          // Extract individual target results from multi_target_result
          const targetStats = response.multi_target_result.statistics;
          const newMultipleResults = [];
          
          Object.keys(targetStats).forEach((targetCell, index) => {
            const stats = targetStats[targetCell];
            
            // 🔍 DEBUG: Check sensitivity data structure
            console.log(`[simulationSlice] 🔍 SENSITIVITY DEBUG for ${targetCell}:`, {
              has_sensitivity_data: !!response.multi_target_result.sensitivity_data,
              sensitivity_data_keys: response.multi_target_result.sensitivity_data ? Object.keys(response.multi_target_result.sensitivity_data) : [],
              target_sensitivity: response.multi_target_result.sensitivity_data?.[targetCell],
              target_sensitivity_length: response.multi_target_result.sensitivity_data?.[targetCell]?.length || 0
            });
            
            const sensitivityData = response.multi_target_result.sensitivity_data?.[targetCell] || [];
            
            // CRITICAL FIX: Use utility function for clean child ID generation
            const childSimulationId = generateChildSimulationId(getParentId(simulationId), index);
            console.log(`[simulationSlice] Generated clean child ID for ${targetCell}:`, childSimulationId);
            
            // Create individual result format that frontend expects
            const individualResult = {
              simulation_id: childSimulationId,
              status: 'completed',
              target_name: targetCell,
              result_cell_coordinate: targetCell,
              mean: stats.mean,
              median: stats.median,
              std: stats.std,
              min: stats.min,
              max: stats.max,
              iterations_run: response.multi_target_result.total_iterations,
              message: `Target ${targetCell} completed successfully`,
              results: {
                mean: stats.mean,
                median: stats.median,
                std_dev: stats.std,
                min_value: stats.min,
                max_value: stats.max,
                percentiles: stats.percentiles,
                histogram: stats.histogram || { bins: [], values: [] },
                raw_values: response.multi_target_result.target_results?.[targetCell] || [],
                iterations_run: response.multi_target_result.total_iterations,
                errors: [],
                sensitivity_analysis: sensitivityData, // Use target-specific sensitivity data
                target_display_name: targetCell
              }
            };
            
            newMultipleResults.push(individualResult);
          });
          
          // Replace multipleResults with the extracted targets
          state.multipleResults = newMultipleResults;
          console.log('[simulationSlice] 🎯 Created', newMultipleResults.length, 'individual results from multi-target');
          
          // Dispatch simulation completed event for sidebar update
          if (typeof window !== 'undefined') {
            const event = new CustomEvent('simulation-completed', {
              detail: {
                simulation_id: simulationId,
                status: 'completed',
                timestamp: new Date().toISOString(),
                targetCount: newMultipleResults.length
              }
            });
            window.dispatchEvent(event);
            console.log('🎉 Multi-target simulation completed event dispatched:', simulationId, 'with', newMultipleResults.length, 'targets');
          }
          
        } else if (response.status === 'completed') {
          // Also dispatch event for non-multi-target completions
          if (typeof window !== 'undefined') {
            const event = new CustomEvent('simulation-completed', {
              detail: {
                simulation_id: simulationId,
                status: 'completed',
                timestamp: new Date().toISOString(),
                targetCount: 1
              }
            });
            window.dispatchEvent(event);
            console.log('🎉 Single simulation completed event dispatched:', simulationId);
          }
          const index = state.multipleResults.findIndex(sim => sim.simulation_id === simulationId);
          if (index !== -1) {
            console.log('[simulationSlice] Updating multipleResults[' + index + '] to completed');
            state.multipleResults[index] = { ...state.multipleResults[index], ...response, status: 'completed' };
          } else {
            console.log('[simulationSlice] ⚠️ Could not find simulation in multipleResults with ID:', simulationId);
            console.log('[simulationSlice] Available simulation IDs:', state.multipleResults.map(s => s.simulation_id));
          }
          
          if (state.currentSimulationId === simulationId) {
            state.status = 'completed';
            state.results = response;
          }
          
          const allCompleted = state.multipleResults.every(sim => 
            sim.status === 'completed' || sim.status === 'failed'
          );
          console.log('[simulationSlice] All simulations completed?', allCompleted);
          
          if (allCompleted) {
            state.status = 'completed';
            state.currentParentSimulationId = null;
            state.currentSimulationId = null;
            console.log('[simulationSlice] Setting main status to completed and clearing current simulation IDs');
            
            // Dispatch simulation completed event for sidebar update
            if (typeof window !== 'undefined') {
              window.dispatchEvent(new CustomEvent('simulation-completed', {
                detail: {
                  simulation_id: simulationId,
                  status: 'completed',
                  timestamp: new Date().toISOString()
                }
              }));
              console.log('🎉 Simulation completed event dispatched:', simulationId);
            }
          }
        } else if (response.status === 'failed') {
          const index = state.multipleResults.findIndex(sim => sim.simulation_id === simulationId);
          if (index !== -1) {
            state.multipleResults[index].status = 'failed';
            state.multipleResults[index].error = response.message || 'Simulation failed';
          }
          
          if (state.currentSimulationId === simulationId) {
            state.status = 'failed';
            state.error = response.message || 'Simulation failed';
          }
        } else {
          const index = state.multipleResults.findIndex(sim => sim.simulation_id === simulationId);
          if (index !== -1) {
            state.multipleResults[index].status = response.status;
          }
          
          if (state.currentSimulationId === simulationId) {
            state.status = response.status;
          }
        }
        
        // CRITICAL FIX: Auto-correct pending status when we have completed results
        // This handles the case where the backend shows completed but frontend is stuck on pending
        if (state.status === 'pending' && state.multipleResults.length > 0) {
          const hasCompletedWithResults = state.multipleResults.some(sim => 
            sim.status === 'completed' && (
              sim.results || 
              sim.mean !== undefined ||
              sim.histogram ||
              (sim.bin_edges && sim.counts) ||
              sim.iterations_run
            )
          );
          
          if (hasCompletedWithResults) {
            console.log('[simulationSlice] 🔧 AUTO-FIX: Correcting pending status to completed (has completed results)');
            state.status = 'completed';
          }
        }
        
        console.log('[simulationSlice] Final state - status:', state.status);
        console.log('[simulationSlice] Final state - multipleResults:', state.multipleResults);
      })
      .addCase(fetchSimulationStatus.rejected, (state, action) => {
        // Don't mark as failed if we already have completed results
        // This prevents auth errors from marking completed simulations as failed
        const hasCompletedResults = state.multipleResults.some(result => 
          result.status === 'completed' && result.results
        );
        
        if (!hasCompletedResults && state.status !== 'completed') {
          state.status = 'failed';
          state.error = action.payload;
        } else {
          // Just log the error but don't change status for completed simulations
          console.warn('[simulationSlice] fetchSimulationStatus failed but simulation appears completed:', action.payload);
        }
      })
      .addCase(cancelSimulation.pending, (state, action) => {
        console.log('[simulationSlice] cancelSimulation.pending');
      })
      .addCase(cancelSimulation.fulfilled, (state, action) => {
        const { simulationId } = action.payload;
        console.log('[simulationSlice] cancelSimulation.fulfilled for:', simulationId);
        
        if (state.currentSimulationId === simulationId) {
          state.status = 'cancelled';
        }
        
        const index = state.multipleResults.findIndex(sim => sim.simulation_id === simulationId);
        if (index !== -1) {
          state.multipleResults[index].status = 'cancelled';
          state.multipleResults[index].error = 'Simulation cancelled by user';
        }
        
        // Clear current simulation IDs on cancellation
        state.currentParentSimulationId = null;
        state.currentSimulationId = null;
        
        console.log('[simulationSlice] Updated simulation status to cancelled and cleared current simulation IDs');
      })
      .addCase(cancelSimulation.rejected, (state, action) => {
        console.error('[simulationSlice] cancelSimulation.rejected:', action.payload);
        state.error = action.payload;
      })
      .addCase(fetchBatchSimulationResults.fulfilled, (state, action) => {
        const results = action.payload;
        console.log('[simulationSlice] fetchBatchSimulationResults.fulfilled - received results for', results.length, 'children');
        
        results.forEach(({ childId, data, error }) => {
          const childIndex = state.multipleResults.findIndex(sim => sim.simulation_id === childId);
          if (childIndex !== -1 && data) {
            console.log('[simulationSlice] 🎯 Updating child simulation:', childId, 'to status:', data.status);
            state.multipleResults[childIndex] = { 
              ...state.multipleResults[childIndex], 
              ...data, 
              status: data.status,
              results: data.results,
              completed_at: data.updated_at,
              progress_percentage: data.progress_percentage || (data.status === 'completed' ? 100 : 0)
            };
          } else if (error) {
            console.error('[simulationSlice] ❌ Failed to update child simulation:', childId, error);
          }
        });
        
        // Clear the pending batch fetch flag
        delete state._pendingBatchFetch;
      })
      .addCase(fetchSimulationProgress.fulfilled, (state, action) => {
        const progressData = action.payload;
        
        if (progressData) {
                  // CRITICAL FIX: Use utility function and add ID validation
        const normalizedId = progressData.normalizedId || getParentId(progressData.simulationId);
        logIdValidationWarning(progressData.simulationId, 'fetchSimulationProgress.fulfilled');
        const isForCurrentParent = state.currentParentSimulationId === normalizedId;
          
          if (isForCurrentParent) {
            // Update progress data with completion tracking
            const newProgressPercentage = progressData.progress_percentage || 0;
            const currentProgressPercentage = state.progressData.progressPercentage || 0;
            
            // INVARIANT: Once progress reaches >=95% (or status==='completed'), ignore any lower updates 
            // unless the parent simulation ID changes - prevents UI flicker from stray initialization updates
            const nearComplete = currentProgressPercentage >= 95;
            const isDecrease = newProgressPercentage < currentProgressPercentage;
            
            // Only allow progress updates that are non-decreasing OR if not near complete and meet other conditions
            const allowProgressUpdate = (!isDecrease) || 
                                        (!nearComplete && 
                                         (progressData.stage !== state.progressData.currentStage || 
                                          !state.progressData.timestamp || 
                                          Date.now() - state.progressData.timestamp > 30000));
            
            if (allowProgressUpdate) {
              // Set hasCompleted when progress reaches 100%
              const hasCompleted = newProgressPercentage >= 100;
              
              state.progressData = {
                progressPercentage: newProgressPercentage,
                currentStage: progressData.stage || 'unknown',
                currentIteration: progressData.current_iteration || 0,
                totalIterations: progressData.total_iterations || 0,
                stageDescription: progressData.stage_description || progressData.message || 'Processing...',
                timestamp: Date.now(),
                hasCompleted: hasCompleted
              };
            }
          }
        }
      });
  },
});

export const { clearSimulation, setIterations, updateSimulationId, removeSimulation, cancelAllRunningSimulations, restoreSimulationResults, updateSimulationProgress } = simulationSlice.actions;

// Selectors
export const selectSimulationStatus = (state) => state.simulation.status;
export const selectIsSimulationLoading = (state) => state.simulation.status === 'pending' || state.simulation.status === 'running';
export const selectSimulationResults = (state) => state.simulation.results;
export const selectMultipleSimulationResults = (state) => state.simulation.multipleResults;
export const selectSimulationError = (state) => state.simulation.error;
export const selectCurrentSimulationId = (state) => state.simulation.currentSimulationId;
export const selectCurrentParentSimulationId = (state) => state.simulation.currentParentSimulationId;
export const selectSimulationProgressData = (state) => state.simulation.progressData;

// Reselect-based selectors for stability
export const selectProgressHasCompleted = createSelector(
  [selectSimulationProgressData],
  (progressData) => progressData.hasCompleted
);

export const selectProgressPercentage = createSelector(
  [selectSimulationProgressData],
  (progressData) => progressData.progressPercentage
);

export default simulationSlice.reducer; 