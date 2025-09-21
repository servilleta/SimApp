import React, { useEffect, useState, useRef, lazy, Suspense } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import CompactSimulationConfig from '../simulation/CompactSimulationConfig';
import VariableDefinitionPopup from '../simulation/VariableDefinitionPopup';
import SheetTabs from './SheetTabs';
import { initializeSetup, resetSetup, addInputVariable, addResultCell, clearAllInputVariables, clearAllResultCells, setIterations } from '../../store/simulationSetupSlice';
import { runSimulation, clearSimulation } from '../../store/simulationSlice';
import './ExcelViewWithConfig.css';
import CertaintyAnalysis from '../simulation/CertaintyAnalysis';
import SimulationResultsDisplay from '../simulation/SimulationResultsDisplay';
import PhaseDebugger from '../simulation/PhaseDebugger';
import EngineSelectionModal from '../simulation/EngineSelectionModal';
import simulationLogger from '../../services/simulationLogger';
import ModelSummaryBar from '../analysis/ModelSummaryBar';
// import { connectWebSocket } from '../../services/simulationService'; // 🚀 EARLY CONNECTION: Now handled in simulationSlice.js

const ExcelGridPro = lazy(() => import('./ExcelGridPro'));

export default function ExcelViewWithConfig({ 
  fileId, 
  selectedSheetData, 
  fileInfo, 
  onReset, 
  sheets, 
  activeSheetName, 
  onSelectSheet 
}) {
  const dispatch = useDispatch();
  const { currentSheetName, inputVariables, resultCells, iterations } = useSelector(state => state.simulationSetup);
  const simulationStatus = useSelector(state => state.simulation?.status);
  const multipleResults = useSelector(state => state.simulation?.multipleResults) || [];
  const currentSimulationId = useSelector(state => state.simulation?.currentSimulationId);
  const [currentMode, setCurrentMode] = useState('idle');
  const [isRunning, setIsRunning] = useState(false);
  const [popupState, setPopupState] = useState({
    isOpen: false,
    cellAddress: null,
    currentValue: null,
    position: null,
    variableType: null
  });
  const [showPhaseDebugger, setShowPhaseDebugger] = useState(false);
  const [showEngineSelection, setShowEngineSelection] = useState(false);
  const [selectedEngine, setSelectedEngine] = useState('ultra');
  
  // Monitor simulation completion to re-enable Run button
  useEffect(() => {
    if (isRunning && multipleResults.length > 0) {
      const allCompleted = multipleResults.every(result => 
        result && (result.status === 'completed' || result.status === 'failed')
      );
      
      if (allCompleted) {
        console.log('✅ All simulations completed, re-enabling Run button');
        setIsRunning(false);
        // DISABLE PhaseDebugger to prevent conflicts with completed simulations
        setShowPhaseDebugger(false);
      }
    }
  }, [multipleResults, isRunning]);


  useEffect(() => {
    if (fileId && selectedSheetData) {
      dispatch(initializeSetup({
        fileId,
        sheetName: selectedSheetData.sheet_name,
      }));
    } else {
      dispatch(resetSetup());
    }
  }, [dispatch, fileId, selectedSheetData]);

  const handleRunSimulation = async () => {
    console.log('🚨 [HANDLER] ExcelViewWithConfig - handleRunSimulation called!');
    console.log('🚨 [HANDLER] Function entry - isRunning:', isRunning);
    
    // Prevent double-clicking
    if (isRunning) {
      console.log('⚠️ Simulation already running, ignoring duplicate request');
      return;
    }
    
    if (!inputVariables || inputVariables.length === 0 || !resultCells || resultCells.length === 0) {
      console.warn('Cannot run simulation: missing input variables or target cells');
      return;
    }

    try {
      setIsRunning(true); // Disable button immediately
      console.log(`🚀 ExcelViewWithConfig - Running simulation with ${selectedEngine} engine`);
      
      // Generate a batch ID for all simulations in this batch
      const batchId = `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      console.log('🚀 ExcelViewWithConfig - Generated batch ID:', batchId);
      
      // Send a single request with all target cells
      const simulationConfig = {
        fileId,
        variables: inputVariables,
        resultCells: resultCells,  // Pass ALL target cells
        targetCells: resultCells.map(rc => rc.name), // Add target_cells field for backend
        iterations,
        // tempId removed - using real IDs directly from backend
        engine_type: selectedEngine, // Use selected engine
        batch_id: batchId  // Add batch ID to group simulations
      };
      
      console.log('🚀 ExcelViewWithConfig - Dispatching runSimulation with config:', simulationConfig);
      
      // Initialize logging for this simulation
      const frontendSimulationId = simulationConfig.tempId;
      simulationLogger.initializeSimulation(frontendSimulationId, simulationConfig);
      
      // Dispatch the action and wait for the response
      // 🚀 EARLY CONNECTION: WebSocket is now connected BEFORE the API call in simulationSlice.js
      const response = await dispatch(runSimulation(simulationConfig));
      console.log('🔍 [EARLY_CONNECTION] Simulation API response received:', response.payload);
      
      // Extract real simulation ID for logging purposes
      const realId = response.payload.realId || response.payload.simulation_id;
      console.log('🔍 [EARLY_CONNECTION] Simulation ID confirmed:', realId);

      // 🚀 EARLY CONNECTION: WebSocket is already connected and receiving updates!
      console.log('✅ [EARLY_CONNECTION] WebSocket already connected, progress updates should be flowing');
      
      if (response.payload && response.payload.batch_simulation_ids) {
        // Backend returned individual simulation IDs
        const simulationIds = response.payload.batch_simulation_ids;
        console.log('🚀 ExcelViewWithConfig - Received individual simulation IDs:', simulationIds);
        
        // The Redux thunk now handles creating individual simulation entries
        // No need to manually create them here
      }
      
      // Button will be re-enabled when all simulations complete (via useEffect above)
      
    } catch (error) {
      console.error('❌ ExcelViewWithConfig - Error running simulation:', error);
      alert('Error running simulation. Please try again.');
      setIsRunning(false); // Re-enable on error
    }
  };

  const handleEngineSelection = (engineType) => {
    console.log(`🚀 Selected engine: ${engineType}`);
    setSelectedEngine(engineType);
    setShowEngineSelection(false);
  };

  const handleShowEngineSelection = () => {
    setShowEngineSelection(true);
  };

  // Fresh Start function moved from SimulationResultsDisplay
  const handleFreshStart = async () => {
    if (!confirm('This will clear all simulation cache and results. Continue?')) {
      return;
    }
    
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch('/api/simulations/ensure-fresh-start', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        // Reset everything
        dispatch(resetSetup());
        dispatch(clearSimulation()); // Clear simulation results too!
        setCurrentMode('idle');
        setIsRunning(false);
        setPopupState({ isOpen: false, cellAddress: null, currentValue: null, position: null, variableType: null });
        if (onReset) onReset();
        alert('Fresh start completed successfully!');
      } else {
        alert('Failed to clear cache. Please try again.');
      }
    } catch (error) {
      console.error('Error during fresh start:', error);
      alert('Error during fresh start. Please try again.');
    }
  };

  // Refresh Status function moved from SimulationResultsDisplay
  const handleRefreshStatus = async () => {
    try {
      const response = await fetch('/api/simulations/status', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        alert('Status refreshed successfully!');
        // Optionally trigger a refresh of the results display
      } else {
        alert('Failed to refresh status. Please try again.');
      }
    } catch (error) {
      console.error('Error refreshing status:', error);
      alert('Error refreshing status. Please try again.');
    }
  };

  // Clear Results function moved from SimulationResultsDisplay
  const handleClearResults = () => {
    if (confirm('Clear all simulation results?')) {
      dispatch(clearSimulation());
      console.log('All simulation results cleared');
    }
  };

  const handleVariableSave = (variableData) => {
    const fullData = {
      ...variableData,
      sheetName: selectedSheetData?.sheet_name || 'Sheet1'
    };

    if (popupState.variableType === 'input') {
      dispatch(addInputVariable(fullData));
      console.log('Adding input variable:', fullData);
    } else if (popupState.variableType === 'target') {
      dispatch(addResultCell(fullData));
      console.log('Adding target cell:', fullData);
    }
    setPopupState({ isOpen: false, cellAddress: null, currentValue: null, position: null, variableType: null });
    // Keep currentMode active so user can continue clicking cells
  };

  const handlePopupClose = () => {
    setPopupState({ isOpen: false, cellAddress: null, currentValue: null, position: null, variableType: null });
    // Keep currentMode active so user can continue clicking cells
  };



  if (!selectedSheetData) {
    return (
      <div className="excel-view-loading">
        <div className="loading-message">
          <h3>No Sheet Selected</h3>
          <p>Please select a sheet to view and configure.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="excel-view-with-config-new">
      {/* Model Summary Bar */}
      <ModelSummaryBar 
        selectedSheetData={selectedSheetData}
        inputVariables={inputVariables}
        resultCells={resultCells}
        sheets={sheets}
        fileInfo={fileInfo}
      />

      {/* Compact Toolbar */}
      <div className="excel-toolbar-compact">
        <div className="toolbar-buttons-compact">
          <button 
            className={`toolbar-btn-compact ${currentMode === 'selectingInput' ? 'active input-active' : ''}`}
            onClick={() => setCurrentMode(currentMode === 'selectingInput' ? 'idle' : 'selectingInput')}
            title="Define input variables"
          >
            <span className="btn-text">Input</span>
            {inputVariables?.length > 0 && (
              <span className="bubble-indicator green-bubble">{inputVariables.length}</span>
            )}
          </button>
          
          <button 
            className={`toolbar-btn-compact ${currentMode === 'selectingTarget' ? 'active target-active' : ''}`}
            onClick={() => setCurrentMode(currentMode === 'selectingTarget' ? 'idle' : 'selectingTarget')}
            title="Define target cells"
          >
            <span className="btn-text">Target</span>
            {resultCells?.length > 0 && (
              <span className="bubble-indicator yellow-bubble">{resultCells.length}</span>
            )}
          </button>
          
          <button 
            className="toolbar-btn-compact engine-btn"
            onClick={handleShowEngineSelection}
            title="Select simulation engine"
          >
            <span className="btn-text">
              {selectedEngine === 'enhanced' ? '⚡ Enhanced' : 
               selectedEngine === 'ultra' ? '🚀 Ultra' : 
               selectedEngine === 'power' ? '🔋 Power' : 
               selectedEngine === 'standard' ? '💻 Standard' : 
               selectedEngine === 'arrow' ? '🏹 Arrow' : 'Select Engine'}
            </span>
          </button>

          
          <button 
            className="toolbar-btn-compact run-btn"
            onClick={() => {
              console.log('🚨 [BUTTON] ExcelViewWithConfig - Run button CLICKED!');
              console.log('🚨 [BUTTON] Button state:', { 
                inputVariables: inputVariables?.length, 
                resultCells: resultCells?.length, 
                isRunning,
                disabled: !inputVariables?.length || !resultCells?.length || isRunning
              });
              handleRunSimulation();
            }}
            disabled={!inputVariables?.length || !resultCells?.length || isRunning}
            title={isRunning ? "Simulation in progress..." : "Run Monte Carlo simulation"}
          >
            <span className="btn-text">{isRunning ? "Running..." : "Run"}</span>
          </button>
          
          <button 
            className="toolbar-btn-compact clear-btn"
            onClick={() => {
              dispatch(clearAllInputVariables());
              dispatch(clearAllResultCells());
            }}
            title="Clear all variables and targets"
          >
            <span className="btn-text">Clear Variables</span>
          </button>

          <button 
            className="toolbar-btn-compact fresh-start-btn"
            onClick={handleFreshStart}
            title="Clear all cache and start completely fresh"
          >
            <span className="btn-text">Fresh Start</span>
          </button>

          <button 
            className="toolbar-btn-compact refresh-btn"
            onClick={handleRefreshStatus}
            title="Refresh pending simulation statuses"
          >
            <span className="btn-text">Refresh Status</span>
          </button>
          
          <button 
            className="toolbar-btn-compact clear-results-btn"
            onClick={handleClearResults}
            title="Clear all simulation results"
          >
            <span className="btn-text">Clear Results</span>
          </button>
          


          {/* Phase Debugger Toggle - DISABLED to prevent conflicts with completed simulations */}
          {false && (
            <button
              className={`toolbar-btn-compact ${showPhaseDebugger ? 'active' : ''}`}
              onClick={() => setShowPhaseDebugger(!showPhaseDebugger)}
              title="Toggle Power Engine Phase Debugger"
            >
              <span className="btn-text">{showPhaseDebugger ? 'Hide' : 'Show'} Debugger</span>
            </button>
          )}
        </div>
        
        <div className="iterations-control-compact">
          <label htmlFor="iterations-slider" className="iterations-label-compact">
            Iterations: <span className="iterations-value">{iterations.toLocaleString()}</span>
          </label>
          <input
            id="iterations-slider"
            type="range"
            min="1000"
            max="100000"
            step="1000"
            value={iterations}
            onChange={(e) => {
              const newIterations = parseInt(e.target.value);
              dispatch(setIterations(newIterations));
            }}
            className="iterations-slider-compact"
          />
          <div className="slider-labels-compact">
            <span>100k</span>
          </div>
        </div>
      </div>

      {/* Excel Grid with integrated functionality - MAXIMIZED SPACE */}
      <div className="excel-container-integrated">
          {/* Sheet Tabs */}
          {sheets && sheets.length > 0 && (
            <SheetTabs 
              sheets={sheets} 
              activeSheetName={activeSheetName} 
              onSelectSheet={onSelectSheet} 
            />
          )}
          
          <div className="grid-area">
            <Suspense fallback={<div className="loading-placeholder">Loading Grid...</div>}>
              <ExcelGridPro 
                sheetData={selectedSheetData} 
                fileId={fileId}
                selectionMode={currentMode}
                onCellClick={(cellAddress, currentValue) => {
                  console.log('ExcelGridPro onCellClick:', { cellAddress, currentValue, currentMode });
                  if (currentMode === 'selectingInput') {
                    setPopupState({
                      isOpen: true,
                      cellAddress,
                      currentValue,
                      position: { x: 0, y: 0 },
                      variableType: 'input'
                    });
                  } else if (currentMode === 'selectingTarget') {
                    setPopupState({
                      isOpen: true,
                      cellAddress,
                      currentValue,
                      position: { x: 0, y: 0 },
                      variableType: 'target'
                    });
                  }
                }}
              />
            </Suspense>
          </div>
      </div>


      {/* Results section with enhanced display */}
      <div className="simulation-results-container">
        <Suspense fallback={<div className="loading-placeholder">Loading Results...</div>}>
          <SimulationResultsDisplay />
        </Suspense>
      </div>

      {/* Variable Definition Popup */}
      <VariableDefinitionPopup
        isOpen={popupState.isOpen}
        onClose={handlePopupClose}
        cellAddress={popupState.cellAddress}
        currentValue={popupState.currentValue}
        onSave={handleVariableSave}
        position={popupState.position}
        variableType={popupState.variableType}
      />

      {/* Phase Debugger */}
      {showPhaseDebugger && fileInfo && (
        <PhaseDebugger
          fileId={fileId || fileInfo.id || fileInfo.fileId}
          variables={inputVariables || []}
          targetCells={resultCells?.map(cell => cell.name) || []}
          targetSheet={selectedSheetData?.sheet_name || 'Sheet1'}
          iterations={iterations || 1000}
        />
      )}


      {/* Engine Selection Modal */}
      <EngineSelectionModal
        isOpen={showEngineSelection}
        onClose={() => setShowEngineSelection(false)}
        onEngineSelect={handleEngineSelection}
        fileComplexity={fileInfo?.complexity || {}}
      />
    </div>
  );
} 