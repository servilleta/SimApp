import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { clearAllResultCells, setIterations, clearAllInputVariables } from '../../store/simulationSetupSlice';
import Button from '../common/Button';
import './CompactSimulationConfig.css';

const CompactSimulationConfig = ({ 
  onRunSimulation, 
  currentMode, 
  onModeChange 
}) => {
  const dispatch = useDispatch();
  const { resultCells, inputVariables, iterations = 1000 } = useSelector(state => state.simulationSetup);
  const { isLoadingInitialRun, isPolling, status } = useSelector(state => state.simulation);
  
  const [iterationCount, setIterationCount] = useState(iterations);

  const isConfigured = resultCells && resultCells.length > 0 && inputVariables && inputVariables.length > 0;
  const hasVariables = inputVariables && inputVariables.length > 0;
  const hasTargets = resultCells && resultCells.length > 0;
  const isRunning = isLoadingInitialRun || isPolling || status === 'running' || status === 'pending';

  const handleDefineInputs = () => {
    if (currentMode === 'selectingInput') {
      onModeChange('idle');
    } else {
      onModeChange('selectingInput');
    }
  };

  const handleDefineTarget = () => {
    if (currentMode === 'selectingTarget') {
      onModeChange('idle');
    } else {
      onModeChange('selectingTarget');
    }
  };

  const handleIterationChange = (value) => {
    setIterationCount(value);
    dispatch(setIterations(value));
  };

  const clearAllVariables = () => {
    dispatch(clearAllInputVariables());
    onModeChange('idle');
  };

  const clearAllTargets = () => {
    dispatch(clearAllResultCells());
    onModeChange('idle');
  };

  const clearAllSelections = () => {
    dispatch(clearAllInputVariables());
    dispatch(clearAllResultCells());
    onModeChange('idle');
  };

  const canRunSimulation = inputVariables.length > 0 && resultCells.length > 0 && !isLoadingInitialRun;

  return (
    <div className="compact-simulation-config">
      {/* Section 1: Variable Definition Controls */}
      <div className="config-mini-section">
        <div className="mini-section-header">
          <h4>Define Variables</h4>
        </div>
        <div className="control-buttons">
          <button
            className={`mode-btn ${currentMode === 'selectingInput' ? 'active' : ''}`}
            onClick={handleDefineInputs}
          >
            {currentMode === 'selectingInput' ? 'Cancel Input' : 'Define Input Variables'}
          </button>
          <button
            className={`mode-btn ${currentMode === 'selectingTarget' ? 'active' : ''}`}
            onClick={handleDefineTarget}
          >
            {currentMode === 'selectingTarget' ? 'Cancel Target' : 'Define Target Cells'}
          </button>
          <button
            className="clear-all-btn"
            onClick={clearAllSelections}
            disabled={inputVariables.length === 0 && resultCells.length === 0}
          >
            Clean Variables
          </button>
        </div>
      </div>

      {/* Section 2: Iterations and Run */}
      <div className="config-mini-section">
        <div className="mini-section-header">
          <h4>Simulation Control</h4>
        </div>
        <div className="simulation-controls">
          <div className="iterations-control">
            <label>Iterations:</label>
            <div className="iteration-buttons">
              <button 
                className={`iteration-btn ${iterationCount === 1000 ? 'active' : ''}`}
                onClick={() => handleIterationChange(1000)}
                disabled={isRunning}
              >
                1K
              </button>
              <button 
                className={`iteration-btn ${iterationCount === 5000 ? 'active' : ''}`}
                onClick={() => handleIterationChange(5000)}
                disabled={isRunning}
              >
                5K
              </button>
              <button 
                className={`iteration-btn ${iterationCount === 10000 ? 'active' : ''}`}
                onClick={() => handleIterationChange(10000)}
                disabled={isRunning}
              >
                10K
              </button>
            </div>
          </div>
          <button
            className="rainbow-run-btn"
            onClick={onRunSimulation}
            disabled={!canRunSimulation}
          >
            {isLoadingInitialRun ? 'Running...' : 'Run Simulation'}
          </button>
        </div>
      </div>

      {/* Section 3: Selected Variables and Targets Info */}
      <div className="config-mini-section">
        <div className="mini-section-header">
          <h4>Selection Summary</h4>
        </div>
        <div className="selection-info">
          <div className="info-item">
            <span className="info-count">{inputVariables.length}</span>
            <span className="info-label">Input Variables</span>
            {inputVariables.length > 0 && (
              <div className="variable-list">
                {inputVariables.map((variable, index) => (
                  <span key={index} className="variable-tag input-var">
                    {variable.name}
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="info-item">
            <span className="info-count">{resultCells.length}</span>
            <span className="info-label">Target Cells</span>
            {resultCells.length > 0 && (
              <div className="variable-list">
                {resultCells.map((cell, index) => (
                  <span key={index} className="variable-tag target-cell">
                    {cell.name}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CompactSimulationConfig; 