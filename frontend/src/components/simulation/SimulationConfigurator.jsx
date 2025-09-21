import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectCurrentGridSelection,
  selectInputVariables,
  selectResultCell,
  selectIterations,
  selectSetupFileId,
  selectCurrentSheetNameForSetup,
  addInputVariable,
  removeInputVariable,
  setResultCell,
  clearResultCell,
  setIterations
} from '../../store/simulationSetupSlice';
import { runSimulation, selectIsSimulationLoading, selectSimulationStatus, clearResults } from '../../store/simulationSlice';
import Button from '../common/Button';
import EngineSelectionModal from './EngineSelectionModal';

console.log('🔧 SimulationConfigurator component loaded');

const SimulationConfigurator = () => {
  console.log('🔧 SimulationConfigurator component rendering');
  
  const dispatch = useDispatch();

  const currentGridSelection = useSelector(selectCurrentGridSelection);
  const inputVariables = useSelector(selectInputVariables);
  const resultCell = useSelector(selectResultCell);
  const iterationsFromSetup = useSelector(selectIterations);
  const fileId = useSelector(selectSetupFileId);
  const currentSheetName = useSelector(selectCurrentSheetNameForSetup);

  const isSimulationRunningOrLoading = useSelector(selectIsSimulationLoading);
  const simulationStatus = useSelector(selectSimulationStatus);

  const [showInputForm, setShowInputForm] = useState(false);
  const [currentInputCell, setCurrentInputCell] = useState(null);
  const [inputMin, setInputMin] = useState('');
  const [inputLikely, setInputLikely] = useState('');
  const [inputMax, setInputMax] = useState('');
  const [showEngineSelection, setShowEngineSelection] = useState(false);
  const [selectedEngine, setSelectedEngine] = useState('ultra');

  useEffect(() => {
    if (currentGridSelection && currentInputCell && 
        (currentGridSelection.name !== currentInputCell.name || currentGridSelection.sheetName !== currentInputCell.sheetName)) {
      setShowInputForm(false);
    }
    if (currentGridSelection) {
        const existingVar = (inputVariables || []).find(v => v.name === currentGridSelection.name && v.sheetName === currentGridSelection.sheetName);
        if (existingVar) {
            setCurrentInputCell({ name: existingVar.name, sheetName: existingVar.sheetName });
            setInputMin(String(existingVar.min_value));
            setInputLikely(String(existingVar.most_likely));
            setInputMax(String(existingVar.max_value));
            setShowInputForm(true);
        } else {
            if (!showInputForm || (currentInputCell && (currentGridSelection.name !== currentInputCell.name || currentGridSelection.sheetName !== currentInputCell.sheetName))) {
                setInputMin('');
                setInputLikely('');
                setInputMax('');
            }
        }
    }
  }, [currentGridSelection, inputVariables, showInputForm, currentInputCell]);

  const handleSetAsInput = () => {
    if (currentGridSelection) {
      setCurrentInputCell({ name: currentGridSelection.name, sheetName: currentGridSelection.sheetName });
      const existingVar = (inputVariables || []).find(v => v.name === currentGridSelection.name && v.sheetName === currentGridSelection.sheetName);
      if (existingVar) {
        setInputMin(String(existingVar.min_value));
        setInputLikely(String(existingVar.most_likely));
        setInputMax(String(existingVar.max_value));
      } else {
        setInputMin('');
        setInputLikely('');
        setInputMax('');
      }
      setShowInputForm(true);
    } else {
      alert('Please select a cell from the grid first.');
    }
  };

  const handleInputFormSubmit = (e) => {
    e.preventDefault();
    if (!currentInputCell) return;
    const min = parseFloat(inputMin);
    const likely = parseFloat(inputLikely);
    const max = parseFloat(inputMax);
    if (isNaN(min) || isNaN(likely) || isNaN(max)) {
      alert('Please enter valid numbers for min, most likely, and max.');
      return;
    }
    if (!(min <= likely && likely <= max)) {
      alert('Values must be in order: Min <= Most Likely <= Max.');
      return;
    }
    dispatch(addInputVariable({
      name: currentInputCell.name,
      sheetName: currentInputCell.sheetName,
      min_value: min,
      most_likely: likely,
      max_value: max
    }));
    setShowInputForm(false);
    setCurrentInputCell(null);
  };

  const handleSetAsResult = () => {
    if (currentGridSelection) {
      dispatch(setResultCell({ name: currentGridSelection.name, sheetName: currentGridSelection.sheetName }));
      setShowInputForm(false);
    } else {
      alert('Please select a cell from the grid first.');
    }
  };

  const handleRunSimulation = async () => {
    console.log('🚨 [HANDLER] SimulationConfigurator - handleRunSimulation called!');
    
    try {
      console.log('🚨 [HANDLER] Starting validation...');
      if (!validateInputs()) {
        console.log('❌ Validation failed, returning early');
        return;
      }

      console.log(`🚀 Running simulation with ${selectedEngine} engine`);
      
      // Create the simulation config in the format expected by runSimulation thunk
      const simulationConfig = {
        variables: inputVariables,
        resultCells: [resultCell],
        iterations: iterationsFromSetup,
        // tempId removed - using real IDs directly from backend
        engine_type: selectedEngine, // Use selected engine
        fileId: fileId
      };
      
      console.log('🚀 Dispatching runSimulation with config:', simulationConfig);
      dispatch(runSimulation(simulationConfig));
      
    } catch (error) {
      console.error('❌ Error running simulation:', error);
      alert('Error running simulation. Please try again.');
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

  const isRunDisabled = !fileId || inputVariables.length === 0 || !resultCell || isSimulationRunningOrLoading || simulationStatus === 'pending' || simulationStatus === 'running';

  const configuratorStyle = {
    padding: '15px',
    border: '1px solid #e0e0e0',
    borderRadius: '4px',
    backgroundColor: '#f9f9f9',
    height: '100%'
  };

  const sectionStyle = { marginBottom: '20px' };
  const itemStyle = { marginBottom: '8px', padding: '5px', border: '1px solid #eee', backgroundColor: '#fff', display: 'flex', justifyContent: 'space-between', alignItems: 'center' };
  const inputStyle = { width: '60px', marginRight: '5px', padding: '4px' }; 

  const validateInputs = () => {
    console.log('🔧 Validating inputs...');
    if (!fileId || inputVariables.length === 0 || !resultCell) {
      console.error('❌ Validation failed:', { fileId, inputVariables: inputVariables.length, resultCell });
      alert('Please ensure you have uploaded a file, defined at least one input variable, and selected a result cell.');
      return false;
    }
    console.log('✅ Validation passed');
    return true;
  };

  if (!fileId) {
    return (
      <div style={configuratorStyle}>
        <p>Please upload an Excel file to begin configuring a simulation.</p>
      </div>
    );
  }

  return (
    <div style={configuratorStyle}>
      <h4>Simulation Setup ({currentSheetName || 'No Sheet Selected'})</h4>

      <div style={sectionStyle}>
        <h5>Selected Cell:</h5>
        {currentGridSelection ? (
          <p>
            <strong>{currentGridSelection.name}</strong> ({currentGridSelection.sheetName})<br />
            Value: <em>{String(currentGridSelection.value).substring(0,100)} {String(currentGridSelection.value).length > 100 ? '...' : ''}</em>
          </p>
        ) : (
          <p><em>No cell selected in the grid.</em></p>
        )}
        {currentGridSelection && (
          <div>
            <Button onClick={handleSetAsInput} size="small" style={{marginRight: '10px'}} disabled={isSimulationRunningOrLoading || simulationStatus === 'pending' || simulationStatus === 'running'}>
              Set/Edit as Input
            </Button>
            <Button onClick={handleSetAsResult} size="small" variant="secondary" disabled={isSimulationRunningOrLoading || simulationStatus === 'pending' || simulationStatus === 'running'}>
              Set as Result
            </Button>
          </div>
        )}
      </div>

      {showInputForm && currentInputCell && (
        <form onSubmit={handleInputFormSubmit} style={{ ...sectionStyle, padding: '10px', border: '1px dashed #ccc' }}>
          <h5>Input: {currentInputCell.name} ({currentInputCell.sheetName})</h5>
          <div>
            <label>Min: <input type="number" step="any" style={inputStyle} value={inputMin} onChange={e => setInputMin(e.target.value)} required /></label>
            <label>Likely: <input type="number" step="any" style={inputStyle} value={inputLikely} onChange={e => setInputLikely(e.target.value)} required /></label>
            <label>Max: <input type="number" step="any" style={inputStyle} value={inputMax} onChange={e => setInputMax(e.target.value)} required /></label>
          </div>
          <Button type="submit" size="small" style={{marginTop: '10px'}}>Save Input</Button>
          <Button type="button" size="small" variant="text" onClick={() => setShowInputForm(false)} style={{marginLeft: '5px'}}>Cancel</Button>
        </form>
      )}

      <div style={sectionStyle}>
        <h5>Input Variables ({inputVariables.length}):</h5>
        {inputVariables.length === 0 && <p><em>None defined.</em></p>}
        <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
          {inputVariables.map(v => (
            <li key={`${v.sheetName}-${v.name}`} style={itemStyle}>
              <span><strong>{v.name}</strong> ({v.sheetName}): {v.min_value}, {v.most_likely}, {v.max_value}</span>
              <Button 
                onClick={() => dispatch(removeInputVariable({ name: v.name, sheetName: v.sheetName }))} 
                size="small" variant="danger" 
                disabled={isSimulationRunningOrLoading || simulationStatus === 'pending' || simulationStatus === 'running'}>X</Button>
            </li>
          ))}
        </ul>
      </div>

      <div style={sectionStyle}>
        <h5>Result Cell:</h5>
        {resultCell ? (
          <div style={itemStyle}>
            <span><strong>{resultCell.name}</strong> ({resultCell.sheetName})</span>
            <Button onClick={() => dispatch(clearResultCell())} size="small" variant="danger" disabled={isSimulationRunningOrLoading || simulationStatus === 'pending' || simulationStatus === 'running'}>X</Button>
          </div>
        ) : (
          <p><em>None selected.</em></p>
        )}
      </div>

      <div style={sectionStyle}>
        <label htmlFor="iterationsInput"><h5>Iterations:</h5></label>
        <input 
          type="number" 
          id="iterationsInput" 
          value={iterationsFromSetup} 
          onChange={e => dispatch(setIterations(parseInt(e.target.value, 10)))}
          style={{width: '100px', padding: '5px'}}
          min="1"
          disabled={isSimulationRunningOrLoading || simulationStatus === 'pending' || simulationStatus === 'running'}
        />
      </div>

      <div style={{marginBottom: '15px'}}>
        <h5>Engine Selection:</h5>
        <Button 
          onClick={handleShowEngineSelection}
          size="small"
          variant="secondary"
          style={{marginRight: '10px'}}
        >
          {selectedEngine === 'enhanced' ? '⚡ Enhanced' : 
           selectedEngine === 'ultra' ? '🚀 Ultra' : 
           selectedEngine === 'standard' ? '💻 Standard' : 'Select Engine'}
        </Button>
        <span style={{color: '#666', fontSize: '0.9em'}}>
          Current: {selectedEngine === 'enhanced' ? 'Enhanced GPU Engine' : 
                   selectedEngine === 'ultra' ? 'Ultra Hybrid Engine' : 
                   selectedEngine === 'standard' ? 'Standard CPU Engine' : 'None'}
        </span>
      </div>

      <Button 
        onClick={() => {
          console.log('🚨 [BUTTON] SimulationConfigurator - Run Simulation BUTTON CLICKED!');
          console.log('🚨 [BUTTON] Current state before click:', { 
            fileId, 
            inputVariables: inputVariables.length, 
            resultCell,
            isRunDisabled,
            simulationStatus
          });
          handleRunSimulation();
        }}
        disabled={isRunDisabled}
        className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg font-medium transition-colors"
      >
        {isSimulationRunningOrLoading || simulationStatus === 'pending' || simulationStatus === 'running' ? `Running (${simulationStatus})...` : 'Run Simulation'}
      </Button>

      {/* Engine Selection Modal */}
      <EngineSelectionModal
        isOpen={showEngineSelection}
        onClose={() => setShowEngineSelection(false)}
        onEngineSelect={handleEngineSelection}
        fileComplexity={{}}
      />
    </div>
  );
};

export default SimulationConfigurator; 