import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { saveSimulation } from '../../services/savedSimulationsService';
import './Modal.css';

const SaveSimulationModal = ({ isOpen, onClose }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Get current simulation state
  const { fileInfo } = useSelector(state => state.excel);
  const { inputVariables, resultCells, iterations } = useSelector(state => state.simulationSetup);
  const { multipleResults, results } = useSelector(state => state.simulation);
  
  // Get slider states from window global context (set by SimulationResultsDisplay)
  const getSliderStates = () => {
    if (typeof window !== 'undefined' && window.simulationSliderStates) {
      return window.simulationSliderStates;
    }
    return {};
  };

  const handleSave = async () => {
    if (!name.trim()) {
      setError('Please enter a simulation name');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const simulationConfig = {
        inputVariables,
        resultCells,
        iterations,
        currentSheetName: fileInfo.sheets?.[0]?.sheet_name || 'Sheet1'
      };

      // Include simulation results if available with slider states
      const sliderStates = getSliderStates();
      console.log('[SaveSimulationModal] Saving slider states:', sliderStates);
      
      const simulationResults = {
        multipleResults: multipleResults?.map(result => ({
          ...result,
          sliderState: sliderStates[result.target_name] || sliderStates[result.result_cell_coordinate]
        })),
        currentResults: results,
        sliderStates: sliderStates
      };

      await saveSimulation({
        name: name.trim(),
        description: description.trim() || null,
        file_id: fileInfo.file_id,
        simulation_config: simulationConfig,
        simulation_results: simulationResults  // Include the results
      });

      alert('Simulation saved successfully!');
      onClose();
    } catch (err) {
      console.error('Failed to save simulation:', err);
      setError(err.response?.data?.detail || 'Failed to save simulation');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>💾 Save Simulation</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          <div className="form-group">
            <label htmlFor="simulation-name">Simulation Name *</label>
            <input
              id="simulation-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter simulation name"
              autoFocus
            />
          </div>

          <div className="form-group">
            <label htmlFor="simulation-description">Description</label>
            <textarea
              id="simulation-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description"
              rows={3}
            />
          </div>

          <div className="simulation-info">
            <h4>Current Simulation:</h4>
            <p><strong>File:</strong> {fileInfo?.original_filename}</p>
            <p><strong>Input Variables:</strong> {inputVariables.length}</p>
            <p><strong>Target Cells:</strong> {resultCells.length}</p>
            <p><strong>Iterations:</strong> {iterations || 1000}</p>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button 
            className="btn-secondary" 
            onClick={onClose}
            disabled={loading}
          >
            Cancel
          </button>
          <button 
            className="btn-primary" 
            onClick={handleSave}
            disabled={loading || !name.trim()}
          >
            {loading ? 'Saving...' : 'Save Simulation'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SaveSimulationModal; 