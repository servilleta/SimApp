import React, { useState, useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { getSavedSimulations, loadSimulation, deleteSimulation } from '../../services/savedSimulationsService';
import { setFileInfo } from '../../store/excelSlice';
import { setSimulationSetup } from '../../store/simulationSetupSlice';
import './Modal.css';

const LoadSimulationModal = ({ isOpen, onClose }) => {
  const [simulations, setSimulations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedSim, setSelectedSim] = useState(null);
  const dispatch = useDispatch();

  useEffect(() => {
    if (isOpen) {
      fetchSimulations();
    }
  }, [isOpen]);

  const fetchSimulations = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await getSavedSimulations();
      setSimulations(data.simulations);
    } catch (err) {
      console.error('Failed to fetch simulations:', err);
      setError('Failed to load saved simulations');
    } finally {
      setLoading(false);
    }
  };

  const handleLoad = async (simulationId) => {
    setLoading(true);
    setError('');
    
    try {
      const loadedSim = await loadSimulation(simulationId);
      
      // Restore Excel file info to Redux store
      // 🔥 FIX: Use filename instead of original_filename to avoid stale data pollution
      dispatch(setFileInfo({
        file_id: loadedSim.file_id,
        filename: loadedSim.original_filename,  // Use filename field for current filename
        sheets: [{ 
          sheet_name: loadedSim.simulation_config.currentSheetName || 'Sheet1',
          data: [] // Will be populated when file is processed
        }]
        // Note: NOT setting original_filename here to avoid stale data
      }));

      // Restore simulation configuration
      dispatch(setSimulationSetup({
        inputVariables: loadedSim.simulation_config.inputVariables || [],
        resultCells: loadedSim.simulation_config.resultCells || [],
        iterations: loadedSim.simulation_config.iterations || 1000,
        currentSheetName: loadedSim.simulation_config.currentSheetName
      }));

      alert(`Simulation "${loadedSim.name}" loaded successfully!`);
      onClose();
      
      // Refresh the page to ensure all components are properly updated
      window.location.reload();
      
    } catch (err) {
      console.error('Failed to load simulation:', err);
      setError(err.response?.data?.detail || 'Failed to load simulation');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (simulationId, simulationName) => {
    if (!confirm(`Are you sure you want to delete "${simulationName}"?`)) {
      return;
    }

    try {
      await deleteSimulation(simulationId);
      alert('Simulation deleted successfully');
      fetchSimulations(); // Refresh the list
    } catch (err) {
      console.error('Failed to delete simulation:', err);
      alert('Failed to delete simulation');
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content large-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>📂 Open Simulation</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          {loading && <div className="loading-spinner">Loading...</div>}
          
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {!loading && simulations.length === 0 && (
            <div className="empty-state">
              <p>No saved simulations found.</p>
              <p>Save your current simulation to see it here.</p>
            </div>
          )}

          {!loading && simulations.length > 0 && (
            <div className="simulations-list">
              {simulations.map((sim) => (
                <div key={sim.id} className="simulation-item">
                  <div className="simulation-info">
                    <h4>{sim.name}</h4>
                    {sim.description && <p className="description">{sim.description}</p>}
                    <div className="metadata">
                      <span className="filename">📄 {sim.original_filename}</span>
                      <span className="date">🕒 {formatDate(sim.created_at)}</span>
                    </div>
                  </div>
                  <div className="simulation-actions">
                    <button 
                      className="btn-primary btn-small"
                      onClick={() => handleLoad(sim.id)}
                      disabled={loading}
                    >
                      Load
                    </button>
                    <button 
                      className="btn-danger btn-small"
                      onClick={() => handleDelete(sim.id, sim.name)}
                      disabled={loading}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoadSimulationModal; 