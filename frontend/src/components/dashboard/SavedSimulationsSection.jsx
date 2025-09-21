import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getSavedSimulations, deleteSimulation } from '../../services/savedSimulationsService';
import { loadSimulation } from '../../services/savedSimulationsService';
import { useDispatch } from 'react-redux';
import { setFileInfo, resetExcelState } from '../../store/excelSlice';
import { setSimulationSetup, resetSetup } from '../../store/simulationSetupSlice';
import { clearSimulation } from '../../store/simulationSlice';
import SimulationCard from './SimulationCard';

const SavedSimulationsSection = () => {
  const [savedSimulations, setSavedSimulations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredSimulations, setFilteredSimulations] = useState([]);
  
  const navigate = useNavigate();
  const dispatch = useDispatch();

  useEffect(() => {
    fetchSavedSimulations();
  }, []);

  useEffect(() => {
    // Filter simulations based on search term
    const filtered = savedSimulations.filter(sim => 
      sim.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sim.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sim.original_filename.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredSimulations(filtered);
  }, [savedSimulations, searchTerm]);

  const fetchSavedSimulations = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await getSavedSimulations();
      setSavedSimulations(data.simulations || []);
    } catch (err) {
      console.error('Failed to fetch saved simulations:', err);
      setError('Failed to load saved simulations');
    } finally {
      setLoading(false);
    }
  };

  const handleLoad = async (simulationId) => {
    try {
      setLoading(true);
      console.log('🔄 Loading simulation:', simulationId);
      const loadedSim = await loadSimulation(simulationId);
      console.log('📥 Loaded simulation data:', loadedSim);
      
      // Check if file_info is included in the response
      let excelData = loadedSim.file_info;
      console.log('📊 Excel data from response:', excelData);
      
      // Always fetch complete Excel data since the response only contains basic info
      if (!excelData || !excelData.sheets || excelData.sheets.length === 0) {
        console.log('⚠️ No complete Excel data in response, fetching full data...');
        const token = localStorage.getItem('authToken');
        const response = await fetch(`/api/excel-parser/files/${loadedSim.file_id}`, {
          headers: { 
            'Authorization': `Bearer ${token}` 
          }
        });
        
        if (!response.ok) {
          throw new Error('Failed to fetch complete Excel data');
        }
        
        excelData = await response.json();
        console.log('📊 Complete Excel data from separate fetch:', excelData);
      }
      
      // Prepare file info object
      // 🔥 FIX: Don't set original_filename when loading saved simulations to avoid filename pollution
      const fileInfoToDispatch = {
        file_id: loadedSim.file_id,
        filename: loadedSim.original_filename,  // Use filename field for current filename
        file_size: excelData.file_size || 0,
        sheet_names: excelData.sheet_names || (excelData.sheets || []).map(s => s.sheet_name),
        sheets: excelData.sheets || [],
        upload_timestamp: new Date().toISOString()
        // Note: NOT setting original_filename here to avoid stale data
      };
      console.log('🎯 Dispatching file info to Redux:', fileInfoToDispatch);
      
      // Restore Excel file info to Redux store with actual data
      dispatch(setFileInfo(fileInfoToDispatch));

      // Prepare simulation config
      const simulationConfig = {
        inputVariables: loadedSim.simulation_config.inputVariables || [],
        resultCells: loadedSim.simulation_config.resultCells || [],
        iterations: loadedSim.simulation_config.iterations || 1000,
        currentSheetName: loadedSim.simulation_config.currentSheetName || (excelData.sheets?.[0]?.sheet_name)
      };
      console.log('⚙️ Dispatching simulation config to Redux:', simulationConfig);

      // Restore simulation configuration
      dispatch(setSimulationSetup(simulationConfig));

      // 📊 RESTORE RESULTS: Load saved simulation results if available
      if (loadedSim.simulation_results) {
        console.log('📊 Restoring saved simulation results:', loadedSim.simulation_results);
        
        // Import simulation slice actions
        const { restoreSimulationResults } = await import('../../store/simulationSlice');
        
        // Restore the saved results to Redux
        dispatch(restoreSimulationResults(loadedSim.simulation_results));
      }
      
      console.log('🧭 Navigating to /simulate...');
      // Navigate to simulation page
      navigate('/simulate');
      
    } catch (err) {
      console.error('Failed to load simulation:', err);
      alert('Failed to load simulation: ' + (err.response?.data?.detail || err.message));
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
      setSavedSimulations(prev => prev.filter(sim => sim.id !== simulationId));
    } catch (err) {
      console.error('Failed to delete simulation:', err);
      alert('Failed to delete simulation: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleViewResults = (simulationId) => {
    // Navigate to simulation report page (to be implemented in Phase 3)
    navigate(`/simulation-report/${simulationId}`);
  };

  const handleNewSimulation = () => {
    // 🔥 FIX: Clear ALL state to ensure fresh start for new simulation
    dispatch(resetExcelState());
    dispatch(resetSetup());
    dispatch(clearSimulation());
    navigate('/simulate');
  };

  if (loading && savedSimulations.length === 0) {
    return (
      <div className="saved-simulations-section">
        <div className="section-header">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-charcoal)' }}>
            📁 My Saved Simulations
          </h2>
        </div>
        <div style={{ 
          textAlign: 'center', 
          padding: '2rem',
          color: 'var(--color-medium-grey)' 
        }}>
          Loading saved simulations...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="saved-simulations-section">
        <div className="section-header">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-charcoal)' }}>
            📁 My Saved Simulations
          </h2>
        </div>
        <div style={{ 
          textAlign: 'center', 
          padding: '2rem',
          color: 'var(--color-error)' 
        }}>
          {error}
          <br />
          <button 
            onClick={fetchSavedSimulations}
            style={{
              marginTop: '1rem',
              padding: '0.5rem 1rem',
              backgroundColor: 'var(--color-braun-orange)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="saved-simulations-section" style={{ marginBottom: '2rem' }}>
      <div className="section-header" style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '1.5rem',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <h2 style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '0.5rem', 
          color: 'var(--color-charcoal)',
          margin: 0,
          fontSize: '1.5rem'
        }}>
          📁 My Saved Simulations
          <span style={{ 
            fontSize: '0.9rem', 
            color: 'var(--color-medium-grey)', 
            fontWeight: 'normal' 
          }}>
            ({filteredSimulations.length})
          </span>
        </h2>
        
        <div className="section-actions" style={{
          display: 'flex',
          gap: '1rem',
          alignItems: 'center'
        }}>
          <input 
            type="search" 
            placeholder="Search simulations..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              padding: '0.5rem 1rem',
              border: '1px solid var(--color-border-light)',
              borderRadius: '6px',
              fontSize: '0.9rem',
              minWidth: '200px'
            }}
          />
          <button 
            onClick={handleNewSimulation}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'var(--color-braun-orange)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.9rem',
              fontWeight: '600',
              whiteSpace: 'nowrap'
            }}
          >
            + New Simulation
          </button>
        </div>
      </div>
      
      {filteredSimulations.length === 0 && !loading ? (
        <div className="empty-state" style={{
          textAlign: 'center',
          padding: '3rem 2rem',
          backgroundColor: 'var(--color-warm-white)',
          borderRadius: '8px',
          border: '1px solid var(--color-border-light)'
        }}>
          {searchTerm ? (
            <>
              <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>🔍</div>
              <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '0.5rem' }}>
                No simulations found
              </h3>
              <p style={{ color: 'var(--color-medium-grey)', margin: 0 }}>
                No saved simulations match "{searchTerm}"
              </p>
            </>
          ) : (
            <>
              <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>💾</div>
              <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '0.5rem' }}>
                No Saved Simulations Yet
              </h3>
              <p style={{ color: 'var(--color-medium-grey)', marginBottom: '1.5rem' }}>
                Save your simulation configurations to quickly run them again later.
              </p>
              <button 
                onClick={handleNewSimulation}
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: 'var(--color-braun-orange)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '1rem',
                  fontWeight: '600'
                }}
              >
                Create Your First Simulation
              </button>
            </>
          )}
        </div>
      ) : (
        <div className="simulations-grid" style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
          gap: '1.5rem'
        }}>
          {filteredSimulations.map(sim => (
            <SimulationCard 
              key={sim.id} 
              simulation={sim}
              onLoad={() => handleLoad(sim.id)}
              onDelete={() => handleDelete(sim.id, sim.name)}
              onViewResults={() => handleViewResults(sim.id)}
              loading={loading}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default SavedSimulationsSection; 