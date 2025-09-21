import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import './ActiveSimulations.css';

const ActiveSimulations = () => {
  const [activeSimulations, setActiveSimulations] = useState([]);
  const [loading, setLoading] = useState(false);
  const isAuthenticated = useSelector(state => state.auth?.isAuthenticated);

  // Fetch active simulations
  const fetchActiveSimulations = async () => {
    setLoading(true);
    try {
      // Get all progress keys from the server
      const response = await fetch('/api/simulations/list-active', {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setActiveSimulations(data || []);
      } else {
        console.warn('Failed to fetch active simulations');
        setActiveSimulations([]);
      }
    } catch (error) {
      console.error('Error fetching active simulations:', error);
      setActiveSimulations([]);
    } finally {
      setLoading(false);
    }
  };

  // Cancel a specific simulation
  const cancelSimulation = async (simulationId) => {
    try {
      const response = await fetch(`/api/simulations/${simulationId}/cancel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        // Remove from local state
        setActiveSimulations(prev => prev.filter(sim => sim.simulation_id !== simulationId));
        console.log(`Simulation ${simulationId} cancelled successfully`);
      } else {
        console.error('Failed to cancel simulation');
      }
    } catch (error) {
      console.error('Error cancelling simulation:', error);
    }
  };

  // Cancel all simulations using stop-all endpoint
  const cancelAllSimulations = async () => {
    if (activeSimulations.length === 0) return;
    
    const confirmCancel = window.confirm(`Are you sure you want to STOP ALL ${activeSimulations.length} active simulations?`);
    if (!confirmCancel) return;

    try {
      const response = await fetch('/api/simulations/stop-all', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        setActiveSimulations([]);
        alert(`✅ Successfully stopped ${result.stopped_count} simulations`);
      } else {
        throw new Error('Failed to stop simulations');
      }
    } catch (error) {
      console.error('Error stopping all simulations:', error);
      alert('❌ Failed to stop simulations');
    }
  };

  // Clean up old simulations
  const cleanupSimulations = async () => {
    try {
      const response = await fetch('/api/simulations/cleanup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        fetchActiveSimulations(); // Refresh the list
        alert(`🧹 Successfully cleaned up ${result.cleaned_count} old simulations`);
      } else {
        throw new Error('Failed to cleanup simulations');
      }
    } catch (error) {
      console.error('Error cleaning up simulations:', error);
      alert('❌ Failed to cleanup simulations');
    }
  };

  // Refresh active simulations every 3 seconds
  useEffect(() => {
    fetchActiveSimulations();
    const interval = setInterval(fetchActiveSimulations, 3000);
    return () => clearInterval(interval);
  }, []);

  // Format progress percentage
  const formatProgress = (sim) => {
    const progress = sim.progress_percentage || 0;
    const current = sim.current_iteration || 0;
    const total = sim.total_iterations || 0;
    return `${Math.round(progress)}% (${current}/${total})`;
  };

  // Format engine type with icon
  const formatEngine = (sim) => {
    const engineInfo = sim.engineInfo || {};
    const engineType = engineInfo.engine_type || 'CPU';
    const gpu = engineInfo.gpu_acceleration;
    let icon = '🖥️';
    if (engineType === 'big' || engineType === 'BIG') {
      icon = '🧬';
    } else if (gpu) {
      icon = '⚡';
    } else if (engineType === 'Hybrid') {
      icon = '🔄';
    } else if (engineType === 'Arrow') {
      icon = '🏹';
    } else if (engineType === 'Streaming') {
      icon = '🌊';
    }
    return `${icon} ${engineType}`;
  };

  if (activeSimulations.length === 0 && !loading) {
    return (
      <div className="active-simulations-container">
        <div className="active-simulations-header">
          <h4 className="active-simulations-title">🔄 Active Simulations</h4>
          <button 
            className="refresh-button"
            onClick={fetchActiveSimulations}
            disabled={loading}
          >
            🔄 Refresh
          </button>
        </div>
        <div className="no-simulations">
          <p>✅ No active simulations running</p>
        </div>
      </div>
    );
  }

  return (
    <div className="active-simulations-container">
      <div className="active-simulations-header">
        <h4 className="active-simulations-title">
          🔄 Active Simulations ({activeSimulations.length})
        </h4>
        <div className="header-buttons">
          <button 
            className="refresh-button"
            onClick={fetchActiveSimulations}
            disabled={loading}
          >
            🔄 Refresh
          </button>
          <button 
            className="cleanup-button"
            onClick={cleanupSimulations}
            disabled={loading}
          >
            🧹 Clean
          </button>
          {activeSimulations.length > 0 && (
            <button 
              className="cancel-all-button"
              onClick={cancelAllSimulations}
              disabled={loading}
            >
              🛑 Stop All
            </button>
          )}
        </div>
      </div>

      {loading && (
        <div className="loading-simulations">
          <p>Loading active simulations...</p>
        </div>
      )}

      <div className="simulations-list">
        {activeSimulations.map((sim) => (
          <div key={sim.simulation_id} className="simulation-item">
            <div className="simulation-info">
              <div className="simulation-id">
                ID: {sim.simulation_id.substring(0, 8)}...
              </div>
              <div className="simulation-details">
                <span className="simulation-progress">
                  {formatProgress(sim)}
                </span>
                <span className="simulation-engine">
                  {formatEngine(sim)}
                </span>
                <span className="simulation-status">
                  {sim.status}
                </span>
              </div>
            </div>
            <button 
              className="cancel-button"
              onClick={() => cancelSimulation(sim.simulation_id)}
              disabled={loading}
              title={`Cancel simulation ${sim.simulation_id}`}
            >
              🛑 Stop
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ActiveSimulations; 