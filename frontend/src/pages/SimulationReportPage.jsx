import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import SimulationResultsDisplay from '../components/simulation/SimulationResultsDisplay';
import apiClient from '../services/api';
import { restoreSimulationResults, updateSimulationProgress } from '../store/simulationSlice';

const SimulationReportPage = () => {
  const { simulationId } = useParams();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [simulationData, setSimulationData] = useState(null);
  
  const simulationStatus = useSelector(state => state.simulation.status);
  
  useEffect(() => {
    if (simulationId) {
      fetchSimulationData();
    }
  }, [simulationId]);

  const fetchSimulationData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log(`🔍 Fetching simulation data for ID: ${simulationId}`);
      
      // Fetch simulation data from the backend
      const response = await apiClient.get(`/simulation/${simulationId}?v=999&t=${Date.now()}`);
      console.log('📊 Simulation data received:', response.data);
      
      setSimulationData(response.data);
      
      // Load the data into Redux store so SimulationResultsDisplay can use it
      if (response.data.status === 'SUCCESS' || response.data.status === 'COMPLETED' || response.data.status === 'completed') {
        console.log('📊 Processing simulation data for results restoration...');
        
        let resultsToRestore = {
          currentResults: response.data,
          multipleResults: []
        };
        
        // Handle multi-target simulations
        if (response.data.multi_target_result) {
          console.log('📊 Processing multi-target simulation results');
          console.log('📊 Multi-target result structure:', response.data.multi_target_result);
          
          const multiTargetData = response.data.multi_target_result;
          
          // Extract targets from the multi_target_result structure
          const targets = multiTargetData.targets || [];
          const targetResults = multiTargetData.target_results || {};
          const statistics = multiTargetData.statistics || {};
          const iterationData = multiTargetData.iteration_data || [];
          const totalIterations = multiTargetData.total_iterations || response.data.iterations_run || 1000;
          
          console.log('📊 Targets found:', targets);
          console.log('📊 Statistics available:', Object.keys(statistics));
          console.log('📊 Target results available:', Object.keys(targetResults));
          console.log('📊 Full multi-target structure keys:', Object.keys(multiTargetData));
          console.log('📊 Total iterations:', totalIterations);
          
          resultsToRestore.multipleResults = targets.map((targetName, index) => {
            const stats = statistics[targetName] || {};
            const targetData = targetResults[targetName] || [];
            
            // Try to calculate basic statistics if not available in stats
            let mean = stats.mean;
            let median = stats.median;
            let std_dev = stats.std;
            let min_value = stats.min;
            let max_value = stats.max;
            
            if (targetData.length > 0 && (mean === undefined || mean === null)) {
              // Calculate basic statistics from raw data
              const validData = targetData.filter(v => v !== null && v !== undefined && !isNaN(v));
              if (validData.length > 0) {
                mean = validData.reduce((a, b) => a + b, 0) / validData.length;
                min_value = Math.min(...validData);
                max_value = Math.max(...validData);
                
                // Calculate median
                const sorted = [...validData].sort((a, b) => a - b);
                const mid = Math.floor(sorted.length / 2);
                median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
                
                // Calculate standard deviation
                const variance = validData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / validData.length;
                std_dev = Math.sqrt(variance);
              }
            }
            
            // Extract histogram data if available
            const histogramData = stats.histogram || { bins: [], values: [], bin_edges: [], counts: [] };
            
            return {
              simulation_id: `restored_${Date.now()}_${index}`,
              temp_id: `restored_temp_${Date.now()}_${index}`,
              status: 'completed',
              target_name: targetName,
              result_cell_coordinate: targetName,
              isRestored: true,
              mean: mean,
              median: median,
              std_dev: std_dev,
              min_value: min_value,
              max_value: max_value,
              iterations_run: totalIterations,
              results: {
                mean: mean,
                median: median,
                std_dev: std_dev,
                min_value: min_value,
                max_value: max_value,
                percentiles: stats.percentiles || {},
                histogram: histogramData,
                iterations_run: totalIterations,
                raw_values: targetData,
                sensitivity_analysis: multiTargetData.sensitivity_data?.[targetName] || [],
                bin_edges: histogramData.bin_edges,
                counts: histogramData.counts
              },
              // Additional fields for compatibility
              histogram: histogramData,
              bin_edges: histogramData.bin_edges,
              counts: histogramData.counts
            };
          });
        } else if (response.data.mean !== null && response.data.mean !== undefined) {
          // Handle single-target simulations
          console.log('📊 Processing single-target simulation results');
          resultsToRestore.multipleResults = [{
            simulation_id: `restored_${Date.now()}_0`,
            temp_id: `restored_temp_${Date.now()}_0`,
            status: 'completed',
            target_name: response.data.target_name || response.data.target_cell || 'Target',
            result_cell_coordinate: response.data.target_cell || 'Unknown',
            isRestored: true,
            mean: response.data.mean,
            median: response.data.median,
            std_dev: response.data.std,
            min_value: response.data.min,
            max_value: response.data.max,
            iterations_run: response.data.iterations_run,
            results: {
              mean: response.data.mean,
              median: response.data.median,
              std_dev: response.data.std,
              min_value: response.data.min,
              max_value: response.data.max,
              percentiles: {},
              histogram: response.data.histogram || { bins: [], values: [] },
              iterations_run: response.data.iterations_run,
              raw_values: [],
              sensitivity_analysis: []
            }
          }];
        }
        
        console.log('📊 Dispatching restored results:', resultsToRestore);
        dispatch(restoreSimulationResults(resultsToRestore));
      }
      
    } catch (err) {
      console.error('❌ Error fetching simulation data:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to load simulation data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="page-header">
          <button 
            onClick={() => navigate('/simulate')}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'var(--color-white)',
              color: 'var(--color-charcoal)',
              border: '1px solid var(--color-border-light)',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.9rem',
              marginBottom: '1rem'
            }}
          >
            ← Back to Simulations
          </button>
          
          <h1 className="page-title">📊 Simulation Report</h1>
          <p className="page-subtitle">
            Loading simulation #{simulationId}...
          </p>
        </div>

        <div className="card" style={{ 
          textAlign: 'center', 
          padding: '3rem 2rem',
          backgroundColor: 'var(--color-warm-white)',
          border: '1px solid var(--color-border-light)'
        }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⏳</div>
          <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem' }}>
            Loading Simulation Data...
          </h3>
          <p style={{ color: 'var(--color-medium-grey)' }}>
            Retrieving historical results and analysis
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="page-header">
          <button 
            onClick={() => navigate('/simulate')}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'var(--color-white)',
              color: 'var(--color-charcoal)',
              border: '1px solid var(--color-border-light)',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.9rem',
              marginBottom: '1rem'
            }}
          >
            ← Back to Simulations
          </button>
          
          <h1 className="page-title">📊 Simulation Report</h1>
          <p className="page-subtitle">
            Error loading simulation #{simulationId}
          </p>
        </div>

        <div className="card" style={{ 
          textAlign: 'center', 
          padding: '3rem 2rem',
          backgroundColor: 'var(--color-warm-white)',
          border: '1px solid var(--color-border-light)'
        }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>❌</div>
          <h3 style={{ color: 'var(--color-error)', marginBottom: '1rem' }}>
            Failed to Load Simulation
          </h3>
          <p style={{ color: 'var(--color-medium-grey)', marginBottom: '2rem' }}>
            {error}
          </p>
          <button 
            onClick={fetchSimulationData}
            style={{
              padding: '0.75rem 1.5rem',
              backgroundColor: 'var(--color-braun-orange)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.9rem',
              fontWeight: '500'
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <button 
          onClick={() => navigate('/simulate')}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: 'var(--color-white)',
            color: 'var(--color-charcoal)',
            border: '1px solid var(--color-border-light)',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '0.9rem',
            marginBottom: '1rem'
          }}
        >
          ← Back to Simulations
        </button>
        
        <h1 className="page-title">📊 Simulation Report</h1>
        <p className="page-subtitle">
          {simulationData?.original_filename || `Simulation #${simulationId}`}
          {simulationData?.created_at && (
            <span style={{ color: 'var(--color-medium-grey)', fontSize: '0.9rem', marginLeft: '1rem' }}>
              • {new Date(simulationData.created_at).toLocaleDateString()}
            </span>
          )}
        </p>
      </div>

      {/* Display the actual simulation results */}
      <SimulationResultsDisplay />
    </div>
  );
};

export default SimulationReportPage; 