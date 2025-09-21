import React from 'react';
// import ResultsDisplay from '../components/results/ResultsDisplay'; // Deleted component
// import ChartComponent from '../components/results/ChartComponent'; // Generic chart component - Commenting out for now
import { useSelector, useDispatch } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { resetExcelState } from '../store/excelSlice';
import { resetSetup } from '../store/simulationSetupSlice';
import { clearSimulation } from '../store/simulationSlice';
// import ResultsComparisonTool from '../components/results/ResultsComparisonTool'; // Advanced feature placeholder

const ResultsPage = () => {
  const { result, status } = useSelector((state) => state.simulation);
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const handleNewSimulation = () => {
    // 🔥 FIX: Clear ALL state to ensure fresh start for new simulation
    dispatch(resetExcelState());
    dispatch(resetSetup());
    dispatch(clearSimulation());
    navigate('/simulate');
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Simulation Results</h1>
        <p className="page-subtitle">
          View detailed statistics, distributions, and charts for your completed Monte Carlo simulations.
          {status === 'running' && result && result.status === 'PENDING' && " Polling for updates..."}
        </p>
      </div>

      <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>
        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
        <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem' }}>
          Results Display Coming Soon
        </h3>
        <p style={{ color: 'var(--color-medium-grey)', marginBottom: '1.5rem' }}>
          Detailed results and visualizations are currently displayed on the main simulation page after running a simulation.
          A dedicated results viewer with enhanced charts and analysis tools is in development.
        </p>
        
        {status === 'succeeded' && result && result.status === 'SUCCESS' && result.histogram_data && (
          <div style={{ marginTop: '2rem' }}>
            <h4 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem' }}>
              Available Visualizations
            </h4>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '1rem',
              marginTop: '1rem'
            }}>
              {Object.keys(result.histogram_data).map(varName => (
                <div key={varName} className="card" style={{ 
                  padding: '1rem',
                  backgroundColor: 'var(--color-warm-white)'
                }}>
                  <h5 style={{ color: 'var(--color-charcoal)', margin: '0 0 0.5rem 0' }}>
                    {varName}
                  </h5>
                  <p style={{ color: 'var(--color-medium-grey)', fontSize: '0.9rem', margin: 0 }}>
                    Distribution chart available
                  </p>
                </div>
              ))}
            </div>
            <p style={{ 
              textAlign: 'center', 
              color: 'var(--color-medium-grey)', 
              fontSize: '0.9rem',
              marginTop: '1.5rem',
              fontStyle: 'italic'
            }}>
              Chart components will be fully configured to display histograms and other relevant visualizations from the simulation results.
            </p>
          </div>
        )}
        
        <div style={{ marginTop: '2rem' }}>
          <button onClick={handleNewSimulation} className="btn-braun-primary">
            Run New Simulation
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultsPage; 