import React, { useState } from 'react';
import axios from 'axios';

const PhaseDebugger = ({ fileId, variables, targetCells, targetSheet, iterations = 1000 }) => {
  const [phases, setPhases] = useState({
    phase1: { status: 'idle', data: null, error: null },
    phase2: { status: 'idle', data: null, error: null },
    phase3: { status: 'idle', data: null, error: null },
    phase4: { status: 'idle', data: null, error: null },
    phase5: { status: 'idle', data: null, error: null },
    phase6: { status: 'idle', data: null, error: null }
  });
  
  const [simulationId, setSimulationId] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  
  // Update phase status
  const updatePhase = (phaseNum, update) => {
    setPhases(prev => ({
      ...prev,
      [`phase${phaseNum}`]: { ...prev[`phase${phaseNum}`], ...update }
    }));
  };
  
  // Run all phases sequentially
  const runAllPhases = async () => {
    if (!fileId || !targetCells || targetCells.length === 0) {
      alert('Missing required parameters');
      return;
    }
    
    setIsRunning(true);
    console.log('Starting phase debugger for:', { fileId, targetCells, targetSheet });
    
    try {
      // Phase 1: Validate File
      updatePhase(1, { status: 'running' });
      const phase1Start = Date.now();
      
      try {
        const response1 = await axios.post('/api/phases/phase1/validate', {
          file_id: fileId
        });
        const phase1Time = Date.now() - phase1Start;
        updatePhase(1, { 
          status: 'completed', 
          data: { ...response1.data, time_ms: phase1Time },
          error: null 
        });
      } catch (error) {
        console.error('Phase 1 error:', error);
        updatePhase(1, { 
          status: 'error', 
          error: error.response?.data?.detail || error.message 
        });
        setIsRunning(false);
        return;
      }
      
      // Phase 2: Parse Excel
      updatePhase(2, { status: 'running' });
      const phase2Start = Date.now();
      
      try {
        const response2 = await axios.post('/api/phases/phase2/parse', {
          file_id: fileId
        });
        const phase2Time = Date.now() - phase2Start;
        updatePhase(2, { 
          status: 'completed', 
          data: { ...response2.data, time_ms: phase2Time },
          error: null 
        });
      } catch (error) {
        console.error('Phase 2 error:', error);
        updatePhase(2, { 
          status: 'error', 
          error: error.response?.data?.detail || error.message 
        });
        setIsRunning(false);
        return;
      }
      
      // Phase 3: Dependency Analysis
      updatePhase(3, { status: 'running' });
      const phase3Start = Date.now();
      
      try {
        const response3 = await axios.post('/api/phases/phase3/dependency', {
          file_id: fileId,
          target_cells: targetCells,
          target_sheet: targetSheet
        });
        const phase3Time = Date.now() - phase3Start;
        updatePhase(3, { 
          status: 'completed', 
          data: { ...response3.data, time_ms: phase3Time },
          error: null 
        });
      } catch (error) {
        console.error('Phase 3 error:', error);
        updatePhase(3, { 
          status: 'error', 
          error: error.response?.data?.detail || error.message 
        });
        setIsRunning(false);
        return;
      }
      
      // Phase 4: Formula Analysis
      updatePhase(4, { status: 'running' });
      const phase4Start = Date.now();
      
      try {
        const response4 = await axios.post('/api/phases/phase4/formula-analysis', {
          file_id: fileId,
          target_cells: targetCells,
          target_sheet: targetSheet
        });
        const phase4Time = Date.now() - phase4Start;
        updatePhase(4, { 
          status: 'completed', 
          data: { ...response4.data, time_ms: phase4Time },
          error: null 
        });
      } catch (error) {
        console.error('Phase 4 error:', error);
        updatePhase(4, { 
          status: 'error', 
          error: error.response?.data?.detail || error.message 
        });
        setIsRunning(false);
        return;
      }
      
      // Phase 5: Run Simulation
      updatePhase(5, { status: 'running' });
      const phase5Start = Date.now();
      
      try {
        // Build simulation request
        const simulationRequest = {
          file_id: fileId,
          result_cell_coordinate: targetCells[0], // Use first target cell
          result_cell_sheet_name: targetSheet,
          target_cells: targetCells,
          variables: variables,
          iterations: iterations,
          engine_type: "power"
        };
        
        const response5 = await axios.post('/api/phases/phase5/simulate', simulationRequest);
        const phase5Time = Date.now() - phase5Start;
        updatePhase(5, { 
          status: 'completed', 
          data: { ...response5.data, time_ms: phase5Time },
          error: null 
        });
        
        // Store simulation ID for phase 6
        setSimulationId(response5.data.simulation_id);
        
        // Wait a bit before checking results
        await new Promise(resolve => setTimeout(resolve, 2000));
        
      } catch (error) {
        console.error('Phase 5 error:', error);
        updatePhase(5, { 
          status: 'error', 
          error: error.response?.data?.detail || error.message 
        });
        setIsRunning(false);
        return;
      }
      
      // Phase 6: Get Results (with polling)
      updatePhase(6, { status: 'running' });
      const phase6Start = Date.now();
      
      if (simulationId || phases.phase5.data?.simulation_id) {
        const simId = simulationId || phases.phase5.data.simulation_id;
        let attempts = 0;
        const maxAttempts = 30; // 30 seconds max
        
        while (attempts < maxAttempts) {
          try {
            const response6 = await axios.get(`/api/phases/phase6/results/${simId}`);
            
            if (response6.data.status === 'success') {
              const phase6Time = Date.now() - phase6Start;
              updatePhase(6, { 
                status: 'completed', 
                data: { ...response6.data, time_ms: phase6Time },
                error: null 
              });
              break;
            } else if (response6.data.status === 'pending') {
              // Still running, wait and retry
              await new Promise(resolve => setTimeout(resolve, 1000));
              attempts++;
            } else {
              throw new Error('Unexpected status: ' + response6.data.status);
            }
          } catch (error) {
            console.error('Phase 6 error:', error);
            updatePhase(6, { 
              status: 'error', 
              error: error.response?.data?.detail || error.message 
            });
            break;
          }
        }
        
        if (attempts >= maxAttempts) {
          updatePhase(6, { 
            status: 'error', 
            error: 'Timeout waiting for results' 
          });
        }
      }
      
    } finally {
      setIsRunning(false);
    }
  };
  
  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return '✅';
      case 'running':
        return '⏳';
      case 'error':
        return '❌';
      default:
        return '⭕';
    }
  };
  
  // Format time
  const formatTime = (ms) => {
    if (!ms) return '';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };
  
  return (
    <div className="phase-debugger" style={{
      padding: '20px',
      background: '#f5f5f5',
      borderRadius: '8px',
      marginTop: '20px'
    }}>
      <h3>Power Engine Phase Debugger</h3>
      
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={runAllPhases}
          disabled={isRunning}
          style={{
            padding: '10px 20px',
            background: isRunning ? '#ccc' : '#1890ff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isRunning ? 'not-allowed' : 'pointer'
          }}
        >
          {isRunning ? 'Running...' : 'Run All Phases'}
        </button>
        
        <button 
          onClick={() => window.location.reload()}
          style={{
            marginLeft: '10px',
            padding: '10px 20px',
            background: '#ff4d4f',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Reset
        </button>
      </div>
      
      <div className="phases-grid" style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '15px'
      }}>
        {Object.entries(phases).map(([phaseKey, phase]) => {
          const phaseNum = phaseKey.replace('phase', '');
          const phaseName = [
            'File Validation',
            'Parse Excel', 
            'Dependency Analysis',
            'Formula Analysis',
            'Run Simulation',
            'Generate Results'
          ][phaseNum - 1];
          
          return (
            <div 
              key={phaseKey}
              style={{
                padding: '15px',
                background: 'white',
                borderRadius: '8px',
                border: '1px solid #e8e8e8',
                boxShadow: phase.status === 'running' ? '0 2px 8px rgba(0,0,0,0.1)' : 'none'
              }}
            >
              <h4 style={{ margin: '0 0 10px 0', display: 'flex', alignItems: 'center' }}>
                <span style={{ marginRight: '8px' }}>{getStatusIcon(phase.status)}</span>
                Phase {phaseNum}: {phaseName}
              </h4>
              
              {phase.status !== 'idle' && (
                <div style={{ fontSize: '14px' }}>
                  {phase.error && (
                    <div style={{ color: '#ff4d4f', marginBottom: '10px' }}>
                      Error: {phase.error}
                    </div>
                  )}
                  
                  {phase.data && (
                    <div>
                      {phase.data.time_ms && (
                        <div style={{ marginBottom: '5px' }}>
                          <strong>Time:</strong> {formatTime(phase.data.time_ms)}
                        </div>
                      )}
                      
                      {/* Phase-specific data */}
                      {phaseNum === '1' && phase.data.file_size && (
                        <div>File size: {(phase.data.file_size / 1024).toFixed(2)} KB</div>
                      )}
                      
                      {phaseNum === '2' && (
                        <>
                          <div>Formulas: {phase.data.formula_count}</div>
                          <div>Constants: {phase.data.constant_count}</div>
                          <div>Sheets: {phase.data.sheets?.join(', ')}</div>
                        </>
                      )}
                      
                      {phaseNum === '3' && (
                        <>
                          <div>Total dependencies: {phase.data.total_dependencies}</div>
                          {phase.data.dependency_info && (
                            <div style={{ marginTop: '5px' }}>
                              {Object.entries(phase.data.dependency_info).map(([cell, info]) => (
                                <div key={cell} style={{ fontSize: '12px' }}>
                                  {cell}: {info.formulas_in_chain} formulas
                                </div>
                              ))}
                            </div>
                          )}
                        </>
                      )}
                      
                      {phaseNum === '4' && phase.data.formula_stats && (
                        <>
                          <div>Simple: {phase.data.formula_stats.simple}</div>
                          <div>Complex: {phase.data.formula_stats.complex}</div>
                          <div>GPU eligible: {phase.data.formula_stats.gpu_eligible}</div>
                          <div>Functions: {phase.data.formula_stats.functions_used?.slice(0, 5).join(', ')}...</div>
                        </>
                      )}
                      
                      {phaseNum === '5' && (
                        <>
                          <div>Simulation ID: {phase.data.simulation_id?.slice(0, 8)}...</div>
                          <div>Iterations: {phase.data.iterations}</div>
                        </>
                      )}
                      
                      {phaseNum === '6' && phase.data.statistics && (
                        <>
                          <div>Mean: {phase.data.statistics.mean?.toFixed(4)}</div>
                          <div>Std Dev: {phase.data.statistics.std?.toFixed(4)}</div>
                          <div>Range: [{phase.data.statistics.min?.toFixed(2)}, {phase.data.statistics.max?.toFixed(2)}]</div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      {/* Debug information */}
      <div style={{ marginTop: '20px', padding: '10px', background: '#f0f0f0', borderRadius: '4px' }}>
        <h5>Debug Info:</h5>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {JSON.stringify({
            fileId,
            targetCells,
            targetSheet,
            variables: variables?.map(v => ({ name: v.name, min: v.min_value, max: v.max_value })),
            iterations
          }, null, 2)}
        </pre>
      </div>
    </div>
  );
};

export default PhaseDebugger; 