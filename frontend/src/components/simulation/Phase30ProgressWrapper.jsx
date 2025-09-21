import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import EnhancedProgressTracker from './EnhancedProgressTracker';
import enhancedWebSocketService from '../../services/enhancedWebSocketService';

/**
 * Phase 30 Progress Wrapper
 * 
 * Integrates the Enhanced Progress Tracker with the existing simulation system
 * to provide smooth progress display without breaking existing functionality.
 */
const Phase30ProgressWrapper = ({ 
  simulationIds = [], 
  targetVariables = [],
  className = '',
  onProgressUpdate = null,
  ...props 
}) => {
  const dispatch = useDispatch();
  const [isPhase30Enabled, setIsPhase30Enabled] = useState(true);
  const [progressData, setProgressData] = useState({});
  
  // Get simulation state from Redux
  const simulationState = useSelector(state => state.simulation);
  const { status, currentSimulationId, multipleResultsCount } = simulationState;

  console.log('🚀 [Phase30] Progress wrapper initialized with:', {
    simulationIds: simulationIds.length,
    targetVariables: targetVariables.length,
    status,
    currentSimulationId
  });

  // Phase 30: Enhanced WebSocket connection management
  useEffect(() => {
    if (!isPhase30Enabled || simulationIds.length === 0) return;

    console.log('🔌 [Phase30] Setting up enhanced WebSocket connections for:', simulationIds);

    const connectPromises = simulationIds.map(simulationId => {
      return enhancedWebSocketService.connect(simulationId, {
        onProgress: (data) => {
          console.log(`📊 [Phase30] Progress update for ${simulationId}:`, data);
          
          setProgressData(prev => ({
            ...prev,
            [simulationId]: data
          }));

          // Call external progress callback if provided
          if (onProgressUpdate) {
            onProgressUpdate(simulationId, data);
          }
        },
        
        onError: (error) => {
          console.error(`❌ [Phase30] WebSocket error for ${simulationId}:`, error);
          // Continue with HTTP polling fallback
        },
        
        onComplete: (data) => {
          console.log(`🎉 [Phase30] Simulation ${simulationId} completed:`, data);
          
          // Force final completion state
          const completionData = {
            ...data,
            progress: 100,
            iteration: data.current_iteration || data.total_iterations || 1000,
            totalIterations: data.total_iterations || 1000,
            completed: true,
            status: 'completed'
          };
          
          setProgressData(prev => ({
            ...prev,
            [simulationId]: completionData
          }));

          // Trigger results check callback
          if (onProgressUpdate) {
            onProgressUpdate(simulationId, completionData);
          }
        },
        
        enableInterpolation: true,
        fallbackToPolling: true
      });
    });

    // Handle connection setup
    Promise.allSettled(connectPromises).then(results => {
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      console.log(`🔌 [Phase30] WebSocket setup complete: ${successful} connected, ${failed} failed (fallback to polling)`);
    });

    // Cleanup on unmount or simulation change
    return () => {
      console.log('🧹 [Phase30] Cleaning up WebSocket connections');
      simulationIds.forEach(id => {
        enhancedWebSocketService.disconnect(id);
      });
    };
  }, [simulationIds, isPhase30Enabled, onProgressUpdate]);

  // Phase 30: Calculate aggregate progress for multiple simulations
  const getAggregateProgress = () => {
    if (Object.keys(progressData).length === 0) {
      return { progress: 0, iteration: 0, totalIterations: 1000 };
    }

    const simulations = Object.values(progressData);
    
    // ENHANCED COMPLETION DETECTION
    const completedSims = simulations.filter(s => 
      s.completed || 
      s.status === 'completed' || 
      s.progress >= 100 ||
      (s.iteration >= s.totalIterations && s.totalIterations > 0)
    );
    
    const isAllCompleted = completedSims.length === simulations.length && simulations.length > 0;
    
    // Calculate progress
    const totalProgress = simulations.reduce((sum, sim) => {
      const progress = isAllCompleted ? 100 : (sim.progress || 0);
      return sum + progress;
    }, 0);
    const averageProgress = isAllCompleted ? 100 : (totalProgress / simulations.length);
    
    // Calculate iterations
    const totalIterations = simulations.reduce((sum, sim) => sum + (sim.totalIterations || 1000), 0);
    const currentIterations = simulations.reduce((sum, sim) => {
      const iteration = isAllCompleted ? (sim.totalIterations || 1000) : (sim.iteration || 0);
      return sum + iteration;
    }, 0);
    
    return {
      progress: Math.min(averageProgress, 100),
      iteration: Math.min(currentIterations, totalIterations),
      totalIterations: totalIterations,
      completedSimulations: completedSims.length,
      totalSimulations: simulations.length,
      allCompleted: isAllCompleted
    };
  };

  // Phase 30: Enhanced progress data for the tracker  
  const enhancedProgressData = getAggregateProgress();
  
  // Phase 30: Aggregate progress data ready for enhanced tracking

  // Phase 30: Toggle between enhanced and legacy mode
  const togglePhase30 = () => {
    setIsPhase30Enabled(!isPhase30Enabled);
    console.log(`🔄 [Phase30] Toggled to ${!isPhase30Enabled ? 'enhanced' : 'legacy'} mode`);
  };

  if (!isPhase30Enabled) {
    // Fallback to original progress display
    return (
      <div className={`phase30-wrapper legacy-mode ${className}`}>
        <div className="mode-toggle">
          <button onClick={togglePhase30} className="toggle-button">
            🚀 Enable Phase 30 Enhanced Progress
          </button>
        </div>
        {/* Original progress component would go here */}
        <div className="legacy-progress">
          <p>Legacy progress display (original UnifiedProgressTracker)</p>
          <p>Progress: {enhancedProgressData.progress.toFixed(1)}%</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`phase30-wrapper enhanced-mode ${className}`}>
      {/* Phase 30: Mode toggle for testing */}
      <div className="mode-toggle">
        <button onClick={togglePhase30} className="toggle-button legacy">
          ⬅️ Use Legacy Progress Display
        </button>
      </div>

      {/* Phase 30: Enhanced Progress Tracker */}
      <EnhancedProgressTracker
        simulationIds={simulationIds}
        targetVariables={targetVariables}
        onProgressUpdate={(data) => {
          console.log('📊 [Phase30] Enhanced tracker progress:', data);
          if (onProgressUpdate) {
            onProgressUpdate('aggregate', data);
          }
        }}
        {...props}
      />

      {/* Phase 30: Debug information (can be removed in production) */}
      {import.meta.env.DEV && (
        <div className="phase30-debug">
          <details>
            <summary>Phase 30 Debug Info</summary>
            <div className="debug-content">
              <h4>WebSocket Status:</h4>
              {simulationIds.map(id => {
                const status = enhancedWebSocketService.getConnectionStatus(id);
                return (
                  <div key={id} className="debug-item">
                    <span className="debug-label">{id.substring(0, 8)}...:</span>
                    <span className={`debug-status ${status.connected ? 'connected' : status.polling ? 'polling' : 'disconnected'}`}>
                      {status.connected ? '🔗 Connected' : status.polling ? '🔄 Polling' : '❌ Disconnected'}
                    </span>
                  </div>
                );
              })}
              
              <h4>Progress Data:</h4>
              <pre>{JSON.stringify(progressData, null, 2)}</pre>
              
              <h4>Aggregate Progress:</h4>
              <pre>{JSON.stringify(enhancedProgressData, null, 2)}</pre>
            </div>
          </details>
        </div>
      )}
    </div>
  );
};

export default Phase30ProgressWrapper;