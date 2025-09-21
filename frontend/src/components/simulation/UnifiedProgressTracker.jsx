import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchSimulationStatus, fetchSimulationProgress, fetchBatchSimulationResults, selectSimulationProgressData, selectProgressHasCompleted } from '../../store/simulationSlice';
import {
  validateSimulationId,
  getParentId,
  logIdValidationWarning
} from '../../utils/simulationIdUtils';
import { 
  CheckCircle, 
  Play, 
  AlertCircle, 
  Clock,
  Loader2,
  X,
  FileText,
  Calculator,
  BarChart3
} from 'lucide-react';
import './UnifiedProgressTracker.css';
import logger from '../../utils/logger';

const UnifiedProgressTracker = ({ simulationIds = [], childSimulationIds = [], targetVariables = [], forceCompleted = false, primaryEngineType = null, restoredResults = [], onResultsCheck = null }) => {
  const dispatch = useDispatch();
  
  // CRITICAL FIX: Add ID validation at component initialization
  useEffect(() => {
    simulationIds.forEach((id, index) => {
      const validation = validateSimulationId(id);
      if (!validation.isValid && validation.isCorrupted) {
        console.warn(
          `🔧 UnifiedProgressTracker: Corrupted simulation ID detected at index ${index}:`,
          {
            corruptedId: id,
            suggestedFix: validation.suggestedFix,
            component: 'UnifiedProgressTracker',
            stackTrace: new Error().stack
          }
        );
      }
    });
  }, [simulationIds]);
  
  // Get progress data from Redux store
  const progressDataFromStore = useSelector(selectSimulationProgressData);
  
  // Use reselect-based selectors for stability
  const progressHasCompleted = useSelector(selectProgressHasCompleted);
  
  // Get global status to improve completion detection
  const globalStatus = useSelector(state => state.simulation.status);
  
  // 🚀 CRITICAL FIX: Listen to Redux for real-time simulation ID updates from WebSocket mapping
  const currentSimulationIdFromRedux = useSelector(state => state.simulation.currentSimulationId);
  
  // CRITICAL FIX: Use Redux currentSimulationId if available (for WebSocket mapping), otherwise use prop
  const primarySimulationId = currentSimulationIdFromRedux || (simulationIds[0] ? getParentId(simulationIds[0]) : null);
  const shouldPoll = (currentSimulationIdFromRedux || simulationIds.length > 0) && !forceCompleted;
  
  // 🔥 DEBUG: Log simulation ID to understand progress tracking issues
  useEffect(() => {
    if (primarySimulationId) {
      console.log('🔥 [PROGRESS_DEBUG] UnifiedProgressTracker using simulation ID:', primarySimulationId);
      console.log('🔥 [PROGRESS_DEBUG] Raw simulationIds:', simulationIds);
      console.log('🔥 [PROGRESS_DEBUG] currentSimulationIdFromRedux:', currentSimulationIdFromRedux);
      console.log('🔥 [PROGRESS_DEBUG] shouldPoll:', shouldPoll);
    }
  }, [primarySimulationId, simulationIds, currentSimulationIdFromRedux, shouldPoll]);
  
  // Log ID validation warnings for debugging
  useEffect(() => {
    if (primarySimulationId) {
      logIdValidationWarning(primarySimulationId, 'UnifiedProgressTracker primarySimulationId');
    }
  }, [primarySimulationId]);

  // 🚀 CRITICAL: Detect WebSocket ID mapping and restart polling immediately
  useEffect(() => {
    if (currentSimulationIdFromRedux && 
        simulationIds.length > 0 && 
        simulationIds[0].startsWith('temp_') && 
        !currentSimulationIdFromRedux.startsWith('temp_')) {
      console.log('🚀 [WebSocket] ID mapping detected! Switching polling from temp to real ID:', simulationIds[0], '->', currentSimulationIdFromRedux);
      
      // Force a progress data reset to ensure fresh polling
      setRenderKey(prev => prev + 1);
      
      // Trigger immediate progress fetch with new ID
      dispatch(fetchSimulationProgress(currentSimulationIdFromRedux));
    }
  }, [currentSimulationIdFromRedux, simulationIds, dispatch]);
  
  // State and refs
  const [isActive, setIsActive] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [pollingError, setPollingError] = useState(null);
  const [hasEverBeenActive, setHasEverBeenActive] = useState(false);
  const [renderKey, setRenderKey] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [childProgress, setChildProgress] = useState({});
  const [smoothProgress, setSmoothProgress] = useState(0);
  const [targetProgress, setTargetProgress] = useState(0);
  
  const pollingIntervalRef = useRef(null);
  const activeSimIdRef = useRef(null);
  const statusPollingIntervalRef = useRef(null);
  const childPollingRefs = useRef({});
  const startTimeRef = useRef(null);
  const elapsedTimerRef = useRef(null);
  const smoothProgressRef = useRef(null);
  
  const pendingBatchFetch = useSelector(state => state.simulation._pendingBatchFetch);
  
  const forceRerender = useCallback(() => {
    setRenderKey(prev => prev + 1);
  }, []);
  
  const [unifiedProgress, setUnifiedProgress] = useState({
    stage: 'initializing',
    overallProgress: 0,
    currentStage: 'Initializing...',
    phases: {
      initialization: { progress: 0, completed: false, stage: 'File Upload & Validation' },
      parsing: { progress: 0, completed: false, stage: 'Parsing Excel File' },
      smart_analysis: { progress: 0, completed: false, stage: 'Smart Dependency Analysis' },
      analysis: { progress: 0, completed: false, stage: 'Formula Analysis' },
      simulation: { progress: 0, completed: false, stage: 'Running Monte Carlo Simulation' },
      results: { progress: 0, completed: false, stage: 'Generating Results' }
    },
    variables: {},
    totalIterations: 0,
    completedIterations: 0,
    startTime: null,
    estimatedTimeRemaining: null,
    engineInfo: {
      engine: null,
      engine_type: null,
      gpu_acceleration: null,
      detected: false
    },
    formulaMetrics: {
      total_formulas: 0,
      relevant_formulas: 0,
      analysis_method: null,
      cache_hits: 0,
      chunks_processed: 0
    },
    backendStartTime: null
  });

  // CRITICAL FIX: Reset progress state when starting tracking for a new simulation
  useEffect(() => {
    if (primarySimulationId && !forceCompleted) {
      // Clear previous progress data to prevent 100% values from previous runs
      setTargetProgress(0);
      setSmoothProgress(0);
      setUnifiedProgress(prev => ({
        ...prev,
        overallProgress: 0,
        currentStage: 'Initializing...',
        completedIterations: 0
      }));
      
      // Reset start time
      startTimeRef.current = Date.now();
      
      // Clear and null elapsed timer
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
        elapsedTimerRef.current = null;
      }
      
      // Cancel any smooth progress animation frame
      if (smoothProgressRef.current) {
        cancelAnimationFrame(smoothProgressRef.current);
        smoothProgressRef.current = null;
      }
      
      // Reset child polling refs
      childPollingRefs.current = {};
      
      logger.debug(`[UnifiedProgressTracker] 🔄 Reset progress state for new simulation: ${primarySimulationId}`);
    }
  }, [primarySimulationId, forceCompleted]);

  // CRITICAL FIX: Improved progress polling effect with better dependency management
  useEffect(() => {
    // DEFENSIVE CHECK: Validate primarySimulationId before starting polling
    if (primarySimulationId) {
      const validation = validateSimulationId(primarySimulationId);
      if (!validation.isValid) {
        console.warn(`[UnifiedProgressTracker] Invalid simulation ID for polling: ${validation.error}`);
        return;
      }
    }

    const shouldStartPolling = isActive && shouldPoll && primarySimulationId && 
      !progressHasCompleted && 
      !(unifiedProgress.overallProgress >= 100) && 
      globalStatus !== 'completed';
    
    // Clear interval immediately on ID change before setting a new one
    if (activeSimIdRef.current !== primarySimulationId) {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
        logger.debug(`[UnifiedProgressTracker] 🔄 Cleared previous polling for ID change: ${activeSimIdRef.current} -> ${primarySimulationId}`);
      }
      activeSimIdRef.current = primarySimulationId;
    }
    
    if (shouldStartPolling) {
      // Guard: Prevent multiple polling intervals
      if (pollingIntervalRef.current) {
        logger.debug(`[UnifiedProgressTracker] ⚠️ Polling interval already exists, skipping creation`);
        return;
      }
      
      logger.debug(`[UnifiedProgressTracker] 🚀 Starting real-time progress polling for ${primarySimulationId}`);
      setIsPolling(true);
      setPollingError(null);
      
      // Add random jitter to polling interval start (±250ms)
      const jitter = Math.random() * 500 - 250; // -250ms to +250ms
      
      // Immediate fetch after jitter delay
      setTimeout(() => {
        dispatch(fetchSimulationProgress(primarySimulationId));
      }, Math.max(0, jitter));
      
      // CRITICAL FIX: Add cleanup guards and better error handling
      pollingIntervalRef.current = setInterval(() => {
        // Compare to activeSimIdRef.current to avoid cross-talk
        if (activeSimIdRef.current === primarySimulationId) {
          // Ensure the parent ID is always used for API calls
          const cleanId = getParentId(primarySimulationId);
          dispatch(fetchSimulationProgress(cleanId))
            .catch(error => {
              logger.warn(`Progress polling error for ${cleanId}:`, error);
              setPollingError(error.message);
            });
        } else {
          logger.debug(`[UnifiedProgressTracker] 🚫 Skipping polling dispatch - ID mismatch: ${activeSimIdRef.current} vs ${primarySimulationId}`);
        }
      }, 2000); // Poll every 2 seconds
      
      return () => {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
          logger.debug(`[UnifiedProgressTracker] 🛑 Stopped progress polling for ${primarySimulationId}`);
        }
        setIsPolling(false);
      };
    } else {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
        logger.debug(`[UnifiedProgressTracker] 🛑 Progress polling cleared - hasCompleted: ${progressDataFromStore.hasCompleted}`);
      }
      setIsPolling(false);
    }
  }, [isActive, shouldPoll, primarySimulationId, progressHasCompleted, globalStatus, unifiedProgress.overallProgress, dispatch]);
  
  // Status polling effect (15s)
  useEffect(() => {
    const shouldStartStatusPolling = isActive && shouldPoll && primarySimulationId && 
      !progressHasCompleted && 
      !(unifiedProgress.overallProgress >= 100) && 
      globalStatus !== 'completed';
    
    if (shouldStartStatusPolling) {
      // Guard: Prevent multiple status polling intervals
      if (statusPollingIntervalRef.current) {
        logger.debug(`[UnifiedProgressTracker] ⚠️ Status polling interval already exists, skipping creation`);
        return;
      }
      
      logger.debug(`[UnifiedProgressTracker] 📊 Starting light status polling for ${primarySimulationId}`);
      
      // CRITICAL FIX: Use parent ID for status polling and improve error handling
      statusPollingIntervalRef.current = setInterval(() => {
        if (activeSimIdRef.current === primarySimulationId) {
          const cleanId = getParentId(primarySimulationId);
          dispatch(fetchSimulationStatus(cleanId))
            .then((result) => {
                              if (result.payload && result.payload.status === 'completed') {
                logger.debug(`[UnifiedProgressTracker] ✅ Status polling detected completion for ${cleanId}`);
                setIsActive(false);
                // Note: fetchSimulationStatus already called in .then block, no need to dispatch again
              }
            })
            .catch(error => {
              logger.warn(`Status polling error for ${cleanId}:`, error);
            });
        }
      }, 15000); // Poll every 15 seconds
      
      return () => {
        if (statusPollingIntervalRef.current) {
          clearInterval(statusPollingIntervalRef.current);
          statusPollingIntervalRef.current = null;
          logger.debug(`[UnifiedProgressTracker] 📊 Stopped status polling for ${primarySimulationId}`);
        }
      };
    } else {
      if (statusPollingIntervalRef.current) {
        clearInterval(statusPollingIntervalRef.current);
        statusPollingIntervalRef.current = null;
        logger.debug(`[UnifiedProgressTracker] 📊 Status polling cleared - hasCompleted: ${progressDataFromStore.hasCompleted}`);
      }
    }
  }, [isActive, shouldPoll, primarySimulationId, progressHasCompleted, globalStatus, unifiedProgress.overallProgress, dispatch]);

  // Redux-driven progress update effect
  useEffect(() => {
    if (progressDataFromStore && primarySimulationId && isActive) {
      const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
      logger.debug(`[UnifiedProgressTracker] 🚀 ${timestamp} REDUX: Processing store data:`, {
        progress: progressDataFromStore.progressPercentage,
        stage: progressDataFromStore.currentStage,
        iteration: progressDataFromStore.currentIteration,
        total: progressDataFromStore.totalIterations
      });
      
      // Update unified progress using store data
      if (progressDataFromStore.progressPercentage !== undefined && progressDataFromStore.progressPercentage >= 0) {
        logger.debug(`[UnifiedProgressTracker] 🔥 REDUX: Updating unified progress: ${progressDataFromStore.progressPercentage}% (${progressDataFromStore.currentStage})`);
        
        // Update main progress from real-time data
        const currentProgress = Math.max(0, Math.min(100, Number(progressDataFromStore.progressPercentage || 0)));
        setUnifiedProgress(prev => ({
          ...prev,
          overallProgress: currentProgress,
          currentStage: progressDataFromStore.stageDescription || progressDataFromStore.currentStage || prev.currentStage,
          totalIterations: progressDataFromStore.totalIterations || prev.totalIterations || 0,
          completedIterations: progressDataFromStore.currentIteration || prev.completedIterations || 0,
          // 🔧 FIX ISSUE 3: Update phases based on actual progress
          phases: {
            ...prev.phases,
            simulation: { 
              progress: currentProgress <= 90 ? Math.min((currentProgress / 90) * 100, 100) : 100, // Maps 0-90% overall to 0-100% simulation
              completed: currentProgress >= 90, 
              stage: 'Running Monte Carlo Simulation' 
            },
            results: { 
              progress: currentProgress >= 90 ? Math.min((currentProgress - 90) * 10, 100) : 0, // Results is 90-100% mapped to 0-100%
              completed: currentProgress >= 100, 
              stage: 'Generating Results' 
            }
          }
        }));
        
        // Update target progress for smooth interpolation
        setTargetProgress(progressDataFromStore.progressPercentage || 0);
        
        // CRITICAL FIX: Ensure smoothProgress is updated immediately for large gaps
        const currentSmoothProgress = smoothProgress;
        const newTargetProgress = progressDataFromStore.progressPercentage || 0;
        const progressGap = Math.abs(newTargetProgress - currentSmoothProgress);
        
        // If there's a large gap (>10%) or we're starting from 0, update smoothProgress immediately
        if (progressGap > 10 || (currentSmoothProgress === 0 && newTargetProgress > 0)) {
          setSmoothProgress(newTargetProgress);
          logger.debug(`[UnifiedProgressTracker] 🚀 IMMEDIATE PROGRESS UPDATE: ${currentSmoothProgress}% → ${newTargetProgress}% (gap: ${progressGap}%)`);
        }
        
        // Update variables object with real-time data
        setUnifiedProgress(prev => ({
          ...prev,
          variables: {
            ...prev.variables,
            [primarySimulationId]: {
              ...prev.variables[primarySimulationId],
              progress: progressDataFromStore.progressPercentage,
              iterations: progressDataFromStore.currentIteration || 0,
              totalIterations: progressDataFromStore.totalIterations || 0,
              stage: progressDataFromStore.currentStage,
              stage_description: progressDataFromStore.stageDescription
            }
          }
        }));
        
        // Handle completion based on progress percentage
        if (progressDataFromStore.progressPercentage >= 100) {
          logger.debug('[UnifiedProgressTracker] ✅ REDUX: Simulation completed, setting inactive');
          setIsActive(false);
          
          // Fetch final results
          if (dispatch && primarySimulationId) {
            logger.debug('[UnifiedProgressTracker] 📤 REDUX: Fetching final results for', primarySimulationId);
            dispatch(fetchSimulationStatus(primarySimulationId));
          }
        }
      }
    }
  }, [progressDataFromStore, primarySimulationId, dispatch, isActive]);

  // Smooth interpolation effect
  useEffect(() => {
    if (hasEverBeenActive && targetProgress !== smoothProgress) {
      const progressGap = Math.abs(targetProgress - smoothProgress);
      
      // Only reset to 0 on new simulation initialization, not for stray 0% updates
      if (targetProgress === 0 && !progressDataFromStore.hasCompleted && smoothProgress < 5) {
        setSmoothProgress(0);
        logger.debug(`[UnifiedProgressTracker] 🌊 Reset progress to 0% (new simulation)`);
        return;
      }
      
      // If target is 100%, jump immediately
      if (targetProgress >= 100) {
        setSmoothProgress(targetProgress);
        logger.debug(`[UnifiedProgressTracker] 🌊 Jumped to completion: ${targetProgress}%`);
        return;
      }
      
      // If gap is very large (>20%), jump halfway immediately to avoid long delays
      if (progressGap > 20) {
        const jumpTo = smoothProgress + (targetProgress - smoothProgress) * 0.7;
        setSmoothProgress(jumpTo);
        logger.debug(`[UnifiedProgressTracker] 🌊 Large gap detected, jumping to: ${jumpTo}%`);
        return;
      }
      
      // 🔧 ULTIMATE MONOTONICITY FIX: Prevent ALL backwards progress after 1%
      if (targetProgress < smoothProgress && smoothProgress > 1) {
        logger.debug(`[UnifiedProgressTracker] 🔧 BLOCKING backwards progress: ${smoothProgress}% → ${targetProgress}% (backend retrieval issue)`);
        return;
      }
      
      // 🔧 ADDITIONAL: Prevent large backwards jumps even from 0%
      if (targetProgress < smoothProgress && (smoothProgress - targetProgress) > 10) {
        logger.debug(`[UnifiedProgressTracker] 🔧 BLOCKING large backwards jump: ${smoothProgress}% → ${targetProgress}% (${smoothProgress - targetProgress}% drop)`);
        return;
      }
      
      // Normal smooth interpolation for smaller gaps (even small ones)
      if (targetProgress > smoothProgress && progressGap > 0.5) {
        logger.debug(`[UnifiedProgressTracker] 🌊 Starting smooth interpolation: ${smoothProgress}% → ${targetProgress}%`);
        
        const startProgress = smoothProgress;
        const progressDiff = targetProgress - startProgress;
        const duration = Math.min(2000, progressGap * 100); // Shorter duration for smaller gaps
        const startTime = Date.now();
        
        const animate = () => {
          const elapsed = Date.now() - startTime;
          const progress = Math.min(elapsed / duration, 1);
          
          // Ease-out function for smooth animation
          const easeOut = 1 - Math.pow(1 - progress, 3);
          const newProgress = startProgress + (progressDiff * easeOut);
          
          setSmoothProgress(newProgress);
          
          if (progress < 1 && newProgress < targetProgress) {
            smoothProgressRef.current = requestAnimationFrame(animate);
          } else {
            setSmoothProgress(targetProgress);
            logger.debug(`[UnifiedProgressTracker] 🌊 Smooth interpolation completed at ${targetProgress}%`);
          }
        };
        
        smoothProgressRef.current = requestAnimationFrame(animate);
        
        return () => {
          if (smoothProgressRef.current) {
            cancelAnimationFrame(smoothProgressRef.current);
          }
        };
      } else if (progressGap <= 1) {
        // Small gap, update immediately
        setSmoothProgress(targetProgress);
      }
    }
  }, [targetProgress, hasEverBeenActive, progressDataFromStore.hasCompleted, smoothProgress]);

  // Elapsed timer effect
  useEffect(() => {
    // Start timer when we have a start time and simulation is active
    if (startTimeRef.current && hasEverBeenActive && isActive) {
      logger.debug(`[UnifiedProgressTracker] 🕐 Starting continuous elapsed time timer`);
      
      // Update elapsed time every second
      elapsedTimerRef.current = setInterval(() => {
        if (startTimeRef.current) {
          const newElapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
          setElapsedTime(newElapsed);
          logger.debug(`[UnifiedProgressTracker] ⏰ Elapsed time tick: ${newElapsed}s`);
        }
      }, 1000);
      
      return () => {
        if (elapsedTimerRef.current) {
          clearInterval(elapsedTimerRef.current);
          logger.debug(`[UnifiedProgressTracker] 🕐 Stopped continuous elapsed time timer`);
        }
      };
    } else if (elapsedTimerRef.current) {
      clearInterval(elapsedTimerRef.current);
      logger.debug(`[UnifiedProgressTracker] 🕐 Cleaned up elapsed time timer (inactive)`);
    }
  }, [hasEverBeenActive, isActive, startTimeRef.current]);

  // Batch fetch trigger effect
  useEffect(() => {
    if (pendingBatchFetch && pendingBatchFetch.length > 0) {
      logger.debug('[UnifiedProgressTracker] 🎯 BATCH FETCH TRIGGERED - Fetching child results for:', pendingBatchFetch);
      dispatch(fetchBatchSimulationResults(pendingBatchFetch));
    }
  }, [dispatch, pendingBatchFetch]);

  // CRITICAL FIX: Improved initialization with ID validation and proper cleanup
  useEffect(() => {
    if (simulationIds.length > 0) {
      const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
      
      // Validate all simulation IDs before proceeding
      const invalidIds = simulationIds.filter(id => {
        const validation = validateSimulationId(id);
        return !validation.isValid;
      });
      
      if (invalidIds.length > 0) {
        console.warn(`[UnifiedProgressTracker] Invalid simulation IDs detected:`, invalidIds);
      }
      
      logger.debug(`[UnifiedProgressTracker] 🚀 ${timestamp} Initializing tracking for`, simulationIds.length, 'simulations:', simulationIds);
      setIsActive(true);
      setHasEverBeenActive(true);
      
      // Set start time for elapsed timer
      if (!startTimeRef.current) {
        startTimeRef.current = Date.now();
        logger.debug('[UnifiedProgressTracker] ⏰ Setting start time for elapsed timer');
      }
      
      // Start at 0% and only show real backend progress
      setUnifiedProgress(prev => ({
        ...prev,
        startTime: Date.now(),
        overallProgress: forceCompleted ? 100 : 0, // Start at 0% - no fake progress
        currentStage: forceCompleted ? 'Completed' : 'Starting simulation...',
        phases: {
          initialization: { progress: 100, completed: true, stage: 'File Upload & Validation' },
          parsing: { progress: 100, completed: true, stage: 'Parsing Excel File' },
          smart_analysis: { progress: 100, completed: true, stage: 'Smart Dependency Analysis' },
          analysis: { progress: 100, completed: true, stage: 'Formula Analysis' },
          simulation: { progress: 0, completed: false, stage: 'Running Monte Carlo Simulation' },
          results: { progress: 0, completed: false, stage: 'Generating Results' }
        },
        // CRITICAL FIX: Clean variable tracking using parent IDs
        variables: (() => {
          if (targetVariables.length > 1 && simulationIds.length === 1) {
            // BATCH SIMULATION: Multiple target variables, single parent ID
            const parentId = getParentId(simulationIds[0]);
            logger.debug('[UnifiedProgressTracker] 🔧 BATCH MODE: Creating', targetVariables.length, 'variable entries for batch simulation with parent ID:', parentId);
            return targetVariables.reduce((acc, targetName, index) => {
              const variableKey = `${parentId}_target_${index}`;
              acc[variableKey] = {
                name: targetName,
                progress: forceCompleted ? 100 : 5,
                status: forceCompleted ? 'completed' : 'running',
                iterations: 0,
                totalIterations: 0,
                targetIndex: index // Track which target this represents
              };
              return acc;
            }, {});
          } else {
            // INDIVIDUAL SIMULATION: One simulation per target variable, use parent IDs
            return simulationIds.reduce((acc, simId, index) => {
              const parentId = getParentId(simId);
              acc[parentId] = {
                name: targetVariables[index] || `Variable ${index + 1}`,
                progress: forceCompleted ? 100 : 5,
                status: forceCompleted ? 'completed' : 'running',
                iterations: 0,
                totalIterations: 0
              };
              return acc;
            }, {});
          }
        })()
      }));
      
      // Reset progress when starting a new simulation (prevent jumps from previous simulation)
      if (!forceCompleted && primarySimulationId) {
        logger.debug(`[UnifiedProgressTracker] 🔄 Resetting progress for new simulation: ${primarySimulationId}`);
        setTargetProgress(0);
        setSmoothProgress(0);
      }
    }
  }, [simulationIds.join(','), forceCompleted]);

  // Add useEffect for render-time logging to prevent side effects
  useEffect(() => {
    logger.debug(`[UnifiedProgressTracker] 🎨 RENDERING PROGRESS BAR: ${unifiedProgress.overallProgress}% (renderKey: ${renderKey})`);
  }, [unifiedProgress.overallProgress, renderKey]);

  // Helper function to format time
  const formatTime = (milliseconds) => {
    if (!milliseconds || milliseconds <= 0) return 'Calculating...';
    
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  // Helper function to get phase icon
  const getPhaseIcon = (phase) => {
    switch (phase) {
      case 'initialization': return <FileText className="w-4 h-4" />;
      case 'parsing': return <FileText className="w-4 h-4" />;
      case 'smart_analysis': return <span className="w-4 h-4 text-center">🧠</span>;
      case 'analysis': return <Calculator className="w-4 h-4" />;
      case 'simulation': return <BarChart3 className="w-4 h-4" />;
      case 'results': return <CheckCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  // Render function
  if (simulationIds.length === 0 && !hasEverBeenActive) {
    return null; // Don't show if never initialized
  }

  return (
    <div 
      key={renderKey}
      className={`unified-progress-tracker ${forceCompleted || unifiedProgress.overallProgress === 100 ? 'completed-state' : ''}`}
      data-progress={unifiedProgress.overallProgress}
      data-completed={unifiedProgress.overallProgress === 100}
    >
      <div className="progress-header">
        <div className="status-section">
          <div className="main-status">
            {isActive ? (
              <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
            ) : (
              <CheckCircle className="w-6 h-6 text-green-500" />
            )}
            <div className="status-info">
              <h3 className="progress-title">
                Monte Carlo Simulation Progress
                {isPolling && (
                  <span className="polling-indicator" style={{ 
                    marginLeft: '8px', 
                    fontSize: '0.8em', 
                    color: '#10b981', 
                    backgroundColor: '#dcfce7', 
                    padding: '2px 6px', 
                    borderRadius: '4px' 
                  }}>
                    📡 Live Updates
                  </span>
                )}
                {pollingError && (
                  <span className="polling-error" style={{ 
                    marginLeft: '8px', 
                    fontSize: '0.8em', 
                    color: '#ef4444', 
                    backgroundColor: '#fee2e2', 
                    padding: '2px 6px', 
                    borderRadius: '4px' 
                  }}>
                    ⚠️ Connection Error
                  </span>
                )}
              </h3>
            </div>
          </div>
        </div>
        
        <div className="progress-stats">
          <div className="stat-item">
            <span className="stat-label">Variables</span>
            <span className="stat-value">{targetVariables.length || unifiedProgress.target_count || Object.keys(unifiedProgress.variables).length || 1}</span>
          </div>
          {/* NEW: Iteration KPIs - Fix completion to show 1000/1000 */}
          {unifiedProgress.totalIterations > 0 && (
            <div className="stat-item">
              <span className="stat-label">Iterations</span>
              <span className="stat-value">
                {(forceCompleted || unifiedProgress.overallProgress === 100) 
                  ? `${unifiedProgress.totalIterations.toLocaleString()}/${unifiedProgress.totalIterations.toLocaleString()}`
                  : `${unifiedProgress.completedIterations.toLocaleString()}/${unifiedProgress.totalIterations.toLocaleString()}`
                }
              </span>
            </div>
          )}
          {/* Elapsed Time - using backend start time for accuracy */}
          {(unifiedProgress.backendStartTime || unifiedProgress.startTime) && (
            <div className="stat-item">
              <span className="stat-label">Elapsed</span>
              <span className="stat-value">
                {elapsedTime > 0 ? `${Math.floor(elapsedTime / 60)}m ${elapsedTime % 60}s` : '0s'}
              </span>
            </div>
          )}
          {unifiedProgress.estimatedTimeRemaining && (
            <div className="stat-item">
              <span className="stat-label">ETA</span>
              <span className="stat-value">{formatTime(unifiedProgress.estimatedTimeRemaining)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Main Progress Bar */}
      <div className="main-progress-container">
        <div className="progress-bar-unified">
          <div className="progress-track">
            <div 
              className="progress-fill"
              style={{ width: `${forceCompleted || unifiedProgress.overallProgress === 100 ? 100 : Math.max(smoothProgress, unifiedProgress.overallProgress || 0)}%` }}
              data-progress={forceCompleted || unifiedProgress.overallProgress === 100 ? 100 : Math.max(smoothProgress, unifiedProgress.overallProgress || 0)}
            >
              <div className="progress-shine"></div>
            </div>
            <div className="progress-percentage">
              {Math.round(forceCompleted || unifiedProgress.overallProgress === 100 ? 100 : Math.max(smoothProgress, unifiedProgress.overallProgress || 0))}%
            </div>
          </div>
        </div>
      </div>

      {/* Phase Progress Indicators */}
      <div className="phases-container">
        <div className="phases-grid">
          {Object.entries(unifiedProgress.phases).map(([phaseKey, phase]) => (
            <div 
              key={phaseKey}
              className={`phase-item ${phase.completed ? 'completed' : phase.progress > 0 ? 'active' : 'pending'}`}
            >
              <div className="phase-icon">
                {getPhaseIcon(phaseKey)}
              </div>
              <div className="phase-details">
                <div className="phase-name">{phase.stage}</div>
                <div className="phase-progress">
                  <div className="phase-progress-bar">
                    <div 
                      className="phase-progress-fill"
                      style={{ width: `${phase.progress}%` }}
                    ></div>
                  </div>
                  <span className="phase-percentage">{Math.round(phase.progress)}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default UnifiedProgressTracker;
