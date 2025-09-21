import React, { useState, useEffect, useRef, useCallback } from 'react';
import { getSimulationProgress, getConnectionStatus, forceHealthCheck } from '../services/simulationService';

/**
 * UnifiedProgressTracker - Resilient progress tracking component
 * 
 * Features:
 * - Smart polling with adaptive frequency
 * - Timeout recovery and offline detection
 * - Progress interpolation for smooth UX
 * - Manual refresh capability
 * - Connection health monitoring
 */
const UnifiedProgressTracker = ({ 
  simulationId, 
  onProgressUpdate, 
  onComplete, 
  onError,
  disabled = false,
  initialProgress = null 
}) => {
  // Core state
  const [progress, setProgress] = useState(initialProgress || {
    progress_percentage: 0,
    status: 'initializing',
    message: 'Preparing simulation...',
    current_iteration: 0,
    total_iterations: 0
  });
  
  // Connection and polling state
  const [isPolling, setIsPolling] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [errorCount, setErrorCount] = useState(0);
  const [lastSuccessfulUpdate, setLastSuccessfulUpdate] = useState(Date.now());
  const [pollingInterval, setPollingInterval] = useState(2000); // Start with 2 second polling
  
  // Interpolation state for smooth progress
  const [interpolatedProgress, setInterpolatedProgress] = useState(0);
  const [lastKnownProgress, setLastKnownProgress] = useState(0);
  const [progressDirection, setProgressDirection] = useState('forward');
  
  // Refs for cleanup
  const pollingTimeoutRef = useRef(null);
  const interpolationIntervalRef = useRef(null);
  const mountedRef = useRef(true);
  
  // Constants
  const MAX_ERROR_COUNT = 5;
  const BASE_POLLING_INTERVAL = 2000; // 2 seconds
  const MAX_POLLING_INTERVAL = 30000; // 30 seconds
  const INTERPOLATION_SPEED = 0.1; // Progress interpolation speed
  const CONNECTION_TIMEOUT = 30000; // 30 seconds before marking as offline
  
  /**
   * Calculate adaptive polling interval based on errors and progress stage
   */
  const calculatePollingInterval = useCallback((currentProgress, errorCount) => {
    let interval = BASE_POLLING_INTERVAL;
    
    // Increase interval based on error count (exponential backoff)
    if (errorCount > 0) {
      interval = Math.min(BASE_POLLING_INTERVAL * Math.pow(2, errorCount), MAX_POLLING_INTERVAL);
    }
    
    // Adjust based on progress stage
    if (currentProgress?.status === 'completed' || currentProgress?.status === 'failed') {
      interval = 10000; // Slower polling for finished simulations
    } else if (currentProgress?.progress_percentage > 80) {
      interval = Math.max(interval, 1500); // Faster polling near completion
    }
    
    return interval;
  }, []);
  
  /**
   * Progress interpolation for smoother UX when updates are sparse
   */
  const startProgressInterpolation = useCallback(() => {
    if (interpolationIntervalRef.current) {
      clearInterval(interpolationIntervalRef.current);
    }
    
    interpolationIntervalRef.current = setInterval(() => {
      if (!mountedRef.current) return;
      
      setInterpolatedProgress(prev => {
        const target = lastKnownProgress;
        const diff = target - prev;
        
        // Only interpolate if we're behind and moving forward
        if (Math.abs(diff) > 0.1 && progressDirection === 'forward') {
          return prev + (diff * INTERPOLATION_SPEED);
        }
        
        return prev;
      });
    }, 100); // Update interpolation every 100ms
  }, [lastKnownProgress, progressDirection]);
  
  /**
   * Stop progress interpolation
   */
  const stopProgressInterpolation = useCallback(() => {
    if (interpolationIntervalRef.current) {
      clearInterval(interpolationIntervalRef.current);
      interpolationIntervalRef.current = null;
    }
  }, []);
  
  /**
   * Handle successful progress update
   */
  const handleProgressSuccess = useCallback((progressData) => {
    if (!mountedRef.current) return;
    
    setProgress(progressData);
    setErrorCount(0);
    setLastSuccessfulUpdate(Date.now());
    setConnectionStatus('connected');
    
    // Update interpolation targets
    const newProgress = progressData.progress_percentage || 0;
    if (newProgress > lastKnownProgress) {
      setProgressDirection('forward');
    } else if (newProgress < lastKnownProgress) {
      setProgressDirection('backward');
    }
    setLastKnownProgress(newProgress);
    setInterpolatedProgress(newProgress);
    
    // Calculate next polling interval
    const nextInterval = calculatePollingInterval(progressData, 0);
    setPollingInterval(nextInterval);
    
    // Callback to parent
    if (onProgressUpdate) {
      onProgressUpdate(progressData);
    }
    
    // Check for completion
    if (progressData.status === 'completed' && onComplete) {
      onComplete(progressData);
    }
  }, [lastKnownProgress, calculatePollingInterval, onProgressUpdate, onComplete]);
  
  /**
   * Handle progress fetch error
   */
  const handleProgressError = useCallback((error, errorType = 'unknown') => {
    if (!mountedRef.current) return;
    
    const newErrorCount = errorCount + 1;
    setErrorCount(newErrorCount);
    
    // Update connection status based on error type
    if (errorType === 'timeout' || errorType === 'network') {
      setConnectionStatus('connecting');
    } else if (errorType === 'not_found') {
      setConnectionStatus('not_found');
    } else {
      setConnectionStatus('error');
    }
    
    // Calculate next polling interval with backoff
    const nextInterval = calculatePollingInterval(progress, newErrorCount);
    setPollingInterval(nextInterval);
    
    console.warn(`Progress fetch error ${newErrorCount}/${MAX_ERROR_COUNT} for ${simulationId}:`, error);
    
    // Stop polling if too many errors
    if (newErrorCount >= MAX_ERROR_COUNT) {
      setIsPolling(false);
      setConnectionStatus('offline');
      
      if (onError) {
        onError(new Error(`Progress tracking failed after ${MAX_ERROR_COUNT} attempts`));
      }
    }
  }, [errorCount, progress, calculatePollingInterval, simulationId, onError]);
  
  /**
   * Fetch progress with error handling
   */
  const fetchProgress = useCallback(async () => {
    if (!simulationId || disabled) return;
    
    try {
      const progressData = await getSimulationProgress(simulationId, 3); // 3 retries
      
      if (progressData) {
        if (progressData.status === 'not_found') {
          handleProgressError(new Error('Simulation not found'), 'not_found');
        } else if (progressData.status === 'timeout') {
          handleProgressError(new Error('Request timeout'), 'timeout');
        } else if (progressData.status === 'connection_error') {
          handleProgressError(new Error('Connection error'), 'network');
        } else if (progressData.status === 'error') {
          handleProgressError(new Error(progressData.message || 'Unknown error'), 'error');
        } else {
          handleProgressSuccess(progressData);
        }
      } else {
        handleProgressError(new Error('No progress data received'), 'empty');
      }
    } catch (error) {
      handleProgressError(error, 'exception');
    }
  }, [simulationId, disabled, handleProgressSuccess, handleProgressError]);
  
  /**
   * Start polling for progress
   */
  const startPolling = useCallback(() => {
    if (!simulationId || disabled || isPolling) return;
    
    setIsPolling(true);
    setConnectionStatus('connecting');
    console.log(`Starting progress polling for ${simulationId} (interval: ${pollingInterval}ms)`);
    
    // Start interpolation
    startProgressInterpolation();
    
    // Initial fetch
    fetchProgress();
    
    // Set up recurring polling
    const poll = () => {
      if (!mountedRef.current || !isPolling) return;
      
      pollingTimeoutRef.current = setTimeout(async () => {
        await fetchProgress();
        poll(); // Schedule next poll
      }, pollingInterval);
    };
    
    poll();
  }, [simulationId, disabled, isPolling, pollingInterval, fetchProgress, startProgressInterpolation]);
  
  /**
   * Stop polling for progress
   */
  const stopPolling = useCallback(() => {
    setIsPolling(false);
    setConnectionStatus('disconnected');
    
    if (pollingTimeoutRef.current) {
      clearTimeout(pollingTimeoutRef.current);
      pollingTimeoutRef.current = null;
    }
    
    stopProgressInterpolation();
    
    console.log(`Stopped progress polling for ${simulationId}`);
  }, [simulationId, stopProgressInterpolation]);
  
  /**
   * Manual refresh progress
   */
  const refreshProgress = useCallback(async () => {
    setErrorCount(0);
    setConnectionStatus('connecting');
    await fetchProgress();
  }, [fetchProgress]);
  
  /**
   * Force health check and retry connection
   */
  const retryConnection = useCallback(async () => {
    console.log('Retrying connection...');
    setConnectionStatus('connecting');
    setErrorCount(0);
    
    try {
      const isHealthy = await forceHealthCheck();
      if (isHealthy) {
        setConnectionStatus('connected');
        if (!isPolling) {
          startPolling();
        }
      } else {
        setConnectionStatus('offline');
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setConnectionStatus('offline');
    }
  }, [isPolling, startPolling]);
  
  /**
   * Check for offline condition
   */
  const checkOfflineStatus = useCallback(() => {
    const timeSinceLastUpdate = Date.now() - lastSuccessfulUpdate;
    
    if (timeSinceLastUpdate > CONNECTION_TIMEOUT && connectionStatus !== 'offline') {
      setConnectionStatus('offline');
      console.warn(`No progress updates for ${timeSinceLastUpdate}ms, marking as offline`);
    }
  }, [lastSuccessfulUpdate, connectionStatus]);
  
  // Effect to start/stop polling based on simulationId and disabled state
  useEffect(() => {
    if (simulationId && !disabled) {
      startPolling();
    } else {
      stopPolling();
    }
    
    return () => {
      stopPolling();
    };
  }, [simulationId, disabled, startPolling, stopPolling]);
  
  // Effect to check for offline status
  useEffect(() => {
    const offlineCheckInterval = setInterval(checkOfflineStatus, 5000);
    
    return () => {
      clearInterval(offlineCheckInterval);
    };
  }, [checkOfflineStatus]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      stopPolling();
    };
  }, [stopPolling]);
  
  // Calculate display progress (use interpolated for smoother UX)
  const displayProgress = Math.min(Math.max(interpolatedProgress, 0), 100);
  
  // Generate status message
  const getStatusMessage = () => {
    if (connectionStatus === 'offline') {
      return 'Connection lost - Click to retry';
    } else if (connectionStatus === 'connecting') {
      return 'Connecting...';
    } else if (connectionStatus === 'not_found') {
      return 'Simulation not found';
    } else if (connectionStatus === 'error') {
      return `Connection issues (${errorCount}/${MAX_ERROR_COUNT} errors)`;
    } else if (progress.status === 'completed') {
      return 'Simulation completed';
    } else if (progress.status === 'failed') {
      return 'Simulation failed';
    } else {
      return progress.message || progress.stage_description || 'Processing...';
    }
  };
  
  // Generate status color
  const getStatusColor = () => {
    if (connectionStatus === 'offline' || connectionStatus === 'error') {
      return 'red';
    } else if (connectionStatus === 'connecting') {
      return 'orange';
    } else if (progress.status === 'completed') {
      return 'green';
    } else if (progress.status === 'failed') {
      return 'red';
    } else {
      return 'blue';
    }
  };
  
  return {
    // State
    progress,
    displayProgress,
    connectionStatus,
    isPolling,
    errorCount,
    
    // Status information
    statusMessage: getStatusMessage(),
    statusColor: getStatusColor(),
    isConnected: connectionStatus === 'connected',
    isOffline: connectionStatus === 'offline',
    
    // Actions
    startPolling,
    stopPolling,
    refreshProgress,
    retryConnection,
    
    // Debug information
    debug: {
      pollingInterval,
      lastSuccessfulUpdate,
      interpolatedProgress,
      lastKnownProgress,
      progressDirection,
      connectionStatus: getConnectionStatus()
    }
  };
};

export default UnifiedProgressTracker;

