import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * 🚀 SIMPLE, BULLETPROOF SIMULATION POLLING HOOK
 * 
 * This hook provides clean, reliable polling for simulation progress
 * with proper cleanup and termination handling.
 */
const useSimulationPolling = (simulationId, isActive = true) => {
  // 📊 Progress state
  const [progressData, setProgressData] = useState({
    progress_percentage: 0,
    stage: 'initialization',
    stage_description: 'Initializing simulation...',
    current_iteration: 0,
    total_iterations: 0,
    target_count: 0,
    status: 'running',
    elapsed_time: 0
  });

  // 🔧 Polling management
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);
  const mountedRef = useRef(true);
  const completedRef = useRef(false);

  // 🧹 Cleanup function
  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsPolling(false);
  }, []);

  // 📡 Fetch progress function
  const fetchProgress = useCallback(async () => {
    if (!simulationId || !mountedRef.current) {
      return null;
    }

    // If we've already completed, skip any further fetches/logs
    if (completedRef.current) {
      return 'completed';
    }

    try {
      // Use the progress endpoint (no auth required)
      const response = await fetch(`/api/simulations/${simulationId}/progress`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (mountedRef.current) {
        setProgressData(prevData => ({
          ...prevData,
          ...data,
          // Ensure we have default values
          progress_percentage: data.progress_percentage || 0,
          stage: data.stage || 'running',
          stage_description: data.message || data.stage_description || 'Processing...',
          current_iteration: data.current_iteration || 0,
          total_iterations: data.total_iterations || 0,
          // Do not hardcode fallback counts; allow UI to infer from its own context
          target_count: data.target_count || data.targets_count || undefined,
          status: data.status || 'running',
          elapsed_time: data.elapsed_time || 0
        }));
        
        setError(null);

        // Check for completion
        if (data.status === 'not_found' || data.progress_percentage >= 100 || data.status === 'completed' || data.status === 'success') {
          completedRef.current = true;
          
          // Update with completion data and stop polling
          if (mountedRef.current) {
            setProgressData({
              ...data,
              progress_percentage: 100,
              status: 'completed',
              target_count: data.target_count || 1
            });
          }
          
          stopPolling();
          return 'completed';
        }
      }

      return data;
    } catch (err) {
      if (mountedRef.current) {
        setError(err.message);
      }
      return null;
    }
  }, [simulationId, stopPolling]);

  // 🚀 Start polling effect
  useEffect(() => {
    // Reset state when simulation ID changes
    if (simulationId) {
      // Reset completion guard
      completedRef.current = false;

      setProgressData({
        progress_percentage: 0,
        stage: 'initialization',
        stage_description: 'Initializing simulation...',
        current_iteration: 0,
        total_iterations: 0,
        target_count: 0,
        status: 'running',
        elapsed_time: 0
      });
      setError(null);
    }
  }, [simulationId]);

  // 📡 Main polling effect
  useEffect(() => {
    // Don't start if no simulation ID or not active
    if (!simulationId || !isActive) {
      stopPolling();
      return;
    }

    setIsPolling(true);

    // Ensure any previous interval is cleared before creating a new one
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Create polling interval (1s cadence)
    intervalRef.current = setInterval(async () => {
      const result = await fetchProgress();
      if (result === 'completed') {
        // Polling will be stopped by fetchProgress
        return;
      }
    }, 1000); // Poll every 1 second for smooth updates

    // 🏃 Immediate first fetch
    fetchProgress();

    // Cleanup on unmount or dependency change
    return () => {
      stopPolling();
    };
  }, [simulationId, isActive]);

  // 🧹 Component unmount cleanup
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopPolling();
    };
  }, [stopPolling]);

  return {
    progressData,
    isPolling,
    error,
    stopPolling,
    refetch: fetchProgress
  };
};

export default useSimulationPolling;
