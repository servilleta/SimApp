/**
 * 🚀 PROGRESS RECOVERY SERVICE
 * Handles progress synchronization when connections are restored
 * Eliminates "black hole" periods where progress is lost
 */

import { getSimulationProgress } from './simulationService';

class ProgressRecovery {
  constructor() {
    this.lastKnownProgress = new Map(); // simulationId -> progress data
    this.recoveryCallbacks = new Map(); // simulationId -> callback function
    this.recoveryTimers = new Map(); // simulationId -> timer
  }

  /**
   * Register a simulation for progress recovery
   */
  registerSimulation(simulationId, callback) {
    console.log(`🔄 [RECOVERY] Registering simulation for recovery: ${simulationId}`);
    this.recoveryCallbacks.set(simulationId, callback);
    
    // Initialize with current progress if available
    this.updateLastKnownProgress(simulationId, {
      progress_percentage: 0,
      status: 'running',
      timestamp: Date.now()
    });
  }

  /**
   * Update last known progress for a simulation
   */
  updateLastKnownProgress(simulationId, progressData) {
    const previousProgress = this.lastKnownProgress.get(simulationId);
    const newProgress = {
      ...progressData,
      lastUpdated: Date.now()
    };
    
    this.lastKnownProgress.set(simulationId, newProgress);
    
    // Log significant progress changes
    if (previousProgress && previousProgress.progress_percentage !== progressData.progress_percentage) {
      console.log(`🔄 [RECOVERY] Progress updated for ${simulationId}: ${previousProgress.progress_percentage}% -> ${progressData.progress_percentage}%`);
    }
  }

  /**
   * Attempt to recover progress after connection restoration
   */
  async recoverProgress(simulationId) {
    console.log(`🔄 [RECOVERY] Attempting progress recovery for: ${simulationId}`);
    
    const callback = this.recoveryCallbacks.get(simulationId);
    if (!callback) {
      console.warn(`🔄 [RECOVERY] No callback registered for ${simulationId}`);
      return;
    }

    const lastKnown = this.lastKnownProgress.get(simulationId);
    if (!lastKnown) {
      console.warn(`🔄 [RECOVERY] No last known progress for ${simulationId}`);
      return;
    }

    try {
      // Fetch current progress from backend
      const currentProgress = await getSimulationProgress(simulationId);
      
      if (!currentProgress || currentProgress.status === 'error') {
        console.warn(`🔄 [RECOVERY] Could not fetch current progress for ${simulationId}`);
        return;
      }

      const lastPercentage = lastKnown.progress_percentage || 0;
      const currentPercentage = currentProgress.progress_percentage || 0;

      // Check if there's a significant gap (> 5%)
      const progressGap = currentPercentage - lastPercentage;
      
      if (progressGap > 5) {
        console.log(`🔄 [RECOVERY] Progress gap detected for ${simulationId}: ${progressGap}% (${lastPercentage}% -> ${currentPercentage}%)`);
        
        // Smooth the transition by interpolating missing progress
        await this.smoothProgressTransition(simulationId, lastKnown, currentProgress, callback);
      } else {
        // Small gap, just update directly
        console.log(`🔄 [RECOVERY] Small progress gap for ${simulationId}: ${progressGap}%`);
        this.updateLastKnownProgress(simulationId, currentProgress);
        callback(currentProgress);
      }

    } catch (error) {
      console.error(`🔄 [RECOVERY] Progress recovery failed for ${simulationId}:`, error);
    }
  }

  /**
   * Smooth progress transition to avoid jarring jumps
   */
  async smoothProgressTransition(simulationId, fromProgress, toProgress, callback) {
    const fromPercentage = fromProgress.progress_percentage || 0;
    const toPercentage = toProgress.progress_percentage || 0;
    const steps = Math.min(Math.ceil((toPercentage - fromPercentage) / 2), 10); // Max 10 steps
    const stepSize = (toPercentage - fromPercentage) / steps;
    const stepDelay = 200; // 200ms between steps

    console.log(`🔄 [RECOVERY] Smoothing progress transition for ${simulationId}: ${fromPercentage}% -> ${toPercentage}% in ${steps} steps`);

    for (let i = 1; i <= steps; i++) {
      const interpolatedPercentage = fromPercentage + (stepSize * i);
      const interpolatedProgress = {
        ...toProgress,
        progress_percentage: interpolatedPercentage,
        stage_description: i === steps ? toProgress.stage_description : `Catching up... (${Math.round(interpolatedPercentage)}%)`
      };

      this.updateLastKnownProgress(simulationId, interpolatedProgress);
      callback(interpolatedProgress);

      // Wait before next step (except for last step)
      if (i < steps) {
        await new Promise(resolve => setTimeout(resolve, stepDelay));
      }
    }

    console.log(`🔄 [RECOVERY] Progress transition completed for ${simulationId}`);
  }

  /**
   * Start periodic recovery checks
   */
  startRecoveryChecks(simulationId, intervalMs = 5000) {
    this.stopRecoveryChecks(simulationId); // Clear any existing timer

    const timer = setInterval(async () => {
      const lastKnown = this.lastKnownProgress.get(simulationId);
      if (!lastKnown) return;

      // Check if progress has been stale for too long
      const staleDuration = Date.now() - lastKnown.lastUpdated;
      if (staleDuration > 10000) { // 10 seconds
        console.log(`🔄 [RECOVERY] Progress stale for ${simulationId}, attempting recovery`);
        await this.recoverProgress(simulationId);
      }
    }, intervalMs);

    this.recoveryTimers.set(simulationId, timer);
    console.log(`🔄 [RECOVERY] Started recovery checks for ${simulationId} every ${intervalMs}ms`);
  }

  /**
   * Stop recovery checks for a simulation
   */
  stopRecoveryChecks(simulationId) {
    const timer = this.recoveryTimers.get(simulationId);
    if (timer) {
      clearInterval(timer);
      this.recoveryTimers.delete(simulationId);
      console.log(`🔄 [RECOVERY] Stopped recovery checks for ${simulationId}`);
    }
  }

  /**
   * Clean up recovery for a completed/failed simulation
   */
  cleanupSimulation(simulationId) {
    console.log(`🔄 [RECOVERY] Cleaning up simulation: ${simulationId}`);
    this.stopRecoveryChecks(simulationId);
    this.recoveryCallbacks.delete(simulationId);
    this.lastKnownProgress.delete(simulationId);
  }

  /**
   * Get recovery statistics
   */
  getStats() {
    return {
      activeSimulations: this.recoveryCallbacks.size,
      activeTimers: this.recoveryTimers.size,
      trackedProgress: this.lastKnownProgress.size
    };
  }
}

// Global progress recovery instance
const progressRecovery = new ProgressRecovery();

export default progressRecovery;
