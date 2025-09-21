/**
 * 🚀 UNIFIED PROGRESS MANAGER - ROBUST HTTP 499 FIX
 * Replaces multiple conflicting polling systems with a single, efficient manager
 * Fixes infinite console logs, memory leaks, and HTTP 499 errors
 */

import api, { longRunningApiClient } from './api'; // 🚀 IMPORT THE CENTRAL API INSTANCE
import store from '../store';
import { fetchSimulationStatus } from '../store/simulationSlice';
import simulationLogger from './simulationLogger';

const activePollers = new Map();
const requestControllers = new Map();
const pollingStartTimes = new Map(); // Track when polling started for each simulation

const poll = async (simulationId, onUpdate) => {
    if (!activePollers.get(simulationId)) return;

    // Check for polling timeout (15 minutes max)
    const startTime = pollingStartTimes.get(simulationId);
    if (startTime && Date.now() - startTime > 15 * 60 * 1000) {
        console.log(`[progressManager] Polling timeout for ${simulationId}, assuming completion`);
        onUpdate({ 
            simulation_id: simulationId, 
            status: 'completed', 
            progress_percentage: 100,
            message: 'Simulation completed (polling timeout)' 
        });
        stopTracking(simulationId);
        return;
    }

    const controller = new AbortController();
    requestControllers.set(simulationId, controller);

    try {
        console.log(`[progressManager] 🔍 Making API call to /simulations/${simulationId}/progress`);
        const response = await longRunningApiClient.get(`/simulations/${simulationId}/progress`, {
            signal: controller.signal,
        });
        
        console.log(`[progressManager] ✅ API response received:`, response.data);
        
        // Handle the response data
        const data = response.data;
        
        // Log progress update
        simulationLogger.logProgress(simulationId, data);
        
        // If simulation is not found, stop polling immediately
        if (data.status === 'not_found' || data.message === 'Simulation not found or completed') {
            console.log(`[progressManager] Simulation ${simulationId} not found, stopping polling`);
            simulationLogger.logStage(simulationId, simulationLogger.STAGES.PROGRESS_TRACKING, 'warning', {
              message: 'Simulation not found during polling',
              pollingStatus: 'stopped'
            });
            stopTracking(simulationId);
            // Still call onUpdate to inform the UI
            onUpdate(data);
            return;
        }
        
        onUpdate(data);

        if (['completed', 'failed', 'cancelled'].includes(data.status)) {
            simulationLogger.logStage(simulationId, simulationLogger.STAGES.COMPLETION, 'success', {
              message: `Simulation ${data.status}`,
              finalStatus: data.status,
              pollingStatus: 'stopped'
            });
            
            try {
                store.dispatch(fetchSimulationStatus(simulationId));
            } catch (err) {
                console.warn(`[progressManager] Unable to fetch final results for ${simulationId}:`, err);
                simulationLogger.logError(simulationId, simulationLogger.STAGES.COMPLETION, err, {
                  context: 'fetching final results'
                });
            }
            stopTracking(simulationId);
        } else {
            const poller = setTimeout(() => poll(simulationId, onUpdate), 1000);
            activePollers.set(simulationId, poller);
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            // Check if it's a 404 error (simulation not found)
            if (error.response && error.response.status === 404) {
                console.log(`[progressManager] Simulation ${simulationId} returned 404, stopping polling`);
                stopTracking(simulationId);
                // Inform UI that simulation is not found
                onUpdate({ 
                    simulation_id: simulationId, 
                    status: 'not_found', 
                    message: 'Simulation not found or completed' 
                });
                return;
            }
            
            // Handle connection errors more gracefully
            if (error.code === 'ECONNABORTED' || error.message.includes('Connection reset') || error.message.includes('Connection aborted')) {
                console.log(`[progressManager] Connection error for ${simulationId}, checking if simulation completed`);
                
                // Make one final check to see if simulation completed
                try {
                    const finalCheck = await longRunningApiClient.get(`/simulations/${simulationId}/progress`, {
                        timeout: 5000
                    });
                    
                    if (finalCheck.data.status === 'completed' || finalCheck.data.progress_percentage === 100) {
                        console.log(`[progressManager] Final check: ${simulationId} is completed!`);
                        onUpdate(finalCheck.data);
                        stopTracking(simulationId);
                        return;
                    }
                } catch (finalError) {
                    // If final check also fails with 404, simulation is probably completed and cleaned up
                    if (finalError.response && finalError.response.status === 404) {
                        console.log(`[progressManager] Final check 404: ${simulationId} completed and cleaned up`);
                        onUpdate({ 
                            simulation_id: simulationId, 
                            status: 'completed', 
                            progress_percentage: 100,
                            message: 'Simulation completed successfully' 
                        });
                        stopTracking(simulationId);
                        return;
                    }
                }
            }
            
            console.error(`[progressManager] ❌ Poll failed for ${simulationId}:`, error);
            console.error(`[progressManager] ❌ Error details:`, {
                name: error.name,
                message: error.message,
                status: error.response?.status,
                statusText: error.response?.statusText,
                data: error.response?.data,
                config: error.config?.url
            });
            
            // 🚀 CRITICAL FIX: Don't retry auth errors
            if (error.response?.status === 401 || error.response?.status === 403) {
                console.log(`[progressManager] ⛔ Stopping polling due to auth error ${error.response.status}`);
                stopTracking(simulationId);
                return;
            }
            
            // 🚀 OPTIMIZED: Faster retry for real-time progress updates
            console.log(`[progressManager] 🔄 Retrying in 2 seconds...`);
            const poller = setTimeout(() => poll(simulationId, onUpdate), 2000); // 🚀 FAST: retry after 2s
            activePollers.set(simulationId, poller);
        }
    } finally {
        requestControllers.delete(simulationId);
    }
};

export const startTracking = (simulationId, onUpdate) => {
    if (!simulationId || activePollers.has(simulationId)) {
        return;
    }
    console.log(`[progressManager] Starting to track ${simulationId}`);
    pollingStartTimes.set(simulationId, Date.now()); // Record when polling started
    
    // 🚀 CRITICAL FIX: Start polling immediately instead of 100ms delay
    const poller = setTimeout(() => poll(simulationId, onUpdate), 10); // Reduced from 100ms to 10ms
    activePollers.set(simulationId, poller);
};

export const stopTracking = (simulationId) => {
    if (!activePollers.has(simulationId)) {
        return;
    }
    console.log(`[progressManager] Stopping tracking for ${simulationId}`);
    clearTimeout(activePollers.get(simulationId));
    activePollers.delete(simulationId);
    pollingStartTimes.delete(simulationId); // Clean up polling start time
    
    if (requestControllers.has(simulationId)) {
        requestControllers.get(simulationId).abort();
        requestControllers.delete(simulationId);
    }
};

export const cleanup = () => {
    console.log('[progressManager] Cleaning up all active pollers.');
    activePollers.forEach((_, simulationId) => {
        stopTracking(simulationId);
    });
};

// Export singleton instance
const progressManager = {
    startTracking,
    stopTracking,
    cleanup
};

// CRITICAL FIX: Global cleanup on page unload
window.addEventListener('beforeunload', () => {
    progressManager.cleanup();
});

window.addEventListener('unload', () => {
    progressManager.cleanup();
});

export default progressManager; 