/**
 * 🚀 COMPREHENSIVE SIMULATION LOGGER
 * Tracks each stage of the simulation process with detailed logging
 * Provides clear visibility into success/failure states for debugging
 */

class SimulationLogger {
    constructor() {
        this.logs = new Map(); // simulationId -> array of log entries
        this.isEnabled = true;
        this.logToConsole = true;
        this.logToUI = true;
        
        // Define all simulation stages
        this.STAGES = {
            FRONTEND_INIT: 'frontend_initialization',
            REQUEST_VALIDATION: 'request_validation', 
            API_CALL: 'api_call',
            BACKEND_RECEIVED: 'backend_received',
            BACKGROUND_TASK: 'background_task_queued',
            ENGINE_INIT: 'engine_initialization',
            EXCEL_PARSING: 'excel_file_parsing',
            DEPENDENCY_ANALYSIS: 'dependency_analysis',
            BATCH_PROCESSING: 'batch_processing',
            FORMULA_EVALUATION: 'formula_evaluation',
            RESULTS_CALCULATION: 'results_calculation',
            PROGRESS_TRACKING: 'progress_tracking',
            COMPLETION: 'simulation_completion',
            ERROR_HANDLING: 'error_handling'
        };
        
        // Stage descriptions for UI display
        this.STAGE_DESCRIPTIONS = {
            [this.STAGES.FRONTEND_INIT]: 'Frontend Initialization',
            [this.STAGES.REQUEST_VALIDATION]: 'Request Validation',
            [this.STAGES.API_CALL]: 'API Call to Backend',
            [this.STAGES.BACKEND_RECEIVED]: 'Backend Processing',
            [this.STAGES.BACKGROUND_TASK]: 'Background Task Queued',
            [this.STAGES.ENGINE_INIT]: 'Engine Initialization',
            [this.STAGES.EXCEL_PARSING]: 'Excel File Parsing',
            [this.STAGES.DEPENDENCY_ANALYSIS]: 'Dependency Analysis',
            [this.STAGES.BATCH_PROCESSING]: 'Batch Processing',
            [this.STAGES.FORMULA_EVALUATION]: 'Formula Evaluation',
            [this.STAGES.RESULTS_CALCULATION]: 'Results Calculation',
            [this.STAGES.PROGRESS_TRACKING]: 'Progress Tracking',
            [this.STAGES.COMPLETION]: 'Simulation Completion',
            [this.STAGES.ERROR_HANDLING]: 'Error Handling'
        };
        
        // Initialize callbacks for UI updates
        this.onLogUpdate = null;
    }
    
    /**
     * Initialize logging for a new simulation
     */
    initializeSimulation(simulationId, config = {}) {
        if (!this.isEnabled) return;
        
        const initialLog = {
            simulationId,
            startTime: new Date().toISOString(),
            config: this._sanitizeConfig(config),
            stages: [],
            currentStage: null,
            status: 'started',
            errors: []
        };
        
        this.logs.set(simulationId, initialLog);
        
        this.logStage(simulationId, this.STAGES.FRONTEND_INIT, 'success', {
            message: 'Simulation logging initialized',
            variables: config.variables?.length || 0,
            targetCells: config.resultCells?.length || 0,
            iterations: config.iterations || 0,
            engine: config.engine_type || 'unknown'
        });
        
        this._notifyUpdate(simulationId);
    }
    
    /**
     * Log a specific stage with status and details
     */
    logStage(simulationId, stage, status, details = {}) {
        if (!this.isEnabled) return;
        
        const simulationLog = this.logs.get(simulationId);
        if (!simulationLog) {
            console.warn(`[SimulationLogger] No log found for simulation ${simulationId}`);
            return;
        }
        
        const timestamp = new Date().toISOString();
        const stageLog = {
            stage,
            status, // 'success', 'failure', 'in_progress', 'warning'
            timestamp,
            duration: this._calculateDuration(simulationLog.startTime, timestamp),
            description: this.STAGE_DESCRIPTIONS[stage] || stage,
            details: details || {},
            message: details.message || ''
        };
        
        // Update current stage
        simulationLog.currentStage = stage;
        simulationLog.stages.push(stageLog);
        
        // Track errors
        if (status === 'failure' || status === 'error') {
            simulationLog.errors.push({
                stage,
                timestamp,
                error: details.error || details.message || 'Unknown error',
                details
            });
            simulationLog.status = 'failed';
        } else if (status === 'success' && stage === this.STAGES.COMPLETION) {
            simulationLog.status = 'completed';
        }
        
        // Console logging
        if (this.logToConsole) {
            const emoji = this._getStatusEmoji(status);
            const stageDesc = this.STAGE_DESCRIPTIONS[stage] || stage;
            const idDisplay = simulationId ? simulationId.substring(0, 8) + '...' : 'unknown-id';
            console.log(`${emoji} [SimulationLogger] ${idDisplay} | ${stageDesc} | ${status.toUpperCase()}`, details);
        }
        
        this._notifyUpdate(simulationId);
    }
    
    /**
     * Log progress updates
     */
    logProgress(simulationId, progressData) {
        if (!this.isEnabled) return;
        
        this.logStage(simulationId, this.STAGES.PROGRESS_TRACKING, 'in_progress', {
            message: `Progress: ${progressData.progress_percentage || 0}%`,
            progress: progressData.progress_percentage || 0,
            iteration: progressData.current_iteration || 0,
            totalIterations: progressData.total_iterations || 0,
            stage: progressData.stage || 'unknown',
            backendStatus: progressData.status || 'unknown'
        });
    }
    
    /**
     * Log API responses
     */
    logApiResponse(simulationId, endpoint, status, responseData, error = null) {
        if (!this.isEnabled) return;
        
        const logStatus = status >= 200 && status < 300 ? 'success' : 'failure';
        
        this.logStage(simulationId, this.STAGES.API_CALL, logStatus, {
            message: `${endpoint} responded with ${status}`,
            endpoint,
            httpStatus: status,
            responseData: this._sanitizeResponse(responseData),
            error: error?.message || error,
            responseTime: responseData?.responseTime || 'unknown'
        });
    }
    
    /**
     * Log errors with full context
     */
    logError(simulationId, stage, error, context = {}) {
        if (!this.isEnabled) return;
        
        this.logStage(simulationId, stage, 'failure', {
            message: `Error in ${this.STAGE_DESCRIPTIONS[stage] || stage}`,
            error: error.message || error.toString(),
            stack: error.stack,
            context,
            timestamp: new Date().toISOString()
        });
    }
    
    /**
     * Get all logs for a simulation
     * If no logs found for the given ID, try to find logs for related simulations
     */
    getSimulationLogs(simulationId) {
        let logs = this.logs.get(simulationId);
        
        // If no logs found for this ID, try to find logs for related simulations
        if (!logs) {
            // Look for logs where the current simulationId might be the parent/batch ID
            for (const [logId, logData] of this.logs.entries()) {
                if (logData && logData.stages) {
                    // Check if any stage mentions this simulation ID as a batch ID
                    const hasRelatedBatch = logData.stages.some(stage => 
                        stage.details && (
                            stage.details.batchId === simulationId ||
                            stage.details.tempId === simulationId ||
                            stage.details.realId === simulationId
                        )
                    );
                    
                    if (hasRelatedBatch) {
                        logs = logData;
                        console.log(`[SimulationLogger] Found logs for related simulation ${logId} when looking for ${simulationId}`);
                        break;
                    }
                }
            }
        }
        
        // If still no logs found, create a minimal log structure indicating no process logs
        if (!logs) {
            console.log(`[SimulationLogger] No process logs found for simulation ${simulationId}`);
            // Return null to indicate no logs found (handled by the modal)
            return null;
        }
        
        return logs;
    }
    
    /**
     * Get logs for all simulations
     */
    getAllLogs() {
        return Array.from(this.logs.values());
    }
    
    /**
     * Get consolidated logs for a batch simulation
     * Combines logs from all related simulations in a batch
     */
    getConsolidatedBatchLogs(parentSimulationId) {
        const relatedLogs = [];
        
        // Find all logs that reference this parent simulation ID
        for (const [logId, logData] of this.logs.entries()) {
            if (logData && logData.stages) {
                const isRelated = logData.stages.some(stage => 
                    stage.details && (
                        stage.details.batchId === parentSimulationId ||
                        stage.details.tempId === parentSimulationId ||
                        logId === parentSimulationId
                    )
                );
                
                if (isRelated) {
                    relatedLogs.push(logData);
                }
            }
        }
        
        if (relatedLogs.length === 0) {
            return null;
        }
        
        // If only one related log, return it
        if (relatedLogs.length === 1) {
            return relatedLogs[0];
        }
        
        // Consolidate multiple logs into one view
        const consolidatedLog = {
            simulationId: parentSimulationId,
            startTime: Math.min(...relatedLogs.map(log => new Date(log.startTime).getTime())),
            config: relatedLogs[0].config || {},
            stages: [],
            currentStage: null,
            status: 'completed',
            errors: [],
            batchInfo: {
                totalSimulations: relatedLogs.length,
                simulationIds: relatedLogs.map(log => log.simulationId)
            }
        };
        
        // Combine all stages from all related logs
        relatedLogs.forEach(log => {
            if (log.stages) {
                consolidatedLog.stages.push(...log.stages);
            }
            if (log.errors) {
                consolidatedLog.errors.push(...log.errors);
            }
        });
        
        // Sort stages by timestamp
        consolidatedLog.stages.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        
        // Update status based on errors
        if (consolidatedLog.errors.length > 0) {
            consolidatedLog.status = 'failed';
        }
        
        // Set current stage to the latest one
        if (consolidatedLog.stages.length > 0) {
            consolidatedLog.currentStage = consolidatedLog.stages[consolidatedLog.stages.length - 1].stage;
        }
        
        return consolidatedLog;
    }
    
    /**
     * Create synthetic process logs from backend progress data
     * Used when frontend process logs are not available
     */
    async createSyntheticLogsFromBackend(simulationId) {
        try {
            // Try to fetch progress data from backend
            const response = await fetch(`/api/simulations/${simulationId}/status`);
            if (!response.ok) {
                return null;
            }
            
            const progressData = await response.json();
            
            if (!progressData || !progressData.simulation_id) {
                return null;
            }
            
            // Create synthetic log structure
            const syntheticLog = {
                simulationId: simulationId,
                startTime: progressData.start_time || new Date().toISOString(),
                config: {
                    variables: Object.keys(progressData.variables || {}).length,
                    targetCellCount: Object.keys(progressData.variables || {}).length,
                    iterations: progressData.total_iterations || 0,
                    engineType: progressData.engineInfo?.engine_type || 'unknown'
                },
                stages: [],
                currentStage: progressData.stage || 'completed',
                status: progressData.status || 'completed',
                errors: [],
                synthetic: true // Mark as synthetic
            };
            
            // Create synthetic stages based on progress data
            const phases = progressData.phases || {};
            Object.keys(phases).forEach((phaseKey, index) => {
                const phase = phases[phaseKey];
                syntheticLog.stages.push({
                    stage: this.STAGES.EXCEL_PARSING + '_' + phaseKey,
                    status: phase.completed ? 'success' : 'in_progress',
                    timestamp: new Date().toISOString(),
                    duration: '0s',
                    description: phase.stage || `Phase ${index + 1}`,
                    details: {
                        message: `${phase.stage}: ${phase.progress}% complete`,
                        progress: phase.progress,
                        completed: phase.completed,
                        synthetic: true
                    }
                });
            });
            
            // Add final completion stage
            syntheticLog.stages.push({
                stage: this.STAGES.COMPLETION,
                status: 'success',
                timestamp: new Date().toISOString(),
                duration: '0s',
                description: 'Simulation Completion',
                details: {
                    message: 'Simulation completed successfully',
                    progress: 100,
                    synthetic: true
                }
            });
            
            console.log(`[SimulationLogger] Created synthetic logs for ${simulationId}`);
            
            // Store the synthetic logs for future use
            this.logs.set(simulationId, syntheticLog);
            
            return syntheticLog;
            
        } catch (error) {
            console.error(`[SimulationLogger] Failed to create synthetic logs for ${simulationId}:`, error);
            return null;
        }
    }
    
    /**
     * Clear logs for a specific simulation
     */
    clearSimulationLogs(simulationId) {
        this.logs.delete(simulationId);
        this._notifyUpdate(simulationId);
    }
    
    /**
     * Clear all logs
     */
    clearAllLogs() {
        this.logs.clear();
        if (this.onLogUpdate) {
            this.onLogUpdate(null);
        }
    }
    
    /**
     * Export logs as JSON for debugging
     */
    exportLogs(simulationId = null) {
        if (simulationId) {
            return JSON.stringify(this.getSimulationLogs(simulationId), null, 2);
        }
        return JSON.stringify(this.getAllLogs(), null, 2);
    }
    
    /**
     * Set callback for UI updates
     */
    setUpdateCallback(callback) {
        this.onLogUpdate = callback;
    }
    
    /**
     * Enable/disable logging
     */
    setEnabled(enabled) {
        this.isEnabled = enabled;
    }
    
    /**
     * Configure logging options
     */
    configure(options = {}) {
        this.logToConsole = options.logToConsole !== undefined ? options.logToConsole : this.logToConsole;
        this.logToUI = options.logToUI !== undefined ? options.logToUI : this.logToUI;
        this.isEnabled = options.enabled !== undefined ? options.enabled : this.isEnabled;
    }
    
    // Private helper methods
    
    _sanitizeConfig(config) {
        return {
            fileId: config.fileId,
            variableCount: config.variables?.length || 0,
            targetCellCount: config.resultCells?.length || 0,
            iterations: config.iterations,
            engineType: config.engine_type,
            batchId: config.batch_id
        };
    }
    
    _sanitizeResponse(response) {
        if (!response) return null;
        
        return {
            simulation_id: response.simulation_id,
            status: response.status,
            message: response.message,
            batch_simulation_ids: response.batch_simulation_ids?.length || 0,
            created_at: response.created_at
        };
    }
    
    _calculateDuration(startTime, endTime) {
        try {
            const start = new Date(startTime);
            const end = new Date(endTime);
            return `${(end - start) / 1000}s`;
        } catch (e) {
            return 'unknown';
        }
    }
    
    _getStatusEmoji(status) {
        const emojis = {
            'success': '✅',
            'failure': '❌',
            'error': '💥',
            'in_progress': '⏳',
            'warning': '⚠️'
        };
        return emojis[status] || '📝';
    }
    
    _notifyUpdate(simulationId) {
        if (this.onLogUpdate && this.logToUI) {
            const logs = this.getSimulationLogs(simulationId);
            this.onLogUpdate(simulationId, logs);
        }
    }
}

// Export singleton instance
const simulationLogger = new SimulationLogger();

export default simulationLogger;
export { SimulationLogger }; 