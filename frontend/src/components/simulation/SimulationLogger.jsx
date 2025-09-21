import React, { useState, useEffect } from 'react';
import simulationLogger from '../../services/simulationLogger';
import './SimulationLogger.css';

const SimulationLogger = ({ simulationId, isVisible, onClose }) => {
    const [logs, setLogs] = useState(null);
    const [expandedStages, setExpandedStages] = useState(new Set());
    const [filter, setFilter] = useState('all'); // all, success, failure, in_progress
    const [autoRefresh, setAutoRefresh] = useState(true);

    useEffect(() => {
        if (!isVisible) return;

        // Set up log update callback
        const handleLogUpdate = (updatedSimulationId, updatedLogs) => {
            if (updatedSimulationId === simulationId) {
                setLogs(updatedLogs);
            }
        };

        simulationLogger.setUpdateCallback(handleLogUpdate);

        // Load initial logs - try consolidated batch logs first
        const loadLogs = async () => {
            let initialLogs = simulationLogger.getConsolidatedBatchLogs(simulationId);
            if (!initialLogs) {
                initialLogs = simulationLogger.getSimulationLogs(simulationId);
            }
            
            // If no logs found, try to create synthetic logs from backend
            if (!initialLogs) {
                console.log(`[SimulationLogger] No frontend logs found for ${simulationId}, attempting to create synthetic logs from backend`);
                initialLogs = await simulationLogger.createSyntheticLogsFromBackend(simulationId);
            }
            
            setLogs(initialLogs);
        };
        
        loadLogs();

        // Auto-refresh interval
        let refreshInterval;
        if (autoRefresh) {
            refreshInterval = setInterval(async () => {
                let currentLogs = simulationLogger.getConsolidatedBatchLogs(simulationId);
                if (!currentLogs) {
                    currentLogs = simulationLogger.getSimulationLogs(simulationId);
                }
                
                // If no logs found during refresh, try to create synthetic logs
                if (!currentLogs) {
                    currentLogs = await simulationLogger.createSyntheticLogsFromBackend(simulationId);
                }
                
                setLogs(currentLogs);
            }, 1000);
        }

        return () => {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        };
    }, [simulationId, isVisible, autoRefresh]);

    const toggleStageExpansion = (index) => {
        const newExpanded = new Set(expandedStages);
        if (newExpanded.has(index)) {
            newExpanded.delete(index);
        } else {
            newExpanded.add(index);
        }
        setExpandedStages(newExpanded);
    };

    const getStatusIcon = (status) => {
        const icons = {
            'success': '✅',
            'failure': '❌',
            'error': '💥',
            'in_progress': '⏳',
            'warning': '⚠️'
        };
        return icons[status] || '📝';
    };

    const getStatusColor = (status) => {
        const colors = {
            'success': '#28a745',
            'failure': '#dc3545',
            'error': '#dc3545',
            'in_progress': '#007bff',
            'warning': '#ffc107'
        };
        return colors[status] || '#6c757d';
    };

    const filteredStages = logs?.stages?.filter(stage => {
        if (filter === 'all') return true;
        return stage.status === filter;
    }) || [];

    const exportLogs = () => {
        const logsJson = simulationLogger.exportLogs(simulationId);
        const blob = new Blob([logsJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `simulation-logs-${simulationId.substring(0, 8)}-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    if (!isVisible) return null;

    return (
        <div className="simulation-logger-overlay">
            <div className="simulation-logger-modal">
                <div className="simulation-logger-header">
                    <h3>🚀 Simulation Process Log</h3>
                    <div className="simulation-logger-controls">
                        <label>
                            <input
                                type="checkbox"
                                checked={autoRefresh}
                                onChange={(e) => setAutoRefresh(e.target.checked)}
                            />
                            Auto-refresh
                        </label>
                        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
                            <option value="all">All Stages</option>
                            <option value="success">Success Only</option>
                            <option value="failure">Failures Only</option>
                            <option value="in_progress">In Progress</option>
                        </select>
                        <button onClick={exportLogs} className="export-btn">
                            📥 Export
                        </button>
                        <button onClick={onClose} className="close-btn">
                            ✕
                        </button>
                    </div>
                </div>

                <div className="simulation-logger-content">
                    {logs ? (
                        <>
                            {logs.synthetic && (
                                <div style={{
                                    background: 'var(--color-warning)',
                                    color: 'white',
                                    padding: '0.75rem',
                                    borderRadius: '8px',
                                    marginBottom: '1rem',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.5rem'
                                }}>
                                    <span>⚠️</span>
                                    <div>
                                        <strong>Synthetic Process Logs</strong>
                                        <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>
                                            These logs were reconstructed from backend progress data since real-time frontend logs are not available.
                                        </div>
                                    </div>
                                </div>
                            )}
                            
                            <div className="simulation-summary">
                                <div className="summary-item">
                                    <strong>Simulation ID:</strong> {logs.simulationId?.substring(0, 16)}...
                                </div>
                                <div className="summary-item">
                                    <strong>Status:</strong> 
                                    <span 
                                        className={`status-badge status-${logs.status}`}
                                        style={{ color: getStatusColor(logs.status) }}
                                    >
                                        {getStatusIcon(logs.status)} {logs.status}
                                    </span>
                                </div>
                                <div className="summary-item">
                                    <strong>Started:</strong> {new Date(logs.startTime).toLocaleString()}
                                </div>
                                <div className="summary-item">
                                    <strong>Current Stage:</strong> {logs.currentStage || 'Not started'}
                                </div>
                                <div className="summary-item">
                                    <strong>Total Stages:</strong> {logs.stages?.length || 0}
                                </div>
                                <div className="summary-item">
                                    <strong>Errors:</strong> 
                                    <span className={logs.errors?.length > 0 ? 'error-count' : 'no-errors'}>
                                        {logs.errors?.length || 0}
                                    </span>
                                </div>
                            </div>

                            <div className="config-summary">
                                <h4>Configuration</h4>
                                <div className="config-grid">
                                    <div>Variables: {logs.config?.variableCount || 0}</div>
                                    <div>Target Cells: {logs.config?.targetCellCount || 0}</div>
                                    <div>Iterations: {logs.config?.iterations || 0}</div>
                                    <div>Engine: {logs.config?.engineType || 'unknown'}</div>
                                </div>
                            </div>

                            {logs.batchInfo && (
                                <div className="batch-info">
                                    <h4>🎯 Batch Simulation Info</h4>
                                    <div style={{ 
                                        background: 'var(--color-light-grey)', 
                                        padding: '1rem', 
                                        borderRadius: '8px',
                                        marginBottom: '1rem'
                                    }}>
                                        <div style={{ marginBottom: '0.5rem' }}>
                                            <strong>Total Simulations:</strong> {logs.batchInfo.totalSimulations}
                                        </div>
                                        <div style={{ fontSize: '0.9rem', color: 'var(--color-medium-grey)' }}>
                                            <strong>Simulation IDs:</strong>
                                            <div style={{ marginTop: '0.25rem' }}>
                                                {logs.batchInfo.simulationIds.map((id, index) => (
                                                    <div key={id} style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                                                        {index + 1}. {id.substring(0, 16)}...
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div className="stages-list">
                                <h4>Process Stages ({filteredStages.length})</h4>
                                {filteredStages.map((stage, index) => (
                                    <div key={index} className="stage-item">
                                        <div 
                                            className="stage-header"
                                            onClick={() => toggleStageExpansion(index)}
                                            style={{ borderLeft: `4px solid ${getStatusColor(stage.status)}` }}
                                        >
                                            <div className="stage-title">
                                                <span className="stage-icon">
                                                    {getStatusIcon(stage.status)}
                                                </span>
                                                <span className="stage-name">
                                                    {stage.description}
                                                </span>
                                                <span className="stage-status">
                                                    {stage.status.toUpperCase()}
                                                </span>
                                            </div>
                                            <div className="stage-meta">
                                                <span className="stage-time">
                                                    {new Date(stage.timestamp).toLocaleTimeString()}
                                                </span>
                                                <span className="stage-duration">
                                                    {stage.duration}
                                                </span>
                                                <span className="expand-icon">
                                                    {expandedStages.has(index) ? '▼' : '▶'}
                                                </span>
                                            </div>
                                        </div>

                                        {expandedStages.has(index) && (
                                            <div className="stage-details">
                                                {stage.message && (
                                                    <div className="detail-item">
                                                        <strong>Message:</strong> {stage.message}
                                                    </div>
                                                )}
                                                
                                                {Object.entries(stage.details || {}).map(([key, value]) => (
                                                    <div key={key} className="detail-item">
                                                        <strong>{key}:</strong> 
                                                        <span className="detail-value">
                                                            {typeof value === 'object' 
                                                                ? JSON.stringify(value, null, 2)
                                                                : String(value)
                                                            }
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>

                            {logs.errors && logs.errors.length > 0 && (
                                <div className="errors-section">
                                    <h4>🚨 Errors ({logs.errors.length})</h4>
                                    {logs.errors.map((error, index) => (
                                        <div key={index} className="error-item">
                                            <div className="error-header">
                                                <strong>{error.stage}</strong>
                                                <span className="error-time">
                                                    {new Date(error.timestamp).toLocaleString()}
                                                </span>
                                            </div>
                                            <div className="error-message">
                                                {error.error}
                                            </div>
                                            {error.details && (
                                                <div className="error-details">
                                                    <pre>{JSON.stringify(error.details, null, 2)}</pre>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="no-logs">
                            <p>No logs available for this simulation.</p>
                            <p>Simulation ID: {simulationId}</p>
                            <p style={{ fontSize: '0.9rem', color: 'var(--color-medium-grey)', marginTop: '1rem' }}>
                                This could mean:
                                <br />• The simulation hasn't started yet
                                <br />• The simulation was run before process logging was enabled
                                <br />• The logs were cleared or the simulation ID is incorrect
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SimulationLogger; 