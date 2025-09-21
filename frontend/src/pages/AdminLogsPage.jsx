import React, { useEffect, useState, useCallback } from 'react';
import api from '../services/api';
import { format } from 'date-fns';
import { Trash2, StopCircle, Eraser, RefreshCw, AlertTriangle, Info } from 'lucide-react';

// The API instance already has the base URL
const HISTORY_API_URL = '/simulations/history';
const ACTIVE_API_URL = '/simulations/active';

const AdminLogsPage = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchLogs = useCallback(async () => {
    try {
      const res = await api.get('/simulations/history');
      setLogs(res.data || []);
      setError('');
    } catch (err) {
      console.error('AdminLogsPage: Fetch logs error', err);
      setError(err.response?.data?.detail || err.message || 'Failed to fetch logs');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLogs();
    const id = setInterval(fetchLogs, 8000); // refresh periodically
    return () => clearInterval(id);
  }, [fetchLogs]);

  // Action handlers
  const handleAction = async (action, simId, confirmMessage) => {
    if (!window.confirm(confirmMessage)) return;
    try {
      await action(simId);
      fetchLogs(); // Refresh logs after action
    } catch (err) {
      alert(err.response?.data?.detail || err.message || 'Action failed');
    }
  };

  const stopSimulation = (simId) => handleAction((id) => api.post(`/simulations/${id}/cancel`), simId, `Stop simulation ${simId}?`);
  
  const deleteSimulation = (log) => {
    const variableCount = log.simulation_count || 1;
    const isMultiVariable = variableCount > 1;
    const confirmMessage = isMultiVariable 
      ? `Delete entire simulation job "${log.file_name || 'Unknown'}" containing ${variableCount} variables?\n\nThis will permanently delete:\n• All ${variableCount} target variable simulations\n• All associated results and cache files\n\nThis action cannot be undone.`
      : `Delete simulation "${log.file_name || 'Unknown'}"?\n\nThis will permanently delete the simulation and all associated files.\n\nThis action cannot be undone.`;
    
    return handleAction((id) => api.delete(`/simulations/${id}`), log.simulation_id, confirmMessage);
  };
  
  const cleanCache = (simId) => handleAction((id) => api.post(`/simulations/${id}/clean-cache`), simId, `Clean cache for simulation ${simId}?`);

  const ActionButton = ({ icon: Icon, onClick, disabled, tooltip, variant = 'default' }) => {
    const [isHovered, setIsHovered] = useState(false);
    
    let buttonClass = 'action-button';
    if (variant === 'delete') buttonClass += ' action-button-delete';
    if (variant === 'stop') buttonClass += ' action-button-stop';
    if (disabled) buttonClass += ' action-button-disabled';

    return (
      <button 
        className={buttonClass}
        onClick={onClick} 
        disabled={disabled} 
        title={tooltip}
        onMouseEnter={() => !disabled && setIsHovered(true)}
        onMouseLeave={() => !disabled && setIsHovered(false)}
        style={{
          background: 'transparent',
          border: 'none',
          padding: '8px',
          borderRadius: '50%',
          cursor: disabled ? 'not-allowed' : 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.2s ease',
          margin: '0 4px',
          color: disabled ? 'var(--color-medium-grey)' : 
                variant === 'delete' ? 'var(--color-error)' :
                variant === 'stop' ? 'var(--color-warning)' : 'var(--color-medium-grey)',
          opacity: disabled ? 0.4 : 1,
          transform: isHovered && !disabled ? 'translateY(-1px)' : 'none',
          backgroundColor: isHovered && !disabled ? 'var(--color-warm-white)' : 'transparent'
        }}
      >
        <Icon size={18} />
      </button>
    );
  }

  const renderStatus = (status) => {
    status = status?.toLowerCase() || 'unknown';
    let color = 'var(--color-medium-grey)';
    if (status === 'completed') color = 'var(--color-success)';
    if (status === 'running' || status === 'pending' || status === 'streaming') color = 'var(--color-braun-orange)';
    if (status === 'failed') color = 'var(--color-error)';
    
    return (
      <span style={{ color, fontWeight: 600, textTransform: 'capitalize' }}>
        {status}
      </span>
    );
  };
  
  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Simulation Logs</h1>
        <p className="page-subtitle">
          Review historical simulation jobs, manage cache and clean up resources. 
          Each job may contain multiple target variables that were simulated together.
        </p>
      </div>
      
      <div className="card" style={{ padding: '0', overflowX: 'auto' }}>
        {loading ? (
          <div style={{ padding: '2rem', textAlign: 'center' }}>
            <p style={{ color: 'var(--color-medium-grey)' }}>Loading logs...</p>
          </div>
        ) : error ? (
          <div style={{ 
            padding: '2rem', 
            color: 'var(--color-error)', 
            display: 'flex', 
            alignItems: 'center' 
          }}>
            <AlertTriangle style={{ marginRight: '8px' }} />
            Error: {error}
          </div>
        ) : logs.length === 0 ? (
          <div style={{ 
            padding: '2rem', 
            textAlign: 'center', 
            color: 'var(--color-medium-grey)', 
            display: 'flex', 
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Info style={{ marginRight: '8px' }} />
            No historical simulations found.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead style={{ borderBottom: '2px solid var(--color-border-light)' }}>
              <tr>
                <th style={{ 
                  padding: '1rem', 
                  color: 'var(--color-charcoal)', 
                  fontWeight: '600',
                  backgroundColor: 'var(--color-warm-white)'
                }}>User</th>
                <th style={{ 
                  padding: '1rem', 
                  color: 'var(--color-charcoal)', 
                  fontWeight: '600',
                  backgroundColor: 'var(--color-warm-white)'
                }}>Simulation ID</th>
                <th style={{ 
                  padding: '1rem', 
                  color: 'var(--color-charcoal)', 
                  fontWeight: '600',
                  backgroundColor: 'var(--color-warm-white)'
                }}>File Name</th>
                <th style={{ 
                  padding: '1rem', 
                  color: 'var(--color-charcoal)', 
                  fontWeight: '600',
                  backgroundColor: 'var(--color-warm-white)'
                }}>Target Variables</th>
                <th style={{ 
                  padding: '1rem', 
                  color: 'var(--color-charcoal)', 
                  fontWeight: '600',
                  backgroundColor: 'var(--color-warm-white)'
                }}>Engine</th>
                <th style={{ 
                  padding: '1rem', 
                  color: 'var(--color-charcoal)', 
                  fontWeight: '600',
                  backgroundColor: 'var(--color-warm-white)'
                }}>Date Created</th>
                <th style={{ 
                  padding: '1rem', 
                  color: 'var(--color-charcoal)', 
                  fontWeight: '600',
                  backgroundColor: 'var(--color-warm-white)'
                }}>Status</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'center',
                  color: 'var(--color-charcoal)', 
                  fontWeight: '600',
                  backgroundColor: 'var(--color-warm-white)'
                }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={log.simulation_id} style={{ borderBottom: '1px solid var(--color-border-light)' }}>
                  <td style={{ padding: '1rem', fontWeight: 500, color: 'var(--color-charcoal)' }}>
                    {log.user || 'unknown'}
                  </td>
                  <td style={{ 
                    padding: '1rem', 
                    color: 'var(--color-medium-grey)', 
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    maxWidth: '200px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    <span title={log.simulation_id}>
                      {log.simulation_id ? log.simulation_id.substring(0, 8) + '...' : 'N/A'}
                    </span>
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)' }}>
                    {log.file_name || 'Unknown'}
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)', fontSize: '0.9rem' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{
                          background: log.simulation_count > 1 ? 'var(--color-braun-orange)' : 'var(--color-light-grey)', 
                          color: log.simulation_count > 1 ? 'white' : 'var(--color-medium-grey)', 
                          padding: '2px 8px', 
                          borderRadius: '12px', 
                          fontSize: '0.8rem',
                          fontWeight: '600'
                        }}>
                          {log.simulation_count || 1} {log.simulation_count > 1 ? 'variables' : 'variable'}
                        </span>
                        {log.simulation_count > 1 && (
                          <span style={{
                            background: 'var(--color-warning)',
                            color: 'white',
                            padding: '2px 6px',
                            borderRadius: '8px',
                            fontSize: '0.7rem',
                            fontWeight: '500'
                          }}>
                            JOB
                          </span>
                        )}
                      </div>
                      <div style={{
                        fontSize: '0.8rem',
                        color: 'var(--color-medium-grey)',
                        marginTop: '2px',
                        lineHeight: '1.2'
                      }}>
                        {log.target_variables || 'Target Variable'}
                      </div>
                    </div>
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)' }}>
                    <span style={{
                      padding: '0.25rem 0.5rem',
                      backgroundColor: 'var(--color-light-grey)',
                      borderRadius: '4px',
                      fontSize: '0.8rem',
                      fontWeight: '500',
                      textTransform: 'lowercase'
                    }}>
                      {log.engine_type || 'power'}
                    </span>
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)', fontSize: '0.9rem' }}>
                    {log.created_at ? new Date(log.created_at).toLocaleString() : 'N/A'}
                  </td>
                  <td style={{ padding: '1rem' }}>
                    {renderStatus(log.status)}
                  </td>
                  <td style={{ padding: '1rem', textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '4px' }}>
                      <ActionButton 
                        icon={StopCircle} 
                        onClick={() => stopSimulation(log.simulation_id)}
                        disabled={!['running', 'pending'].includes(log.status?.toLowerCase())}
                        tooltip="Stop simulation"
                        variant="stop"
                      />
                      <ActionButton 
                        icon={Eraser} 
                        onClick={() => cleanCache(log.simulation_id)}
                        disabled={false}
                        tooltip="Clean cache"
                      />
                      <ActionButton 
                        icon={Trash2} 
                        onClick={() => deleteSimulation(log)}
                        disabled={false}
                        tooltip="Delete simulation"
                        variant="delete"
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default AdminLogsPage; 