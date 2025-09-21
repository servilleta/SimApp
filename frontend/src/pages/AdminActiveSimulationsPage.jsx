import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import axios from 'axios';

const Loading = () => (
  <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>
    <p style={{ color: 'var(--color-medium-grey)' }}>Loading...</p>
  </div>
);

const AdminActiveSimulationsPage = () => {
  const token = useSelector(state => state.auth?.token);
  const [loading, setLoading] = useState(false);
  const [sims, setSims] = useState([]);
  const [error, setError] = useState(null);

  const fetchActive = async () => {
    if (!token) return;
    try {
      setLoading(true);
      const res = await axios.get('/api/simulations/active', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSims(res.data || []);
      setError(null);
    } catch (err) {
      console.error('AdminActiveSimulations: fetch error', err);
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchActive();
    const id = setInterval(fetchActive, 5000);
    return () => clearInterval(id);
  }, [token]);

  const cancelSim = async (simId) => {
    if (!window.confirm(`Cancel simulation ${simId}?`)) return;
    try {
      await axios.post(`/api/simulations/${simId}/cancel`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchActive();
    } catch (err) {
      alert('Cancel failed: ' + (err.response?.data?.detail || err.message));
    }
  };

  if (!token) {
    return (
      <div className="page-container">
        <div className="card error-card">
          <p>Please login as admin.</p>
        </div>
      </div>
    );
  }

  if (loading && sims.length === 0) return <Loading />;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Active Simulations</h1>
        <p className="page-subtitle">
          Monitor and manage currently running simulations across the platform
        </p>
      </div>
      
      {error && (
        <div className="card error-card" style={{ marginBottom: '1rem' }}>
          <p style={{ color: 'var(--color-error)' }}>{error}</p>
        </div>
      )}
      
      {sims.length === 0 ? (
        <div className="card" style={{ 
          padding: '3rem', 
          textAlign: 'center',
          backgroundColor: 'var(--color-warm-white)'
        }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚡</div>
          <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '0.5rem' }}>
            No Active Simulations
          </h3>
          <p style={{ color: 'var(--color-medium-grey)' }}>
            All simulations are currently completed or idle.
          </p>
        </div>
      ) : (
        <div className="card" style={{ padding: '1.5rem', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ 
                borderBottom: '2px solid var(--color-border-light)',
                backgroundColor: 'var(--color-warm-white)'
              }}>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>ID</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>User</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Status</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Started</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Message</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {sims.map(sim => (
                <tr key={sim.simulation_id} style={{ 
                  borderBottom: '1px solid var(--color-border-light)'
                }}>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)', fontFamily: 'monospace' }}>
                    {sim.simulation_id}
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--color-charcoal)', fontWeight: '500' }}>
                    {sim.user}
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.75rem',
                      borderRadius: '20px',
                      fontSize: '0.8rem',
                      fontWeight: '600',
                      backgroundColor: sim.status === 'running' ? 'var(--color-braun-orange)' : 'var(--color-warning)',
                      color: 'white',
                      textTransform: 'capitalize'
                    }}>
                      {sim.status}
                    </span>
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)', fontSize: '0.9rem' }}>
                    {new Date(sim.created_at).toLocaleString()}
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)', fontSize: '0.9rem' }}>
                    {sim.message || 'Processing...'}
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <button 
                      onClick={() => cancelSim(sim.simulation_id)}
                      style={{
                        fontSize: '0.8rem',
                        padding: '0.25rem 0.75rem',
                        backgroundColor: 'transparent',
                        border: '1px solid var(--color-error)',
                        color: 'var(--color-error)',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Cancel
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AdminActiveSimulationsPage; 