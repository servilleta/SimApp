import React, { useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import WebhookExamples from '../components/webhook/WebhookExamples';
import '../styles/colors.css';
import '../styles/WebhookBraun.css';

const WebhookManagementPage = () => {
  const { getAccessTokenSilently, isAuthenticated } = useAuth0();
  
  // State management
  const [webhooks, setWebhooks] = useState([]);
  const [eventTypes, setEventTypes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedWebhook, setSelectedWebhook] = useState(null);
  const [deliveries, setDeliveries] = useState([]);
  const [loadingDeliveries, setLoadingDeliveries] = useState(false);
  const [activeTab, setActiveTab] = useState('manage');

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    url: '',
    events: [],
    secret: '',
    enabled: true
  });

  // Fetch webhook data on component mount
  useEffect(() => {
    if (isAuthenticated) {
      loadWebhooks();
      loadEventTypes();
    }
  }, [isAuthenticated]);

  const getAuthHeaders = async () => {
    const token = await getAccessTokenSilently();
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  };

  const loadWebhooks = async () => {
    try {
      setLoading(true);
      const headers = await getAuthHeaders();
      const response = await fetch('/api/webhooks/', { headers });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setWebhooks(data);
    } catch (err) {
      console.error('Failed to load webhooks:', err);
      setError('Failed to load webhooks: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadEventTypes = async () => {
    try {
      const response = await fetch('/api/webhooks/events/types');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setEventTypes(data.events || []);
    } catch (err) {
      console.error('Failed to load event types:', err);
    }
  };

  const loadDeliveries = async (webhookId) => {
    try {
      setLoadingDeliveries(true);
      const headers = await getAuthHeaders();
      const response = await fetch(`/api/webhooks/${webhookId}/deliveries`, { headers });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setDeliveries(data);
    } catch (err) {
      console.error('Failed to load deliveries:', err);
      setError('Failed to load delivery history: ' + err.message);
    } finally {
      setLoadingDeliveries(false);
    }
  };

  const handleCreateWebhook = async (e) => {
    e.preventDefault();
    
    if (!formData.name || !formData.url || formData.events.length === 0) {
      setError('Please fill in all required fields');
      return;
    }

    try {
      const headers = await getAuthHeaders();
      const response = await fetch('/api/webhooks/', {
        method: 'POST',
        headers,
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const newWebhook = await response.json();
      setWebhooks([...webhooks, newWebhook]);
      setShowCreateForm(false);
      setFormData({ name: '', url: '', events: [], secret: '', enabled: true });
      setError(null);
    } catch (err) {
      console.error('Failed to create webhook:', err);
      setError('Failed to create webhook: ' + err.message);
    }
  };

  const handleUpdateWebhook = async (webhookId, updates) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`/api/webhooks/${webhookId}`, {
        method: 'PUT',
        headers,
        body: JSON.stringify(updates)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const updatedWebhook = await response.json();
      setWebhooks(webhooks.map(w => w.id === webhookId ? updatedWebhook : w));
      setError(null);
    } catch (err) {
      console.error('Failed to update webhook:', err);
      setError('Failed to update webhook: ' + err.message);
    }
  };

  const handleDeleteWebhook = async (webhookId) => {
    if (!window.confirm('Are you sure you want to delete this webhook?')) {
      return;
    }

    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`/api/webhooks/${webhookId}`, {
        method: 'DELETE',
        headers
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      setWebhooks(webhooks.filter(w => w.id !== webhookId));
      if (selectedWebhook?.id === webhookId) {
        setSelectedWebhook(null);
        setDeliveries([]);
      }
      setError(null);
    } catch (err) {
      console.error('Failed to delete webhook:', err);
      setError('Failed to delete webhook: ' + err.message);
    }
  };

  const handleTestWebhook = async (webhookId) => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`/api/webhooks/${webhookId}/test`, {
        method: 'POST',
        headers
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        alert('✅ Test webhook sent successfully!');
        // Reload deliveries to show the test
        if (selectedWebhook?.id === webhookId) {
          loadDeliveries(webhookId);
        }
      } else {
        alert('❌ Test webhook failed. Check your endpoint.');
      }
    } catch (err) {
      console.error('Failed to test webhook:', err);
      alert('Failed to test webhook: ' + err.message);
    }
  };

  const handleEventToggle = (eventType) => {
    const newEvents = formData.events.includes(eventType)
      ? formData.events.filter(e => e !== eventType)
      : [...formData.events, eventType];
    
    setFormData({ ...formData, events: newEvents });
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };


  if (loading) {
    return (
      <div className="page-container">
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem' }}>
          <div className="spinner" style={{ marginBottom: '1rem' }}></div>
          <p style={{ color: 'var(--color-text-secondary)' }}>Loading webhooks...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Webhook Management</h1>
        <p className="page-subtitle">Configure webhooks to receive real-time notifications about your simulations</p>
        
        <div className="braun-tabs">
          <button 
            className={`tab-button ${activeTab === 'manage' ? 'active' : ''}`}
            onClick={() => setActiveTab('manage')}
          >
            Manage Webhooks
          </button>
          <button 
            className={`tab-button ${activeTab === 'examples' ? 'active' : ''}`}
            onClick={() => setActiveTab('examples')}
          >
            Integration Guide
          </button>
        </div>
        
        {error && (
          <div className="alert-braun alert-braun-error">
            {error}
            <button onClick={() => setError(null)} className="alert-close">×</button>
          </div>
        )}
        
        {activeTab === 'manage' && (
          <button 
            className="btn-braun-primary" 
            onClick={() => setShowCreateForm(true)}
            disabled={showCreateForm}
          >
            Add New Webhook
          </button>
        )}
      </div>

      {/* Tab Content */}
      {activeTab === 'examples' ? (
        <WebhookExamples />
      ) : (
        <div className="webhook-management-content">

      {/* Create/Edit Form */}
      {showCreateForm && (
        <div className="card-braun" style={{ marginBottom: '2rem' }}>
          <form onSubmit={handleCreateWebhook}>
            <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1.5rem' }}>Create New Webhook</h3>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <label htmlFor="name" className="label-braun">Name *</label>
              <input
                type="text"
                id="name"
                className="input-braun"
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                placeholder="e.g., Production Notifications"
                required
              />
            </div>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <label htmlFor="url" className="label-braun">Webhook URL *</label>
              <input
                type="url"
                id="url"
                className="input-braun"
                value={formData.url}
                onChange={(e) => setFormData({...formData, url: e.target.value})}
                placeholder="https://api.yourcompany.com/webhooks/simulations"
                required
              />
            </div>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <label htmlFor="secret" className="label-braun">Secret (Optional)</label>
              <input
                type="password"
                id="secret"
                className="input-braun"
                value={formData.secret}
                onChange={(e) => setFormData({...formData, secret: e.target.value})}
                placeholder="Leave empty to use default secret"
              />
              <small style={{ color: 'var(--color-text-tertiary)', fontSize: '0.875rem', marginTop: '0.25rem', display: 'block' }}>Used for HMAC signature verification</small>
            </div>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <label className="label-braun">Event Types *</label>
              <div style={{ display: 'grid', gap: '0.75rem', marginTop: '0.5rem' }}>
                {eventTypes.map(event => (
                  <div key={event.type} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                    <input
                      type="checkbox"
                      id={event.type}
                      checked={formData.events.includes(event.type)}
                      onChange={() => handleEventToggle(event.type)}
                      style={{ marginTop: '0.25rem' }}
                    />
                    <label htmlFor={event.type} style={{ flex: 1, cursor: 'pointer' }}>
                      <div style={{ color: 'var(--color-charcoal)', fontWeight: 500, fontSize: '0.9rem' }}>{event.type}</div>
                      <div style={{ color: 'var(--color-text-tertiary)', fontSize: '0.8rem', marginTop: '0.25rem' }}>{event.description}</div>
                    </label>
                  </div>
                ))}
              </div>
            </div>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontWeight: 500 }}>
                <input
                  type="checkbox"
                  checked={formData.enabled}
                  onChange={(e) => setFormData({...formData, enabled: e.target.checked})}
                />
                Enable webhook immediately
              </label>
            </div>
            
            <div style={{ display: 'flex', gap: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--color-border-light)' }}>
              <button type="submit" className="btn-braun-primary">
                Create Webhook
              </button>
              <button 
                type="button" 
                className="btn-braun-secondary"
                onClick={() => {
                  setShowCreateForm(false);
                  setFormData({ name: '', url: '', events: [], secret: '', enabled: true });
                }}
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Webhooks List */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', minHeight: '500px' }}>
        <div>
          <h2 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem', fontSize: '1.5rem' }}>Your Webhooks ({webhooks.length})</h2>
          
          {webhooks.length === 0 ? (
            <div className="card-braun" style={{ textAlign: 'center', padding: '3rem 2rem', border: '2px dashed var(--color-border-light)' }}>
              <h3 style={{ color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>No webhooks configured</h3>
              <p style={{ color: 'var(--color-text-tertiary)' }}>Create your first webhook to start receiving simulation notifications</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {webhooks.map(webhook => (
                <div 
                  key={webhook.id} 
                  className={`card-braun hover-lift ${selectedWebhook?.id === webhook.id ? 'selected' : ''}`}
                  style={{ 
                    cursor: 'pointer', 
                    borderColor: selectedWebhook?.id === webhook.id ? 'var(--color-braun-orange)' : 'var(--color-border-light)',
                    backgroundColor: selectedWebhook?.id === webhook.id ? 'var(--color-warm-white)' : 'var(--color-white)'
                  }}
                  onClick={() => {
                    setSelectedWebhook(webhook);
                    loadDeliveries(webhook.id);
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div 
                        style={{ 
                          width: '8px', 
                          height: '8px', 
                          borderRadius: '50%', 
                          backgroundColor: webhook.enabled ? 'var(--color-success)' : 'var(--color-medium-grey)' 
                        }}
                      />
                      <h3 style={{ margin: 0, color: 'var(--color-charcoal)', fontSize: '1.1rem' }}>{webhook.name}</h3>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '0.25rem' }}>
                      <button
                        className="action-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleTestWebhook(webhook.id);
                        }}
                        title="Test webhook"
                      >
                        Test
                      </button>
                      
                      <button
                        className="action-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleUpdateWebhook(webhook.id, { enabled: !webhook.enabled });
                        }}
                        title={webhook.enabled ? 'Disable' : 'Enable'}
                      >
                        {webhook.enabled ? 'Disable' : 'Enable'}
                      </button>
                      
                      <button
                        className="action-button action-button-delete"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteWebhook(webhook.id);
                        }}
                        title="Delete webhook"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
                      <strong style={{ color: 'var(--color-charcoal)' }}>URL:</strong> {webhook.url}
                    </div>
                    
                    <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
                      <strong style={{ color: 'var(--color-charcoal)' }}>Events:</strong> {webhook.events.join(', ')}
                    </div>
                    
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', fontSize: '0.8rem' }}>
                      <span style={{ color: 'var(--color-text-tertiary)' }}>
                        {webhook.total_deliveries || 0} deliveries
                      </span>
                      {webhook.failed_deliveries > 0 && (
                        <span style={{ color: 'var(--color-error)' }}>
                          {webhook.failed_deliveries} failed
                        </span>
                      )}
                      {webhook.last_delivery_at && (
                        <span style={{ color: 'var(--color-text-tertiary)' }}>
                          Last: {formatTimestamp(webhook.last_delivery_at)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Delivery History */}
        {selectedWebhook && (
          <div className="card-braun">
            <h2 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem', fontSize: '1.5rem' }}>
              Delivery History - {selectedWebhook.name}
            </h2>
            
            {loadingDeliveries ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '1rem', color: 'var(--color-text-tertiary)' }}>
                <div className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }}></div>
                <span>Loading delivery history...</span>
              </div>
            ) : deliveries.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--color-text-tertiary)' }}>
                <p>No deliveries yet. Test your webhook to see delivery attempts here.</p>
              </div>
            ) : (
              <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
                <table className="table-braun">
                  <thead>
                    <tr>
                      <th>Status</th>
                      <th>Event</th>
                      <th>Simulation ID</th>
                      <th>Attempt</th>
                      <th>Response</th>
                      <th>Time</th>
                      <th>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {deliveries.map(delivery => (
                      <tr key={delivery.id}>
                        <td>
                          <span className={`badge-braun ${delivery.status === 'delivered' ? 'badge-braun-primary' : 'badge-braun-secondary'}`}>
                            {delivery.status}
                          </span>
                        </td>
                        <td style={{ fontFamily: 'Monaco, Menlo, monospace', fontSize: '0.8rem', color: 'var(--color-info)' }}>
                          {delivery.event_type}
                        </td>
                        <td style={{ fontFamily: 'Monaco, Menlo, monospace', fontSize: '0.8rem', color: 'var(--color-success)' }}>
                          {delivery.simulation_id}
                        </td>
                        <td>#{delivery.attempt}</td>
                        <td>
                          {delivery.response_status ? (
                            <span 
                              style={{ 
                                padding: '0.125rem 0.375rem', 
                                borderRadius: '4px', 
                                fontFamily: 'Monaco, Menlo, monospace', 
                                fontSize: '0.8rem', 
                                fontWeight: 600,
                                backgroundColor: delivery.response_status >= 200 && delivery.response_status < 300 ? 'var(--color-success-bg)' : 'var(--color-error-bg)',
                                color: delivery.response_status >= 200 && delivery.response_status < 300 ? 'var(--color-success)' : 'var(--color-error)'
                              }}
                            >
                              {delivery.response_status}
                            </span>
                          ) : (
                            <span style={{ color: 'var(--color-text-tertiary)' }}>-</span>
                          )}
                          {delivery.response_time_ms && (
                            <span style={{ color: 'var(--color-text-tertiary)', fontSize: '0.75rem', marginLeft: '0.25rem' }}>
                              ({delivery.response_time_ms}ms)
                            </span>
                          )}
                        </td>
                        <td style={{ fontSize: '0.8rem', color: 'var(--color-text-tertiary)', whiteSpace: 'nowrap' }}>
                          {delivery.delivered_at ? formatTimestamp(delivery.delivered_at) : '-'}
                        </td>
                        <td style={{ fontSize: '0.8rem', color: 'var(--color-text-tertiary)', whiteSpace: 'nowrap' }}>
                          {formatTimestamp(delivery.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
        </div>
      )}
    </div>
  );
};

export default WebhookManagementPage;
