import React, { useState } from 'react';

const WebhooksPage = () => {
  const [webhooks, setWebhooks] = useState([
    {
      id: 'webhook_1',
      name: 'Production Notifications',
      url: 'https://api.yourcompany.com/webhooks/simulations',
      events: ['simulation.completed', 'simulation.failed'],
      status: 'active',
      lastDelivery: '2024-01-20 14:23:00',
      deliveryStatus: 'success'
    },
    {
      id: 'webhook_2',
      name: 'Development Testing',
      url: 'https://dev-api.yourcompany.com/webhooks/test',
      events: ['simulation.completed'],
      status: 'inactive',
      lastDelivery: '2024-01-18 09:15:00',
      deliveryStatus: 'failed'
    }
  ]);

  const pageStyle = {
    padding: '2rem',
    backgroundColor: 'var(--color-white)',
    minHeight: '100vh'
  };

  const headerStyle = {
    marginBottom: '2rem'
  };

  const titleStyle = {
    fontSize: '2.5rem',
    fontWeight: '600',
    color: 'var(--color-charcoal)',
    marginBottom: '0.5rem',
    letterSpacing: '-0.02em'
  };

  const subtitleStyle = {
    fontSize: '1.1rem',
    color: 'var(--color-medium-grey)',
    lineHeight: '1.5'
  };

  const cardStyle = {
    backgroundColor: 'var(--color-white)',
    borderRadius: '8px',
    padding: '1.5rem',
    marginBottom: '1rem',
    border: '1px solid var(--color-border-light)',
    boxShadow: 'var(--shadow-sm)',
    transition: 'all var(--transition-base)'
  };

  const buttonStyle = {
    padding: '0.75rem 1.5rem',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: 'var(--color-braun-orange)',
    color: 'white',
    fontSize: '0.875rem',
    fontWeight: '600',
    cursor: 'pointer',
    marginBottom: '1.5rem',
    transition: 'all var(--transition-base)'
  };

  const statusBadgeStyle = (status) => ({
    display: 'inline-block',
    padding: '0.25rem 0.5rem',
    borderRadius: '12px',
    fontSize: '0.75rem',
    fontWeight: '600',
    backgroundColor: status === 'active' ? 'var(--color-success-bg)' : status === 'inactive' ? 'var(--color-light-grey)' : 'var(--color-error-bg)',
    color: status === 'active' ? 'var(--color-success)' : status === 'inactive' ? 'var(--color-medium-grey)' : 'var(--color-error)',
    border: `1px solid ${status === 'active' ? 'var(--color-success)' : status === 'inactive' ? 'var(--color-medium-grey)' : 'var(--color-error)'}`
  });

  const deliveryBadgeStyle = (status) => ({
    display: 'inline-block',
    padding: '0.25rem 0.5rem',
    borderRadius: '12px',
    fontSize: '0.75rem',
    fontWeight: '600',
    backgroundColor: status === 'success' ? 'var(--color-success-bg)' : 'var(--color-error-bg)',
    color: status === 'success' ? 'var(--color-success)' : 'var(--color-error)',
    border: `1px solid ${status === 'success' ? 'var(--color-success)' : 'var(--color-error)'}`
  });

  const codeBlockStyle = {
    backgroundColor: 'var(--color-charcoal)',
    color: '#e5e7eb',
    padding: '1rem',
    borderRadius: '4px',
    fontSize: '0.875rem',
    fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
    overflow: 'auto',
    marginTop: '1rem',
    border: '1px solid var(--color-border-light)'
  };

  return (
    <div style={pageStyle}>
      <div style={headerStyle}>
        <h1 style={titleStyle}>📡 Webhooks</h1>
        <p style={subtitleStyle}>
          Configure webhooks to receive real-time notifications about your simulations
        </p>
      </div>

      <button 
        style={buttonStyle}
        onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--color-braun-orange-dark)'}
        onMouseLeave={(e) => e.target.style.backgroundColor = 'var(--color-braun-orange)'}
      >
        + Add New Webhook
      </button>

      <div>
        {webhooks.map((webhook) => (
          <div key={webhook.id} style={cardStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
              <div>
                <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '8px' }}>
                  {webhook.name}
                </h3>
                <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
                  <span style={statusBadgeStyle(webhook.status)}>
                    {webhook.status.toUpperCase()}
                  </span>
                  <span style={deliveryBadgeStyle(webhook.deliveryStatus)}>
                    Last: {webhook.deliveryStatus.toUpperCase()}
                  </span>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button style={{ 
                  padding: '6px 12px', 
                  borderRadius: '4px', 
                  border: `1px solid ${'var(--color-border-light)'}`,
                  backgroundColor: 'var(--color-warm-white)',
                  fontSize: '12px',
                  cursor: 'pointer'
                }}>
                  Test
                </button>
                <button style={{ 
                  padding: '6px 12px', 
                  borderRadius: '4px', 
                  border: `1px solid ${'var(--color-border-light)'}`,
                  backgroundColor: 'var(--color-warm-white)',
                  fontSize: '12px',
                  cursor: 'pointer'
                }}>
                  Edit
                </button>
                <button style={{ 
                  padding: '6px 12px', 
                  borderRadius: '4px', 
                  border: '1px solid #ef4444',
                  backgroundColor: '#fef2f2',
                  color: '#dc2626',
                  fontSize: '12px',
                  cursor: 'pointer'
                }}>
                  Delete
                </button>
              </div>
            </div>
            
            <div style={{ 
              backgroundColor: 'var(--color-warm-white)',
              padding: '12px',
              borderRadius: '6px',
              marginBottom: '16px',
              fontSize: '14px',
              fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace'
            }}>
              {webhook.url}
            </div>
            
            <div style={{ marginBottom: '16px' }}>
              <strong style={{ color: 'var(--color-charcoal)' }}>Events:</strong>
              <div style={{ marginTop: '8px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {webhook.events.map((event, index) => (
                  <span key={index} style={{
                    padding: '4px 8px',
                    backgroundColor: 'var(--color-warm-white)',
                    borderRadius: '4px',
                    fontSize: '12px',
                    border: `1px solid ${'var(--color-border-light)'}`
                  }}>
                    {event}
                  </span>
                ))}
              </div>
            </div>
            
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '16px',
              fontSize: '14px',
              color: 'var(--color-text-secondary)'
            }}>
              <div>
                <strong>Last Delivery:</strong> {webhook.lastDelivery}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div style={cardStyle}>
        <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '16px' }}>
          🔧 Webhook Events
        </h3>
        <div style={{ marginBottom: '16px' }}>
          <p style={{ color: 'var(--color-text-secondary)', marginBottom: '12px' }}>
            Available events you can subscribe to:
          </p>
          <ul style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
            <li><code>simulation.started</code> - When a simulation begins processing</li>
            <li><code>simulation.completed</code> - When a simulation finishes successfully</li>
            <li><code>simulation.failed</code> - When a simulation encounters an error</li>
            <li><code>simulation.progress</code> - Progress updates (every 25%)</li>
          </ul>
        </div>

        <h4 style={{ color: 'var(--color-charcoal)', marginBottom: '12px' }}>
          Example Payload
        </h4>
        <div style={codeBlockStyle}>
{`{
  "event": "simulation.completed",
  "timestamp": "2024-01-20T14:23:00Z",
  "data": {
    "simulation_id": "sim_789",
    "model_id": "model_456",
    "status": "completed",
    "results": {
      "mean": 1250.67,
      "std_dev": 234.89,
      "var_95": 890.23,
      "var_99": 645.12
    },
    "download_links": {
      "detailed_csv": "https://.../sim_789.csv",
      "summary_pdf": "https://.../sim_789.pdf"
    }
  }
}`}
        </div>
      </div>

      <div style={cardStyle}>
        <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '16px' }}>
          🛡️ Security & Verification
        </h3>
        <ul style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
          <li>All webhook requests include an <code>X-Signature</code> header for verification</li>
          <li>Webhooks timeout after 30 seconds</li>
          <li>Failed deliveries are retried up to 3 times</li>
          <li>Use HTTPS endpoints for production</li>
          <li>Return HTTP 200 status to acknowledge receipt</li>
        </ul>
      </div>
    </div>
  );
};

export default WebhooksPage;
