import React, { useState } from 'react';
import { getWebhookExamples, generateTestPayload, eventDescriptions } from '../../utils/webhookTester';

const WebhookExamples = () => {
  const [selectedLanguage, setSelectedLanguage] = useState('javascript');
  const [selectedEvent, setSelectedEvent] = useState('simulation.completed');

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('📋 Code copied to clipboard!');
    }).catch(() => {
      alert('❌ Failed to copy to clipboard');
    });
  };

  const payloadExample = generateTestPayload(selectedEvent);

  return (
    <div className="braun-container">
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h2 style={{ color: 'var(--color-charcoal)', fontSize: '2rem', marginBottom: '0.5rem' }}>Webhook Integration Examples</h2>
        <p style={{ color: 'var(--color-text-secondary)', fontSize: '1.1rem' }}>Learn how to receive and process webhook notifications in your application</p>
      </div>

      {/* Event Type Selector */}
      <div className="card-braun" style={{ marginBottom: '3rem' }}>
        <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1.5rem', fontSize: '1.5rem' }}>Event Types</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
          {Object.entries(eventDescriptions).map(([eventType, description]) => (
            <button
              key={eventType}
              className={selectedEvent === eventType ? 'btn-braun-primary' : 'btn-braun-secondary'}
              style={{ textAlign: 'left', padding: '1rem' }}
              onClick={() => setSelectedEvent(eventType)}
            >
              <div style={{ fontWeight: 500, fontSize: '0.9rem', marginBottom: '0.25rem' }}>{eventType}</div>
              <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>{description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Sample Payload */}
      <div className="card-braun" style={{ marginBottom: '3rem' }}>
        <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1.5rem', fontSize: '1.5rem' }}>Sample Payload - {selectedEvent}</h3>
        <div style={{ backgroundColor: '#0f172a', borderRadius: '8px', overflow: 'hidden' }}>
          <div style={{ backgroundColor: '#1e293b', padding: '0.75rem 1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #334155' }}>
            <span style={{ color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 600 }}>JSON Payload</span>
            <button 
              className="btn-braun-primary"
              style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
              onClick={() => copyToClipboard(JSON.stringify(payloadExample, null, 2))}
            >
              Copy
            </button>
          </div>
          <pre style={{ color: '#e2e8f0', padding: '1rem', margin: 0, overflowX: 'auto', fontSize: '0.85rem', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(payloadExample, null, 2)}
          </pre>
        </div>
      </div>

      {/* Server Code Examples */}
      <div className="card-braun" style={{ marginBottom: '3rem' }}>
        <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1.5rem', fontSize: '1.5rem' }}>Server Implementation</h3>
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
          <button
            className={selectedLanguage === 'javascript' ? 'btn-braun-primary' : 'btn-braun-secondary'}
            style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}
            onClick={() => setSelectedLanguage('javascript')}
          >
            JavaScript (Node.js)
          </button>
          <button
            className={selectedLanguage === 'python' ? 'btn-braun-primary' : 'btn-braun-secondary'}
            style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}
            onClick={() => setSelectedLanguage('python')}
          >
            Python (Flask)
          </button>
          <button
            className={selectedLanguage === 'curl' ? 'btn-braun-primary' : 'btn-braun-secondary'}
            style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}
            onClick={() => setSelectedLanguage('curl')}
          >
            cURL (Testing)
          </button>
        </div>
        
        <div className="code-block">
          <div className="code-header">
            <span>{selectedLanguage.charAt(0).toUpperCase() + selectedLanguage.slice(1)} Example</span>
            <button onClick={() => copyToClipboard(getWebhookExamples(selectedLanguage))}>
              📋 Copy
            </button>
          </div>
          <pre className="code-content">
            {getWebhookExamples(selectedLanguage)}
          </pre>
        </div>
      </div>

      {/* Security Information */}
      <div className="card-braun" style={{ marginBottom: '3rem' }}>
        <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1.5rem', fontSize: '1.5rem' }}>Security Headers</h3>
        <div className="security-info">
          <p>All webhook requests include these security headers:</p>
          <ul>
            <li><code>X-SimApp-Signature</code> - HMAC-SHA256 signature for payload verification</li>
            <li><code>X-SimApp-Event</code> - The event type being delivered</li>
            <li><code>X-SimApp-Delivery</code> - Unique delivery attempt ID</li>
            <li><code>X-SimApp-Timestamp</code> - ISO timestamp of the event</li>
            <li><code>User-Agent</code> - Always <code>SimApp-Webhook/1.0</code></li>
          </ul>
        </div>
      </div>

      {/* Best Practices */}
      <div className="example-section">
        <h3>✅ Best Practices</h3>
        <div className="best-practices">
          <div className="practice-item">
            <h4>🔐 Always Verify Signatures</h4>
            <p>Use the HMAC signature to ensure the webhook came from SimApp and hasn't been tampered with.</p>
          </div>
          
          <div className="practice-item">
            <h4>⚡ Respond Quickly</h4>
            <p>Return HTTP 200 within 30 seconds. Process heavy tasks asynchronously after responding.</p>
          </div>
          
          <div className="practice-item">
            <h4>🔄 Handle Retries</h4>
            <p>We retry failed deliveries up to 3 times. Make your endpoint idempotent.</p>
          </div>
          
          <div className="practice-item">
            <h4>🌐 Use HTTPS</h4>
            <p>Always use HTTPS endpoints in production to protect sensitive simulation data.</p>
          </div>
          
          <div className="practice-item">
            <h4>📊 Log Everything</h4>
            <p>Log webhook deliveries for debugging and monitoring your integration.</p>
          </div>
        </div>
      </div>

      {/* Testing Tools */}
      <div className="example-section">
        <h3>🧪 Testing Your Webhook</h3>
        <div className="testing-tools">
          <p>Use these tools to test your webhook endpoint:</p>
          <ul>
            <li><strong>ngrok</strong> - Expose your local server to the internet for testing</li>
            <li><strong>webhook.site</strong> - Create temporary webhook URLs to inspect payloads</li>
            <li><strong>PostBin</strong> - Another service for testing webhook deliveries</li>
            <li><strong>SimApp Test Button</strong> - Use the test button in your webhook configuration</li>
          </ul>
          
          <div className="testing-command">
            <h4>Quick ngrok setup:</h4>
            <code>ngrok http 3000</code>
            <p>Then use the https URL (e.g., https://abc123.ngrok.io/webhooks) in your webhook configuration.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WebhookExamples;
