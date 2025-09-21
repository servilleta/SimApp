import React, { useState } from 'react';

const APIDocumentationPage = () => {
  const [activeTab, setActiveTab] = useState('auth');
  const [apiKey, setApiKey] = useState(import.meta.env.VITE_DEMO_API_KEY || '');

  const pageStyle = {
    padding: '2rem',
    backgroundColor: 'var(--color-white)',
    minHeight: '100vh',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", sans-serif'
  };

  const headerStyle = {
    marginBottom: '2rem',
    paddingBottom: '1.5rem',
    borderBottom: `2px solid var(--color-braun-orange)`
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

  const tabContainerStyle = {
    display: 'flex',
    marginBottom: '1.5rem',
    backgroundColor: 'var(--color-warm-white)',
    borderRadius: '8px',
    padding: '4px',
    border: '1px solid var(--color-border-light)'
  };

  const tabStyle = (isActive) => ({
    padding: '0.75rem 1.5rem',
    borderRadius: '4px',
    backgroundColor: isActive ? 'var(--color-white)' : 'transparent',
    color: isActive ? 'var(--color-charcoal)' : 'var(--color-medium-grey)',
    border: 'none',
    cursor: 'pointer',
    fontSize: '0.875rem',
    fontWeight: isActive ? '600' : '500',
    transition: 'all var(--transition-base)',
    boxShadow: isActive ? 'var(--shadow-sm)' : 'none'
  });

  const contentStyle = {
    backgroundColor: 'var(--color-white)',
    borderRadius: '8px',
    padding: '2rem',
    border: '1px solid var(--color-border-light)',
    boxShadow: 'var(--shadow-sm)'
  };

  const codeBlockStyle = {
    backgroundColor: 'var(--color-charcoal)',
    color: '#e5e7eb',
    padding: '1rem',
    borderRadius: '4px',
    fontSize: '0.875rem',
    fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
    overflow: 'auto',
    margin: '1rem 0',
    border: '1px solid var(--color-border-light)'
  };

  const endpointStyle = {
    backgroundColor: 'var(--color-warm-white)',
    borderRadius: '8px',
    padding: '1.5rem',
    marginBottom: '1.5rem',
    border: '1px solid var(--color-border-light)'
  };

  const methodBadgeStyle = (method) => ({
    display: 'inline-block',
    padding: '0.25rem 0.5rem',
    borderRadius: '4px',
    fontSize: '0.75rem',
    fontWeight: '600',
    color: 'white',
    backgroundColor: method === 'GET' ? 'var(--color-success)' : method === 'POST' ? 'var(--color-braun-orange)' : method === 'DELETE' ? 'var(--color-error)' : 'var(--color-medium-grey)',
    marginRight: '0.75rem'
  });

  const apiKeyInputStyle = {
    width: '100%',
    padding: '0.75rem 1rem',
    borderRadius: '4px',
    border: '1px solid var(--color-border-light)',
    backgroundColor: 'var(--color-white)',
    fontSize: '0.875rem',
    fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
    marginBottom: '1rem',
    color: 'var(--color-text-primary)'
  };

  const copyButtonStyle = {
    padding: '0.5rem 1rem',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: 'var(--color-braun-orange)',
    color: 'white',
    fontSize: '0.75rem',
    fontWeight: '500',
    cursor: 'pointer',
    marginLeft: '0.5rem',
    transition: 'all var(--transition-base)'
  };

  const alertStyle = {
    backgroundColor: 'rgba(255, 215, 0, 0.1)',
    border: '1px solid var(--color-warning)',
    borderRadius: '4px',
    padding: '1rem',
    marginBottom: '1.5rem',
    color: 'var(--color-dark-grey)'
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    // You could add a toast notification here
  };


  const renderAuthenticationWithOverview = () => (
    <div>
      {/* Authentication Section */}
      <h2 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem', fontSize: '1.75rem', fontWeight: '600' }}>
        🔐 Authentication Setup
      </h2>
      
      <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '0.75rem', fontSize: '1.25rem', fontWeight: '600' }}>Your API Key</h3>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
        <input 
          type="text" 
          value={apiKey} 
          onChange={(e) => setApiKey(e.target.value)}
          style={apiKeyInputStyle}
          placeholder="Enter your API key"
        />
        <button 
          style={copyButtonStyle}
          onClick={() => copyToClipboard(apiKey)}
          onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--color-braun-orange-dark)'}
          onMouseLeave={(e) => e.target.style.backgroundColor = 'var(--color-braun-orange)'}
        >
          Copy
        </button>
      </div>

      <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '0.75rem', fontSize: '1.25rem', fontWeight: '600' }}>
        Base URL & Authentication
      </h3>
      <div style={codeBlockStyle}>
{`Base URL: http://209.51.170.185:8000/simapp-api
Authorization Header: Bearer ${apiKey}`}
      </div>

      <h3 style={{ color: 'var(--color-charcoal)', marginTop: '1.5rem', marginBottom: '0.75rem', fontSize: '1.25rem', fontWeight: '600' }}>
        Quick Test
      </h3>
      <div style={codeBlockStyle}>
{`curl -H "Authorization: Bearer ${apiKey}" \\
     http://209.51.170.185:8000/simapp-api/health`}
      </div>

    </div>
  );

  const renderAuthentication = () => (
    <div>
      <h2 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem', fontSize: '1.5rem', fontWeight: '600' }}>
        🔐 Authentication
      </h2>
      
      <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '0.75rem', fontSize: '1.25rem', fontWeight: '600' }}>Your API Key</h3>
      <input 
        type="text" 
        value={apiKey} 
        onChange={(e) => setApiKey(e.target.value)}
        style={apiKeyInputStyle}
        placeholder="Enter your API key"
      />
      <button 
        style={copyButtonStyle}
        onClick={() => copyToClipboard(apiKey)}
        onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--color-braun-orange-dark)'}
        onMouseLeave={(e) => e.target.style.backgroundColor = 'var(--color-braun-orange)'}
      >
        Copy
      </button>

      <h3 style={{ color: 'var(--color-charcoal)', marginTop: '1.5rem', marginBottom: '0.75rem', fontSize: '1.25rem', fontWeight: '600' }}>
        Header Format
      </h3>
      <div style={codeBlockStyle}>
        Authorization: Bearer {apiKey}
      </div>

      <h3 style={{ color: 'var(--color-charcoal)', marginTop: '1.5rem', marginBottom: '0.75rem', fontSize: '1.25rem', fontWeight: '600' }}>
        Test Authentication
      </h3>
      <div style={codeBlockStyle}>
{`curl -H "Authorization: Bearer ${apiKey}" \\
     http://209.51.170.185:8000/simapp-api/models`}
      </div>

      <div style={{ ...alertStyle, backgroundColor: 'var(--color-success-bg)', borderColor: 'var(--color-success)' }}>
        <strong>💡 Pro Tip:</strong> Store your API key securely and never commit it to version control.
      </div>
    </div>
  );

  const renderEndpoints = () => (
    <div>
      <h2 style={{ color: 'var(--color-charcoal)', marginBottom: '1.5rem', fontSize: '1.5rem', fontWeight: '600' }}>
        📋 API Endpoints
      </h2>

      {/* Health Check */}
      <div style={endpointStyle}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
          <span style={methodBadgeStyle('GET')}>GET</span>
          <code style={{ fontSize: '16px' }}>/health</code>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
          Check API health status (no authentication required)
        </p>
        <div style={codeBlockStyle}>
{`      curl http://209.51.170.185:8000/simapp-api/health

Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "gpu_available": true,
  "system_metrics": {
    "uptime": "1h 23m",
    "memory_usage": "2.1GB",
    "active_simulations": 0
  }
}`}
        </div>
      </div>

      {/* List Models */}
      <div style={endpointStyle}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
          <span style={methodBadgeStyle('GET')}>GET</span>
          <code style={{ fontSize: '16px' }}>/models</code>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
          List your uploaded Excel models
        </p>
        <div style={codeBlockStyle}>
{`curl -H "Authorization: Bearer ${apiKey}" \\
     http://209.51.170.185:8000/simapp-api/models

Response:
{
  "models": [
    {
      "model_id": "mdl_abc123def456",
      "filename": "portfolio_risk.xlsx",
      "formulas_count": 45,
      "created_at": "2024-01-15T09:00:00Z",
      "status": "ready",
      "variables_detected": 8
    }
  ]
}`}
        </div>
      </div>

      {/* Upload Model */}
      <div style={endpointStyle}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
          <span style={methodBadgeStyle('POST')}>POST</span>
          <code style={{ fontSize: '16px' }}>/models</code>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
          Upload a new Excel model for simulation
        </p>
        <div style={codeBlockStyle}>
{`curl -X POST \\
     -H "Authorization: Bearer ${apiKey}" \\
     -F "file=@portfolio_model.xlsx" \\
     http://209.51.170.185:8000/simapp-api/models

Response:
{
  "model_id": "mdl_abc123def456",
  "status": "uploaded",
  "processing_time_estimate": "< 5 minutes",
  "formulas_count": 32,
  "variables_detected": [
    {
      "cell": "B5",
      "sheet": "Main",
      "current_value": 0.15,
      "is_input": true
    },
    {
      "cell": "C8", 
      "sheet": "Main",
      "current_value": 100000,
      "is_input": true
    }
  ],
  "created_at": "2024-01-15T09:00:00Z"
}`}
        </div>
      </div>

      {/* Run Simulation */}
      <div style={endpointStyle}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
          <span style={methodBadgeStyle('POST')}>POST</span>
          <code style={{ fontSize: '16px' }}>/simulations</code>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
          Start a Monte Carlo simulation on your model
        </p>
        <div style={{ padding: '1rem', backgroundColor: 'var(--color-info-bg)', border: '1px solid var(--color-info-border)', borderRadius: '6px', marginBottom: '1rem' }}>
          <h4 style={{ color: 'var(--color-info)', margin: '0 0 0.5rem 0' }}>📋 Multi-Sheet Excel Support</h4>
          <p style={{ margin: '0', fontSize: '14px', color: 'var(--color-text-secondary)' }}>
            <strong>Cell References:</strong> Use <code>Sheet!Cell</code> format (e.g., <code>"Sheet2!C10"</code>) for multi-sheet files. 
            Simple references (e.g., <code>"B5"</code>) default to the first sheet.
          </p>
        </div>
        <div style={codeBlockStyle}>
{`curl -X POST \\
     -H "Authorization: Bearer ${apiKey}" \\
     -H "Content-Type: application/json" \\
     -d '{
       "model_id": "mdl_abc123def456",
       "simulation_config": {
         "iterations": 10000,
         "variables": [
           {
             "cell": "B5",
             "distribution": {
               "type": "triangular",
               "min": 0.05,
               "mode": 0.15,
               "max": 0.35
             }
           },
           {
             "cell": "Sheet2!C10",
             "distribution": {
               "type": "normal",
               "mean": 0.12,
               "std": 0.03
             }
           }
         ],
         "output_cells": ["J25", "Sheet2!K25"],
         "confidence_levels": [0.95, 0.99]
       }
     }' \\
     http://209.51.170.185:8000/simapp-api/simulations

Response:
{
  "simulation_id": "sim_b2b_789abc12",
  "status": "queued",
  "estimated_completion": "2024-01-15T10:45:00Z",
  "progress_url": "/simapp-api/simulations/sim_b2b_789abc12",
  "credits_consumed": 50.0
}`}
        </div>
      </div>

      {/* Get Results */}
      <div style={endpointStyle}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
          <span style={methodBadgeStyle('GET')}>GET</span>
          <code style={{ fontSize: '16px' }}>/simulations/{'{simulation_id}'}</code>
        </div>
        <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
          Get simulation status and results (results included when status is "completed")
        </p>
        <div style={codeBlockStyle}>
{`curl -H "Authorization: Bearer ${apiKey}" \\
     http://209.51.170.185:8000/simapp-api/simulations/sim_b2b_789abc12

Response:
{
  "simulation_id": "sim_b2b_789abc12",
  "status": "completed",
  "completion_time": "2024-01-15T10:42:30Z",
  "execution_time_seconds": 45.2,
  "credits_consumed": 50.0,
  "results": {
    "J25": {
      "mean": 1250.67,
      "std": 234.89,
      "min": 445.12,
      "max": 2890.23,
      "percentiles": {
        "5": 678.45,
        "25": 980.12,
        "50": 1250.67,
        "75": 1520.89,
        "95": 1890.23
      },
      "var_95": 890.23,
      "var_99": 645.12
    },
    "K25": {
      "mean": 15.6,
      "std": 4.2,
      "min": 8.1,
      "max": 24.3,
      "percentiles": {
        "5": 9.2,
        "25": 12.8,
        "50": 15.6,
        "75": 18.4,
        "95": 22.1
      },
      "var_95": 9.8,
      "var_99": 8.5
    }
  },
  "download_links": {
    "detailed_csv": "https://api.domain.com/downloads/sim_b2b_789abc12.csv",
    "summary_pdf": "https://api.domain.com/downloads/sim_b2b_789abc12.pdf"
  }
}`}
        </div>
      </div>
    </div>
  );

  const renderExamples = () => (
    <div>
      <h2 style={{ color: 'var(--color-charcoal)', marginBottom: '1.5rem', fontSize: '1.5rem', fontWeight: '600' }}>
        💡 Code Examples
      </h2>

      <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem', fontSize: '1.25rem', fontWeight: '600' }}>Python</h3>
      <div style={codeBlockStyle}>
{`import requests
import time

# Configuration
API_KEY = "${apiKey}"
BASE_URL = "http://209.51.170.185:8000/simapp-api"
headers = {"Authorization": f"Bearer {API_KEY}"}

# 1. Upload Excel model
with open("portfolio_model.xlsx", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/models", 
                           headers=headers, files=files)
    model_data = response.json()
    model_id = model_data["model_id"]

# 2. Start simulation
simulation_data = {
    "model_id": model_id,
    "simulation_config": {
        "iterations": 10000,
        "variables": [
            {
                "cell": "B5",
                "distribution": {
                    "type": "triangular",
                    "min": 0.05,
                    "mode": 0.15,
                    "max": 0.35
                }
            }
        ],
        "output_cells": ["J25", "K25"],
        "confidence_levels": [0.95, 0.99]
    }
}
response = requests.post(f"{BASE_URL}/simulations", 
                        headers=headers, json=simulation_data)
sim_data = response.json()
simulation_id = sim_data["simulation_id"]

# 3. Poll for completion and get results
while True:
    response = requests.get(f"{BASE_URL}/simulations/{simulation_id}", 
                           headers=headers)
    status_data = response.json()
    
    if status_data["status"] == "completed":
        # Results are included in the status response
        results = status_data["results"]
        print(f"Mean: {results['mean']}")
        print(f"VaR 95%: {results['var_95']}")
        break
    elif status_data["status"] == "failed":
        print("Simulation failed!")
        break
    
    time.sleep(5)  # Wait 5 seconds`}
      </div>

      <h3 style={{ color: 'var(--color-charcoal)', marginTop: '2rem', marginBottom: '1rem', fontSize: '1.25rem', fontWeight: '600' }}>JavaScript</h3>
      <div style={codeBlockStyle}>
{`const API_KEY = "${apiKey}";
const BASE_URL = "http://209.51.170.185:8000/simapp-api";

// Upload model and run simulation
async function runMonteCarloSimulation(excelFile) {
  // 1. Upload model
  const formData = new FormData();
  formData.append('file', excelFile);
  
  const uploadResponse = await fetch(\`\${BASE_URL}/models\`, {
    method: 'POST',
    headers: { 'Authorization': \`Bearer \${API_KEY}\` },
    body: formData
  });
  const modelData = await uploadResponse.json();
  
  // 2. Start simulation
  const simResponse = await fetch(\`\${BASE_URL}/simulations\`, {
    method: 'POST',
    headers: {
      'Authorization': \`Bearer \${API_KEY}\`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model_id: modelData.model_id,
      simulation_config: {
        iterations: 10000,
        variables: [
          {
            cell: "B5",
            distribution: {
              type: "triangular",
              min: 0.05,
              mode: 0.15,
              max: 0.35
            }
          }
        ],
        output_cells: ["J25", "K25"],
        confidence_levels: [0.95, 0.99]
      }
    })
  });
  const simData = await simResponse.json();
  
  // 3. Poll for completion
  return new Promise((resolve) => {
    const checkStatus = async () => {
      const statusResponse = await fetch(
        \`\${BASE_URL}/simulations/\${simData.simulation_id}\`,
        { headers: { 'Authorization': \`Bearer \${API_KEY}\` } }
      );
      const status = await statusResponse.json();
      
      if (status.status === 'completed') {
        // Results are included in the status response
        resolve(status.results);
      } else if (status.status !== 'failed') {
        setTimeout(checkStatus, 5000); // Check again in 5 seconds
      }
    };
    checkStatus();
  });
}`}
      </div>
    </div>
  );


  const renderContent = () => {
    switch (activeTab) {
      case 'auth':
        return renderAuthenticationWithOverview();
      case 'endpoints':
        return renderEndpoints();
      case 'examples':
        return renderExamples();
      default:
        return renderAuthenticationWithOverview();
    }
  };

  return (
    <div style={pageStyle}>
      <div style={headerStyle}>
        <h1 style={titleStyle}>API Documentation</h1>
        <p style={subtitleStyle}>
          Integrate Monte Carlo simulations into your applications with our powerful B2B API
        </p>
      </div>

      <div style={tabContainerStyle}>
        <button 
          style={tabStyle(activeTab === 'auth')}
          onClick={() => setActiveTab('auth')}
        >
          🔑 Authentication
        </button>
        <button 
          style={tabStyle(activeTab === 'endpoints')}
          onClick={() => setActiveTab('endpoints')}
        >
          📋 API Endpoints
        </button>
        <button 
          style={tabStyle(activeTab === 'examples')}
          onClick={() => setActiveTab('examples')}
        >
          💡 Code Examples
        </button>
      </div>

      <div style={contentStyle}>
        {renderContent()}
      </div>
    </div>
  );
};

export default APIDocumentationPage;
