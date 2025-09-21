import React, { useState, useEffect } from 'react';
import { FaUpload, FaPlay, FaDownload, FaSpinner, FaCheck, FaExclamationTriangle, FaCopy } from 'react-icons/fa';

const APITestPage = () => {
  // State management
  const [apiKey, setApiKey] = useState(import.meta.env.VITE_DEMO_API_KEY || '');
  const [selectedFile, setSelectedFile] = useState(null);
  const [modelId, setModelId] = useState('');
  const [simulationId, setSimulationId] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('setup'); // setup, upload, configure, simulate, results
  const [results, setResults] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  // API Configuration
  const API_BASE_URL = '/simapp-api';
  
  // Default simulation configuration
  const [simulationConfig, setSimulationConfig] = useState({
    iterations: 10000,
    variables: [
      {
        cell: 'B5',
        name: 'Market_Volatility',
        distribution: {
          type: 'triangular',
          min: 0.05,
          mode: 0.15,
          max: 0.35
        }
      },
      {
        cell: 'C7',
        name: 'Interest_Rate',
        distribution: {
          type: 'normal',
          mean: 0.03,
          std: 0.01
        }
      }
    ],
    output_cells: ['J25', 'K25'],
    confidence_levels: [0.95, 0.99]
  });

  const [detectedVariables, setDetectedVariables] = useState([]);
  const [downloadToken, setDownloadToken] = useState(null);

  // Add log entry
  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    console.log('🧪 [API_TEST] Adding log:', message, 'Type:', type);
    setLogs(prev => {
      const newLogs = [...prev, { timestamp, message, type }];
      console.log('🧪 [API_TEST] Updated logs array:', newLogs);
      return newLogs;
    });
  };

  // API Health Check
  const checkAPIHealth = async () => {
    console.log('🧪 [API_TEST] Health check button clicked!');
    try {
      setIsLoading(true);
      addLog('Checking API health...', 'info');
      console.log('🧪 [API_TEST] Making request to:', `${API_BASE_URL}/health`);
      
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiKey}`
        }
      });

      if (response.ok) {
        const health = await response.json();
        console.log('🧪 [API_TEST] Health response:', health);
        addLog(`✅ API is healthy - GPU Available: ${health.gpu_available}`, 'success');
        
        // Clear success popup
        alert(`🎉 API HEALTH CHECK SUCCESS! 🎉

✅ Status: ${health.status.toUpperCase()}
🖥️ GPU Available: ${health.gpu_available ? 'YES' : 'NO'}
📅 Timestamp: ${health.timestamp}
🔧 Version: ${health.version}
📊 System Metrics: ${JSON.stringify(health.system_metrics, null, 2)}

🚀 Your Monte Carlo API is working perfectly!`);
        
        return true;
      } else {
        console.log('🧪 [API_TEST] Health check failed:', response.status, response.statusText);
        addLog(`❌ API health check failed: ${response.status}`, 'error');
        
        alert(`❌ API HEALTH CHECK FAILED! ❌

🚫 HTTP Status: ${response.status}
📝 Status Text: ${response.statusText}
⚠️ The API is not responding correctly.

Please check your API configuration.`);
        
        return false;
      }
    } catch (error) {
      console.log('🧪 [API_TEST] Health check error:', error);
      addLog(`❌ Connection error: ${error.message}`, 'error');
      
      alert(`❌ API CONNECTION ERROR! ❌

🔌 Error: ${error.message}
🌐 Could not connect to the API
⚠️ Please check your network connection.

Debug info: Trying to connect to ${API_BASE_URL}/health`);
      
      return false;
    } finally {
      console.log('🧪 [API_TEST] Health check finished, setting loading to false');
      setIsLoading(false);
    }
  };

  // Upload Excel Model
  const uploadModel = async () => {
    if (!selectedFile) {
      setError('Please select an Excel file');
      return;
    }

    try {
      setIsLoading(true);
      setCurrentStep('upload');
      addLog('Uploading Excel model...', 'info');

      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${API_BASE_URL}/models`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`
        },
        body: formData
      });

      if (response.ok) {
        const uploadResult = await response.json();
        setModelId(uploadResult.model_id);
        addLog(`✅ Model uploaded successfully - ID: ${uploadResult.model_id}`, 'success');
        addLog(`📊 Formulas detected: ${uploadResult.formulas_count}`, 'info');
        
        // Update detected variables from the API response
        if (uploadResult.variables_detected && uploadResult.variables_detected.length > 0) {
          setDetectedVariables(uploadResult.variables_detected);
          addLog(`🔍 Detected ${uploadResult.variables_detected.length} potential variables`, 'info');
          
          // Update simulation config with detected variables
          const dynamicVariables = uploadResult.variables_detected.map(variable => ({
            cell: variable.cell,
            name: variable.name || `Variable_${variable.cell}`,
            distribution: {
              type: variable.suggested_distribution || 'normal',
              mean: variable.current_value || 0,
              std: variable.current_value * 0.1 || 0.01
            }
          }));
          
          setSimulationConfig(prev => ({
            ...prev,
            variables: dynamicVariables
          }));
        }
        
        setCurrentStep('configure');
        setError('');
      } else {
        const errorData = await response.json();
        addLog(`❌ Upload failed: ${errorData.detail}`, 'error');
        setError(errorData.detail);
      }
    } catch (error) {
      addLog(`❌ Upload error: ${error.message}`, 'error');
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Run Simulation
  const runSimulation = async () => {
    if (!modelId) {
      setError('No model uploaded');
      return;
    }

    try {
      setIsLoading(true);
      setCurrentStep('simulate');
      addLog('Starting Monte Carlo simulation...', 'info');

      const simulationRequest = {
        model_id: modelId,
        simulation_config: simulationConfig
      };

      const response = await fetch(`${API_BASE_URL}/simulations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(simulationRequest)
      });

      if (response.ok) {
        const simResult = await response.json();
        setSimulationId(simResult.simulation_id);
        addLog(`🚀 Simulation started - ID: ${simResult.simulation_id}`, 'success');
        addLog(`💰 Credits consumed: ${simResult.credits_consumed}`, 'info');
        
        // Start polling for progress
        startProgressPolling(simResult.simulation_id);
        setError('');
      } else {
        const errorData = await response.json();
        addLog(`❌ Simulation failed: ${errorData.detail}`, 'error');
        setError(errorData.detail);
      }
    } catch (error) {
      addLog(`❌ Simulation error: ${error.message}`, 'error');
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Progress Polling
  const startProgressPolling = (simId) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/simulations/${simId}/progress`, {
          headers: {
            'Authorization': `Bearer ${apiKey}`
          }
        });

        if (response.ok) {
          const progress = await response.json();
          setSimulationProgress(progress.progress.percentage || 0);
          
          if (progress.status === 'completed') {
            clearInterval(pollInterval);
            addLog('✅ Simulation completed!', 'success');
            fetchResults(simId);
          } else if (progress.status === 'failed') {
            clearInterval(pollInterval);
            addLog('❌ Simulation failed', 'error');
          } else {
            addLog(`⏳ Progress: ${Math.round(progress.progress.percentage)}%`, 'info');
          }
        }
      } catch (error) {
        addLog(`⚠️ Progress check error: ${error.message}`, 'warning');
      }
    }, 2000);

    // Clear interval after 10 minutes
    setTimeout(() => clearInterval(pollInterval), 600000);
  };

  // Fetch Results
  // Generate download token for browser-friendly downloads
  const generateDownloadToken = async (simId) => {
    try {
      addLog('🎫 Generating download token for browser access...', 'info');
      
      const response = await fetch(`${API_BASE_URL}/simulations/${simId}/generate-download-token`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to generate download token: ${response.status}`);
      }
      
      const tokenData = await response.json();
      setDownloadToken(tokenData);
      addLog(`✅ Download token generated (expires: ${new Date(tokenData.expires_at).toLocaleString()})`, 'success');
      
      return tokenData;
    } catch (error) {
      addLog(`❌ Token generation failed: ${error.message}`, 'error');
      throw error;
    }
  };

  const fetchResults = async (simId) => {
    try {
      addLog('Fetching simulation results...', 'info');
      
      const response = await fetch(`${API_BASE_URL}/simulations/${simId}/results`, {
        headers: {
          'Authorization': `Bearer ${apiKey}`
        }
      });

      if (response.ok) {
        const resultsData = await response.json();
        setResults(resultsData);
        setCurrentStep('results');
        addLog('📊 Results retrieved successfully!', 'success');
        
        // Generate download token for browser-friendly downloads
        await generateDownloadToken(simId);
      } else {
        addLog('❌ Failed to fetch results', 'error');
      }
    } catch (error) {
      addLog(`❌ Results error: ${error.message}`, 'error');
    }
  };

  // Load sample Excel file
  const loadSampleFile = async (filename = 'business-model.xlsx') => {
    try {
      addLog(`📁 Loading sample file: ${filename}...`, 'info');
      
      // Fetch the sample file from public/examples
      const response = await fetch(`/examples/${filename}`);
      if (response.ok) {
        const blob = await response.blob();
        const file = new File([blob], filename, { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
        setSelectedFile(file);
        addLog(`✅ Sample file loaded: ${filename} (${Math.round(file.size / 1024)} KB)`, 'success');
      } else {
        addLog(`❌ Failed to load sample file: ${filename}`, 'error');
      }
    } catch (error) {
      addLog(`❌ Error loading sample file: ${error.message}`, 'error');
      // Fallback to simulated file
      setSelectedFile({ name: filename, size: 45000 });
    }
  };

  // Copy API key
  const copyApiKey = () => {
    navigator.clipboard.writeText(apiKey);
    addLog('📋 API key copied to clipboard', 'info');
  };

  // Reset everything
  const resetTest = () => {
    setModelId('');
    setSimulationId('');
    setSelectedFile(null);
    setResults(null);
    setCurrentStep('setup');
    setSimulationProgress(0);
    setUploadProgress(0);
    setError('');
    setLogs([]);
    setDetectedVariables([]);
    
    // Reset to default simulation config
    setSimulationConfig({
      iterations: 10000,
      variables: [
        {
          cell: 'B5',
          name: 'Market_Volatility',
          distribution: {
            type: 'triangular',
            min: 0.05,
            mode: 0.15,
            max: 0.35
          }
        },
        {
          cell: 'C7',
          name: 'Interest_Rate',
          distribution: {
            type: 'normal',
            mean: 0.03,
            std: 0.01
          }
        }
      ],
      output_cells: ['J25', 'K25'],
      confidence_levels: [0.95, 0.99]
    });
    
    addLog('🔄 Test environment reset', 'info');
  };

  useEffect(() => {
    console.log('🧪 [API_TEST] Component initialized - API_BASE_URL:', API_BASE_URL);
    addLog('🧪 API Test Environment Initialized', 'info');
    addLog('💡 This page simulates how users interact with our Monte Carlo API', 'info');
  }, []);

  return (
    <div style={{ 
      padding: '2rem',
      maxWidth: '1200px',
      margin: '0 auto',
      backgroundColor: '#f8fafc',
      minHeight: '100vh'
    }}>
      {/* Header */}
      <div style={{ 
        textAlign: 'center',
        marginBottom: '2rem',
        backgroundColor: 'white',
        padding: '2rem',
        borderRadius: '12px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ 
          fontSize: '2.5rem',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '0.5rem'
        }}>
          🧪 Monte Carlo API Test Environment
        </h1>
        <p style={{ color: '#64748b', fontSize: '1.1rem' }}>
          Interactive simulation of how users integrate with our API
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        
        {/* Left Panel - Configuration & Controls */}
        <div style={{ backgroundColor: 'white', borderRadius: '12px', padding: '2rem', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
          <h2 style={{ marginBottom: '1.5rem', color: '#1e293b' }}>🛠 Test Configuration</h2>
          
          {/* API Key Section */}
          <div style={{ marginBottom: '2rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', color: '#374151' }}>
              API Key
            </label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="text"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                style={{
                  flex: 1,
                  padding: '0.75rem',
                  border: '2px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '0.9rem',
                  fontFamily: 'monospace'
                }}
                placeholder="Enter your API key"
              />
              <button onClick={copyApiKey} style={{
                padding: '0.75rem',
                backgroundColor: '#f3f4f6',
                border: '2px solid #e5e7eb',
                borderRadius: '8px',
                cursor: 'pointer'
              }}>
                <FaCopy />
              </button>
            </div>
          </div>

          {/* Step 1: Health Check */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ color: '#374151', marginBottom: '1rem' }}>Step 1: Health Check</h3>
            <button
              onClick={checkAPIHealth}
              disabled={isLoading}
              style={{
                width: '100%',
                padding: '1rem',
                backgroundColor: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem'
              }}
            >
              {isLoading ? <FaSpinner className="animate-spin" /> : <FaCheck />}
              Check API Health
            </button>
          </div>

          {/* Step 2: File Upload */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ color: '#374151', marginBottom: '1rem' }}>Step 2: Upload Excel Model</h3>
            
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: '#6b7280' }}>
                Choose your Excel file:
              </label>
              <input
                type="file"
                accept=".xlsx,.xlsm,.xls"
                onChange={(e) => setSelectedFile(e.target.files[0])}
                style={{ 
                  width: '100%',
                  padding: '0.5rem',
                  border: '2px dashed #d1d5db',
                  borderRadius: '8px',
                  backgroundColor: '#f9fafb'
                }}
              />
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <p style={{ fontSize: '0.9rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                Or try one of our sample models:
              </p>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button
                  onClick={() => loadSampleFile('business-model.xlsx')}
                  style={{
                    padding: '0.5rem 1rem',
                    backgroundColor: '#e0f2fe',
                    border: '1px solid #0891b2',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    color: '#0891b2'
                  }}
                >
                  📊 Business Model
                </button>
                <button
                  onClick={() => loadSampleFile('investment-returns.xlsx')}
                  style={{
                    padding: '0.5rem 1rem',
                    backgroundColor: '#f0fdf4',
                    border: '1px solid #16a34a',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    color: '#16a34a'
                  }}
                >
                  💰 Investment Returns
                </button>
                <button
                  onClick={() => loadSampleFile('project-costs.xlsx')}
                  style={{
                    padding: '0.5rem 1rem',
                    backgroundColor: '#fef3c7',
                    border: '1px solid #d97706',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    color: '#d97706'
                  }}
                >
                  🏗️ Project Costs
                </button>
              </div>
            </div>
            
            {selectedFile && (
              <div style={{ 
                padding: '1rem',
                backgroundColor: '#f0f9ff',
                borderRadius: '8px',
                marginBottom: '1rem',
                border: '1px solid #bae6fd'
              }}>
                <p><strong>File:</strong> {selectedFile.name}</p>
                <p><strong>Size:</strong> {Math.round(selectedFile.size / 1024)} KB</p>
              </div>
            )}

            <button
              onClick={uploadModel}
              disabled={!selectedFile || isLoading}
              style={{
                width: '100%',
                padding: '1rem',
                backgroundColor: selectedFile ? '#3b82f6' : '#9ca3af',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem',
                cursor: selectedFile && !isLoading ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem'
              }}
            >
              {isLoading && currentStep === 'upload' ? <FaSpinner className="animate-spin" /> : <FaUpload />}
              Upload Model
            </button>
          </div>

          {/* Step 3: Configure Simulation */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ color: '#374151', marginBottom: '1rem' }}>Step 3: Configure Simulation</h3>
            
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: '#6b7280' }}>
                Iterations
              </label>
              <input
                type="number"
                value={simulationConfig.iterations}
                onChange={(e) => setSimulationConfig(prev => ({ ...prev, iterations: parseInt(e.target.value) }))}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px'
                }}
                min="1000"
                max="1000000"
              />
            </div>

            {/* Input Variables Configuration */}
            <div style={{ 
              backgroundColor: '#f9fafb',
              padding: '1rem',
              borderRadius: '8px',
              marginBottom: '1rem',
              border: '1px solid #e5e7eb'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h4 style={{ fontSize: '0.9rem', color: '#374151', margin: 0 }}>
                  Input Variables Configuration:
                </h4>
                <button
                  onClick={() => {
                    setSimulationConfig(prev => ({
                      ...prev,
                      variables: [...prev.variables, {
                        cell: 'A1',
                        name: 'New_Variable',
                        distribution: { type: 'normal', mean: 0, std: 1 }
                      }]
                    }));
                    addLog('➕ Added new input variable', 'info');
                  }}
                  style={{
                    padding: '0.25rem 0.5rem',
                    backgroundColor: '#10b981',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '0.8rem',
                    cursor: 'pointer'
                  }}
                >
                  ➕ Add Variable
                </button>
              </div>
              
              {simulationConfig.variables.map((variable, index) => (
                <div key={index} style={{ 
                  backgroundColor: 'white',
                  padding: '0.75rem',
                  borderRadius: '6px',
                  marginBottom: '0.75rem',
                  border: '1px solid #e5e7eb'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <strong style={{ fontSize: '0.85rem', color: '#374151' }}>Variable {index + 1}</strong>
                    <button
                      onClick={() => {
                        setSimulationConfig(prev => ({
                          ...prev,
                          variables: prev.variables.filter((_, i) => i !== index)
                        }));
                        addLog(`🗑️ Removed variable ${index + 1}`, 'info');
                      }}
                      style={{
                        padding: '0.25rem 0.5rem',
                        backgroundColor: '#ef4444',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        fontSize: '0.7rem',
                        cursor: 'pointer'
                      }}
                    >
                      🗑️ Remove
                    </button>
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <div>
                      <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                        Cell Reference
                      </label>
                      <input
                        type="text"
                        value={variable.cell}
                        onChange={(e) => {
                          const newVariables = [...simulationConfig.variables];
                          newVariables[index].cell = e.target.value;
                          setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                        }}
                        style={{
                          width: '100%',
                          padding: '0.375rem',
                          border: '1px solid #d1d5db',
                          borderRadius: '4px',
                          fontSize: '0.8rem'
                        }}
                        placeholder="e.g., B5"
                      />
                    </div>
                    <div>
                      <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                        Variable Name
                      </label>
                      <input
                        type="text"
                        value={variable.name}
                        onChange={(e) => {
                          const newVariables = [...simulationConfig.variables];
                          newVariables[index].name = e.target.value;
                          setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                        }}
                        style={{
                          width: '100%',
                          padding: '0.375rem',
                          border: '1px solid #d1d5db',
                          borderRadius: '4px',
                          fontSize: '0.8rem'
                        }}
                        placeholder="e.g., Market_Volatility"
                      />
                    </div>
                  </div>
                  
                  <div style={{ marginBottom: '0.5rem' }}>
                    <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                      Distribution Type
                    </label>
                    <select
                      value={variable.distribution.type}
                      onChange={(e) => {
                        const newVariables = [...simulationConfig.variables];
                        const newType = e.target.value;
                        if (newType === 'normal') {
                          newVariables[index].distribution = { type: 'normal', mean: 0, std: 1 };
                        } else if (newType === 'triangular') {
                          newVariables[index].distribution = { type: 'triangular', min: 0, mode: 0.5, max: 1 };
                        } else if (newType === 'uniform') {
                          newVariables[index].distribution = { type: 'uniform', min: 0, max: 1 };
                        }
                        setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                      }}
                      style={{
                        width: '100%',
                        padding: '0.375rem',
                        border: '1px solid #d1d5db',
                        borderRadius: '4px',
                        fontSize: '0.8rem'
                      }}
                    >
                      <option value="normal">Normal Distribution</option>
                      <option value="triangular">Triangular Distribution</option>
                      <option value="uniform">Uniform Distribution</option>
                    </select>
                  </div>
                  
                  {/* Distribution Parameters */}
                  {variable.distribution.type === 'normal' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                      <div>
                        <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                          Mean
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={variable.distribution.mean || 0}
                          onChange={(e) => {
                            const newVariables = [...simulationConfig.variables];
                            newVariables[index].distribution.mean = parseFloat(e.target.value) || 0;
                            setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                          }}
                          style={{
                            width: '100%',
                            padding: '0.375rem',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                          }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                          Std Dev
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={variable.distribution.std || 0}
                          onChange={(e) => {
                            const newVariables = [...simulationConfig.variables];
                            newVariables[index].distribution.std = parseFloat(e.target.value) || 0;
                            setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                          }}
                          style={{
                            width: '100%',
                            padding: '0.375rem',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                          }}
                        />
                      </div>
                    </div>
                  )}
                  
                  {variable.distribution.type === 'triangular' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.5rem' }}>
                      <div>
                        <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                          Min
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={variable.distribution.min || 0}
                          onChange={(e) => {
                            const newVariables = [...simulationConfig.variables];
                            newVariables[index].distribution.min = parseFloat(e.target.value) || 0;
                            setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                          }}
                          style={{
                            width: '100%',
                            padding: '0.375rem',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                          }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                          Mode
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={variable.distribution.mode || 0}
                          onChange={(e) => {
                            const newVariables = [...simulationConfig.variables];
                            newVariables[index].distribution.mode = parseFloat(e.target.value) || 0;
                            setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                          }}
                          style={{
                            width: '100%',
                            padding: '0.375rem',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                          }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                          Max
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={variable.distribution.max || 0}
                          onChange={(e) => {
                            const newVariables = [...simulationConfig.variables];
                            newVariables[index].distribution.max = parseFloat(e.target.value) || 0;
                            setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                          }}
                          style={{
                            width: '100%',
                            padding: '0.375rem',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                          }}
                        />
                      </div>
                    </div>
                  )}
                  
                  {variable.distribution.type === 'uniform' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                      <div>
                        <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                          Min
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={variable.distribution.min || 0}
                          onChange={(e) => {
                            const newVariables = [...simulationConfig.variables];
                            newVariables[index].distribution.min = parseFloat(e.target.value) || 0;
                            setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                          }}
                          style={{
                            width: '100%',
                            padding: '0.375rem',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                          }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: '0.75rem', color: '#6b7280', display: 'block', marginBottom: '0.25rem' }}>
                          Max
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={variable.distribution.max || 0}
                          onChange={(e) => {
                            const newVariables = [...simulationConfig.variables];
                            newVariables[index].distribution.max = parseFloat(e.target.value) || 0;
                            setSimulationConfig(prev => ({ ...prev, variables: newVariables }));
                          }}
                          style={{
                            width: '100%',
                            padding: '0.375rem',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Output Cells Configuration */}
            <div style={{ 
              backgroundColor: '#f9fafb',
              padding: '1rem',
              borderRadius: '8px',
              marginBottom: '1rem',
              border: '1px solid #e5e7eb'
            }}>
              <h4 style={{ fontSize: '0.9rem', color: '#374151', marginBottom: '0.5rem' }}>
                Output Cells to Analyze:
              </h4>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
                <input
                  type="text"
                  value={simulationConfig.output_cells.join(', ')}
                  onChange={(e) => {
                    const cells = e.target.value.split(',').map(cell => cell.trim()).filter(cell => cell);
                    setSimulationConfig(prev => ({ ...prev, output_cells: cells }));
                  }}
                  style={{
                    flex: 1,
                    padding: '0.5rem',
                    border: '1px solid #d1d5db',
                    borderRadius: '4px',
                    fontSize: '0.8rem'
                  }}
                  placeholder="e.g., J25, K25, NPV_Cell"
                />
                <button
                  onClick={() => {
                    setSimulationConfig(prev => ({ 
                      ...prev, 
                      output_cells: [...prev.output_cells, 'A1'] 
                    }));
                    addLog('➕ Added output cell', 'info');
                  }}
                  style={{
                    padding: '0.5rem',
                    backgroundColor: '#10b981',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '0.8rem',
                    cursor: 'pointer'
                  }}
                >
                  ➕
                </button>
              </div>
              <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>
                💡 Separate multiple cells with commas (e.g., J25, K25, NPV_Cell)
              </p>
            </div>

            <button
              onClick={runSimulation}
              disabled={!modelId || isLoading}
              style={{
                width: '100%',
                padding: '1rem',
                backgroundColor: modelId ? '#8b5cf6' : '#9ca3af',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem',
                cursor: modelId && !isLoading ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem'
              }}
            >
              {isLoading && currentStep === 'simulate' ? <FaSpinner className="animate-spin" /> : <FaPlay />}
              Run Simulation
            </button>
          </div>

          {/* Progress */}
          {simulationProgress > 0 && (
            <div style={{ marginBottom: '2rem' }}>
              <h4 style={{ color: '#374151', marginBottom: '0.5rem' }}>Simulation Progress</h4>
              <div style={{
                width: '100%',
                height: '20px',
                backgroundColor: '#f3f4f6',
                borderRadius: '10px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${simulationProgress}%`,
                  height: '100%',
                  backgroundColor: '#10b981',
                  transition: 'width 0.3s ease'
                }} />
              </div>
              <p style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.5rem' }}>
                {Math.round(simulationProgress)}% Complete
              </p>
            </div>
          )}

          {/* Reset Button */}
          <button
            onClick={resetTest}
            style={{
              width: '100%',
              padding: '0.75rem',
              backgroundColor: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '0.9rem',
              cursor: 'pointer'
            }}
          >
            🔄 Reset Test Environment
          </button>
        </div>

        {/* Right Panel - Logs & Results */}
        <div>
          {/* API Usage Examples */}
          <div style={{ 
            backgroundColor: 'white',
            borderRadius: '12px',
            padding: '2rem',
            marginBottom: '2rem',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}>
            <h2 style={{ marginBottom: '1.5rem', color: '#1e293b' }}>💡 API Usage Examples</h2>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ color: '#374151', marginBottom: '0.5rem' }}>1. Health Check</h4>
              <div style={{
                backgroundColor: '#1e293b',
                borderRadius: '6px',
                padding: '1rem',
                fontFamily: 'monospace',
                fontSize: '0.8rem',
                color: '#e5e7eb',
                overflow: 'auto'
              }}>
{`curl -X GET \\
  -H "Authorization: Bearer ${apiKey}" \\
  ${API_BASE_URL}/health`}
              </div>
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ color: '#374151', marginBottom: '0.5rem' }}>2. Upload Model</h4>
              <div style={{
                backgroundColor: '#1e293b',
                borderRadius: '6px',
                padding: '1rem',
                fontFamily: 'monospace',
                fontSize: '0.8rem',
                color: '#e5e7eb',
                overflow: 'auto'
              }}>
{`curl -X POST \\
  -H "Authorization: Bearer ${apiKey}" \\
  -F "file=@model.xlsx" \\
  ${API_BASE_URL}/models`}
              </div>
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ color: '#374151', marginBottom: '0.5rem' }}>3. Run Simulation</h4>
              <div style={{
                backgroundColor: '#1e293b',
                borderRadius: '6px',
                padding: '1rem',
                fontFamily: 'monospace',
                fontSize: '0.8rem',
                color: '#e5e7eb',
                overflow: 'auto'
              }}>
{`curl -X POST \\
  -H "Authorization: Bearer ${apiKey}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "model_123",
    "simulation_config": {
      "iterations": 10000,
      "variables": [...],
      "output_cells": ["J25", "K25"]
    }
  }' \\
  ${API_BASE_URL}/simulations`}
              </div>
            </div>
          </div>
          {/* Logs Section */}
          <div style={{ 
            backgroundColor: 'white',
            borderRadius: '12px',
            padding: '2rem',
            marginBottom: '2rem',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}>
            <h2 style={{ marginBottom: '1.5rem', color: '#1e293b' }}>📋 Activity Log</h2>
            <div style={{
              height: '300px',
              overflowY: 'auto',
              backgroundColor: '#1e293b',
              borderRadius: '8px',
              padding: '1rem',
              fontFamily: 'monospace',
              fontSize: '0.85rem'
            }}>
              {logs.map((log, index) => (
                <div key={index} style={{
                  color: log.type === 'error' ? '#f87171' : 
                        log.type === 'success' ? '#34d399' :
                        log.type === 'warning' ? '#fbbf24' : '#e5e7eb',
                  marginBottom: '0.5rem'
                }}>
                  <span style={{ color: '#9ca3af' }}>[{log.timestamp}]</span> {log.message}
                </div>
              ))}
            </div>
          </div>

          {/* Results Section */}
          {results && (
            <div style={{ 
              backgroundColor: 'white',
              borderRadius: '12px',
              padding: '2rem',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
            }}>
              <h2 style={{ marginBottom: '1.5rem', color: '#1e293b' }}>📊 Simulation Results</h2>
              
              <div style={{ 
                backgroundColor: '#f0f9ff',
                padding: '1.5rem',
                borderRadius: '8px',
                border: '1px solid #bae6fd'
              }}>
                <h3 style={{ color: '#1e40af', marginBottom: '1rem' }}>Simulation: {results.simulation_id}</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <p><strong>Status:</strong> {results.status}</p>
                    <p><strong>Execution Time:</strong> {results.execution_time}</p>
                  </div>
                  <div>
                    <p><strong>Iterations:</strong> {results.iterations_completed?.toLocaleString()}</p>
                    <p><strong>Created:</strong> {new Date(results.created_at).toLocaleString()}</p>
                  </div>
                </div>
              </div>

              {Object.entries(results.results || {}).map(([cellName, cellResults]) => (
                <div key={cellName} style={{ 
                  marginTop: '1.5rem',
                  padding: '1.5rem',
                  backgroundColor: '#f8fafc',
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0'
                }}>
                  <h4 style={{ color: '#374151', marginBottom: '1rem' }}>{cellResults.cell_name}</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', fontSize: '0.9rem' }}>
                    <div>
                      <p><strong>Mean:</strong> {cellResults.statistics.mean?.toFixed(2)}</p>
                      <p><strong>Std Dev:</strong> {cellResults.statistics.std?.toFixed(2)}</p>
                    </div>
                    <div>
                      <p><strong>Min:</strong> {cellResults.statistics.min?.toFixed(2)}</p>
                      <p><strong>Max:</strong> {cellResults.statistics.max?.toFixed(2)}</p>
                    </div>
                    <div>
                      <p><strong>VaR 95%:</strong> {cellResults.statistics.var_95?.toFixed(2)}</p>
                      <p><strong>VaR 99%:</strong> {cellResults.statistics.var_99?.toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              ))}

              {(results.download_links || downloadToken) && (
                <div style={{ marginTop: '2rem' }}>
                  <h4 style={{ color: '#374151', marginBottom: '1rem' }}>📁 Download Options</h4>
                  <div style={{ display: 'flex', gap: '1rem' }}>
                    {downloadToken ? (
                      // Use token-based downloads (browser-friendly)
                      Object.entries(downloadToken.download_links).map(([format, url]) => (
                        <button
                          key={format}
                          onClick={() => window.open(url, '_blank')}
                          style={{
                            padding: '0.75rem 1rem',
                            backgroundColor: '#059669',
                            color: 'white',
                            border: 'none',
                            borderRadius: '0.375rem',
                            cursor: 'pointer',
                            textTransform: 'uppercase',
                            fontWeight: 'bold'
                          }}
                        >
                          📄 Download {format.toUpperCase()}
                        </button>
                      ))
                    ) : (
                      // Fallback to original download links
                      results.download_links && Object.entries(results.download_links).map(([format, url]) => (
                        <button
                          key={format}
                          onClick={() => window.open(url, '_blank')}
                        style={{
                          padding: '0.75rem 1.5rem',
                          backgroundColor: '#059669',
                          color: 'white',
                          border: 'none',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem'
                        }}
                      >
                        <FaDownload />
                        {format.toUpperCase()}
                      </button>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          position: 'fixed',
          bottom: '2rem',
          right: '2rem',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '8px',
          padding: '1rem',
          maxWidth: '400px',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
        }}>
          <FaExclamationTriangle style={{ color: '#ef4444' }} />
          <span style={{ color: '#dc2626' }}>{error}</span>
          <button
            onClick={() => setError('')}
            style={{
              marginLeft: 'auto',
              background: 'none',
              border: 'none',
              fontSize: '1.2rem',
              cursor: 'pointer',
              color: '#dc2626'
            }}
          >
            ×
          </button>
        </div>
      )}

      {/* Status Indicators */}
      <div style={{
        position: 'fixed',
        top: '2rem',
        right: '2rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        {['setup', 'upload', 'configure', 'simulate', 'results'].map((step, index) => (
          <div
            key={step}
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: 
                currentStep === step ? '#3b82f6' :
                ['setup', 'upload', 'configure', 'simulate', 'results'].indexOf(currentStep) > index ? '#10b981' : '#e5e7eb'
            }}
            title={step.charAt(0).toUpperCase() + step.slice(1)}
          />
        ))}
      </div>
    </div>
  );
};

export default APITestPage;
