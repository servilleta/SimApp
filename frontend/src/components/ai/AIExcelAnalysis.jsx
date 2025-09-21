import React, { useState, useEffect } from 'react';
import aiService from '../../services/aiService';
import AIAnalysisStatus from './AIAnalysisStatus';
import AIVariableSuggestions from './AIVariableSuggestions';
import AIResultsInsights from './AIResultsInsights';

export default function AIExcelAnalysis({ fileId, sheetName, isVisible, onSuggestionsReceived }) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [suggestions, setSuggestions] = useState(null);
  const [error, setError] = useState(null);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);

  // Auto-analyze when component becomes visible
  useEffect(() => {
    if (isVisible && fileId && !analysis && !isAnalyzing) {
      handleAnalyze();
    }
  }, [isVisible, fileId, analysis, isAnalyzing]);

  const pollForSuggestions = async (analysisId, maxAttempts = 60, interval = 2000) => {
    console.log('🔄 [AIExcelAnalysis] Starting to poll for suggestions:', analysisId);
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        console.log(`🔄 [AIExcelAnalysis] Poll attempt ${attempt}/${maxAttempts} for ${analysisId}`);
        const suggestionsResult = await aiService.getVariableSuggestions(analysisId);
        console.log('✅ [AIExcelAnalysis] Suggestions retrieved successfully!', suggestionsResult);
        return suggestionsResult;
      } catch (error) {
        // Handle expected 404 errors during polling without logging noise
        if (error.isExpectedPollingError || error.isNotFound || error.message.includes('Analysis not found')) {
          console.log(`⏳ [AIExcelAnalysis] Analysis still processing... (attempt ${attempt}/${maxAttempts})`);
          if (attempt < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, interval));
            continue;
          }
        }
        // Only log unexpected errors
        if (!error.isExpectedPollingError) {
          console.error(`❌ [AIExcelAnalysis] Unexpected error during polling:`, error);
        }
        throw error;
      }
    }
    
    throw new Error(`Analysis timeout after ${maxAttempts} attempts. The analysis is processing 1,601 variables and may take several minutes.`);
  };

  const handleAnalyze = async () => {
    if (!fileId) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      console.log('🤖 [AIExcelAnalysis] Starting analysis for:', { fileId, sheetName });
      const analysisResult = await aiService.analyzeExcel(fileId, sheetName);
      setAnalysis(analysisResult);
      
      // Auto-fetch suggestions with polling
      if (analysisResult.analysis_id) {
        setIsLoadingSuggestions(true);
        try {
          const suggestionsResult = await pollForSuggestions(analysisResult.analysis_id);
          setSuggestions(suggestionsResult);
          
          // Notify parent component about suggestions
          if (onSuggestionsReceived) {
            onSuggestionsReceived(suggestionsResult);
          }
        } catch (suggestionsError) {
          console.error('🤖 [AIExcelAnalysis] Suggestions error:', suggestionsError);
          setError(`AI analysis is taking longer than expected. This complex model has ${suggestionsError.message.includes('1601') ? '1,601 variables' : 'many variables'} to analyze.`);
        } finally {
          setIsLoadingSuggestions(false);
        }
      }
    } catch (error) {
      console.error('🤖 [AIExcelAnalysis] Analysis error:', error);
      setError(error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div style={{
      backgroundColor: '#f8fafc',
      border: '1px solid #e2e8f0',
      borderRadius: '8px',
      padding: '16px',
      margin: '16px 0',
      maxHeight: '400px',
      overflowY: 'auto'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '12px'
      }}>
        <h3 style={{
          margin: 0,
          fontSize: '16px',
          fontWeight: '600',
          color: '#1e293b',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          🤖 AI Excel Analysis
        </h3>
        
        {!analysis && !isAnalyzing && (
          <button
            onClick={handleAnalyze}
            style={{
              padding: '6px 12px',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '12px',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            Analyze
          </button>
        )}
      </div>

      {isAnalyzing && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          color: '#6b7280',
          fontSize: '14px'
        }}>
          <div style={{
            width: '16px',
            height: '16px',
            border: '2px solid #e5e7eb',
            borderTop: '2px solid #3b82f6',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }}></div>
          Analyzing Excel file with AI...
        </div>
      )}

      {error && (
        <div style={{
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '6px',
          padding: '12px',
          color: '#dc2626',
          fontSize: '14px'
        }}>
          <strong>Analysis Error:</strong> {error}
        </div>
      )}

      {analysis && (
        <div style={{ fontSize: '14px', lineHeight: '1.5' }}>
          {/* File Overview */}
          {analysis.file_overview && (
            <div style={{ marginBottom: '16px' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', fontWeight: '600', color: '#374151' }}>
                📊 File Overview
              </h4>
              <div style={{
                backgroundColor: 'white',
                padding: '12px',
                borderRadius: '6px',
                border: '1px solid #e5e7eb'
              }}>
                {analysis.file_overview.summary && (
                  <p style={{ margin: '0 0 8px 0', color: '#4b5563' }}>
                    {analysis.file_overview.summary}
                  </p>
                )}
                {analysis.file_overview.key_metrics && analysis.file_overview.key_metrics.length > 0 && (
                  <div>
                    <strong style={{ color: '#374151' }}>Key Metrics:</strong>
                    <ul style={{ margin: '4px 0 0 16px', color: '#6b7280' }}>
                      {analysis.file_overview.key_metrics.map((metric, index) => (
                        <li key={index}>{metric}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Variable Suggestions */}
          {isLoadingSuggestions && (
            <div style={{
              backgroundColor: '#f0f9ff',
              border: '1px solid #bfdbfe',
              borderRadius: '6px',
              padding: '12px',
              marginBottom: '16px'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: '#1e40af',
                fontSize: '14px',
                marginBottom: '8px'
              }}>
                <div style={{
                  width: '14px',
                  height: '14px',
                  border: '2px solid #bfdbfe',
                  borderTop: '2px solid #2563eb',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}></div>
                <strong>AI is analyzing your Excel model...</strong>
              </div>
              <div style={{ fontSize: '12px', color: '#6b7280', lineHeight: '1.4' }}>
                Processing Excel model formulas and dependencies. This may take 1-2 minutes for complex models.
                <br />
                <em>🧠 Enhanced with Ultra Engine insights for intelligent variable classification</em>
              </div>
            </div>
          )}

          {suggestions && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                marginBottom: '20px',
                paddingBottom: '12px',
                borderBottom: '2px solid #e5e7eb'
              }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  backgroundColor: '#3b82f6',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginRight: '12px',
                  fontSize: '18px'
                }}>
                  📊
                </div>
                <h4 style={{ margin: '0', fontSize: '18px', fontWeight: '700', color: '#1f2937' }}>
                  Excel Model Analysis
                </h4>
              </div>
              
              {/* File Description and Model Overview */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{
                  backgroundColor: 'white',
                  padding: '20px',
                  borderRadius: '12px',
                  border: '1px solid #e5e7eb',
                  boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
                }}>
                  <label style={{ 
                    display: 'flex',
                    alignItems: 'center', 
                    fontSize: '14px', 
                    fontWeight: '600', 
                    color: '#374151', 
                    marginBottom: '12px'
                  }}>
                    <span style={{
                      width: '24px',
                      height: '24px',
                      backgroundColor: '#f3f4f6',
                      borderRadius: '6px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      marginRight: '8px',
                      fontSize: '12px'
                    }}>
                      📋
                    </span>
                    File Analysis Overview:
                  </label>
                  
                  {/* File Statistics */}
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', 
                    gap: '12px', 
                    marginBottom: '16px',
                    padding: '12px',
                    backgroundColor: '#f8fafc',
                    borderRadius: '8px',
                    border: '1px solid #e5e7eb'
                  }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#1f2937' }}>
                        {suggestions.model_kpis?.active_sheets || 0}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280', fontWeight: '500' }}>Sheets</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#059669' }}>
                        {suggestions.model_kpis?.total_cells || 0}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280', fontWeight: '500' }}>Total Cells</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#dc2626' }}>
                        {suggestions.model_kpis?.formula_cells || 0}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280', fontWeight: '500' }}>Formulas</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#7c3aed' }}>
                        {suggestions.model_kpis?.input_cells || 0}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280', fontWeight: '500' }}>Inputs</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#ea580c' }}>
                        {suggestions.model_kpis?.output_cells || 0}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280', fontWeight: '500' }}>Outputs</div>
                    </div>
                  </div>

                  {/* Model Description */}
                  {(suggestions.model_description || suggestions.model_kpis?.structure_description) && (
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
                        Purpose & Structure:
                      </div>
                      <textarea
                        readOnly
                        value={suggestions.model_description || suggestions.model_kpis?.structure_description || 'No description available'}
                        style={{
                          width: '100%',
                          minHeight: '80px',
                          padding: '12px',
                          fontSize: '13px',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          backgroundColor: '#f8fafc',
                          color: '#374151',
                          resize: 'vertical',
                          lineHeight: '1.5',
                          fontFamily: 'system-ui, -apple-system, sans-serif'
                        }}
                      />
                    </div>
                  )}
                </div>
              </div>


              {/* Input Variables Table */}
              {suggestions.suggested_variables && suggestions.suggested_variables.length > 0 && (
                <div style={{ marginBottom: '20px' }}>
                  <div style={{
                    backgroundColor: 'white',
                    padding: '20px',
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
                  }}>
                    <label style={{ 
                      display: 'flex',
                      alignItems: 'center', 
                      fontSize: '14px', 
                      fontWeight: '600', 
                      color: '#374151', 
                      marginBottom: '12px' 
                    }}>
                      <span style={{
                        width: '24px',
                        height: '24px',
                        backgroundColor: '#dcfce7',
                        borderRadius: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginRight: '8px',
                        fontSize: '12px'
                      }}>
                        📊
                      </span>
                      Input Variables ({suggestions.suggested_variables.length}):
                    </label>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        fontSize: '13px',
                        border: 'none',
                        borderRadius: '8px',
                        overflow: 'hidden'
                      }}>
                        <thead>
                          <tr style={{ backgroundColor: '#f8fafc' }}>
                            <th style={{ 
                              padding: '12px 16px', 
                              textAlign: 'left', 
                              fontWeight: '600',
                              color: '#374151',
                              borderBottom: '1px solid #e5e7eb',
                              minWidth: '80px'
                            }}>Cell</th>
                            <th style={{ 
                              padding: '12px 16px', 
                              textAlign: 'left', 
                              fontWeight: '600',
                              color: '#374151',
                              borderBottom: '1px solid #e5e7eb',
                              minWidth: '200px'
                            }}>Variable Name</th>
                            <th style={{ 
                              padding: '12px 16px', 
                              textAlign: 'right', 
                              fontWeight: '600',
                              color: '#374151',
                              borderBottom: '1px solid #e5e7eb',
                              minWidth: '120px'
                            }}>Current Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {suggestions.suggested_variables.map((variable, index) => (
                            <tr key={index} style={{ 
                              backgroundColor: index % 2 === 0 ? 'white' : '#fafbfc',
                              borderBottom: index < suggestions.suggested_variables.length - 1 ? '1px solid #f3f4f6' : 'none'
                            }}>
                              <td style={{ 
                                padding: '12px 16px', 
                                fontWeight: '600',
                                color: '#1f2937',
                                fontFamily: 'monospace'
                              }}>
                                {variable.cell_address}
                              </td>
                              <td style={{ 
                                padding: '12px 16px', 
                                color: '#374151',
                                fontWeight: '500'
                              }}>
                                {variable.variable_name || `Variable_${variable.cell_address}`}
                              </td>
                              <td style={{ 
                                padding: '12px 16px', 
                                textAlign: 'right',
                                color: '#059669',
                                fontWeight: '600',
                                fontFamily: 'monospace'
                              }}>
                                {variable.current_value !== null && variable.current_value !== undefined ? 
                                  (typeof variable.current_value === 'number' ? 
                                    variable.current_value.toLocaleString() : 
                                    variable.current_value
                                  ) : 'N/A'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* Output Target Variables Table */}
              {suggestions.suggested_targets && suggestions.suggested_targets.length > 0 && (
                <div style={{ marginBottom: '20px' }}>
                  <div style={{
                    backgroundColor: 'white',
                    padding: '20px',
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
                  }}>
                    <label style={{ 
                      display: 'flex',
                      alignItems: 'center', 
                      fontSize: '14px', 
                      fontWeight: '600', 
                      color: '#374151', 
                      marginBottom: '12px' 
                    }}>
                      <span style={{
                        width: '24px',
                        height: '24px',
                        backgroundColor: '#fecaca',
                        borderRadius: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginRight: '8px',
                        fontSize: '12px'
                      }}>
                        🎯
                      </span>
                      Output Target Variables ({suggestions.suggested_targets.length}):
                    </label>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        fontSize: '13px',
                        border: 'none',
                        borderRadius: '8px',
                        overflow: 'hidden'
                      }}>
                        <thead>
                          <tr style={{ backgroundColor: '#f8fafc' }}>
                            <th style={{ 
                              padding: '12px 16px', 
                              textAlign: 'left', 
                              fontWeight: '600',
                              color: '#374151',
                              borderBottom: '1px solid #e5e7eb',
                              minWidth: '80px'
                            }}>Cell</th>
                            <th style={{ 
                              padding: '12px 16px', 
                              textAlign: 'left', 
                              fontWeight: '600',
                              color: '#374151',
                              borderBottom: '1px solid #e5e7eb',
                              minWidth: '200px'
                            }}>Target Name</th>
                            <th style={{ 
                              padding: '12px 16px', 
                              textAlign: 'right', 
                              fontWeight: '600',
                              color: '#374151',
                              borderBottom: '1px solid #e5e7eb',
                              minWidth: '120px'
                            }}>Current Value</th>
                            <th style={{ 
                              padding: '12px 16px', 
                              textAlign: 'left', 
                              fontWeight: '600',
                              color: '#374151',
                              borderBottom: '1px solid #e5e7eb',
                              minWidth: '200px'
                            }}>Formula</th>
                          </tr>
                        </thead>
                        <tbody>
                          {suggestions.suggested_targets.map((target, index) => (
                            <tr key={index} style={{ 
                              backgroundColor: index % 2 === 0 ? 'white' : '#fafbfc',
                              borderBottom: index < suggestions.suggested_targets.length - 1 ? '1px solid #f3f4f6' : 'none'
                            }}>
                              <td style={{ 
                                padding: '12px 16px', 
                                fontWeight: '600',
                                color: '#1f2937',
                                fontFamily: 'monospace'
                              }}>
                                {target.cell_address}
                              </td>
                              <td style={{ 
                                padding: '12px 16px', 
                                color: '#374151',
                                fontWeight: '500'
                              }}>
                                {target.variable_name || target.target_name || `Target_${target.cell_address}`}
                              </td>
                              <td style={{ 
                                padding: '12px 16px', 
                                textAlign: 'right',
                                color: '#dc2626',
                                fontWeight: '600',
                                fontFamily: 'monospace'
                              }}>
                                {target.current_value !== null && target.current_value !== undefined ? 
                                  (typeof target.current_value === 'number' ? 
                                    target.current_value.toLocaleString() : 
                                    target.current_value
                                  ) : 'N/A'}
                              </td>
                              <td style={{ 
                                padding: '12px 16px', 
                                color: '#6b7280',
                                fontFamily: 'monospace',
                                fontSize: '12px'
                              }}>
                                {target.formula || 'N/A'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}

          {/* Analysis Status */}
          <div style={{
            fontSize: '12px',
            color: '#6b7280',
            fontStyle: 'italic',
            borderTop: '1px solid #e5e7eb',
            paddingTop: '8px'
          }}>
            Analysis ID: {analysis.analysis_id} | Status: {analysis.status}
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
