import React, { useState, useEffect } from 'react';
import aiService from '../../services/aiService';

export default function AIResultsAnalysis({ simulationId, targetResult, isVisible, onAnalysisReceived }) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);

  // Auto-analyze when component becomes visible
  useEffect(() => {
    if (isVisible && simulationId && !analysis && !isAnalyzing) {
      handleAnalyze();
    }
  }, [isVisible, simulationId, analysis, isAnalyzing]);

  const handleAnalyze = async () => {
    if (!simulationId) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      console.log('🤖 [AIResultsAnalysis] Starting analysis for:', simulationId);
      const analysisResult = await aiService.analyzeResults(simulationId, true);
      setAnalysis(analysisResult);
      
      // Auto-fetch summary if analysis has ID
      if (analysisResult.analysis_id) {
        setIsLoadingSummary(true);
        try {
          const summaryResult = await aiService.getResultsSummary(analysisResult.analysis_id);
          setSummary(summaryResult);
          
          // Notify parent component about analysis
          if (onAnalysisReceived) {
            onAnalysisReceived({
              analysis: analysisResult,
              summary: summaryResult
            });
          }
        } catch (summaryError) {
          console.error('🤖 [AIResultsAnalysis] Summary error:', summaryError);
          // Don't show error for summary, it's not critical
        } finally {
          setIsLoadingSummary(false);
        }
      }
    } catch (error) {
      console.error('🤖 [AIResultsAnalysis] Analysis error:', error);
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
        <h4 style={{
          margin: 0,
          fontSize: '16px',
          fontWeight: '600',
          color: '#1e293b',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          🤖 AI Results Analysis
        </h4>
        
        {!analysis && !isAnalyzing && (
          <button
            onClick={handleAnalyze}
            style={{
              padding: '6px 12px',
              backgroundColor: '#10b981',
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
            borderTop: '2px solid #10b981',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }}></div>
          Analyzing simulation results with AI...
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
          {/* Key Insights */}
          {analysis.insights && analysis.insights.length > 0 && (
            <div style={{ marginBottom: '16px' }}>
              <h5 style={{ margin: '0 0 8px 0', fontSize: '14px', fontWeight: '600', color: '#374151' }}>
                💡 Key Insights
              </h5>
              <div style={{
                backgroundColor: 'white',
                padding: '12px',
                borderRadius: '6px',
                border: '1px solid #e5e7eb'
              }}>
                <ul style={{ margin: '0', paddingLeft: '20px', color: '#4b5563' }}>
                  {analysis.insights.map((insight, index) => (
                    <li key={index} style={{ marginBottom: '4px' }}>
                      {insight}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Risk Factors */}
          {analysis.risk_factors && analysis.risk_factors.length > 0 && (
            <div style={{ marginBottom: '16px' }}>
              <h5 style={{ margin: '0 0 8px 0', fontSize: '14px', fontWeight: '600', color: '#dc2626' }}>
                ⚠️ Risk Factors
              </h5>
              <div style={{
                backgroundColor: 'white',
                padding: '12px',
                borderRadius: '6px',
                border: '1px solid #fee2e2'
              }}>
                <ul style={{ margin: '0', paddingLeft: '20px', color: '#991b1b' }}>
                  {analysis.risk_factors.map((risk, index) => (
                    <li key={index} style={{ marginBottom: '4px' }}>
                      {risk}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Executive Summary */}
          {isLoadingSummary && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              color: '#6b7280',
              fontSize: '14px',
              marginBottom: '16px'
            }}>
              <div style={{
                width: '14px',
                height: '14px',
                border: '2px solid #e5e7eb',
                borderTop: '2px solid #10b981',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></div>
              Loading executive summary...
            </div>
          )}

          {summary && (
            <div style={{ marginBottom: '16px' }}>
              <h5 style={{ margin: '0 0 8px 0', fontSize: '14px', fontWeight: '600', color: '#374151' }}>
                📊 Executive Summary
              </h5>
              <div style={{
                backgroundColor: 'white',
                padding: '12px',
                borderRadius: '6px',
                border: '1px solid #e5e7eb'
              }}>
                {summary.summary && (
                  <div style={{ marginBottom: '12px' }}>
                    <p style={{ margin: '0', color: '#4b5563', lineHeight: '1.6' }}>
                      {summary.summary}
                    </p>
                  </div>
                )}

                {/* Key Metrics Summary */}
                {summary.key_metrics && (
                  <div style={{ marginBottom: '12px' }}>
                    <h6 style={{ margin: '0 0 6px 0', fontSize: '13px', fontWeight: '600', color: '#059669' }}>
                      📈 Key Metrics
                    </h6>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                      gap: '8px',
                      fontSize: '12px'
                    }}>
                      {Object.entries(summary.key_metrics).map(([metric, value]) => (
                        <div key={metric} style={{
                          backgroundColor: '#f0f9ff',
                          padding: '6px 8px',
                          borderRadius: '4px',
                          textAlign: 'center'
                        }}>
                          <div style={{ fontWeight: '500', color: '#0369a1', textTransform: 'capitalize' }}>
                            {metric.replace(/_/g, ' ')}
                          </div>
                          <div style={{ color: '#1e40af', fontWeight: '600' }}>
                            {typeof value === 'number' ? value.toLocaleString() : value}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {summary.recommendations && summary.recommendations.length > 0 && (
                  <div>
                    <h6 style={{ margin: '0 0 6px 0', fontSize: '13px', fontWeight: '600', color: '#7c2d12' }}>
                      🎯 Recommendations
                    </h6>
                    <ul style={{ margin: '0', paddingLeft: '16px', fontSize: '12px', color: '#92400e' }}>
                      {summary.recommendations.map((recommendation, index) => (
                        <li key={index} style={{ marginBottom: '2px' }}>
                          {recommendation}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
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
