import React, { useState, useEffect } from 'react';

export default function AISmartVariableHelper({ 
  aiSuggestions, 
  cellAddress, 
  currentValue, 
  variableType,
  onApplySuggestion,
  isVisible 
}) {
  const [matchingSuggestion, setMatchingSuggestion] = useState(null);

  useEffect(() => {
    if (!aiSuggestions || !cellAddress) {
      setMatchingSuggestion(null);
      return;
    }

    // Find matching suggestion for the current cell
    const suggestions = variableType === 'input' 
      ? aiSuggestions.input_suggestions 
      : aiSuggestions.target_suggestions;

    if (suggestions) {
      const match = suggestions.find(s => s.cell_address === cellAddress);
      setMatchingSuggestion(match);
    }
  }, [aiSuggestions, cellAddress, variableType]);

  if (!isVisible || !matchingSuggestion) {
    return null;
  }

  const handleApplySuggestion = () => {
    if (onApplySuggestion && matchingSuggestion) {
      onApplySuggestion(matchingSuggestion);
    }
  };

  return (
    <div style={{
      backgroundColor: '#f0f9ff',
      border: '1px solid #0ea5e9',
      borderRadius: '8px',
      padding: '12px',
      margin: '8px 0',
      fontSize: '14px'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '8px'
      }}>
        <h4 style={{
          margin: 0,
          fontSize: '14px',
          fontWeight: '600',
          color: '#0369a1',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          🤖 AI Suggestion for {cellAddress}
        </h4>
        <button
          onClick={handleApplySuggestion}
          style={{
            padding: '4px 8px',
            backgroundColor: '#0ea5e9',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            fontSize: '12px',
            cursor: 'pointer',
            fontWeight: '500'
          }}
        >
          Apply
        </button>
      </div>

      <div style={{ color: '#0c4a6e', lineHeight: '1.4' }}>
        <div style={{ marginBottom: '6px' }}>
          <strong>Description:</strong> {matchingSuggestion.description}
        </div>
        
        {matchingSuggestion.business_justification && (
          <div style={{ 
            fontSize: '12px', 
            color: '#075985', 
            fontStyle: 'italic',
            backgroundColor: '#e0f2fe',
            padding: '6px 8px',
            borderRadius: '4px',
            marginTop: '6px'
          }}>
            💡 <strong>Why this matters:</strong> {matchingSuggestion.business_justification}
          </div>
        )}

        {/* Distribution suggestions for input variables */}
        {variableType === 'input' && matchingSuggestion.suggested_distribution && (
          <div style={{ 
            fontSize: '12px', 
            marginTop: '6px',
            color: '#164e63'
          }}>
            <strong>Suggested Distribution:</strong>
            <div style={{
              backgroundColor: '#e0f2fe',
              padding: '6px 8px',
              borderRadius: '4px',
              marginTop: '4px',
              fontFamily: 'monospace'
            }}>
              {matchingSuggestion.suggested_distribution.type} 
              {matchingSuggestion.suggested_distribution.parameters && 
                ` (${Object.entries(matchingSuggestion.suggested_distribution.parameters)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join(', ')})`
              }
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
