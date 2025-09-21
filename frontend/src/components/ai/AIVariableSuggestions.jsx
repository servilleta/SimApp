import React from 'react';

const AIVariableSuggestions = ({ suggestions }) => {
  if (!suggestions || suggestions.length === 0) {
    return (
      <div className="bg-gray-50 p-4 rounded-lg">
        <p className="text-gray-600">No variable suggestions available.</p>
      </div>
    );
  }

  const formatDistributionType = (type) => {
    if (!type) return 'Not specified';
    return type.replace('DistributionType.', '').toLowerCase().replace('_', ' ');
  };

  const formatValue = (value) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') return value.toFixed(2);
    return String(value);
  };

  const getRiskColor = (risk) => {
    switch (risk?.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        🎯 AI Variable Suggestions ({suggestions.length})
      </h3>
      
      {suggestions.map((suggestion, index) => (
        <div key={index} className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
          {/* Header */}
          <div className="flex justify-between items-start mb-3">
            <div>
              <h4 className="font-medium text-gray-900">
                {suggestion.variable_name || `Variable_${suggestion.cell_address}`}
              </h4>
              <p className="text-sm text-gray-600">
                📍 Cell: {suggestion.cell_address} | Sheet: {suggestion.sheet_name}
              </p>
            </div>
            <span className={`px-2 py-1 text-xs font-medium rounded-full ${getRiskColor(suggestion.risk_category)}`}>
              {suggestion.risk_category || 'unknown'} risk
            </span>
          </div>

          {/* Current Value */}
          <div className="mb-3">
            <span className="text-sm font-medium text-gray-700">Current Value: </span>
            <span className="text-sm text-gray-900">{formatValue(suggestion.current_value)}</span>
          </div>

          {/* Distribution Information */}
          {suggestion.distribution && (
            <div className="bg-blue-50 p-3 rounded-md mb-3">
              <h5 className="text-sm font-medium text-blue-900 mb-2">📊 Suggested Distribution</h5>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="font-medium text-blue-800">Type: </span>
                  <span className="text-blue-700 capitalize">
                    {formatDistributionType(suggestion.distribution.distribution_type)}
                  </span>
                </div>
                {suggestion.distribution.min_value !== null && (
                  <div>
                    <span className="font-medium text-blue-800">Min: </span>
                    <span className="text-blue-700">{formatValue(suggestion.distribution.min_value)}</span>
                  </div>
                )}
                {suggestion.distribution.max_value !== null && (
                  <div>
                    <span className="font-medium text-blue-800">Max: </span>
                    <span className="text-blue-700">{formatValue(suggestion.distribution.max_value)}</span>
                  </div>
                )}
                {suggestion.distribution.most_likely !== null && (
                  <div>
                    <span className="font-medium text-blue-800">Most Likely: </span>
                    <span className="text-blue-700">{formatValue(suggestion.distribution.most_likely)}</span>
                  </div>
                )}
                {suggestion.distribution.mean !== null && suggestion.distribution.mean !== 0 && (
                  <div>
                    <span className="font-medium text-blue-800">Mean: </span>
                    <span className="text-blue-700">{formatValue(suggestion.distribution.mean)}</span>
                  </div>
                )}
                {suggestion.distribution.std_dev !== null && suggestion.distribution.std_dev !== 1 && (
                  <div>
                    <span className="font-medium text-blue-800">Std Dev: </span>
                    <span className="text-blue-700">{formatValue(suggestion.distribution.std_dev)}</span>
                  </div>
                )}
              </div>
              
              {/* AI Reasoning */}
              {suggestion.distribution.reasoning && (
                <div className="mt-2">
                  <span className="font-medium text-blue-800">AI Reasoning: </span>
                  <span className="text-blue-700 text-sm">{suggestion.distribution.reasoning}</span>
                </div>
              )}
            </div>
          )}

          {/* Business Justification */}
          {suggestion.business_justification && (
            <div className="bg-green-50 p-3 rounded-md">
              <h5 className="text-sm font-medium text-green-900 mb-1">💼 Business Justification</h5>
              <p className="text-sm text-green-800">{suggestion.business_justification}</p>
            </div>
          )}

          {/* Correlation Candidates */}
          {suggestion.correlation_candidates && suggestion.correlation_candidates.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <h5 className="text-sm font-medium text-gray-700 mb-1">🔗 Correlation Candidates</h5>
              <div className="flex flex-wrap gap-1">
                {suggestion.correlation_candidates.map((candidate, idx) => (
                  <span key={idx} className="inline-block bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded">
                    {candidate}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default AIVariableSuggestions;

