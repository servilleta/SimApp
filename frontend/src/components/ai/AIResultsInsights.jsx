import React from 'react';

const AIResultsInsights = ({ modelInsights, confidenceScore, analysisId }) => {
  if (!modelInsights) {
    return (
      <div className="bg-gray-50 p-4 rounded-lg">
        <p className="text-gray-600">No model insights available.</p>
      </div>
    );
  }

  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const formatConfidence = (score) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Analysis Header */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">🧠 AI Model Insights</h3>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(confidenceScore || 0)}`}>
            Confidence: {formatConfidence(confidenceScore || 0)}
          </div>
        </div>
        
        {analysisId && (
          <p className="text-xs text-gray-500 mb-3">Analysis ID: {analysisId}</p>
        )}

        {/* Model Type and Complexity */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="bg-blue-50 p-3 rounded-md">
            <h4 className="text-sm font-medium text-blue-900 mb-1">📊 Model Type</h4>
            <p className="text-sm text-blue-800 capitalize">
              {modelInsights.model_type?.replace(/_/g, ' ') || 'Unknown'}
            </p>
          </div>
          
          <div className="bg-purple-50 p-3 rounded-md">
            <h4 className="text-sm font-medium text-purple-900 mb-1">🎯 Complexity Score</h4>
            <p className="text-sm text-purple-800">
              {typeof modelInsights.complexity_score === 'number' 
                ? `${(modelInsights.complexity_score * 100).toFixed(1)}%`
                : 'N/A'
              }
            </p>
          </div>
        </div>
      </div>

      {/* Key Drivers */}
      {modelInsights.key_drivers && modelInsights.key_drivers.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-md font-medium text-gray-900 mb-3">🚀 Key Drivers</h4>
          <div className="flex flex-wrap gap-2">
            {modelInsights.key_drivers.map((driver, index) => (
              <span key={index} className="inline-block bg-green-100 text-green-800 text-sm px-3 py-1 rounded-full">
                {driver}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Output Variables */}
      {modelInsights.output_variables && modelInsights.output_variables.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-md font-medium text-gray-900 mb-3">📈 Output Variables</h4>
          <div className="flex flex-wrap gap-2">
            {modelInsights.output_variables.map((output, index) => (
              <span key={index} className="inline-block bg-orange-100 text-orange-800 text-sm px-3 py-1 rounded-full">
                {output}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Potential Risks */}
      {modelInsights.potential_risks && modelInsights.potential_risks.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-md font-medium text-gray-900 mb-3">⚠️ Potential Risks</h4>
          <div className="space-y-2">
            {modelInsights.potential_risks.map((risk, index) => (
              <div key={index} className="flex items-center space-x-2">
                <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                <span className="text-sm text-gray-700">{risk}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dependencies */}
      {modelInsights.dependencies && Object.keys(modelInsights.dependencies).length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-md font-medium text-gray-900 mb-3">🔗 Dependencies</h4>
          <div className="space-y-2">
            {Object.entries(modelInsights.dependencies).map(([key, deps], index) => (
              <div key={index} className="text-sm">
                <span className="font-medium text-gray-700">{key}: </span>
                <span className="text-gray-600">
                  {Array.isArray(deps) ? deps.join(', ') : String(deps)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional Metadata */}
      {(modelInsights.total_cells || modelInsights.formula_cells || modelInsights.input_cells) && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-md font-medium text-gray-900 mb-3">📊 Model Statistics</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {modelInsights.total_cells && (
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{modelInsights.total_cells}</div>
                <div className="text-sm text-gray-600">Total Cells</div>
              </div>
            )}
            {modelInsights.formula_cells && (
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{modelInsights.formula_cells}</div>
                <div className="text-sm text-gray-600">Formula Cells</div>
              </div>
            )}
            {modelInsights.input_cells && (
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{modelInsights.input_cells}</div>
                <div className="text-sm text-gray-600">Input Cells</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AIResultsInsights;

