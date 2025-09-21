import React from 'react';

const AIAnalysisStatus = ({ 
  isAnalyzing, 
  isLoadingSuggestions, 
  error, 
  analysis,
  suggestions,
  onRetry 
}) => {
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <span className="text-red-400 text-xl">❌</span>
          </div>
          <div className="ml-3 flex-1">
            <h3 className="text-sm font-medium text-red-800">
              Analysis Error
            </h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
            {onRetry && (
              <div className="mt-3">
                <button
                  onClick={onRetry}
                  className="bg-red-600 text-white px-3 py-1 text-sm rounded hover:bg-red-700 transition-colors"
                >
                  Retry Analysis
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">
              🔬 Analyzing Excel Model...
            </h3>
            <p className="mt-1 text-sm text-blue-700">
              AI is examining your Excel model structure, formulas, and dependencies. This may take 1-2 minutes for complex models.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (isLoadingSuggestions) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <div className="animate-pulse text-yellow-600 text-xl">🧠</div>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-yellow-800">
              🤖 Generating AI Suggestions...
            </h3>
            <p className="mt-1 text-sm text-yellow-700">
              DeepSeek AI is analyzing your model and generating intelligent variable suggestions with distribution parameters.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (analysis && !suggestions) {
    return (
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <span className="text-orange-400 text-xl">⏳</span>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-orange-800">
              Analysis Complete - Loading Suggestions
            </h3>
            <p className="mt-1 text-sm text-orange-700">
              Model analysis finished. Retrieving AI-generated variable suggestions...
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (suggestions) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <span className="text-green-400 text-xl">✅</span>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-green-800">
              AI Analysis Complete!
            </h3>
            <p className="mt-1 text-sm text-green-700">
              Successfully analyzed your Excel model and generated {suggestions.suggested_variables?.length || 0} variable suggestions.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <span className="text-gray-400 text-xl">🤖</span>
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-gray-800">
            AI Excel Analysis Ready
          </h3>
          <p className="mt-1 text-sm text-gray-700">
            Click the AI button to analyze your Excel model and get intelligent variable suggestions for Monte Carlo simulation.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AIAnalysisStatus;

