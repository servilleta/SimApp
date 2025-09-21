import React, { useState, useEffect } from 'react';
// Note: AIResultsInsights component removed due to UI library dependencies
// The main AI functionality is available in SimulationResultsDisplay via AIResultsAnalysis

// Example integration with your existing results display
const SimulationResultsWithAI = ({ simulationId, results }) => {
    const [showAIInsights, setShowAIInsights] = useState(false);

    useEffect(() => {
        // Automatically show AI insights when simulation completes
        if (simulationId && results) {
            setShowAIInsights(true);
        }
    }, [simulationId, results]);

    // Handle exporting AI insights
    const handleExportInsights = (insights) => {
        // Create a comprehensive report
        const report = {
            simulation_id: simulationId,
            generated_at: new Date().toISOString(),
            executive_summary: insights.executive_summary,
            risk_assessment: insights.risk_assessment,
            opportunities: insights.opportunities,
            recommendations: insights.recommendations,
            key_insights: insights.key_insights,
            
            // Include statistical data
            statistical_results: results
        };

        // Export as JSON
        const blob = new Blob([JSON.stringify(report, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ai-insights-${simulationId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log('AI insights exported:', report);
    };

    return (
        <div className="space-y-6">
            {/* Your existing results display */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Statistical Summary Card */}
                <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4">Statistical Summary</h3>
                    {results && (
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span>Mean:</span>
                                <span className="font-medium">{results.mean?.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Median:</span>
                                <span className="font-medium">{results.median?.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Std Dev:</span>
                                <span className="font-medium">{results.std_dev?.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>5th Percentile:</span>
                                <span className="font-medium">{results.percentile_5?.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>95th Percentile:</span>
                                <span className="font-medium">{results.percentile_95?.toFixed(2)}</span>
                            </div>
                        </div>
                    )}
                </div>

                {/* Success Probability Card */}
                <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4">Success Metrics</h3>
                    {results && (
                        <div className="space-y-2">
                            <div className="text-3xl font-bold text-green-600">
                                {(results.success_probability * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-gray-600">Success Probability</div>
                            
                            <div className="mt-4">
                                <div className="text-lg font-semibold">
                                    {results.total_iterations?.toLocaleString()}
                                </div>
                                <div className="text-sm text-gray-600">Total Iterations</div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Risk Metrics Card */}
                <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4">Risk Metrics</h3>
                    {results && (
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span>VaR (5%):</span>
                                <span className="font-medium text-red-600">
                                    {results.value_at_risk_5?.toFixed(2)}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Expected Shortfall:</span>
                                <span className="font-medium text-red-600">
                                    {results.expected_shortfall_5?.toFixed(2)}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Skewness:</span>
                                <span className="font-medium">
                                    {results.skewness?.toFixed(3)}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Kurtosis:</span>
                                <span className="font-medium">
                                    {results.kurtosis?.toFixed(3)}
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Your existing charts would go here */}
            <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-semibold mb-4">Distribution Chart</h3>
                {/* Your existing histogram/distribution chart component */}
                <div className="h-64 bg-gray-100 rounded flex items-center justify-center">
                    <span className="text-gray-500">Histogram Chart Component</span>
                </div>
            </div>

            {/* AI Insights Section */}
            {showAIInsights && simulationId && (
                <AIResultsInsights
                    simulationId={simulationId}
                    onExportInsights={handleExportInsights}
                    className="mt-6"
                />
            )}

            {/* Additional Actions */}
            <div className="flex space-x-4">
                <button 
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    onClick={() => {
                        // Your existing export functionality
                        console.log('Export results');
                    }}
                >
                    Export Results
                </button>
                
                <button 
                    className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                    onClick={() => {
                        // Your existing save functionality
                        console.log('Save simulation');
                    }}
                >
                    Save Simulation
                </button>
                
                {showAIInsights && (
                    <button 
                        className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
                        onClick={() => setShowAIInsights(!showAIInsights)}
                    >
                        {showAIInsights ? 'Hide' : 'Show'} AI Insights
                    </button>
                )}
            </div>
        </div>
    );
};

export default SimulationResultsWithAI;
