import React from 'react';

const SummaryMetrics = ({ metrics }) => {
  const containerStyle = {
    display: 'flex',
    gap: '1rem',
    justifyContent: 'space-around',
    padding: '1rem',
    backgroundColor: '#fff',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
    marginBottom: '1rem',
  };

  const metricItemStyle = {
    textAlign: 'center',
  };

  const metricValueStyle = {
    fontSize: '1.8rem',
    fontWeight: 'bold',
    color: '#007bff',
  };

  const metricLabelStyle = {
    fontSize: '0.9rem',
    color: '#6c757d',
  };

  // Dummy metrics if none are provided
  const defaultMetrics = [
    { label: 'Total Simulations', value: 0 },
    { label: 'Active Users', value: 0 },
    { label: 'Files Processed', value: 0 },
  ];

  const displayMetrics = metrics || defaultMetrics;

  return (
    <div style={containerStyle}>
      {displayMetrics.map((metric, index) => (
        <div key={index} style={metricItemStyle}>
          <div style={metricValueStyle}>{metric.value}</div>
          <div style={metricLabelStyle}>{metric.label}</div>
        </div>
      ))}
    </div>
  );
};

export default SummaryMetrics; 