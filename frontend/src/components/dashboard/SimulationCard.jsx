import React from 'react';

const SimulationCard = ({ simulation, onLoad, onDelete, onViewResults, loading }) => {
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getConfigSummary = (config) => {
    const inputVars = config?.inputVariables?.length || 0;
    const resultCells = config?.resultCells?.length || 0;
    const iterations = config?.iterations || 1000;
    
    return { inputVars, resultCells, iterations };
  };

  const { inputVars, resultCells, iterations } = getConfigSummary(simulation.simulation_config);

  return (
    <div className="simulation-card" style={{
      backgroundColor: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '8px',
      padding: '1.5rem',
      transition: 'all 0.2s ease',
      cursor: 'pointer',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
      ':hover': {
        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
        borderColor: 'var(--color-medium-grey)'
      }
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.1)';
      e.currentTarget.style.borderColor = 'var(--color-medium-grey)';
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.05)';
      e.currentTarget.style.borderColor = 'var(--color-border-light)';
    }}
    >
      <div className="card-header" style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: '1rem'
      }}>
        <h3 style={{
          margin: 0,
          color: 'var(--color-charcoal)',
          fontSize: '1.1rem',
          fontWeight: '600',
          lineHeight: '1.3',
          flex: 1,
          marginRight: '1rem'
        }}>
          {simulation.name}
        </h3>
        <span style={{
          fontSize: '0.8rem',
          color: 'var(--color-medium-grey)',
          whiteSpace: 'nowrap'
        }}>
          {formatDate(simulation.created_at)}
        </span>
      </div>
      
      <div className="card-body" style={{ marginBottom: '1.5rem' }}>
        {simulation.description && (
          <p style={{
            margin: '0 0 1rem 0',
            color: 'var(--color-medium-grey)',
            fontSize: '0.9rem',
            lineHeight: '1.4'
          }}>
            {simulation.description}
          </p>
        )}
        
        <div className="metadata" style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-medium-grey)'
        }}>
          <span style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem',
            padding: '0.25rem 0.5rem',
            backgroundColor: 'var(--color-warm-white)',
            borderRadius: '4px',
            border: '1px solid var(--color-border-light)'
          }}>
            📄 {simulation.original_filename}
          </span>
          <span style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem',
            padding: '0.25rem 0.5rem',
            backgroundColor: 'var(--color-warm-white)',
            borderRadius: '4px',
            border: '1px solid var(--color-border-light)'
          }}>
            🎯 {resultCells} target{resultCells !== 1 ? 's' : ''}
          </span>
          <span style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem',
            padding: '0.25rem 0.5rem',
            backgroundColor: 'var(--color-warm-white)',
            borderRadius: '4px',
            border: '1px solid var(--color-border-light)'
          }}>
            📊 {inputVars} variable{inputVars !== 1 ? 's' : ''}
          </span>
          <span style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem',
            padding: '0.25rem 0.5rem',
            backgroundColor: 'var(--color-warm-white)',
            borderRadius: '4px',
            border: '1px solid var(--color-border-light)'
          }}>
            🔄 {iterations.toLocaleString()} iterations
          </span>
        </div>
      </div>
      
      <div className="card-actions" style={{
        display: 'flex',
        gap: '0.5rem',
        justifyContent: 'flex-end'
      }}>
        <button 
          onClick={(e) => {
            e.stopPropagation();
            onLoad();
          }}
          disabled={loading}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: 'var(--color-braun-orange)',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '0.85rem',
            fontWeight: '500',
            opacity: loading ? 0.6 : 1,
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (!loading) {
              e.target.style.backgroundColor = 'var(--color-braun-orange-dark)';
            }
          }}
          onMouseLeave={(e) => {
            if (!loading) {
              e.target.style.backgroundColor = 'var(--color-braun-orange)';
            }
          }}
        >
          📂 Load
        </button>
        
        <button 
          onClick={(e) => {
            e.stopPropagation();
            onViewResults();
          }}
          disabled={loading}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: 'var(--color-white)',
            color: 'var(--color-charcoal)',
            border: '1px solid var(--color-border-light)',
            borderRadius: '6px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '0.85rem',
            fontWeight: '500',
            opacity: loading ? 0.6 : 1,
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (!loading) {
              e.target.style.backgroundColor = 'var(--color-warm-white)';
              e.target.style.borderColor = 'var(--color-medium-grey)';
            }
          }}
          onMouseLeave={(e) => {
            if (!loading) {
              e.target.style.backgroundColor = 'var(--color-white)';
              e.target.style.borderColor = 'var(--color-border-light)';
            }
          }}
        >
          📊 Results
        </button>
        
        <button 
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          disabled={loading}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: 'var(--color-white)',
            color: 'var(--color-error)',
            border: '1px solid var(--color-error-border)',
            borderRadius: '6px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '0.85rem',
            fontWeight: '500',
            opacity: loading ? 0.6 : 1,
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (!loading) {
              e.target.style.backgroundColor = 'var(--color-error-bg)';
            }
          }}
          onMouseLeave={(e) => {
            if (!loading) {
              e.target.style.backgroundColor = 'var(--color-white)';
            }
          }}
        >
          🗑️ Delete
        </button>
      </div>
      
      {/* Results Preview - Placeholder for Phase 2 */}
      {simulation.latest_results && (
        <div className="results-preview" style={{
          marginTop: '1rem',
          padding: '0.75rem',
          backgroundColor: 'var(--color-success-bg)',
          border: '1px solid var(--color-success-border)',
          borderRadius: '6px',
          fontSize: '0.85rem'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ color: 'var(--color-success)' }}>
              ✅ Last run: {formatDate(simulation.latest_results.completed_at)}
            </span>
            <span style={{ color: 'var(--color-success)', fontWeight: '600' }}>
              Mean: {simulation.latest_results.mean?.toFixed(2)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimulationCard; 