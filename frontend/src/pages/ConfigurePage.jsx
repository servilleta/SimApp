import React, { lazy, Suspense } from 'react';
import ErrorBoundary from '../components/common/ErrorBoundary';
// import VariableConfigurator from '../components/simulation/VariableConfigurator'; // Advanced component
// import FormulaEditor from '../components/simulation/FormulaEditor'; // Advanced component

const ExcelUploader = lazy(() => import('../components/excel-parser/ExcelUploader'));

const ConfigurePage = () => {
  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Monte Carlo Simulation Setup</h1>
        <p className="page-subtitle">
          Upload your Excel file to begin configuring input variables, their distributions, and the output cell for simulation.
        </p>
      </div>
      
      <div className="card" style={{ padding: '2rem' }}>
        <Suspense fallback={
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <p style={{ color: 'var(--color-medium-grey)' }}>Loading Excel Uploader...</p>
          </div>
        }>
          <ErrorBoundary>
            <ExcelUploader />
          </ErrorBoundary>
        </Suspense>
      </div>
      
      <div className="card" style={{ 
        padding: '1.5rem', 
        marginTop: '1rem',
        backgroundColor: 'var(--color-warm-white)'
      }}>
        <h3 style={{ color: 'var(--color-charcoal)', marginBottom: '1rem' }}>
          Advanced Configuration
        </h3>
        <p style={{ color: 'var(--color-medium-grey)', fontSize: '0.9rem' }}>
          Advanced variable configuration and formula editing tools are planned for future releases.
          Currently, the platform automatically detects and configures variables from your Excel file.
        </p>
      </div>
      
      {/* 
        More advanced configuration could be broken down:
        <VariableConfigurator /> 
        <FormulaEditor /> 
      */}
    </div>
  );
};

export default ConfigurePage; 