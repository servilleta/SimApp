import React, { useEffect, useState, lazy, Suspense } from 'react';
import { useSelector } from 'react-redux';
import ErrorBoundary from '../components/common/ErrorBoundary';
import './SimulatePage.css';

// Lazy-load uploader to break circular compile deps
const ExcelUploader = lazy(() => import('../components/excel-parser/ExcelUploader'));

const SimulatePage = () => {
  // Get file info from Redux to determine if file is uploaded
  const { fileInfo } = useSelector(state => state.excel);
  const hasFileUploaded = !!fileInfo;

  const [currentUser, setCurrentUser] = useState(null);
  const [currentFileInfo, setCurrentFileInfo] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);

  useEffect(() => {
    if (isAuthenticated && user) {
      setCurrentUser(user);
      
      if (fileInfo) {
        setCurrentFileInfo(fileInfo);
      }
    }
  }, [isAuthenticated, user, fileInfo]);

  const stepData = [
    {
      number: "1",
      title: "Build a Formula-Correct Excel File",
      description: "Create an Excel file with your business model, including:",
      details: [
        "Input variables (costs, prices, quantities, etc.)",
        "Formulas that calculate your target outcomes",
        "Clear cell references and proper Excel functions",
        "Realistic data ranges for your variables"
      ],
      icon: "📊",
      tips: [
        "Use named ranges for important cells",
        "Keep formulas simple and well-documented", 
        "Test your model with sample values first"
      ]
    },
    {
      number: "2", 
      title: "Upload Your Excel File",
      description: "Our platform will automatically:",
      details: [
        "Detect input variables and their current values",
        "Identify formula cells as potential targets",
        "Parse your Excel structure and dependencies",
        "Show you an interactive preview"
      ],
      icon: "📤",
      tips: [
        "Supported formats: .xlsx, .xls",
        "File size limit: 10MB",
        "Complex files may take longer to process"
      ]
    },
    {
      number: "3",
      title: "Configure & Run Simulation", 
      description: "Set up your scenario analysis:",
      details: [
        "Choose which cells to vary (input variables)",
        "Set probability distributions for each variable",
        "Select target cells to analyze",
        "Run thousands of simulations instantly"
      ],
      icon: "⚡",
      tips: [
        "Start with 1,000 iterations for quick results",
        "Use 10,000+ iterations for final analysis",
        "Our Ultra engine provides fastest performance"
      ]
    },
    {
      number: "4",
      title: "Analyze & Export Results",
      description: "Get comprehensive insights:",
      details: [
        "Statistical summaries (mean, median, percentiles)",
        "Interactive charts and histograms", 
        "Sensitivity analysis and correlations",
        "Professional PDF reports for sharing"
      ],
      icon: "📈",
      tips: [
        "Use percentiles to understand risk ranges",
        "Check sensitivity analysis for key drivers",
        "Export PDFs for presentations and reports"
      ]
    }
  ];

  return (
    <div className="page-container" style={{ paddingTop: hasFileUploaded ? '2rem' : '1rem' }}>
      <div style={{ maxWidth: hasFileUploaded ? '100%' : '1200px', width: '100%', margin: '0 auto' }}>
        
        {/* Upload Section - AT THE TOP */}
        <div className={hasFileUploaded ? '' : 'upload-section card-braun'} style={{ 
          padding: hasFileUploaded ? '0' : '0',
          marginBottom: hasFileUploaded ? '0' : '2rem',
          marginTop: hasFileUploaded ? '0' : '1rem'
        }}>
          {!hasFileUploaded && (
            <div className="upload-header">
              <h2 className="text-headline">Upload Your Excel File</h2>
              <p className="text-subheadline">
                Drag and drop your .xlsx or .xls file, or click to browse.
                <br />
                <small className="text-tertiary">Our platform will automatically detect variables and formulas for simulation.</small>
              </p>
            </div>
          )}
          
          <div className={hasFileUploaded ? '' : 'upload-card-content'}>
            <Suspense fallback={
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p className="text-secondary">Loading Excel Uploader...</p>
              </div>
            }>
              <ErrorBoundary>
                <ExcelUploader />
              </ErrorBoundary>
            </Suspense>
          </div>
        </div>

        {/* Process Steps - BELOW UPLOAD */}
        {!hasFileUploaded && (
          <div className="process-steps">
            <h3 className="text-headline">How It Works</h3>
            <div className="steps-overview">
              <div className="step-item">
                <span className="step-number">1</span>
                <span className="step-text">Build Excel model</span>
              </div>
              <div className="step-arrow">→</div>
              <div className="step-item">
                <span className="step-number">2</span>
                <span className="step-text">Upload</span>
              </div>
              <div className="step-arrow">→</div>
              <div className="step-item">
                <span className="step-number">3</span>
                <span className="step-text">Select inputs and targets</span>
              </div>
              <div className="step-arrow">→</div>
              <div className="step-item">
                <span className="step-number">4</span>
                <span className="step-text">Simulate and export results</span>
              </div>
            </div>
          </div>
        )}

        {/* What is Simulation Section - BELOW STEPS */}
        {!hasFileUploaded && (
          <div className="concept-explanation">
            <h3 className="text-headline">What is Excel Simulation?</h3>
            <div className="explanation-grid braun-grid braun-grid-3">
              <div className="explanation-card card-braun hover-lift">
                <h4 className="concept-title">The Concept</h4>
                <p className="text-secondary">
                  Instead of single values, simulation runs your model 
                  thousands of times with different inputs to explore all possible outcomes and scenarios.
                </p>
              </div>
              
              <div className="explanation-card card-braun hover-lift">
                <h4 className="concept-title">Why Use It?</h4>
                <p className="text-secondary">
                  Test assumptions, explore possibilities, optimize performance, 
                  and make informed decisions with comprehensive scenario analysis.
                </p>
              </div>
              
              <div className="explanation-card card-braun hover-lift">
                <h4 className="concept-title">Our Advantage</h4>
                <p className="text-secondary">
                  Ultra-fast processing, automatic Excel integration, 
                  interactive charts, and professional reports - no coding required.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Example Files Section - MOVED TO BOTTOM */}
        {!hasFileUploaded && (
          <div className="example-files card-braun">
            <h3 className="text-primary">Don't have an Excel file ready?</h3>
            <p className="text-secondary">Download our example templates to get started quickly:</p>
            <div className="example-buttons braun-grid braun-grid-3">
              <button className="btn-braun-secondary" onClick={() => window.open('/examples/business-model.xlsx', '_blank')}>
                📊 Business Revenue Model
              </button>
              <button className="btn-braun-secondary" onClick={() => window.open('/examples/project-costs.xlsx', '_blank')}>
                💰 Project Cost Analysis
              </button>
              <button className="btn-braun-secondary" onClick={() => window.open('/examples/investment-returns.xlsx', '_blank')}>
                📈 Investment Returns
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default SimulatePage; 