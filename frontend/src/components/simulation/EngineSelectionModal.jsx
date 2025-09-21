import React, { useState, useEffect } from 'react';
import { X, Zap, Cpu, CheckCircle, AlertCircle } from 'lucide-react';
import './EngineSelectionModal.css';

const EngineSelectionModal = ({ isOpen, onClose, onEngineSelect, fileComplexity = {} }) => {
  const [selectedEngine, setSelectedEngine] = useState('ultra');
  const [engineRecommendation, setEngineRecommendation] = useState(null);

  // Engine specifications - ADDED: Ultra engine
  const engines = {
    enhanced: {
      id: 'enhanced',
      name: 'Enhanced GPU Engine',
      shortName: 'Enhanced',
      icon: <Zap className="engine-icon" />,
      architecture: 'GPU-Accelerated Hybrid',
      computeUnits: 'CUDA Cores + CPU Threads',
      memoryModel: 'GPU Memory Pool + RAM',
      maxCells: '10M+',
      maxFormulas: '1M+',
      maxIterations: '1M',
      avgSpeed: '50,000 iter/sec',
      memoryEfficiency: '85%',
      parallelization: 'Massive (1000+ threads)',
      status: 'RECOMMENDED',
      statusColor: 'success',
      scientificBasis: 'GPU-accelerated pseudorandom number generation with CURAND library, parallel formula evaluation using CUDA kernels',
      bestFor: 'Complex financial models, risk analysis, heavy calculations with medium to large datasets',
      limitations: 'Requires CUDA-compatible GPU, higher memory usage during processing',
      useCases: ['Trading simulations', 'Portfolio optimization', 'Complex derivatives', 'Medium-large Excel files'],
      pros: ['Fastest proven performance', 'GPU acceleration', 'Mature implementation', 'Advanced features'],
      cons: ['Requires GPU hardware', 'Higher memory usage', 'Complex deployment']
    },
    ultra: {
      id: 'ultra',
      name: 'Ultra Hybrid Engine',
      shortName: 'Ultra',
      icon: <Zap className="engine-icon" />,
      architecture: 'Next-Gen GPU-CPU Hybrid',
      computeUnits: 'CUDA Cores + Multi-Core CPU',
      memoryModel: 'Unified Memory + Smart Caching',
      maxCells: '100M+',
      maxFormulas: '10M+',
      maxIterations: '10M',
      avgSpeed: '100,000+ iter/sec',
      memoryEfficiency: '95%',
      parallelization: 'Massive GPU + CPU (2000+ threads)',
      status: 'NEXT-GEN',
      statusColor: 'success',
      scientificBasis: 'Research-validated GPU acceleration with complete dependency analysis, database-first architecture, and advanced Excel parsing',
      bestFor: 'All file sizes with maximum performance, reliability, and accuracy',
      limitations: 'Cutting-edge technology, requires modern hardware for optimal performance',
      useCases: ['Enterprise simulations', 'Large-scale analysis', 'Complex multi-sheet models', 'Research applications'],
      pros: ['Maximum performance', 'Complete dependency analysis', 'Database-first reliability', 'Scientific validation'],
      cons: ['Newest technology', 'Requires modern GPU for best performance']
    },
    standard: {
      id: 'standard',
      name: 'Standard CPU Engine',
      shortName: 'Standard',
      icon: <Cpu className="engine-icon" />,
      architecture: 'Multi-threaded CPU Processing',
      computeUnits: 'CPU Threads + Thread Pooling',
      memoryModel: 'Standard RAM Allocation',
      maxCells: '1M',
      maxFormulas: '100K',
      maxIterations: '100K',
      avgSpeed: '5,000 iter/sec',
      memoryEfficiency: '70%',
      parallelization: 'Thread-based (4-16 threads)',
      status: 'STABLE',
      statusColor: 'info',
      scientificBasis: 'Traditional Monte Carlo simulation with multi-threaded CPU processing and standard pseudorandom number generation',
      bestFor: 'Simple models, debugging, guaranteed compatibility, development and testing',
      limitations: 'Limited scalability, slower performance for complex models, CPU-only processing',
      useCases: ['Prototyping', 'Simple business models', 'Educational purposes', 'Compatibility testing'],
      pros: ['Universal compatibility', 'Reliable', 'Simple deployment', 'Good for testing'],
      cons: ['Limited scalability', 'Slower performance', 'No GPU acceleration']
    }
  };

  // Get engine recommendation based on file complexity
  useEffect(() => {
    if (fileComplexity && Object.keys(fileComplexity).length > 0) {
      const { formula_cells = 0, complexity_score = 0 } = fileComplexity;
      
      let recommended = 'standard';
      let reason = 'Standard engine is sufficient for small files with simple calculations.';
      
      // UPDATED: Simplified recommendation logic preferring Ultra for better multi-target support
      if (formula_cells > 500) {
        recommended = 'ultra';
        reason = 'Ultra hybrid engine provides excellent performance and multi-target support for medium to large files.';
      }
      
      setEngineRecommendation({
        recommended,
        reason,
        complexity_score,
        formula_cells
      });
      
      setSelectedEngine(recommended);
    }
  }, [fileComplexity]);

  const handleConfirm = () => {
    onEngineSelect(selectedEngine);
    onClose();
  };

  const getStatusBadge = (engine) => {
    const colors = {
      success: 'status-recommended',
      warning: 'status-beta',
      info: 'status-stable',
      secondary: 'status-experimental'
    };
    
    return (
      <div className={`engine-status ${colors[engine.statusColor]}`}>
        {engine.status}
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="engine-selection-overlay">
      <div className="engine-selection-modal">
        <div className="engine-selection-header">
          <div className="header-content">
            <h2 className="modal-title">🚀 Select Monte Carlo Engine</h2>
            <p className="modal-subtitle">Choose the optimal engine for your simulation requirements</p>
          </div>
          <button className="close-button" onClick={onClose}>
            <X size={24} />
          </button>
        </div>

        <div className="engine-selection-body">
          {/* File Analysis Section */}
          {engineRecommendation && (
            <div className="file-analysis-section">
              <h3 className="section-title">📊 File Analysis</h3>
              <div className="analysis-grid">
                <div className="analysis-item">
                  <span className="analysis-label">Formulas:</span>
                  <span className="analysis-value">{engineRecommendation.formula_cells}</span>
                </div>
                <div className="analysis-item">
                  <span className="analysis-label">Complexity Score:</span>
                  <span className="analysis-value">{engineRecommendation.complexity_score}</span>
                </div>
              </div>
              <div className="recommendation-banner">
                <CheckCircle className="recommendation-icon" />
                <div className="recommendation-text">
                  <strong>Recommended:</strong> {engines[engineRecommendation.recommended].name}
                  <br />
                  <span className="recommendation-reason">{engineRecommendation.reason}</span>
                </div>
              </div>
            </div>
          )}

          {/* Engine Selection Grid */}
          <div className="engines-section">
            <h3 className="section-title">🛠️ Available Engines</h3>
            <div className="engines-grid">
              {Object.values(engines).map((engine) => (
                <div 
                  key={engine.id}
                  className={`engine-card ${selectedEngine === engine.id ? 'selected' : ''}`}
                  onClick={() => setSelectedEngine(engine.id)}
                >
                  <div className="engine-card-header">
                    <div className="engine-icon-container">
                      {engine.icon}
                    </div>
                    <div className="engine-title-section">
                      <h4 className="engine-title">{engine.name}</h4>
                      {getStatusBadge(engine)}
                    </div>
                  </div>

                  <div className="engine-specs">
                    <div className="specs-row">
                      <span className="spec-label">Architecture:</span>
                      <span className="spec-value">{engine.architecture}</span>
                    </div>
                    <div className="specs-row">
                      <span className="spec-label">Max Formulas:</span>
                      <span className="spec-value">{engine.maxFormulas}</span>
                    </div>
                    <div className="specs-row">
                      <span className="spec-label">Avg Speed:</span>
                      <span className="spec-value">{engine.avgSpeed}</span>
                    </div>
                    <div className="specs-row">
                      <span className="spec-label">Memory Efficiency:</span>
                      <span className="spec-value">{engine.memoryEfficiency}</span>
                    </div>
                  </div>

                  <div className="engine-description">
                    <p className="engine-best-for">
                      <strong>Best for:</strong> {engine.bestFor}
                    </p>
                    <p className="engine-limitations">
                      <strong>Limitations:</strong> {engine.limitations}
                    </p>
                  </div>

                  <div className="engine-pros-cons">
                    <div className="pros">
                      <h5>✅ Pros:</h5>
                      <ul>
                        {engine.pros.map((pro, index) => (
                          <li key={index}>{pro}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="cons">
                      <h5>❌ Cons:</h5>
                      <ul>
                        {engine.cons.map((con, index) => (
                          <li key={index}>{con}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="engine-selection-footer">
          <div className="selected-engine-info">
            <strong>Selected:</strong> {engines[selectedEngine].name}
          </div>
          <div className="footer-buttons">
            <button className="cancel-button" onClick={onClose}>
              Cancel
            </button>
            <button className="confirm-button" onClick={handleConfirm}>
              Use {engines[selectedEngine].shortName} Engine
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EngineSelectionModal; 