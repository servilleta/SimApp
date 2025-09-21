import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import './VariableDefinitionPopup.css';
import { AISmartVariableHelper } from '../ai';

const VariableDefinitionPopup = ({ 
  isOpen, 
  onClose, 
  cellAddress, 
  currentValue, 
  onSave,
  position,
  variableType = 'input', // 'input' or 'target'
  aiSuggestions = null // AI suggestions from parent component
}) => {
  const [variableName, setVariableName] = useState('');
  const [format, setFormat] = useState('decimal');
  const [decimalPlaces, setDecimalPlaces] = useState(2);
  const [minValue, setMinValue] = useState('');
  const [maxValue, setMaxValue] = useState('');
  const [likelyValue, setLikelyValue] = useState('');
  const [distribution, setDistribution] = useState('uniform');
  const [errors, setErrors] = useState({});

  // Get existing variables from Redux state
  const inputVariables = useSelector(state => state.simulationSetup?.inputVariables) || [];
  const resultCells = useSelector(state => state.simulationSetup?.resultCells) || [];
  const currentSheetName = useSelector(state => state.simulationSetup?.currentSheetName) || 'Sheet1';

  // Helper function to detect if the cell is formatted as percentage
  const detectPercentageFormat = (value) => {
    if (!value) return false;
    
    // Check if the value string contains '%' symbol
    const valueStr = String(value);
    if (valueStr.includes('%')) {
      return true;
    }
    
    // Check if it's a small decimal that could be a percentage (0.05 = 5%, 0.15 = 15%, etc.)
    const numValue = parseFloat(value);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 1 && numValue.toString().includes('.')) {
      // Additional heuristic: if it's a small decimal with 2-3 decimal places, likely a percentage
      const decimalPart = valueStr.split('.')[1];
      if (decimalPart && decimalPart.length <= 3) {
        return true;
      }
    }
    
    return false;
  };

  // Helper function to convert percentage string to decimal
  const parsePercentageValue = (value) => {
    if (!value) return '';
    const valueStr = String(value);
    if (valueStr.includes('%')) {
      // Remove % and convert to decimal
      const numValue = parseFloat(valueStr.replace('%', ''));
      return (numValue / 100).toString();
    }
    return value.toString();
  };

  // Helper function to convert decimal to percentage for display
  const formatAsPercentage = (value) => {
    if (!value || isNaN(parseFloat(value))) return '';
    const numValue = parseFloat(value);
    return (numValue * 100).toString();
  };

  useEffect(() => {
    if (isOpen && cellAddress) {
      // Check if there's an existing variable for this cell
      let existingVariable = null;
      
      if (variableType === 'input') {
        existingVariable = inputVariables.find(v => 
          v.name === cellAddress && (v.sheetName === currentSheetName || !v.sheetName)
        );
      } else {
        existingVariable = resultCells.find(v => 
          v.name === cellAddress && (v.sheetName === currentSheetName || !v.sheetName)
        );
      }
      
      if (existingVariable) {
        // Load existing variable data
        console.log('📝 Loading existing variable data:', existingVariable);
        setVariableName(existingVariable.display_name || existingVariable.variableName || existingVariable.cell || cellAddress);
        setFormat(existingVariable.format || 'decimal');
        setDecimalPlaces(existingVariable.decimalPlaces || 2);
        
        if (variableType === 'input' && existingVariable.min_value !== undefined) {
          const isPercentage = existingVariable.format === 'percentage';
          setMinValue(isPercentage ? formatAsPercentage(existingVariable.min_value) : existingVariable.min_value.toString());
          setMaxValue(isPercentage ? formatAsPercentage(existingVariable.max_value) : existingVariable.max_value.toString());
          setLikelyValue(isPercentage ? formatAsPercentage(existingVariable.most_likely) : existingVariable.most_likely.toString());
          setDistribution(existingVariable.distribution || 'triangular');
        }
      } else {
        // Set default variable name based on cell address
        setVariableName(cellAddress || '');
        
        if (currentValue !== null && currentValue !== undefined) {
          // Auto-detect format based on Excel cell format
          const isPercentageFormat = detectPercentageFormat(currentValue);
          console.log('🔍 Auto-detected format for', cellAddress, ':', isPercentageFormat ? 'percentage' : 'decimal', 'based on value:', currentValue);
          
          setFormat(isPercentageFormat ? 'percentage' : 'decimal');
          
          // Parse the value appropriately
          let numericValue;
          if (isPercentageFormat) {
            // Convert percentage to decimal for calculations, but display as percentage
            numericValue = parseFloat(parsePercentageValue(currentValue));
            if (!isNaN(numericValue)) {
              // Display values as percentages (multiply by 100)
              const percentageValue = numericValue * 100;
              setMinValue((percentageValue * 0.8).toString());
              setMaxValue((percentageValue * 1.2).toString());
              setLikelyValue(percentageValue.toString());
            }
          } else {
            // Regular decimal handling
            numericValue = parseFloat(currentValue);
            if (!isNaN(numericValue)) {
              setMinValue((numericValue * 0.8).toString());
              setMaxValue((numericValue * 1.2).toString());
              setLikelyValue(numericValue.toString());
            }
          }
          
          if (isNaN(numericValue)) {
            setMinValue('');
            setMaxValue('');
            setLikelyValue('');
            setFormat('decimal');
          }
        }
      }
    }
  }, [isOpen, currentValue, cellAddress, variableType, inputVariables, resultCells, currentSheetName]);

  const validateInputs = () => {
    const newErrors = {};
    
    if (!variableName.trim()) {
      newErrors.name = 'Variable name is required';
    }

    if (variableType === 'input') {
      const min = parseFloat(minValue);
      const max = parseFloat(maxValue);
      const likely = parseFloat(likelyValue);

      if (!minValue || isNaN(min)) {
        newErrors.min = 'Minimum value is required';
      }
      
      if (!maxValue || isNaN(max)) {
        newErrors.max = 'Maximum value is required';
      }

      if (!likelyValue || isNaN(likely)) {
        newErrors.likely = 'Likeliest value is required';
      }
      
      if (!newErrors.min && !newErrors.max && min >= max) {
        newErrors.range = 'Maximum must be greater than minimum';
      }

      if (!newErrors.min && !newErrors.likely && !newErrors.max && (likely < min || likely > max)) {
        newErrors.likelyRange = 'Likeliest value must be between minimum and maximum';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSave = () => {
    if (validateInputs()) {
      const variableData = {
        cell: cellAddress,
        name: cellAddress, // This stays as the cell reference for backend compatibility
        display_name: variableName.trim(), // This is the user-defined name for display
        variableName: variableName.trim(), // Additional field for compatibility
        format: format,
        decimalPlaces: format === 'decimal' ? parseInt(decimalPlaces) : 2
      };

      if (variableType === 'input') {
        // Convert percentage values to decimals for backend
        if (format === 'percentage') {
          variableData.min_value = parseFloat(minValue) / 100;
          variableData.max_value = parseFloat(maxValue) / 100;
          variableData.most_likely = parseFloat(likelyValue) / 100;
        } else {
          variableData.min_value = parseFloat(minValue);
          variableData.max_value = parseFloat(maxValue);
          variableData.most_likely = parseFloat(likelyValue);
        }
        variableData.distribution = distribution;
      }

      console.log('💾 Saving variable data:', variableData);
      console.log('🔍 Format:', format, 'Values being sent to backend:', {
        min: variableData.min_value,
        max: variableData.max_value,
        likely: variableData.most_likely
      });
      onSave(variableData);
      handleClose();
    }
  };

  const handleClose = () => {
    setVariableName('');
    setFormat('decimal');
    setDecimalPlaces(2);
    setMinValue('');
    setMaxValue('');
    setLikelyValue('');
    setDistribution('uniform');
    setErrors({});
    onClose();
  };

  const formatValue = (value) => {
    if (!value || isNaN(parseFloat(value))) return value;
    
    const numValue = parseFloat(value);
    if (format === 'percentage') {
      // Values are already in percentage form (e.g., 10 for 10%)
      return `${numValue.toFixed(decimalPlaces)}%`;
    } else {
      return numValue.toFixed(decimalPlaces);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      handleClose();
    }
  };

  const handleApplyAISuggestion = (suggestion) => {
    console.log('🤖 [VariableDefinitionPopup] Applying AI suggestion:', suggestion);
    
    // Apply the suggestion data
    if (suggestion.description) {
      setVariableName(suggestion.description);
    }
    
    // Apply distribution suggestions for input variables
    if (variableType === 'input' && suggestion.suggested_distribution) {
      const dist = suggestion.suggested_distribution;
      if (dist.type) {
        setDistribution(dist.type);
      }
      
      // Apply parameter suggestions if available
      if (dist.parameters) {
        if (dist.parameters.min !== undefined) {
          setMinValue(String(dist.parameters.min));
        }
        if (dist.parameters.max !== undefined) {
          setMaxValue(String(dist.parameters.max));
        }
        if (dist.parameters.most_likely !== undefined) {
          setLikelyValue(String(dist.parameters.most_likely));
        }
      }
    }
    
    console.log('🤖 [VariableDefinitionPopup] AI suggestion applied successfully');
  };

  if (!isOpen) return null;

  return (
    <>
      <div className="popup-overlay" onClick={handleClose} />
      <div 
        className="variable-definition-popup-landscape"
        style={{
          position: 'fixed',
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 1000
        }}
      >
        <div className={`popup-header-landscape ${variableType === 'input' ? 'input-header' : 'target-header'}`}>
          <h3>{variableType === 'input' ? 'Define Input Variable' : 'Define Target Cell'}</h3>
          <button className="popup-close" onClick={handleClose}>✕</button>
        </div>

        <div className="popup-content-landscape">
          {/* Left Column - Basic Info */}
          <div className="popup-column-left">
            <div className="cell-info-landscape">
              <div className="cell-badge">
                <span className="cell-label">Cell:</span>
                <span className="cell-address">{cellAddress}</span>
              </div>
              {currentValue !== null && currentValue !== undefined && (
                <div className="value-badge">
                  <span className="current-label">Value:</span>
                  <span className="current-value">{currentValue}</span>
                </div>
              )}
            </div>

            {/* AI Smart Variable Helper */}
            <AISmartVariableHelper
              aiSuggestions={aiSuggestions}
              cellAddress={cellAddress}
              currentValue={currentValue}
              variableType={variableType}
              onApplySuggestion={handleApplyAISuggestion}
              isVisible={true}
            />

            <div className="input-group-landscape">
              <label className="input-label-compact">Variable Name</label>
              <input
                type="text"
                value={variableName}
                onChange={(e) => setVariableName(e.target.value)}
                onKeyPress={handleKeyPress}
                className={`popup-input-compact ${errors.name ? 'error' : ''}`}
                placeholder="Enter name..."
                autoFocus
              />
              {errors.name && <span className="error-text-compact">{errors.name}</span>}
            </div>

            <div className="format-row">
              <div className="input-group-compact">
                <label className="input-label-compact">Format</label>
                <select
                  value={format}
                  onChange={(e) => setFormat(e.target.value)}
                  className="popup-select-compact"
                >
                  <option value="decimal">Decimal</option>
                  <option value="percentage">Percentage</option>
                </select>
              </div>
              <div className="input-group-compact">
                <label className="input-label-compact">Decimals</label>
                <input
                  type="number"
                  value={decimalPlaces}
                  onChange={(e) => setDecimalPlaces(Math.max(0, Math.min(10, parseInt(e.target.value) || 0)))}
                  className="popup-input-compact"
                  min="0"
                  max="10"
                />
              </div>
            </div>
          </div>

          {/* Right Column - Variable Parameters (only for input variables) */}
          {variableType === 'input' && (
            <div className="popup-column-right">
              <div className="values-row">
                <div className="input-group-compact">
                  <label className="input-label-compact">
                    Min Value{format === 'percentage' ? ' (%)' : ''}
                  </label>
                  <div style={{ position: 'relative' }}>
                    <input
                      type="number"
                      value={minValue}
                      onChange={(e) => setMinValue(e.target.value)}
                      onKeyPress={handleKeyPress}
                      className={`popup-input-compact ${errors.min ? 'error' : ''}`}
                      step="any"
                      placeholder={format === 'percentage' ? 'e.g., 8' : 'Enter value'}
                    />
                    {format === 'percentage' && (
                      <span style={{ 
                        position: 'absolute', 
                        right: '8px', 
                        top: '50%', 
                        transform: 'translateY(-50%)', 
                        color: '#666',
                        pointerEvents: 'none' 
                      }}>%</span>
                    )}
                  </div>
                  {errors.min && <span className="error-text-compact">{errors.min}</span>}
                </div>

                <div className="input-group-compact">
                  <label className="input-label-compact">
                    Most Likely{format === 'percentage' ? ' (%)' : ''}
                  </label>
                  <div style={{ position: 'relative' }}>
                    <input
                      type="number"
                      value={likelyValue}
                      onChange={(e) => setLikelyValue(e.target.value)}
                      onKeyPress={handleKeyPress}
                      className={`popup-input-compact ${errors.likely ? 'error' : ''}`}
                      step="any"
                      placeholder={format === 'percentage' ? 'e.g., 10' : 'Enter value'}
                    />
                    {format === 'percentage' && (
                      <span style={{ 
                        position: 'absolute', 
                        right: '8px', 
                        top: '50%', 
                        transform: 'translateY(-50%)', 
                        color: '#666',
                        pointerEvents: 'none' 
                      }}>%</span>
                    )}
                  </div>
                  {errors.likely && <span className="error-text-compact">{errors.likely}</span>}
                </div>

                <div className="input-group-compact">
                  <label className="input-label-compact">
                    Max Value{format === 'percentage' ? ' (%)' : ''}
                  </label>
                  <div style={{ position: 'relative' }}>
                    <input
                      type="number"
                      value={maxValue}
                      onChange={(e) => setMaxValue(e.target.value)}
                      onKeyPress={handleKeyPress}
                      className={`popup-input-compact ${errors.max ? 'error' : ''}`}
                      step="any"
                      placeholder={format === 'percentage' ? 'e.g., 12' : 'Enter value'}
                    />
                    {format === 'percentage' && (
                      <span style={{ 
                        position: 'absolute', 
                        right: '8px', 
                        top: '50%', 
                        transform: 'translateY(-50%)', 
                        color: '#666',
                        pointerEvents: 'none' 
                      }}>%</span>
                    )}
                  </div>
                  {errors.max && <span className="error-text-compact">{errors.max}</span>}
                </div>
              </div>

              <div className="input-group-landscape">
                <label className="input-label-compact">Distribution</label>
                <select
                  value={distribution}
                  onChange={(e) => setDistribution(e.target.value)}
                  className="popup-select-compact"
                >
                  <option value="uniform">Uniform</option>
                  <option value="normal">Normal</option>
                  <option value="triangular">Triangular</option>
                </select>
              </div>

              {(errors.range || errors.likelyRange) && (
                <div className="error-message-compact">
                  {errors.range || errors.likelyRange}
                </div>
              )}

              {minValue && maxValue && likelyValue && !errors.min && !errors.max && !errors.likely && !errors.range && !errors.likelyRange && (
                <div className="preview-compact">
                  Range: <strong>{formatValue(minValue)}</strong> to <strong>{formatValue(maxValue)}</strong>, most likely <strong>{formatValue(likelyValue)}</strong>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="popup-actions-landscape">
          <button className="popup-btn-landscape cancel" onClick={handleClose}>
            Cancel
          </button>
          <button 
            className="popup-btn-landscape save" 
            onClick={handleSave}
            disabled={Object.keys(errors).length > 0}
          >
            {variableType === 'input' ? 'Add Variable' : 'Add Target'}
          </button>
        </div>
      </div>
    </>
  );
};

export default VariableDefinitionPopup; 