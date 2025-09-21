import React, { useMemo } from 'react';
import './ModelSummaryBar.css';

const ModelSummaryBar = ({ 
  selectedSheetData, 
  inputVariables = [], 
  resultCells = [], 
  sheets = [],
  fileInfo = null
}) => {
  const stats = useMemo(() => {
    if (!selectedSheetData || !selectedSheetData.grid_data) {
      return {
        sheets: sheets.length || 0,
        totalCells: 0,
        formulas: 0,
        inputs: inputVariables.length,
        outputs: resultCells.length
      };
    }

    let totalCells = 0;
    let formulas = 0;
    let potentialInputs = 0;

    // Count actual cells and formulas
    selectedSheetData.grid_data.forEach((row, rowIndex) => {
      if (row) {
        row.forEach((cell, colIndex) => {
          if (cell && (cell.value !== null && cell.value !== undefined && cell.value !== '')) {
            totalCells++;
            if (cell.formula || cell.is_formula_cell) {
              formulas++;
            } else if (typeof cell.value === 'number' && cell.value !== 0) {
              potentialInputs++;
            }
          }
        });
      }
    });
    
    return {
      sheets: sheets.length || 1,
      totalCells,
      formulas,
      inputs: inputVariables.length > 0 ? inputVariables.length : potentialInputs
    };
  }, [selectedSheetData, inputVariables.length, resultCells.length, sheets.length]);

  const formatFileSize = (bytes) => {
    if (!bytes || bytes === 0) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFileDisplayInfo = () => {
    if (!fileInfo) return "No file loaded";
    
    const fileName = fileInfo.filename || fileInfo.name || "Unknown file";
    const fileSize = formatFileSize(fileInfo.file_size);
    
    return `${fileName}${fileSize ? ` • ${fileSize}` : ""}`;
  };

  return (
    <div className="model-summary-bar">
      <div className="model-summary-content">
        <div className="model-description">
          <span className="model-icon">📊</span>
          <span className="model-text">{getFileDisplayInfo()}</span>
        </div>
        
        <div className="model-stats">
          <div className="stat-group">
            <div className="stat-item">
              <span className="stat-value">{stats.sheets}</span>
              <span className="stat-label">Sheets</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{stats.totalCells}</span>
              <span className="stat-label">Cells</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{stats.formulas}</span>
              <span className="stat-label">Formulas</span>
            </div>
            <div className="stat-item inputs">
              <span className="stat-value">{stats.inputs}</span>
              <span className="stat-label">Inputs</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelSummaryBar;
