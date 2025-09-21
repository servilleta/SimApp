import React from 'react';
import './CompactFileInfo.css';

const CompactFileInfo = ({ fileInfo, selectedSheetData }) => {
  if (!fileInfo) return null;

  // Calculate file size in KB/MB
  const formatFileSize = (bytes) => {
    if (!bytes || bytes === 0) return 'Unknown';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Get current sheet dimensions
  const getCurrentSheetDimensions = () => {
    if (!selectedSheetData || !selectedSheetData.grid_data) {
      return { rows: 0, cols: 0 };
    }
    const rows = selectedSheetData.grid_data.length;
    const cols = rows > 0 ? selectedSheetData.grid_data[0]?.length || 0 : 0;
    return { rows, cols };
  };

  const dimensions = getCurrentSheetDimensions();
  const sheetCount = fileInfo.sheet_names ? fileInfo.sheet_names.length : 1; // Show at least 1

  return (
    <div className="file-info-compact">
      <div className="file-info-content">
        <div className="primary-info">
          <span className="file-name">{fileInfo.filename}</span>
          <span className="file-size">{formatFileSize(fileInfo.file_size)}</span>
        </div>
        <div className="secondary-info">
          <span className="sheet-info">{sheetCount} sheet{sheetCount !== 1 ? 's' : ''}</span>
          <span className="dimensions">{dimensions.rows} × {dimensions.cols}</span>
          <span className="cell-count">{dimensions.rows * dimensions.cols} cells</span>
        </div>
      </div>
    </div>
  );
};

export default CompactFileInfo; 