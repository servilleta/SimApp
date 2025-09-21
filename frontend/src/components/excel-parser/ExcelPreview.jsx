import React from 'react';
import { useSelector } from 'react-redux';

const ExcelPreview = () => {
  const { fileInfo, loading } = useSelector((state) => state.excel);

  const tableStyle = {
    width: '100%',
    borderCollapse: 'collapse',
    marginTop: '1rem',
    fontSize: '0.9rem',
  };

  const thTdStyle = {
    border: '1px solid #ddd',
    padding: '8px',
    textAlign: 'left',
  };

  const thStyle = {
    ...thTdStyle,
    backgroundColor: '#f2f2f2',
    fontWeight: 'bold',
  };

  if (loading === 'pending') {
    return <p>Loading preview...</p>;
  }

  if (!fileInfo || !fileInfo.preview || fileInfo.preview.length === 0) {
    return <p>No preview data available. Upload an Excel file to see a preview.</p>;
  }

  const headers = fileInfo.columns || Object.keys(fileInfo.preview[0] || {});

  return (
    <div>
      <h4>Preview of '{fileInfo.filename}' (First few rows)</h4>
      <table style={tableStyle}>
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={index} style={thStyle}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fileInfo.preview.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {headers.map((header, colIndex) => (
                <td key={colIndex} style={thTdStyle}>{row[header]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p>Total Rows: {fileInfo.row_count}, Total Columns: {headers.length}</p>
    </div>
  );
};

export default ExcelPreview; 