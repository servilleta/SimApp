import React from 'react';
import SimpleSpreadsheet from './SimpleSpreadsheet';

const ExcelGridDisplay = () => {
  return (
    <div>
      <div style={{ padding: '20px' }}>
        <h4>Spreadsheet Grid</h4>
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#e8f5e8', 
          border: '1px solid #4CAF50', 
          borderRadius: '4px', 
          marginBottom: '20px',
          fontSize: '14px',
          color: '#2e7d2e'
        }}>
          ✅ <strong>Reliable Spreadsheet System Active</strong><br/>
          <span style={{ fontSize: '12px', opacity: 0.9 }}>
            Pure React implementation • No external dependencies • Always works
          </span>
      </div>
      </div>

      <SimpleSpreadsheet />
    </div>
  );
};

export default ExcelGridDisplay; 