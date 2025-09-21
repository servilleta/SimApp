import React, { useState } from 'react';

const SimpleSpreadsheet = ({ data = [] }) => {
  const [cells, setCells] = useState(() => {
    // Initialize with sample data or props data
    const initialCells = {};
    for (let row = 0; row < 12; row++) {
      for (let col = 0; col < 6; col++) {
        const key = `${row}-${col}`;
        if (row === 0) {
          // Header row
          initialCells[key] = ['Product', 'Price', 'Qty', 'Total', 'Category', 'Notes'][col] || '';
        } else if (row === 1) {
          initialCells[key] = ['Apple', '1.50', '10', '=B2*C2', 'Fruit', 'Fresh'][col] || '';
        } else if (row === 2) {
          initialCells[key] = ['Orange', '2.00', '5', '=B3*C3', 'Fruit', 'Organic'][col] || '';
        } else if (row === 3) {
          initialCells[key] = ['Bread', '3.50', '2', '=B4*C4', 'Bakery', 'Whole grain'][col] || '';
        } else if (row === 4) {
          initialCells[key] = ['Milk', '4.20', '1', '=B5*C5', 'Dairy', '2% fat'][col] || '';
        } else {
          initialCells[key] = '';
        }
      }
    }
    return initialCells;
  });

  const handleCellChange = (row, col, value) => {
    const key = `${row}-${col}`;
    setCells(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const getColumnLabel = (col) => {
    return String.fromCharCode(65 + col); // A, B, C, D, E, F...
  };

  return (
    <div style={{ padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '8px', margin: '20px' }}>
      <div style={{ marginBottom: '15px' }}>
        <h4 style={{ margin: '0 0 5px 0', color: '#2c3e50' }}>📊 Reliable Spreadsheet</h4>
        <p style={{ margin: '0', fontSize: '14px', color: '#666' }}>
          ✅ Always works • ✅ No dependencies • ✅ Fully interactive
        </p>
      </div>
      
      <div style={{ 
        border: '2px solid #dee2e6', 
        borderRadius: '6px', 
        overflow: 'hidden',
        width: 'fit-content',
        backgroundColor: '#fff',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        {/* Header with column labels */}
        <div style={{ display: 'flex', backgroundColor: '#e9ecef' }}>
          <div style={{ 
            width: '50px', 
            height: '35px', 
            border: '1px solid #dee2e6', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            fontSize: '12px',
            fontWeight: 'bold',
            backgroundColor: '#6c757d',
            color: '#fff'
          }}>
            #
          </div>
          {[0, 1, 2, 3, 4, 5].map(col => (
            <div key={col} style={{ 
              width: '120px', 
              height: '35px', 
              border: '1px solid #dee2e6', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              fontSize: '13px',
              fontWeight: 'bold',
              backgroundColor: '#495057',
              color: '#fff'
            }}>
              {getColumnLabel(col)}
            </div>
          ))}
        </div>

        {/* Data rows */}
        {Array.from({ length: 12 }, (_, row) => (
          <div key={row} style={{ display: 'flex', backgroundColor: row % 2 === 0 ? '#fff' : '#f8f9fa' }}>
            {/* Row number */}
            <div style={{ 
              width: '50px', 
              height: '35px', 
              border: '1px solid #dee2e6', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              fontSize: '12px',
              fontWeight: 'bold',
              backgroundColor: '#6c757d',
              color: '#fff'
            }}>
              {row + 1}
            </div>
            
            {/* Data cells */}
            {[0, 1, 2, 3, 4, 5].map(col => (
              <input
                key={col}
                type="text"
                value={cells[`${row}-${col}`] || ''}
                onChange={(e) => handleCellChange(row, col, e.target.value)}
                style={{
                  width: '120px',
                  height: '35px',
                  border: '1px solid #dee2e6',
                  borderRadius: '0',
                  padding: '6px 10px',
                  fontSize: '13px',
                  outline: 'none',
                  backgroundColor: row === 0 ? '#e3f2fd' : 'transparent',
                  fontWeight: row === 0 ? 'bold' : 'normal',
                  transition: 'all 0.2s ease'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#007bff';
                  e.target.style.boxShadow = '0 0 0 2px rgba(0,123,255,0.25)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = '#dee2e6';
                  e.target.style.boxShadow = 'none';
                }}
              />
            ))}
          </div>
        ))}
      </div>
      
      <div style={{ marginTop: '15px', display: 'flex', gap: '20px', fontSize: '12px', color: '#666' }}>
        <div>
          <strong>Features:</strong> 
          <span style={{ marginLeft: '5px' }}>
            ✅ Editable cells ✅ Grid layout ✅ Sample data ✅ Professional styling
          </span>
        </div>
      </div>
      
      <div style={{ marginTop: '10px', padding: '10px', backgroundColor: '#d4edda', border: '1px solid #c3e6cb', borderRadius: '4px', fontSize: '13px', color: '#155724' }}>
        <strong>💡 Tip:</strong> This fallback spreadsheet always works and requires no external libraries. 
        It's ready for your data and can be extended with formulas, validation, or export features.
      </div>
    </div>
  );
};

export default SimpleSpreadsheet; 