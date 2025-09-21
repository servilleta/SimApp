import React from 'react';

const SheetTabs = ({ sheets, onSelectSheet, activeSheetName }) => {
  if (!sheets || sheets.length === 0) {
    // Don't render tabs if there are no sheets
    return null;
  }

  const tabStyle = {
    padding: '10px 15px',
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderBottom: 'none',
    marginRight: '5px',
    backgroundColor: '#f9f9f9',
    borderRadius: '4px 4px 0 0',
  };

  const activeTabStyle = {
    ...tabStyle,
    backgroundColor: '#fff',
    borderBottom: '1px solid #fff', // Make it look like it merges with the content below
    fontWeight: 'bold',
  };

  return (
    <div style={{ display: 'flex', marginBottom: '10px', borderBottom: '1px solid #ccc', paddingBottom: '0' }}>
      {sheets.map((sheet) => (
        <button
          key={sheet.sheet_name}
          onClick={() => onSelectSheet(sheet.sheet_name)}
          style={sheet.sheet_name === activeSheetName ? activeTabStyle : tabStyle}
        >
          {sheet.sheet_name}
        </button>
      ))}
    </div>
  );
};

export default SheetTabs; 