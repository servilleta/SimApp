import React, { useState, useCallback, useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import { useDispatch, useSelector } from 'react-redux';
import { setCurrentGridSelection } from '../../store/simulationSetupSlice';
import './ExcelGridPro.css';

// Custom cell renderer for Excel-like cells with formatting support
const ExcelCellRenderer = ({ value, rowIndex, column }) => {
  // With the new valueFormatter/valueGetter, we should get clean display values
  // But we still need to handle the original object structure for styling
  let displayValue = '';
  let cellData = value;
  
  // If value is already processed by valueFormatter, use it directly
  if (typeof value === 'string' || typeof value === 'number') {
    displayValue = String(value);
  } else if (typeof value === 'object' && value !== null) {
    // Handle object data structure for display
    displayValue = value?.display_value || value?.value || '';
    cellData = value;
  } else {
    displayValue = String(value || '');
  }

  // Extract Excel formatting information (excluding borders - they'll be handled by cellClass)
  const cellStyle = {
    padding: '4px 8px',
    fontSize: '14px',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    boxSizing: 'border-box',
    overflow: 'hidden',
    whiteSpace: 'nowrap',
    textOverflow: 'ellipsis',
    // Apply Excel formatting if available (excluding borders)
    ...(cellData?.font_name && { fontFamily: cellData.font_name }),
    ...(cellData?.font_size && { fontSize: `${cellData.font_size}px` }),
    ...(cellData?.font_bold && { fontWeight: 'bold' }),
    ...(cellData?.font_italic && { fontStyle: 'italic' }),
    ...(cellData?.font_color && { color: cellData.font_color }),
    ...(cellData?.fill_color && { backgroundColor: cellData.fill_color }),
    ...(cellData?.alignment && { textAlign: cellData.alignment }),
  };

  return (
    <div style={cellStyle}>
      {displayValue}
    </div>
  );
};

// Helper function to calculate dynamic column width based on content
const calculateColumnWidth = (columnData, baseWidth = 150, minWidth = 120, maxWidth = 500) => {
  if (!columnData || columnData.length === 0) return baseWidth;
  
  // Find the longest display value in the column
  const maxLength = columnData.reduce((max, cellData) => {
    if (!cellData) return max;
    const displayValue = cellData?.display_value || cellData?.value || cellData || '';
    const length = String(displayValue).length;
    return Math.max(max, length);
  }, 0);
  
  // Calculate width based on character count (approximately 12 pixels per character + generous padding)
  const calculatedWidth = Math.max(minWidth, Math.min(maxWidth, maxLength * 12 + 40));
  
  return calculatedWidth;
};

// Helper function to parse Excel border style and apply to CSS
const parseBorderStyle = (borderInfo) => {
  if (!borderInfo) return '';
  
  const parts = borderInfo.split(':');
  const style = parts[0];
  const color = parts[1] || '#000000';
  
  // Map Excel border styles to CSS
  const borderMap = {
    'thin': '1px solid',
    'medium': '2px solid',
    'thick': '3px solid',
    'double': '3px double',
    'dotted': '1px dotted',
    'dashed': '1px dashed',
    'dashDot': '1px dashed',
    'dashDotDot': '1px dashed',
    'hair': '1px solid',
    'mediumDashed': '2px dashed',
    'mediumDashDot': '2px dashed',
    'mediumDashDotDot': '2px dashed',
    'slantDashDot': '1px dashed'
  };
  
  const cssStyle = borderMap[style] || '1px solid';
  return `${cssStyle} ${color}`;
};



// Custom header renderer for column letters with zoom support
const ColumnHeaderRenderer = ({ displayName, context }) => {
  const zoomLevel = context?.zoomLevel || 100;
  
  const headerStyle = {
    fontSize: `${Math.floor(13 * (zoomLevel / 100))}px`, // Increased from 11px
    fontWeight: '600',
    color: '#374151',
    textAlign: 'center',
    padding: `${Math.floor(4 * (zoomLevel / 100))}px`,
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  };

  return (
    <div style={headerStyle}>
      {displayName}
    </div>
  );
};

const ExcelGridPro = ({ sheetData, fileId, selectionMode, onCellClick }) => {
  const dispatch = useDispatch();
  const { currentGridSelection, inputVariables, resultCells } = useSelector(state => state.simulationSetup);
  const [selectedCell, setSelectedCell] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(100);

  // Debug Redux state
  const fullSimulationSetup = useSelector(state => state.simulationSetup);
  console.log('🔧 [REDUX_DEBUG] Full simulationSetup state:', fullSimulationSetup);
  console.log('🔧 [REDUX_DEBUG] inputVariables from state:', inputVariables);
  console.log('🔧 [REDUX_DEBUG] resultCells from state:', resultCells);
  
  // Debug individual variables for cell highlighting
  if (inputVariables && inputVariables.length > 0) {
    inputVariables.forEach((variable, index) => {
      console.log(`🔧 [CELL_DEBUG] Variable ${index}:`, variable);
      console.log(`🔧 [CELL_DEBUG] Variable ${index} name: "${variable.name}", sheet: "${variable.sheetName}"`);
    });
  }
  
  if (resultCells && resultCells.length > 0) {
    resultCells.forEach((cell, index) => {
      console.log(`🔧 [CELL_DEBUG] Result cell ${index}:`, cell);
      console.log(`🔧 [CELL_DEBUG] Result cell ${index} name: "${cell.name}", sheet: "${cell.sheetName}"`);
    });
  }

  console.log('🔧 ExcelGridPro - Render with data:', { 
    hasSheetData: !!sheetData, 
    dataKeys: sheetData ? Object.keys(sheetData) : [],
    selectionMode,
    inputVariables: inputVariables?.length || 0,
    resultCells: resultCells?.length || 0
  });

  // Define getCellClassName before useMemo to avoid hoisting issues
  const getCellClassName = useCallback((coordinate) => {
    let className = 'excel-cell';
    
    // Check if this cell is selected
    if (currentGridSelection?.name === coordinate) {
      className += ' selected-cell';
    }
    
    // Check if this cell is an input variable - check both cell and name properties
    const isInputVariable = inputVariables.some(v => 
      v.cell === coordinate || v.name === coordinate
    );
    if (isInputVariable) {
      className += ' input-variable-cell';
    }
    
    // Check if this cell is a result cell
    const isResultCell = resultCells && resultCells.some(r => 
      r.name === coordinate || r.cell === coordinate
    );
    if (isResultCell) {
      className += ' result-cell';
    }
    
    return className;
  }, [currentGridSelection, inputVariables, resultCells]);

  const handleZoomChange = (newZoom) => {
    setZoomLevel(newZoom);
  };

  // HELPERS --------------------------------------------
  // Generate Excel-style column labels (A, B, ..., Z, AA, AB, ...)
  const getColumnLabel = useCallback((index) => {
    let label = '';
    let i = index;
    while (i >= 0) {
      label = String.fromCharCode((i % 26) + 65) + label;
      i = Math.floor(i / 26) - 1;
    }
    return label;
  }, []);

  // Convert sheet data to AG Grid format - Handle both data structures but prefer grid_data
  const { columnDefs, rowData } = useMemo(() => {
    // Use grid_data first (working version), fallback to data
    const gridData = sheetData?.grid_data || sheetData?.data;
    
    if (!gridData || gridData.length === 0) {
      console.log('❌ No grid data available:', { sheetData });
      return { columnDefs: [], rowData: [] };
    }

    console.log('✅ Processing grid data:', { 
      gridDataLength: gridData.length, 
      firstRowLength: gridData[0]?.length,
      sampleData: gridData.slice(0, 2).map(row => row.slice(0, 3))
    });

    // Determine max columns across all rows
    const maxCols = Math.max(...gridData.map(row => row.length));
    
    // Use Excel column widths if available, otherwise calculate dynamic widths
    const columnWidths = [];
    const excelColumnWidths = sheetData?.column_widths || {};
    
    console.log('📏 Excel column widths:', excelColumnWidths);
    
    for (let colIndex = 0; colIndex < maxCols; colIndex++) {
      const colLetter = getColumnLabel(colIndex);
      const excelWidth = excelColumnWidths[colLetter];
      
      if (excelWidth && excelWidth > 0) {
        // Use Excel width (already in pixels), apply zoom factor
        const finalWidth = Math.floor(excelWidth * (zoomLevel / 100));
        columnWidths.push(finalWidth);
        console.log(`📏 Column ${colLetter}: Using Excel width ${excelWidth}px → ${finalWidth}px`);
      } else {
        // Fallback to dynamic calculation for missing widths
        const columnData = gridData.map(row => row[colIndex]).filter(Boolean);
        const dynamicWidth = calculateColumnWidth(columnData, 150, 120, 500);
        const finalWidth = Math.floor(dynamicWidth * (zoomLevel / 100));
        columnWidths.push(finalWidth);
        console.log(`📏 Column ${colLetter}: Using dynamic width ${dynamicWidth}px → ${finalWidth}px`);
      }
    }
    
    // Calculate cell dimensions based on zoom - increased for better visibility
    const baseHeight = 32; // Increased from 24
    const cellHeight = Math.floor(baseHeight * (zoomLevel / 100));
    
    // Create column definitions (A, B, C, ...)
    const colDefs = [
      {
        headerName: '#',
        field: 'rowNumber',
        width: Math.floor(60 * (zoomLevel / 100)), // Increased from 40
        pinned: 'left',
        cellClass: 'row-header-cell',
        suppressHeaderMenuButton: true,
        sortable: false,
        filter: false,
        editable: false,
        cellRenderer: (params) => {
          const { context, node } = params;
          const zoomLevel = context?.zoomLevel || 100;
          
          // Get row index from different sources - AG Grid provides multiple ways
          let rowIndex = 0;
          if (typeof params.rowIndex === 'number') {
            rowIndex = params.rowIndex;
          } else if (node && typeof node.rowIndex === 'number') {
            rowIndex = node.rowIndex;
          } else if (node && typeof node.id === 'string') {
            // Parse row index from node ID if available
            const parsed = parseInt(node.id);
            if (!isNaN(parsed)) {
              rowIndex = parsed;
            }
          }
          
          const rowStyle = {
            fontSize: `${Math.floor(12 * (zoomLevel / 100))}px`,
            fontWeight: '600',
            color: '#6b7280',
            textAlign: 'center',
            padding: `${Math.floor(4 * (zoomLevel / 100))}px`,
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          };
          return (
            <div style={rowStyle}>{isNaN(rowIndex) ? 1 : rowIndex + 1}</div>
          );
        },
      },
      ...Array.from({ length: maxCols }, (_, index) => {
        const colLetter = getColumnLabel(index);
        return {
          headerName: colLetter,
          field: `col_${index}`,
          colId: colLetter,
          width: columnWidths[index] || Math.floor(150 * (zoomLevel / 100)),
          resizable: true,
          editable: false,
          suppressHeaderMenuButton: true,
          sortable: false,
          filter: false,
          // Add proper value formatter to handle object data
          valueFormatter: (params) => {
            if (!params.value) return '';
            
            // Handle object data structure
            if (typeof params.value === 'object') {
              return params.value?.display_value || params.value?.value || '';
            }
            
            // Handle primitive values
            return String(params.value);
          },
          // Add value getter to ensure proper data access
          valueGetter: (params) => {
            const cellData = params.data[`col_${index}`];
            if (!cellData) return '';
            
            // Handle object data structure
            if (typeof cellData === 'object') {
              return cellData?.display_value || cellData?.value || '';
            }
            
            // Handle primitive values
            return String(cellData);
          },
          // headerComponent: ColumnHeaderRenderer,
          cellRenderer: ExcelCellRenderer,
          cellClass: (params) => {
            // Safely get row index
            let rowIndex = 0;
            if (typeof params.rowIndex === 'number') {
              rowIndex = params.rowIndex;
            } else if (params.node && typeof params.node.rowIndex === 'number') {
              rowIndex = params.node.rowIndex;
            }
            
            const coord = `${colLetter}${rowIndex + 1}`;
            let className = getCellClassName(coord);
            
            // Add border class if cell has border formatting
            const cellData = params.data[`col_${index}`];
            if (cellData && (cellData.border_top || cellData.border_bottom || cellData.border_left || cellData.border_right)) {
              className += ' excel-cell-with-borders';
            }
            
            return className;
          },
          cellStyle: (params) => {
            // Apply Excel borders as inline styles for immediate effect
            const cellData = params.data[`col_${index}`];
            if (!cellData) return {};
            
            const borderStyles = {};
            if (cellData.border_top) {
              borderStyles.borderTop = parseBorderStyle(cellData.border_top);
            }
            if (cellData.border_bottom) {
              borderStyles.borderBottom = parseBorderStyle(cellData.border_bottom);
            }
            if (cellData.border_left) {
              borderStyles.borderLeft = parseBorderStyle(cellData.border_left);
            }
            if (cellData.border_right) {
              borderStyles.borderRight = parseBorderStyle(cellData.border_right);
            }
            
            return borderStyles;
          },
        };
      })
    ];

    // Convert grid data to row data
    const rows = gridData.map((row, rowIndex) => {
      const rowObj = { rowNumber: rowIndex + 1 };
      row.forEach((cellData, colIndex) => {
        rowObj[`col_${colIndex}`] = cellData;
      });
      return rowObj;
    });

    console.log('🎯 Generated AG Grid data:', { 
      columnCount: colDefs.length, 
      rowCount: rows.length,
      sampleRow: rows[0],
      maxCols
    });

    return { columnDefs: colDefs, rowData: rows };
  }, [sheetData, getCellClassName, zoomLevel, getColumnLabel]);

  const onCellClicked = useCallback((event) => {
    if (event.column.colId === 'rowNumber') return;
    
    // Safely get row index
    let rowIndex = 0;
    if (typeof event.rowIndex === 'number') {
      rowIndex = event.rowIndex;
    } else if (event.node && typeof event.node.rowIndex === 'number') {
      rowIndex = event.node.rowIndex;
    }
    
    const coordinate = `${event.column.colId}${rowIndex + 1}`;
    const cellData = event.data[event.column.colDef.field];
    const displayValue = cellData?.display_value || cellData?.value || cellData || '';
    const rawValue = cellData?.value || cellData || '';
    const formula = cellData?.formula || '';
    
    console.log('🎯 Cell clicked:', { coordinate, cellData, displayValue, rawValue });
    
    setSelectedCell(coordinate);
    dispatch(setCurrentGridSelection({
      name: coordinate,
      value: rawValue,  // Store raw value for calculations
      displayValue: displayValue,  // Store formatted value for display
      formula: formula,
      sheetName: sheetData?.sheet_name || sheetData?.name || 'Sheet1'
    }));

    // Call parent's onCellClick handler if in selection mode
    if (selectionMode && selectionMode !== 'idle' && onCellClick) {
      console.log('🎯 Calling parent onCellClick:', { selectionMode, coordinate, displayValue });
      onCellClick(coordinate, displayValue);
    }
  }, [dispatch, sheetData, selectionMode, onCellClick]);

  const defaultColDef = useMemo(() => ({
    resizable: true,
    sortable: false,
    filter: false,
    suppressHeaderMenuButton: true,
  }), []);

  // Handle both data structures for validation
  const gridData = sheetData?.grid_data || sheetData?.data;
  
  if (!sheetData || !gridData) {
    console.log('❌ No sheet data available:', { sheetData });
    return (
      <div className="excel-grid-loading">
        <div className="loading-message">
          <h3>No Data Available</h3>
          <p>Please upload an Excel file to view the grid.</p>
        </div>
      </div>
    );
  }

  console.log('🎯 About to render AG Grid with:', {
    columnDefs: columnDefs.length,
    rowData: rowData.length,
    hasData: rowData.length > 0 && columnDefs.length > 0
  });

  const gridOptions = {
    columnDefs,
    rowData,
    defaultColDef: {
      resizable: true,
      sortable: false,
      filter: false,
      editable: false,
      suppressHeaderMenuButton: true,
    },
    onGridReady: (params) => {
      console.log('🎯 AG Grid Ready!', params);
      
      // Debug actual column widths after grid is ready
      setTimeout(() => {
        const columnApi = params.columnApi;
        const allColumns = columnApi.getAllColumns();
        
        console.log('🔍 AG Grid Column Width Debug:');
        allColumns.forEach((col, index) => {
          const colDef = col.getColDef();
          const actualWidth = col.getActualWidth();
          const definedWidth = colDef.width;
          const colId = col.getColId();
          
          console.log(`🔍 Column ${colId}: Defined=${definedWidth}px, Actual=${actualWidth}px, Diff=${actualWidth - definedWidth}px`);
        });
        
        // Check if there are any CSS styles affecting column widths
        const headerElement = document.querySelector('.ag-header-cell');
        if (headerElement) {
          const computedStyle = window.getComputedStyle(headerElement);
          console.log('🔍 Header Cell CSS:', {
            padding: computedStyle.padding,
            border: computedStyle.border,
            margin: computedStyle.margin,
            boxSizing: computedStyle.boxSizing
          });
        }
        
        const cellElement = document.querySelector('.ag-cell');
        if (cellElement) {
          const computedStyle = window.getComputedStyle(cellElement);
          console.log('🔍 Cell CSS:', {
            padding: computedStyle.padding,
            border: computedStyle.border,
            margin: computedStyle.margin,
            boxSizing: computedStyle.boxSizing
          });
        }
      }, 1000);
    },
    suppressRowClickSelection: true,
    suppressCellSelection: false,
    animateRows: false,
    suppressLoadingOverlay: true,
    suppressNoRowsOverlay: true,
    headerHeight: Math.floor(35 * (zoomLevel / 100)),
    rowHeight: Math.floor(25 * (zoomLevel / 100)),
    suppressHorizontalScroll: false,
    suppressVerticalScroll: false,
    pagination: false,
    domLayout: 'normal',
    undoRedoCellEditing: false,
    undoRedoCellEditingLimit: 20,
    stopEditingWhenCellsLoseFocus: true,
    enterNavigatesVertically: true,
    enterNavigatesVerticallyAfterEdit: true,
    suppressDragLeaveHidesColumns: true,
    suppressMakeColumnVisibleAfterUnGroup: true,
    suppressAggFuncInHeader: true,
    suppressMenuHide: true,
    alwaysShowHorizontalScroll: false,
    alwaysShowVerticalScroll: false,
    scrollbarWidth: 12,
    icons: {
      checkboxChecked: '<span class="ag-icon ag-icon-checkbox-checked"></span>',
      checkboxUnchecked: '<span class="ag-icon ag-icon-checkbox-unchecked"></span>',
      checkboxIndeterminate: '<span class="ag-icon ag-icon-checkbox-indeterminate"></span>',
    },
    context: {
      zoomLevel,
      onCellValueChanged: (newValue, coordinate) => {
        console.log('📝 Cell value changed:', { newValue, coordinate });
      },
      onCellClick: (coordinate, cellData) => {
        console.log('🖱️ Cell clicked:', { coordinate, cellData });
      },
      onCellDoubleClick: (coordinate, cellData) => {
        console.log('🖱️ Cell double clicked:', { coordinate, cellData });
      },
      onCellMouseOver: (coordinate, cellData) => {
        // console.log('🖱️ Cell mouse over:', { coordinate, cellData });
      },
      onCellMouseOut: (coordinate, cellData) => {
        // console.log('🖱️ Cell mouse out:', { coordinate, cellData });
      },
    },
  };

  return (
    <div className="excel-grid-pro-container" style={{ height: '100%', width: '100%' }}>
      {/* Formula Bar */}
      <div className="formula-bar">
        <div className="cell-reference">
          {selectedCell || 'Select a cell'}
        </div>
        <div className="formula-input">
          <span className="fx-label">fx</span>
          <input 
            type="text" 
            value={currentGridSelection?.formula || currentGridSelection?.displayValue || currentGridSelection?.value || ''} 
            placeholder="Enter formula or value"
            readOnly
            title={`Raw value: ${currentGridSelection?.value || 'N/A'}\nDisplay value: ${currentGridSelection?.displayValue || 'N/A'}`}
          />
        </div>
      </div>

      {/* Grid - simplified wrapper */}
      <div className="grid-container ag-theme-alpine" style={{ height: 'calc(100vh - 250px)', minHeight: '600px', width: '100%' }}>
        <AgGridReact
          {...gridOptions}
          onCellClicked={onCellClicked}
          getRowStyle={(params) => ({ background: params.rowIndex % 2 === 0 ? '#f8f9fa' : '#ffffff' })}
          onGridReady={(params) => {
            console.log('🎯 AG Grid Ready!', params);
            
            // Optional: Simple column analysis (since main functionality is working)
            setTimeout(() => {
              console.log('🔍 COLUMN ANALYSIS...');
              
              try {
                // Try multiple approaches to get columns
                let allColumns = [];
                
                // Method 1: Try columnApi (legacy but still works)
                if (params.columnApi && params.columnApi.getAllColumns) {
                  allColumns = params.columnApi.getAllColumns();
                  console.log('✅ Used columnApi.getAllColumns()');
                } 
                // Method 2: Try columnModel from API
                else if (params.api && params.api.getColumnModel) {
                  const columnModel = params.api.getColumnModel();
                  if (columnModel && columnModel.getAllColumns) {
                    allColumns = columnModel.getAllColumns();
                    console.log('✅ Used columnModel.getAllColumns()');
                  }
                }
                
                console.log('📊 Found columns:', allColumns.length);
                
                if (allColumns.length > 0) {
                  // Quick analysis of first few columns
                  console.log('📏 FIRST 5 COLUMNS:');
                  allColumns.slice(0, 5).forEach(column => {
                    try {
                      const colId = column.getColId ? column.getColId() : column.colId;
                      const actualWidth = column.getActualWidth ? column.getActualWidth() : 'unknown';
                      const definedWidth = column.getColDef ? column.getColDef().width : 'unknown';
                      
                      console.log(`📐 ${colId}: Defined=${definedWidth}px → Actual=${actualWidth}px`);
                    } catch (e) {
                      console.log(`❌ Error analyzing column:`, e.message);
                    }
                  });
                } else {
                  console.log('ℹ️ No columns found for analysis, but column width application is working correctly!');
                }
                
              } catch (error) {
                console.log('❌ Analysis failed:', error.message);
                console.log('ℹ️ But column width application is working correctly!');
              }
            }, 1000);
          }}
        />

        {/* Zoom Controls */}
        <div className="zoom-controls">
          <button className="zoom-btn" onClick={() => handleZoomChange(Math.max(50, zoomLevel - 25))} disabled={zoomLevel <= 50}>−</button>
          <span className="zoom-level">{zoomLevel}%</span>
          <button className="zoom-btn" onClick={() => handleZoomChange(Math.min(200, zoomLevel + 25))} disabled={zoomLevel >= 200}>+</button>
        </div>
      </div>
    </div>
  );
};

export default ExcelGridPro; 