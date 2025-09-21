# Monte Carlo Simulation Web Platform
# Frontend Developer Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Development Environment Setup](#development-environment-setup)
5. [Component Architecture](#component-architecture)
6. [State Management](#state-management)
7. [Data Visualization with Plotly.js](#data-visualization-with-plotlyjs)
8. [Excel File Upload and Processing](#excel-file-upload-and-processing)
9. [Simulation Configuration](#simulation-configuration)
10. [Results Visualization](#results-visualization)
11. [Responsive Design](#responsive-design)
12. [Performance Optimization](#performance-optimization)
13. [Testing](#testing)
14. [Deployment](#deployment)
15. [References](#references)

## Introduction

This document provides comprehensive guidelines for frontend development of the Monte Carlo Simulation Web Platform. The frontend is responsible for providing an intuitive user interface for uploading Excel files, configuring simulation parameters, running simulations, and visualizing results.

### System Requirements

- React for UI development
- Plotly.js for data visualization
- Modern state management
- Responsive design for various devices
- Efficient handling of large datasets

## Architecture Overview

The frontend follows a component-based architecture with the following key features:

1. **Excel Upload**: Interface for uploading and previewing Excel files
2. **Variable Configuration**: UI for setting triangular probability distributions
3. **Simulation Control**: Interface for running and managing simulations
4. **Results Visualization**: Interactive charts and graphs for simulation results
5. **Responsive Layout**: Adaptive design for different screen sizes

### Data Flow

1. User uploads Excel file
2. Frontend displays file preview and available variables
3. User configures variables with triangular distributions
4. User initiates simulation
5. Frontend displays simulation results with interactive visualizations

## Project Structure

Follow this structure for better organization and maintainability:

```
frontend/
├── public/                  # Static assets
│   ├── index.html           # HTML template
│   └── favicon.ico          # Favicon
├── src/
│   ├── assets/              # Images, fonts, etc.
│   ├── components/          # Reusable UI components
│   │   ├── common/          # Shared components
│   │   ├── excel/           # Excel-related components
│   │   ├── simulation/      # Simulation-related components
│   │   └── visualization/   # Visualization components
│   ├── hooks/               # Custom React hooks
│   ├── pages/               # Page components
│   │   ├── Dashboard.jsx    # Main dashboard
│   │   ├── Upload.jsx       # File upload page
│   │   ├── Configure.jsx    # Simulation configuration
│   │   └── Results.jsx      # Results visualization
│   ├── services/            # API services
│   │   ├── api.js           # API client
│   │   ├── excelService.js  # Excel-related API calls
│   │   └── simulationService.js # Simulation-related API calls
│   ├── store/               # State management
│   │   ├── index.js         # Store configuration
│   │   ├── excelSlice.js    # Excel-related state
│   │   └── simulationSlice.js # Simulation-related state
│   ├── utils/               # Utility functions
│   ├── App.jsx              # Root component
│   ├── index.jsx            # Entry point
│   └── routes.jsx           # Route definitions
├── .env                     # Environment variables
├── package.json             # Dependencies and scripts
└── README.md                # Project documentation
```

## Development Environment Setup

### Prerequisites

- Node.js 18.x or later
- npm 9.x or later
- Git

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/monte-carlo-app.git
   cd monte-carlo-app/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Start the development server:
   ```bash
   npm start
   ```

### Key Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "react-plotly.js": "^2.6.0",
    "plotly.js": "^2.27.1",
    "@reduxjs/toolkit": "^2.0.1",
    "react-redux": "^9.0.4",
    "axios": "^1.6.2",
    "formik": "^2.4.5",
    "yup": "^1.3.2",
    "tailwindcss": "^3.3.5",
    "react-dropzone": "^14.2.3"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "eslint": "^8.54.0",
    "prettier": "^3.1.0",
    "vitest": "^0.34.6",
    "@testing-library/react": "^14.1.2"
  }
}
```

## Component Architecture

### Functional Components with Hooks

Use functional components with hooks instead of class components:

```jsx
// Bad: Class component
class ExcelUploader extends React.Component {
  constructor(props) {
    super(props);
    this.state = { file: null };
  }
  
  handleFileChange = (file) => {
    this.setState({ file });
  }
  
  render() {
    return (
      <div>
        <input type="file" onChange={(e) => this.handleFileChange(e.target.files[0])} />
      </div>
    );
  }
}

// Good: Functional component with hooks
const ExcelUploader = () => {
  const [file, setFile] = useState(null);
  
  const handleFileChange = (file) => {
    setFile(file);
  };
  
  return (
    <div>
      <input type="file" onChange={(e) => handleFileChange(e.target.files[0])} />
    </div>
  );
};
```

### Component Composition

Break down complex components into smaller, reusable ones:

```jsx
// Parent component
const SimulationPage = () => {
  return (
    <div className="simulation-page">
      <Header title="Monte Carlo Simulation" />
      <VariableConfiguration />
      <SimulationControls />
      <ResultsVisualization />
    </div>
  );
};

// Child component
const VariableConfiguration = () => {
  const variables = useSelector(state => state.simulation.variables);
  
  return (
    <div className="variable-config">
      <h2>Configure Variables</h2>
      {variables.map(variable => (
        <VariableInput key={variable.id} variable={variable} />
      ))}
    </div>
  );
};
```

### Custom Hooks

Create custom hooks for reusable logic:

```jsx
// Custom hook for API calls
const useApi = (initialUrl, initialData = null) => {
  const [url, setUrl] = useState(initialUrl);
  const [data, setData] = useState(initialData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await axios.get(url);
        setData(response.data);
        setError(null);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    if (url) {
      fetchData();
    }
  }, [url]);

  return { data, loading, error, setUrl };
};

// Usage
const VariableList = () => {
  const { data, loading, error } = useApi('/api/variables');
  
  if (loading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  
  return (
    <ul>
      {data.map(variable => (
        <li key={variable.id}>{variable.name}</li>
      ))}
    </ul>
  );
};
```

## State Management

### Redux Toolkit

Use Redux Toolkit for global state management:

```jsx
// store/index.js
import { configureStore } from '@reduxjs/toolkit';
import excelReducer from './excelSlice';
import simulationReducer from './simulationSlice';

export const store = configureStore({
  reducer: {
    excel: excelReducer,
    simulation: simulationReducer,
  },
});

// store/excelSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { uploadExcelFile } from '../services/excelService';

export const uploadExcel = createAsyncThunk(
  'excel/uploadExcel',
  async (file, { rejectWithValue }) => {
    try {
      const response = await uploadExcelFile(file);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);

const excelSlice = createSlice({
  name: 'excel',
  initialState: {
    fileId: null,
    fileName: null,
    columns: [],
    loading: false,
    error: null,
  },
  reducers: {
    resetExcelState: (state) => {
      state.fileId = null;
      state.fileName = null;
      state.columns = [];
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(uploadExcel.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(uploadExcel.fulfilled, (state, action) => {
        state.loading = false;
        state.fileId = action.payload.fileId;
        state.fileName = action.payload.fileName;
        state.columns = action.payload.columns;
      })
      .addCase(uploadExcel.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || 'Failed to upload file';
      });
  },
});

export const { resetExcelState } = excelSlice.actions;
export default excelSlice.reducer;
```

### React Context for UI State

Use React Context for UI-specific state:

```jsx
// contexts/UIContext.js
import { createContext, useContext, useState } from 'react';

const UIContext = createContext();

export const UIProvider = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState('light');
  
  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);
  const toggleTheme = () => setTheme(theme === 'light' ? 'dark' : 'light');
  
  return (
    <UIContext.Provider value={{ sidebarOpen, theme, toggleSidebar, toggleTheme }}>
      {children}
    </UIContext.Provider>
  );
};

export const useUI = () => useContext(UIContext);

// Usage
const Header = () => {
  const { sidebarOpen, toggleSidebar } = useUI();
  
  return (
    <header>
      <button onClick={toggleSidebar}>
        {sidebarOpen ? 'Close' : 'Open'} Sidebar
      </button>
    </header>
  );
};
```

### Form State Management

Use Formik for form state management:

```jsx
import { Formik, Form, Field, ErrorMessage } from 'formik';
import * as Yup from 'yup';

const VariableConfigSchema = Yup.object().shape({
  name: Yup.string().required('Variable name is required'),
  minValue: Yup.number().required('Minimum value is required'),
  mostLikely: Yup.number().required('Most likely value is required'),
  maxValue: Yup.number()
    .required('Maximum value is required')
    .moreThan(Yup.ref('minValue'), 'Max value must be greater than min value'),
});

const VariableConfigForm = ({ onSubmit, initialValues }) => {
  return (
    <Formik
      initialValues={initialValues || { name: '', minValue: 0, mostLikely: 0, maxValue: 0 }}
      validationSchema={VariableConfigSchema}
      onSubmit={onSubmit}
    >
      {({ isSubmitting }) => (
        <Form>
          <div className="form-group">
            <label htmlFor="name">Variable Name</label>
            <Field type="text" name="name" className="form-control" />
            <ErrorMessage name="name" component="div" className="error" />
          </div>
          
          <div className="form-group">
            <label htmlFor="minValue">Minimum Value</label>
            <Field type="number" name="minValue" className="form-control" />
            <ErrorMessage name="minValue" component="div" className="error" />
          </div>
          
          <div className="form-group">
            <label htmlFor="mostLikely">Most Likely Value</label>
            <Field type="number" name="mostLikely" className="form-control" />
            <ErrorMessage name="mostLikely" component="div" className="error" />
          </div>
          
          <div className="form-group">
            <label htmlFor="maxValue">Maximum Value</label>
            <Field type="number" name="maxValue" className="form-control" />
            <ErrorMessage name="maxValue" component="div" className="error" />
          </div>
          
          <button type="submit" disabled={isSubmitting}>
            {isSubmitting ? 'Saving...' : 'Save Variable'}
          </button>
        </Form>
      )}
    </Formik>
  );
};
```

## Data Visualization with Plotly.js

### Setting Up Plotly.js with React

Use the `react-plotly.js` library for React integration:

```jsx
import React from 'react';
import dynamic from 'next/dynamic';

// Use dynamic import for better performance
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const HistogramChart = ({ data }) => {
  const layout = {
    title: 'Simulation Results Histogram',
    xaxis: {
      title: 'Value',
    },
    yaxis: {
      title: 'Frequency',
    },
    autosize: true,
    height: 500,
    margin: { l: 50, r: 50, b: 100, t: 100, pad: 4 },
  };

  const plotData = [
    {
      x: data.values,
      type: 'histogram',
      marker: {
        color: 'rgba(100, 150, 200, 0.7)',
        line: {
          color: 'rgba(100, 150, 200, 1)',
          width: 1,
        },
      },
    },
  ];

  return (
    <div className="chart-container">
      <Plot data={plotData} layout={layout} useResizeHandler={true} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};

export default HistogramChart;
```

### Creating Reusable Chart Components

Create reusable chart components for different visualization types:

```jsx
// components/visualization/LineChart.jsx
import React from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const LineChart = ({ 
  title, 
  xData, 
  yData, 
  xLabel, 
  yLabel,
  seriesName = 'Series 1',
  color = 'rgb(75, 192, 192)'
}) => {
  const layout = {
    title: title,
    xaxis: {
      title: xLabel,
    },
    yaxis: {
      title: yLabel,
    },
    autosize: true,
    height: 400,
    margin: { l: 50, r: 50, b: 100, t: 100, pad: 4 },
  };

  const data = [
    {
      x: xData,
      y: yData,
      type: 'scatter',
      mode: 'lines+markers',
      name: seriesName,
      line: {
        color: color,
        width: 2,
      },
      marker: {
        color: color,
        size: 6,
      },
    },
  ];

  return (
    <div className="chart-container">
      <Plot data={data} layout={layout} useResizeHandler={true} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};

export default LineChart;
```

### Cross-Linking Charts

Implement cross-linking between charts for synchronized zooming and filtering:

```jsx
import React, { useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const CrossLinkedCharts = ({ simulationResults }) => {
  const [dateRange, setDateRange] = useState({
    startDate: simulationResults.minDate,
    endDate: simulationResults.maxDate,
  });

  const handleRelayout = (event) => {
    if (event['xaxis.range[0]'] && event['xaxis.range[1]']) {
      setDateRange({
        startDate: new Date(event['xaxis.range[0]']),
        endDate: new Date(event['xaxis.range[1]']),
      });
    } else if (event['xaxis.autorange']) {
      // Reset to full range
      setDateRange({
        startDate: simulationResults.minDate,
        endDate: simulationResults.maxDate,
      });
    }
  };

  const layout1 = {
    title: 'Variable A Over Time',
    xaxis: {
      title: 'Time',
      range: [dateRange.startDate, dateRange.endDate],
    },
    yaxis: {
      title: 'Value',
    },
  };

  const layout2 = {
    title: 'Variable B Over Time',
    xaxis: {
      title: 'Time',
      range: [dateRange.startDate, dateRange.endDate],
    },
    yaxis: {
      title: 'Value',
    },
  };

  const data1 = [{
    x: simulationResults.timePoints,
    y: simulationResults.variableA,
    type: 'scatter',
    mode: 'lines',
    name: 'Variable A',
  }];

  const data2 = [{
    x: simulationResults.timePoints,
    y: simulationResults.variableB,
    type: 'scatter',
    mode: 'lines',
    name: 'Variable B',
  }];

  return (
    <div className="charts-container">
      <div className="chart">
        <Plot
          data={data1}
          layout={layout1}
          onRelayout={handleRelayout}
          useResizeHandler={true}
          style={{ width: '100%', height: '400px' }}
        />
      </div>
      <div className="chart">
        <Plot
          data={data2}
          layout={layout2}
          onRelayout={handleRelayout}
          useResizeHandler={true}
          style={{ width: '100%', height: '400px' }}
        />
      </div>
    </div>
  );
};

export default CrossLinkedCharts;
```

### Advanced Visualization Features

Implement advanced visualization features for better user experience:

```jsx
import React from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const AdvancedHistogram = ({ data, title }) => {
  // Calculate statistics
  const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
  const sortedData = [...data].sort((a, b) => a - b);
  const median = sortedData[Math.floor(data.length / 2)];
  
  const layout = {
    title: title,
    xaxis: {
      title: 'Value',
    },
    yaxis: {
      title: 'Frequency',
    },
    shapes: [
      // Add vertical line for mean
      {
        type: 'line',
        x0: mean,
        y0: 0,
        x1: mean,
        y1: 1,
        yref: 'paper',
        line: {
          color: 'red',
          width: 2,
          dash: 'dash',
        },
      },
      // Add vertical line for median
      {
        type: 'line',
        x0: median,
        y0: 0,
        x1: median,
        y1: 1,
        yref: 'paper',
        line: {
          color: 'green',
          width: 2,
          dash: 'dash',
        },
      },
    ],
    annotations: [
      // Add annotation for mean
      {
        x: mean,
        y: 1,
        xref: 'x',
        yref: 'paper',
        text: `Mean: ${mean.toFixed(2)}`,
        showarrow: true,
        arrowhead: 7,
        ax: 0,
        ay: -40,
      },
      // Add annotation for median
      {
        x: median,
        y: 0.9,
        xref: 'x',
        yref: 'paper',
        text: `Median: ${median.toFixed(2)}`,
        showarrow: true,
        arrowhead: 7,
        ax: 0,
        ay: -40,
      },
    ],
    autosize: true,
    height: 500,
    margin: { l: 50, r: 50, b: 100, t: 100, pad: 4 },
  };

  const plotData = [
    {
      x: data,
      type: 'histogram',
      marker: {
        color: 'rgba(100, 150, 200, 0.7)',
        line: {
          color: 'rgba(100, 150, 200, 1)',
          width: 1,
        },
      },
    },
  ];

  return (
    <div className="chart-container">
      <Plot 
        data={plotData} 
        layout={layout} 
        useResizeHandler={true} 
        style={{ width: '100%', height: '100%' }}
        config={{
          displayModeBar: true,
          responsive: true,
          toImageButtonOptions: {
            format: 'png',
            filename: 'simulation_histogram',
            height: 500,
            width: 700,
            scale: 2,
          },
        }}
      />
    </div>
  );
};

export default AdvancedHistogram;
```

## Excel File Upload and Processing

### File Upload Component

Create a file upload component using `react-dropzone`:

```jsx
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useDispatch, useSelector } from 'react-redux';
import { uploadExcel } from '../store/excelSlice';

const ExcelUploader = () => {
  const dispatch = useDispatch();
  const { loading, error } = useSelector((state) => state.excel);

  const onDrop = useCallback(
    (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (file) {
        dispatch(uploadExcel(file));
      }
    },
    [dispatch]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    maxFiles: 1,
    multiple: false,
  });

  return (
    <div className="excel-uploader">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${error ? 'error' : ''}`}
      >
        <input {...getInputProps()} />
        {loading ? (
          <p>Uploading...</p>
        ) : (
          <>
            <p>Drag & drop an Excel file here, or click to select one</p>
            <p className="file-hint">Accepts .xlsx and .xls files</p>
          </>
        )}
      </div>
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default ExcelUploader;
```

### Excel Preview Component

Create a component to preview uploaded Excel files:

```jsx
import React from 'react';
import { useSelector } from 'react-redux';

const ExcelPreview = () => {
  const { fileName, columns, preview } = useSelector((state) => state.excel);

  if (!fileName) {
    return null;
  }

  return (
    <div className="excel-preview">
      <h2>File Preview: {fileName}</h2>
      
      <div className="table-container">
        <table className="preview-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((column) => (
                  <td key={`${rowIndex}-${column}`}>{row[column]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <p className="preview-note">Showing first 5 rows of data</p>
    </div>
  );
};

export default ExcelPreview;
```

## Simulation Configuration

### Variable Configuration Component

Create a component for configuring simulation variables:

```jsx
import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { addVariable, updateVariable, removeVariable } from '../store/simulationSlice';
import VariableConfigForm from './VariableConfigForm';

const VariableConfiguration = () => {
  const dispatch = useDispatch();
  const { variables } = useSelector((state) => state.simulation);
  const { columns } = useSelector((state) => state.excel);
  const [editingVariable, setEditingVariable] = useState(null);

  const handleAddVariable = (values) => {
    dispatch(addVariable(values));
  };

  const handleUpdateVariable = (values) => {
    dispatch(updateVariable(values));
    setEditingVariable(null);
  };

  const handleRemoveVariable = (id) => {
    dispatch(removeVariable(id));
  };

  const handleEditVariable = (variable) => {
    setEditingVariable(variable);
  };

  return (
    <div className="variable-configuration">
      <h2>Configure Simulation Variables</h2>
      
      <div className="available-columns">
        <h3>Available Columns</h3>
        <ul>
          {columns.map((column) => (
            <li key={column}>{column}</li>
          ))}
        </ul>
      </div>
      
      <div className="variable-form">
        <h3>{editingVariable ? 'Edit Variable' : 'Add New Variable'}</h3>
        <VariableConfigForm
          onSubmit={editingVariable ? handleUpdateVariable : handleAddVariable}
          initialValues={editingVariable}
        />
      </div>
      
      <div className="configured-variables">
        <h3>Configured Variables</h3>
        {variables.length === 0 ? (
          <p>No variables configured yet</p>
        ) : (
          <table className="variables-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Min Value</th>
                <th>Most Likely</th>
                <th>Max Value</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {variables.map((variable) => (
                <tr key={variable.id}>
                  <td>{variable.name}</td>
                  <td>{variable.minValue}</td>
                  <td>{variable.mostLikely}</td>
                  <td>{variable.maxValue}</td>
                  <td>
                    <button onClick={() => handleEditVariable(variable)}>Edit</button>
                    <button onClick={() => handleRemoveVariable(variable.id)}>Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default VariableConfiguration;
```

### Formula Configuration Component

Create a component for configuring simulation formulas:

```jsx
import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { setFormula } from '../store/simulationSlice';

const FormulaConfiguration = () => {
  const dispatch = useDispatch();
  const { formula } = useSelector((state) => state.simulation);
  const { variables } = useSelector((state) => state.simulation);
  const [localFormula, setLocalFormula] = useState(formula || '');

  const handleFormulaChange = (e) => {
    setLocalFormula(e.target.value);
  };

  const handleSaveFormula = () => {
    dispatch(setFormula(localFormula));
  };

  const insertVariable = (variableName) => {
    setLocalFormula((prev) => prev + variableName);
  };

  return (
    <div className="formula-configuration">
      <h2>Configure Simulation Formula</h2>
      
      <div className="formula-input">
        <label htmlFor="formula">Formula:</label>
        <div className="formula-editor">
          <textarea
            id="formula"
            value={localFormula}
            onChange={handleFormulaChange}
            placeholder="Enter formula (e.g., x + y)"
            rows={3}
          />
          <div className="formula-buttons">
            <button onClick={() => insertVariable('+')}>+</button>
            <button onClick={() => insertVariable('-')}>-</button>
            <button onClick={() => insertVariable('*')}>*</button>
            <button onClick={() => insertVariable('/')}>/</button>
            <button onClick={() => insertVariable('(')}>(</button>
            <button onClick={() => insertVariable(')')}>)</button>
          </div>
        </div>
        <button onClick={handleSaveFormula}>Save Formula</button>
      </div>
      
      <div className="available-variables">
        <h3>Available Variables</h3>
        <div className="variable-buttons">
          {variables.map((variable) => (
            <button
              key={variable.id}
              onClick={() => insertVariable(variable.name)}
              className="variable-button"
            >
              {variable.name}
            </button>
          ))}
        </div>
      </div>
      
      {formula && (
        <div className="current-formula">
          <h3>Current Formula</h3>
          <div className="formula-display">{formula}</div>
        </div>
      )}
    </div>
  );
};

export default FormulaConfiguration;
```

## Results Visualization

### Results Dashboard Component

Create a dashboard for displaying simulation results:

```jsx
import React from 'react';
import { useSelector } from 'react-redux';
import HistogramChart from '../components/visualization/HistogramChart';
import StatisticsTable from '../components/visualization/StatisticsTable';
import PercentileChart from '../components/visualization/PercentileChart';

const ResultsDashboard = () => {
  const { results, loading, error } = useSelector((state) => state.simulation);

  if (loading) {
    return <div className="loading">Running simulation...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!results) {
    return <div className="no-results">No simulation results available</div>;
  }

  return (
    <div className="results-dashboard">
      <h2>Simulation Results</h2>
      
      <div className="results-summary">
        <div className="summary-card">
          <h3>Mean</h3>
          <div className="summary-value">{results.mean.toFixed(2)}</div>
        </div>
        <div className="summary-card">
          <h3>Median</h3>
          <div className="summary-value">{results.median.toFixed(2)}</div>
        </div>
        <div className="summary-card">
          <h3>Std Dev</h3>
          <div className="summary-value">{results.stdDev.toFixed(2)}</div>
        </div>
        <div className="summary-card">
          <h3>Iterations</h3>
          <div className="summary-value">{results.iterations.toLocaleString()}</div>
        </div>
      </div>
      
      <div className="results-charts">
        <div className="chart-container">
          <h3>Distribution Histogram</h3>
          <HistogramChart 
            data={{
              values: results.histogram.values,
              bins: results.histogram.bins,
            }}
          />
        </div>
        
        <div className="chart-container">
          <h3>Percentile Chart</h3>
          <PercentileChart percentiles={results.percentiles} />
        </div>
      </div>
      
      <div className="results-details">
        <h3>Detailed Statistics</h3>
        <StatisticsTable statistics={results} />
      </div>
      
      <div className="export-options">
        <button className="export-button">Export as CSV</button>
        <button className="export-button">Export as PDF</button>
        <button className="export-button">Export Charts</button>
      </div>
    </div>
  );
};

export default ResultsDashboard;
```

### Statistics Table Component

Create a component for displaying detailed statistics:

```jsx
import React from 'react';

const StatisticsTable = ({ statistics }) => {
  return (
    <table className="statistics-table">
      <thead>
        <tr>
          <th>Statistic</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Mean</td>
          <td>{statistics.mean.toFixed(4)}</td>
        </tr>
        <tr>
          <td>Median</td>
          <td>{statistics.median.toFixed(4)}</td>
        </tr>
        <tr>
          <td>Standard Deviation</td>
          <td>{statistics.stdDev.toFixed(4)}</td>
        </tr>
        <tr>
          <td>Minimum</td>
          <td>{statistics.min.toFixed(4)}</td>
        </tr>
        <tr>
          <td>Maximum</td>
          <td>{statistics.max.toFixed(4)}</td>
        </tr>
        <tr>
          <td>10th Percentile</td>
          <td>{statistics.percentiles['10'].toFixed(4)}</td>
        </tr>
        <tr>
          <td>25th Percentile</td>
          <td>{statistics.percentiles['25'].toFixed(4)}</td>
        </tr>
        <tr>
          <td>50th Percentile</td>
          <td>{statistics.percentiles['50'].toFixed(4)}</td>
        </tr>
        <tr>
          <td>75th Percentile</td>
          <td>{statistics.percentiles['75'].toFixed(4)}</td>
        </tr>
        <tr>
          <td>90th Percentile</td>
          <td>{statistics.percentiles['90'].toFixed(4)}</td>
        </tr>
        <tr>
          <td>Iterations</td>
          <td>{statistics.iterations.toLocaleString()}</td>
        </tr>
      </tbody>
    </table>
  );
};

export default StatisticsTable;
```

## Responsive Design

### Responsive Layout with CSS Grid and Flexbox

Use CSS Grid and Flexbox for responsive layouts:

```css
/* Base layout */
.app-container {
  display: grid;
  grid-template-columns: 250px 1fr;
  grid-template-rows: auto 1fr auto;
  grid-template-areas:
    "header header"
    "sidebar main"
    "footer footer";
  min-height: 100vh;
}

.header {
  grid-area: header;
}

.sidebar {
  grid-area: sidebar;
}

.main-content {
  grid-area: main;
  padding: 1rem;
}

.footer {
  grid-area: footer;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .app-container {
    grid-template-columns: 1fr;
    grid-template-areas:
      "header"
      "main"
      "footer";
  }
  
  .sidebar {
    display: none;
  }
  
  .sidebar.open {
    display: block;
    position: fixed;
    top: 0;
    left: 0;
    width: 250px;
    height: 100vh;
    z-index: 1000;
    background-color: white;
    box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
  }
}

/* Flexbox for card layouts */
.card-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}

.card {
  flex: 1 1 300px;
  max-width: 100%;
  padding: 1rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Responsive charts */
.chart-container {
  width: 100%;
  height: 400px;
  margin-bottom: 2rem;
}

@media (max-width: 576px) {
  .chart-container {
    height: 300px;
  }
}
```

### Responsive Component Design

Implement responsive components:

```jsx
import React, { useState, useEffect } from 'react';
import { useMediaQuery } from '../hooks/useMediaQuery';

const ResponsiveDataGrid = ({ data, columns }) => {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const [visibleColumns, setVisibleColumns] = useState(columns);
  
  useEffect(() => {
    if (isMobile) {
      // Show fewer columns on mobile
      setVisibleColumns(columns.filter(col => col.priority === 'high'));
    } else {
      setVisibleColumns(columns);
    }
  }, [isMobile, columns]);
  
  return (
    <div className="responsive-grid">
      {isMobile ? (
        // Card view for mobile
        <div className="card-list">
          {data.map((item, index) => (
            <div key={index} className="data-card">
              {visibleColumns.map(column => (
                <div key={column.id} className="card-field">
                  <span className="field-label">{column.label}:</span>
                  <span className="field-value">{item[column.id]}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      ) : (
        // Table view for desktop
        <table className="data-table">
          <thead>
            <tr>
              {visibleColumns.map(column => (
                <th key={column.id}>{column.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index}>
                {visibleColumns.map(column => (
                  <td key={column.id}>{item[column.id]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default ResponsiveDataGrid;
```

### Media Query Hook

Create a custom hook for media queries:

```jsx
import { useState, useEffect } from 'react';

export const useMediaQuery = (query) => {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia(query);
    setMatches(mediaQuery.matches);

    const handler = (event) => {
      setMatches(event.matches);
    };

    mediaQuery.addEventListener('change', handler);

    return () => {
      mediaQuery.removeEventListener('change', handler);
    };
  }, [query]);

  return matches;
};
```

## Performance Optimization

### Memoization with React.memo and useMemo

Use memoization to prevent unnecessary re-renders:

```jsx
import React, { useMemo } from 'react';

// Memoize component to prevent re-renders when props don't change
const MemoizedChart = React.memo(({ data, title }) => {
  return (
    <div className="chart">
      <h3>{title}</h3>
      {/* Chart implementation */}
    </div>
  );
});

const SimulationResults = ({ results }) => {
  // Memoize expensive calculations
  const statistics = useMemo(() => {
    return {
      mean: calculateMean(results),
      median: calculateMedian(results),
      stdDev: calculateStdDev(results),
    };
  }, [results]);
  
  return (
    <div className="results">
      <MemoizedChart data={results} title="Simulation Results" />
      <div className="statistics">
        <p>Mean: {statistics.mean}</p>
        <p>Median: {statistics.median}</p>
        <p>Standard Deviation: {statistics.stdDev}</p>
      </div>
    </div>
  );
};
```

### Virtualized Lists for Large Datasets

Use virtualization for rendering large lists:

```jsx
import React from 'react';
import { FixedSizeList as List } from 'react-window';

const VirtualizedDataTable = ({ data, height, width }) => {
  const Row = ({ index, style }) => {
    const item = data[index];
    return (
      <div className="table-row" style={style}>
        <div className="table-cell">{item.id}</div>
        <div className="table-cell">{item.name}</div>
        <div className="table-cell">{item.value}</div>
      </div>
    );
  };

  return (
    <div className="virtualized-table">
      <div className="table-header">
        <div className="table-cell">ID</div>
        <div className="table-cell">Name</div>
        <div className="table-cell">Value</div>
      </div>
      <List
        height={height}
        width={width}
        itemCount={data.length}
        itemSize={50}
      >
        {Row}
      </List>
    </div>
  );
};

export default VirtualizedDataTable;
```

### Code Splitting and Lazy Loading

Use code splitting and lazy loading for better initial load times:

```jsx
import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Loading from './components/common/Loading';

// Lazy load pages
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Upload = lazy(() => import('./pages/Upload'));
const Configure = lazy(() => import('./pages/Configure'));
const Results = lazy(() => import('./pages/Results'));

const App = () => {
  return (
    <Router>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/configure" element={<Configure />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </Suspense>
    </Router>
  );
};

export default App;
```

## Testing

### Component Testing with React Testing Library

Test components with React Testing Library:

```jsx
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import ExcelUploader from '../components/ExcelUploader';

const mockStore = configureStore([]);

describe('ExcelUploader', () => {
  let store;

  beforeEach(() => {
    store = mockStore({
      excel: {
        loading: false,
        error: null,
      },
    });
    store.dispatch = jest.fn();
  });

  test('renders upload area', () => {
    render(
      <Provider store={store}>
        <ExcelUploader />
      </Provider>
    );

    expect(screen.getByText(/drag & drop an excel file here/i)).toBeInTheDocument();
    expect(screen.getByText(/accepts .xlsx and .xls files/i)).toBeInTheDocument();
  });

  test('shows loading state', () => {
    store = mockStore({
      excel: {
        loading: true,
        error: null,
      },
    });

    render(
      <Provider store={store}>
        <ExcelUploader />
      </Provider>
    );

    expect(screen.getByText(/uploading.../i)).toBeInTheDocument();
  });

  test('shows error message', () => {
    store = mockStore({
      excel: {
        loading: false,
        error: 'Failed to upload file',
      },
    });

    render(
      <Provider store={store}>
        <ExcelUploader />
      </Provider>
    );

    expect(screen.getByText(/failed to upload file/i)).toBeInTheDocument();
  });
});
```

### Integration Testing

Test integration between components:

```jsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import SimulationPage from '../pages/SimulationPage';

const mockStore = configureStore([thunk]);

describe('SimulationPage Integration', () => {
  let store;

  beforeEach(() => {
    store = mockStore({
      excel: {
        fileId: '123',
        fileName: 'test.xlsx',
        columns: ['A', 'B', 'C'],
        loading: false,
        error: null,
      },
      simulation: {
        variables: [
          { id: '1', name: 'A', minValue: 10, mostLikely: 20, maxValue: 30 },
        ],
        formula: 'A * 2',
        loading: false,
        error: null,
        results: null,
      },
    });
    store.dispatch = jest.fn();
  });

  test('can configure and run simulation', async () => {
    render(
      <Provider store={store}>
        <SimulationPage />
      </Provider>
    );

    // Check if variables are displayed
    expect(screen.getByText(/A/i)).toBeInTheDocument();
    expect(screen.getByText(/10/i)).toBeInTheDocument();
    expect(screen.getByText(/20/i)).toBeInTheDocument();
    expect(screen.getByText(/30/i)).toBeInTheDocument();

    // Check if formula is displayed
    expect(screen.getByText(/A \* 2/i)).toBeInTheDocument();

    // Click run simulation button
    fireEvent.click(screen.getByText(/run simulation/i));

    // Check if dispatch was called with the correct action
    await waitFor(() => {
      expect(store.dispatch).toHaveBeenCalledWith(
        expect.objectContaining({
          type: expect.stringContaining('runSimulation'),
        })
      );
    });
  });
});
```

## Deployment

### Build Process

Configure the build process in `package.json`:

```json
{
  "scripts": {
    "start": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint src",
    "format": "prettier --write \"src/**/*.{js,jsx,ts,tsx,json,css}\""
  }
}
```

### Environment Configuration

Create environment configuration files:

```
# .env.development
VITE_API_URL=http://localhost:8000/api
VITE_DEBUG=true

# .env.production
VITE_API_URL=/api
VITE_DEBUG=false
```

### Docker Configuration

Create a Dockerfile for the frontend:

```dockerfile
# Build stage
FROM node:18-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

Create an nginx configuration file:

```
# nginx.conf
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Serve static files
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## References

1. React Documentation: https://react.dev/
2. Plotly.js Documentation: https://plotly.com/javascript/
3. React-Plotly.js Documentation: https://github.com/plotly/react-plotly.js/
4. Redux Toolkit Documentation: https://redux-toolkit.js.org/
5. Formik Documentation: https://formik.org/docs/overview
6. React Router Documentation: https://reactrouter.com/
7. Tailwind CSS Documentation: https://tailwindcss.com/docs
8. React Testing Library Documentation: https://testing-library.com/docs/react-testing-library/intro/
9. Vite Documentation: https://vitejs.dev/guide/
10. React Dropzone Documentation: https://react-dropzone.js.org/
