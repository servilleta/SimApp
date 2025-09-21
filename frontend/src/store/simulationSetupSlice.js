import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  fileId: null,
  currentSheetName: null, // To know the context of cell selections
  inputVariables: [], // Array of { name (cell coord), sheetName, min_value, most_likely, max_value }
  resultCells: [], // Array of Objects { name (cell coord), sheetName }
  iterations: 1000,
  currentGridSelection: null, // Object { name (cell coord A1-style), value (string for display), sheetName (context) }
  isRestoring: false, // Flag to prevent initialization from clearing restored variables
  // Could also store selection state if needed, e.g., which cell is currently selected in the grid
  // selectedCell: { name (coord), sheetName } // For UI display before committing as input/result
};

const simulationSetupSlice = createSlice({
  name: 'simulationSetup',
  initialState,
  reducers: {
    initializeSetup: (state, action) => {
      // Payload: { fileId, sheetName }
      state.fileId = action.payload.fileId;
      state.currentSheetName = action.payload.sheetName;
      
      // Only reset fields if we're not in restoration mode
      if (!state.isRestoring) {
        state.inputVariables = [];
        state.resultCells = [];
        state.iterations = 1000;
      }
      state.currentGridSelection = null; // Always reset selection on new sheet/file
    },
    resetSetup: (state) => {
      Object.assign(state, initialState);
    },
    setCurrentGridSelection: (state, action) => {
      // payload: { name, value, sheetName } or null
      state.currentGridSelection = action.payload;
    },
    addInputVariable: (state, action) => {
      // Payload: { name (cell coord), sheetName, min_value, most_likely, max_value }
      const newVar = { ...action.payload }; // { name, sheetName, min_value, most_likely, max_value }
      const existingIndex = state.inputVariables.findIndex(
        item => item.name === newVar.name && item.sheetName === newVar.sheetName
      );
      if (existingIndex !== -1) {
        state.inputVariables[existingIndex] = newVar;
      } else {
        state.inputVariables.push(newVar);
      }
      // If this cell was a result cell, remove it from results
      state.resultCells = state.resultCells.filter(
        item => !(item.name === newVar.name && item.sheetName === newVar.sheetName)
      );
    },
    removeInputVariable: (state, action) => {
      // Payload: { name (cell coord), sheetName }
      state.inputVariables = state.inputVariables.filter(
        item => !(item.name === action.payload.name && item.sheetName === action.payload.sheetName)
      );
    },
    addResultCell: (state, action) => {
      // Payload: { name (cell coord), sheetName, display_name, variableName, format, decimalPlaces }
      const newResult = { ...action.payload }; // Preserve all fields including display_name
      const existingIndex = state.resultCells.findIndex(
        item => item.name === newResult.name && item.sheetName === newResult.sheetName
      );
      if (existingIndex !== -1) {
        // Update existing result cell with new data (including display_name)
        state.resultCells[existingIndex] = newResult;
      } else {
        // Add new result cell
        state.resultCells.push(newResult);
      }
      
      // If this cell was an input variable, remove it from inputs
      state.inputVariables = state.inputVariables.filter(
        item => !(item.name === newResult.name && item.sheetName === newResult.sheetName)
      );
    },
    removeResultCell: (state, action) => {
      // Payload: { name (cell coord), sheetName }
      state.resultCells = state.resultCells.filter(
        item => !(item.name === action.payload.name && item.sheetName === action.payload.sheetName)
      );
    },
    clearAllResultCells: (state) => {
      state.resultCells = [];
    },
    clearAllInputVariables: (state) => {
      state.inputVariables = [];
    },
    setIterations: (state, action) => {
      // Payload: number
      state.iterations = Math.max(1, parseInt(action.payload, 10) || 1000);
    },
    // Legacy support for single result cell (for backward compatibility)
    setResultCell: (state, action) => {
      // Payload: { name (cell coord), sheetName }
      const newResult = { ...action.payload }; // { name, sheetName }
      const existingIndex = state.resultCells.findIndex(
        item => item.name === newResult.name && item.sheetName === newResult.sheetName
      );
      if (existingIndex === -1) {
        state.resultCells.push(newResult);
        // If this cell was an input variable, remove it from inputs
        state.inputVariables = state.inputVariables.filter(
          item => !(item.name === newResult.name && item.sheetName === newResult.sheetName)
        );
      }
    },
    clearResultCell: (state) => {
      state.resultCells = [];
    },
    setSimulationSetup: (state, action) => {
      // Payload: { inputVariables, resultCells, iterations, currentSheetName }
      const { inputVariables, resultCells, iterations, currentSheetName } = action.payload;
      
      // Set restoration mode to prevent initialization from clearing variables
      state.isRestoring = true;
      
      if (inputVariables) state.inputVariables = inputVariables;
      if (resultCells) state.resultCells = resultCells;
      if (iterations) state.iterations = iterations;
      if (currentSheetName) state.currentSheetName = currentSheetName;
    },
    clearRestoringFlag: (state) => {
      // Clear the restoration flag after setup is complete
      state.isRestoring = false;
    },
  }
});

export const {
  initializeSetup,
  resetSetup,
  setCurrentGridSelection,
  addInputVariable,
  removeInputVariable,
  addResultCell,
  removeResultCell,
  clearAllResultCells,
  clearAllInputVariables,
  setIterations,
  setSimulationSetup,
  clearRestoringFlag,
  // Legacy support
  setResultCell,
  clearResultCell
} = simulationSetupSlice.actions;

// Selectors
export const selectSimulationSetupState = (state) => state.simulationSetup;
export const selectInputVariables = (state) => state.simulationSetup.inputVariables;
export const selectResultCells = (state) => state.simulationSetup.resultCells;
export const selectResultCell = (state) => state.simulationSetup.resultCells?.[0] || null; // Legacy support
export const selectIterations = (state) => state.simulationSetup.iterations;
export const selectSetupFileId = (state) => state.simulationSetup.fileId;
export const selectCurrentSheetNameForSetup = (state) => state.simulationSetup.currentSheetName; // Renamed for clarity
export const selectCurrentGridSelection = (state) => state.simulationSetup.currentGridSelection;

export default simulationSetupSlice.reducer; 