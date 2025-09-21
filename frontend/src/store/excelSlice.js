import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { uploadExcelFile } from '../services/excelService'; // Assuming excelService.js is in ../services

// Async thunk for uploading an Excel file
export const uploadExcel = createAsyncThunk(
  'excel/uploadExcel', // Action type prefix
  async (file, { rejectWithValue }) => {
    try {
      // responseData here is expected to be ExcelFileResponse from backend
      // { file_id: string, filename: string, sheets: List[SheetData] }
      // SheetData: { sheet_name: string, grid_data: List[List[CellData | null]] }
      // CellData: { value: any, formula: string | null, is_formula_cell: boolean, coordinate: string }
      const responseData = await uploadExcelFile(file);
      return responseData;
    } catch (error) {
      return rejectWithValue(error.message || 'Upload failed');
    }
  }
);

const initialState = {
  fileInfo: null, // Will store { file_id, filename, sheets: [SheetData] }
  isLoading: false,
  error: null,
};

const excelSlice = createSlice({
  name: 'excel',
  initialState,
  reducers: {
    resetExcelState: (state) => { // Renamed from clearExcelData for clarity
      state.fileInfo = null;
      state.isLoading = false;
      state.error = null;
    },
    setFileInfo: (state, action) => {
      // Payload: fileInfo object
      state.fileInfo = action.payload;
      state.error = null;
    },
    // Reducer to update cell data in the grid if needed (e.g., after a successful save of edits)
    // updateCellData: (state, action) => {
    //   const { sheetIndex, rowIndex, colIndex, newCellData } = action.payload;
    //   if (state.fileInfo && state.fileInfo.sheets[sheetIndex] && 
    //       state.fileInfo.sheets[sheetIndex].grid_data[rowIndex]) {
    //     state.fileInfo.sheets[sheetIndex].grid_data[rowIndex][colIndex] = newCellData;
    //   }
    // },
  },
  extraReducers: (builder) => {
    builder
      .addCase(uploadExcel.pending, (state) => {
        state.isLoading = true;
        state.fileInfo = null; // Clear previous file info
        state.error = null;
      })
      .addCase(uploadExcel.fulfilled, (state, action) => {
        state.isLoading = false;
        state.fileInfo = action.payload; // action.payload is the full ExcelFileResponse
        state.error = null;
        
        // Dispatch custom event to notify other components (like sidebar) that file was uploaded
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('excel-file-uploaded', {
            detail: { 
              filename: action.payload.filename,
              file_id: action.payload.file_id,
              timestamp: new Date().toISOString()
            }
          }));
          console.log('📁 Excel file uploaded event dispatched:', action.payload.filename);
        }
      })
      .addCase(uploadExcel.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload;
        state.fileInfo = null;
      });
  },
});

export const { resetExcelState, setFileInfo } = excelSlice.actions;

// Selectors
export const selectFileInfo = (state) => state.excel.fileInfo;
export const selectExcelLoading = (state) => state.excel.isLoading;
export const selectExcelError = (state) => state.excel.error;

// Convenience selector for the first sheet's data, often used for display
// Ensure to handle cases where fileInfo or sheets might be null/empty
export const selectFirstSheetData = (state) => {
  if (state.excel.fileInfo && state.excel.fileInfo.sheets && state.excel.fileInfo.sheets.length > 0) {
    return state.excel.fileInfo.sheets[0];
  }
  return null;
};

export default excelSlice.reducer; 