import axios from 'axios'; // Assuming axios is your HTTP client

// Get API base URL from environment variables, defaulting for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const EXCEL_API_URL = `${API_BASE_URL}/excel-parser`;

/**
 * Uploads an Excel file to the backend.
 * @param {File} file - The Excel file to upload.
 * @returns {Promise<object>} A promise that resolves to the backend's response data.
 *                             Expected structure includes file_id, filename, sheet_names, columns, row_count, formulas_count, preview.
 * @throws {Error} If the upload fails or the server returns an error.
 */
export const uploadExcelFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const token = localStorage.getItem('authToken'); // Use 'authToken' to match authService.js
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }

    const response = await axios.post(`${EXCEL_API_URL}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
        'Authorization': `Bearer ${token}`, // Add the Authorization header
    },
  });
    return response.data; // This should be the ExcelFileResponse from the backend
  } catch (error) {
    // Axios wraps errors, error.response contains server response if available
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to upload Excel file.');
    }
    throw new Error(error.message || 'Network error or failed to upload Excel file.');
  }
};

/**
 * Fetches metadata/information about a previously uploaded and parsed Excel file.
 * @param {string} fileId - The ID of the file.
 * @returns {Promise<object>} A promise that resolves to the file information.
 */
export const getFileInfo = async (fileId) => {
  try {
    const response = await axios.get(`${EXCEL_API_URL}/files/${fileId}/info`);
    return response.data;
  } catch (error) {
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to fetch file info.');
    }
    throw new Error(error.message || 'Network error or failed to fetch file info.');
  }
};

/**
 * Fetches available variables (e.g., column names) from a parsed Excel file.
 * @param {string} fileId - The ID of the file.
 * @returns {Promise<string[]>} A promise that resolves to a list of variable names.
 */
export const getFileVariables = async (fileId) => {
  try {
    const response = await axios.get(`${EXCEL_API_URL}/files/${fileId}/variables`);
    return response.data;
  } catch (error) {
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to fetch file variables.');
    }
    throw new Error(error.message || 'Network error or failed to fetch file variables.');
  }
};

/**
 * Fetches a specific formula from a parsed Excel file.
 * @param {string} fileId - The ID of the file.
 * @param {string} cellCoordinate - The cell coordinate (e.g., "A1").
 * @returns {Promise<object>} A promise that resolves to an object containing the formula.
 */
export const getFormula = async (fileId, cellCoordinate) => {
  try {
    const response = await axios.get(`${EXCEL_API_URL}/files/${fileId}/formulas/${cellCoordinate}`);
    return response.data;
  } catch (error) {
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to fetch formula.');
    }
    throw new Error(error.message || 'Network error or failed to fetch formula.');
  }
}; 