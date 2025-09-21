// Background PDF Service
// Handles background PDF generation and instant downloads

import { getToken } from './authService';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

/**
 * Start background PDF generation for a simulation
 * @param {Object} simulationData - Complete simulation data
 * @returns {Promise<Object>} - Response with generation status
 */
export const startBackgroundPdfGeneration = async (simulationData) => {
  try {
    console.log('🔄 [BACKGROUND_PDF] Starting background PDF generation...');
    console.log('🔄 [BACKGROUND_PDF] Simulation data:', simulationData);

    const token = getToken();
    if (!token) {
      throw new Error('Authentication token not found');
    }

    const response = await fetch(`${API_BASE_URL}/pdf/generate-background`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        simulation_id: simulationData.simulationId,
        results_data: simulationData.results,
        user_id: 1 // Will be set by backend from token
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    console.log('✅ [BACKGROUND_PDF] Background generation started:', result);
    return result;

  } catch (error) {
    console.error('❌ [BACKGROUND_PDF] Failed to start background generation:', error);
    throw error;
  }
};

/**
 * Check the status of PDF generation
 * @param {string} simulationId - Simulation ID
 * @returns {Promise<Object>} - Status information
 */
export const checkPdfStatus = async (simulationId) => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Authentication token not found');
    }

    const response = await fetch(`${API_BASE_URL}/pdf/status/${simulationId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const status = await response.json();
    return status;

  } catch (error) {
    console.error('❌ [BACKGROUND_PDF] Failed to check PDF status:', error);
    throw error;
  }
};

/**
 * Download pre-generated PDF instantly
 * @param {string} simulationId - Simulation ID
 * @returns {Promise<void>} - Triggers download
 */
export const downloadPdfInstant = async (simulationId) => {
  try {
    console.log('⚡ [INSTANT_PDF] Starting instant PDF download...');

    const token = getToken();
    if (!token) {
      throw new Error('Authentication token not found');
    }

    const response = await fetch(`${API_BASE_URL}/pdf/download/${simulationId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    // Get the blob and create download
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    
    // Extract filename from response headers or use default
    const contentDisposition = response.headers.get('content-disposition');
    const filename = contentDisposition 
      ? contentDisposition.split('filename=')[1]?.replace(/"/g, '') 
      : `simulation_${simulationId}.pdf`;

    // Create download link
    const downloadLink = document.createElement('a');
    downloadLink.href = url;
    downloadLink.download = filename;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
    
    // Clean up
    window.URL.revokeObjectURL(url);
    
    console.log('✅ [INSTANT_PDF] PDF downloaded instantly:', filename);

  } catch (error) {
    console.error('❌ [INSTANT_PDF] Failed to download PDF:', error);
    throw error;
  }
};

/**
 * Poll PDF status until ready, then download
 * @param {string} simulationId - Simulation ID
 * @param {Function} onStatusUpdate - Callback for status updates
 * @returns {Promise<void>}
 */
export const waitAndDownloadPdf = async (simulationId, onStatusUpdate = null) => {
  const maxAttempts = 30; // 30 seconds max
  let attempts = 0;
  
  while (attempts < maxAttempts) {
    try {
      const status = await checkPdfStatus(simulationId);
      
      if (onStatusUpdate) {
        onStatusUpdate(status);
      }
      
      if (status.file_ready && status.status === 'completed') {
        // PDF is ready, download instantly
        await downloadPdfInstant(simulationId);
        return;
      }
      
      if (status.status === 'failed') {
        throw new Error(status.error || 'PDF generation failed');
      }
      
      // Wait 1 second before next check
      await new Promise(resolve => setTimeout(resolve, 1000));
      attempts++;
      
    } catch (error) {
      console.error('❌ [BACKGROUND_PDF] Error while waiting for PDF:', error);
      throw error;
    }
  }
  
  throw new Error('PDF generation timeout - please try again');
};
