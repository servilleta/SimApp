// PowerPoint Export Service
// Handles exporting simulation results as editable PowerPoint presentations

import { getToken } from './authService';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

/**
 * Export simulation results as an editable PowerPoint presentation
 * @param {Object} simulationData - Complete simulation data including results and metadata
 * @returns {Promise<Blob>} - PowerPoint file as blob for download
 */
export const exportToPowerPoint = async (simulationData) => {
  try {
    console.log('🎯 [PPT_EXPORT] Starting PowerPoint export...');
    console.log('🎯 [PPT_EXPORT] Simulation data:', simulationData);

    // Prepare request body - same format as PDF export but for PowerPoint
    const requestBody = {
      simulationId: simulationData.simulationId || 'ppt_export',
      results: simulationData.results || {},
      metadata: {
        iterations_run: simulationData.metadata?.iterations_run || simulationData.iterations_run || 1000,
        engine_type: simulationData.metadata?.engine_type || simulationData.requested_engine_type || 'Ultra',
        timestamp: new Date().toISOString(),
        export_type: 'powerpoint',
        aspect_ratio: '16:9'
      }
    };

    console.log('🎯 [PPT_EXPORT] Request body prepared:', requestBody);

    // Make API request
    const response = await fetch(`${API_BASE_URL}/ppt/export`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${getToken()}`
      },
      body: JSON.stringify(requestBody)
    });

    console.log('🎯 [PPT_EXPORT] API response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('🎯 [PPT_EXPORT] API error:', errorText);
      throw new Error(`PowerPoint export failed: ${response.status} - ${errorText}`);
    }

    // Get the PowerPoint file as blob
    const blob = await response.blob();
    console.log('🎯 [PPT_EXPORT] PowerPoint blob received, size:', blob.size);

    // Verify it's a PowerPoint file
    if (blob.type && !blob.type.includes('presentation')) {
      console.warn('🎯 [PPT_EXPORT] Unexpected content type:', blob.type);
    }

    return blob;

  } catch (error) {
    console.error('🎯 [PPT_EXPORT] Export failed:', error);
    throw error;
  }
};

/**
 * Download PowerPoint blob as file
 * @param {Blob} blob - PowerPoint file blob
 * @param {string} filename - Optional filename (will generate if not provided)
 */
export const downloadPowerPoint = (blob, filename = null) => {
  try {
    // Generate filename if not provided
    if (!filename) {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      filename = `monte_carlo_simulation_${timestamp}.pptx`;
    }

    // Ensure .pptx extension
    if (!filename.endsWith('.pptx')) {
      filename += '.pptx';
    }

    console.log('🎯 [PPT_EXPORT] Downloading PowerPoint as:', filename);

    // Create download link
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    
    // Cleanup
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);

    console.log('🎯 [PPT_EXPORT] PowerPoint download completed');

  } catch (error) {
    console.error('🎯 [PPT_EXPORT] Download failed:', error);
    throw error;
  }
};

/**
 * Check if PowerPoint export is available
 * @returns {Promise<Object>} - Service status and capabilities
 */
export const getPowerPointExportStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/ppt/status`, {
      headers: {
        'Authorization': `Bearer ${getToken()}`
      }
    });

    if (!response.ok) {
      throw new Error(`Status check failed: ${response.status}`);
    }

    const status = await response.json();
    console.log('🎯 [PPT_EXPORT] Service status:', status);
    
    return status;

  } catch (error) {
    console.error('🎯 [PPT_EXPORT] Status check failed:', error);
    return {
      service_available: false,
      error: error.message
    };
  }
};

/**
 * Complete PowerPoint export workflow
 * @param {Object} simulationData - Simulation data to export
 * @param {string} filename - Optional custom filename
 * @returns {Promise<void>}
 */
export const exportAndDownloadPowerPoint = async (simulationData, filename = null) => {
  try {
    console.log('🎯 [PPT_EXPORT] Starting complete PowerPoint export workflow...');

    // Check service availability first
    const status = await getPowerPointExportStatus();
    if (!status.service_available) {
      throw new Error('PowerPoint export service is not available');
    }

    // Export to PowerPoint
    const blob = await exportToPowerPoint(simulationData);

    // Download the file
    downloadPowerPoint(blob, filename);

    console.log('🎯 [PPT_EXPORT] ✅ PowerPoint export workflow completed successfully');

  } catch (error) {
    console.error('🎯 [PPT_EXPORT] ❌ Export workflow failed:', error);
    throw error;
  }
};
