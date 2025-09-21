/**
 * PDF Export Service
 * Handles backend PDF generation using headless Chrome
 */

import { getToken } from './authService';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

class PDFExportService {
  /**
   * Export simulation results to PDF using backend headless Chrome
   * @param {string} simulationId - The simulation ID
   * @param {Array} resultsData - The simulation results data
   * @param {Object} options - Export options
   * @returns {Promise<Blob>} - PDF file as blob
   */
  async exportToPDF(simulationId, resultsData, options = {}) {
    try {
      const requestBody = {
        simulation_id: simulationId,
        results_data: resultsData,
        export_type: options.exportType || 'results_page',
        format: {
          format: 'A4',
          margin: options.margin || { top: '20mm', right: '20mm', bottom: '20mm', left: '20mm' },
          printBackground: true,
          scale: options.scale || 1
        }
      };

      const response = await fetch(`${API_BASE_URL}/pdf/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${getToken()}`
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'PDF export failed');
      }

      const result = await response.json();
      
      // Download the PDF file
      return await this.downloadPDF(result.download_url);
    } catch (error) {
      console.error('PDF export failed:', error);
      throw error;
    }
  }

  /**
   * Download PDF file from the provided URL
   * @param {string} downloadUrl - The download URL
   * @returns {Promise<Blob>} - PDF file as blob
   */
  async downloadPDF(downloadUrl) {
    try {
      const response = await fetch(`${API_BASE_URL}${downloadUrl}`, {
        headers: {
          'Authorization': `Bearer ${getToken()}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to download PDF');
      }

      return await response.blob();
    } catch (error) {
      console.error('PDF download failed:', error);
      throw error;
    }
  }

  /**
   * Trigger PDF download in browser
   * @param {Blob} pdfBlob - PDF blob
   * @param {string} filename - Filename for download
   */
  triggerDownload(pdfBlob, filename = null) {
    const url = URL.createObjectURL(pdfBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || `monte-carlo-results-${new Date().toISOString().split('T')[0]}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }


  /**
   * Check PDF service status
   * @returns {Promise<Object>} - Service status
   */
  async getServiceStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}/pdf/status`, {
        headers: {
          'Authorization': `Bearer ${getToken()}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to get PDF service status');
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get PDF service status:', error);
      return { available: false, error: error.message };
    }
  }
}

// Export singleton instance
export const pdfExportService = new PDFExportService();
export default PDFExportService;
