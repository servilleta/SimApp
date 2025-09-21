/**
 * Modern PDF Export Utility with 100% Visual Fidelity
 * Uses backend Playwright service for perfect HTML to PDF conversion
 */

import { toast } from 'react-toastify';

class PDFExportService {
  constructor() {
    // Use the same API base URL pattern as other services
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
  }

  /**
   * Export simulation results to PDF with perfect visual fidelity
   * @param {string} simulationId - The simulation ID
   * @param {Object} resultsData - The simulation results data
   * @param {Object} options - Export options
   * @returns {Promise<Blob>} - PDF blob for download
   */
  async exportResultsToPDF(simulationId, resultsData, options = {}) {
    try {
      // Show loading toast
      const loadingToast = toast.loading('🔄 Generating high-quality PDF...');

      const exportRequest = {
        simulation_id: simulationId,
        results_data: resultsData,
        export_type: 'results',
        ...options
      };

      const response = await fetch(`${this.baseURL}/pdf/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken')}` // Use the same token key as other services
        },
        body: JSON.stringify(exportRequest)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.message || 'PDF generation failed');
      }

      // Download the PDF
      const pdfBlob = await this.downloadPDF(result.download_url);
      
      // Update toast
      toast.update(loadingToast, {
        render: '✅ PDF generated successfully!',
        type: 'success',
        isLoading: false,
        autoClose: 3000
      });

      return pdfBlob;

    } catch (error) {
      console.error('PDF export failed:', error);
      toast.error(`❌ PDF export failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Export current webpage URL to PDF
   * @param {string} url - The URL to capture
   * @param {string} simulationId - Simulation ID for filename
   * @param {Object} options - Export options
   * @returns {Promise<Blob>} - PDF blob for download
   */
  async exportURLToPDF(url, simulationId, options = {}) {
    try {
      const loadingToast = toast.loading('🔄 Capturing webpage as PDF...');

      const params = new URLSearchParams({
        url: url,
        simulation_id: simulationId,
        ...options
      });

      const response = await fetch(`${this.baseURL}/pdf/export-url?${params}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.message || 'PDF generation failed');
      }

      const pdfBlob = await this.downloadPDF(result.download_url);
      
      toast.update(loadingToast, {
        render: '✅ Webpage captured as PDF!',
        type: 'success',
        isLoading: false,
        autoClose: 3000
      });

      return pdfBlob;

    } catch (error) {
      console.error('URL PDF export failed:', error);
      toast.error(`❌ URL PDF export failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Download PDF from the backend
   * @private
   */
  async downloadPDF(downloadUrl) {
    const response = await fetch(`${this.baseURL}${downloadUrl}`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
      }
    });

    if (!response.ok) {
      throw new Error(`Failed to download PDF: HTTP ${response.status}`);
    }

    return await response.blob();
  }

  /**
   * Trigger browser download of PDF blob
   * @param {Blob} pdfBlob - The PDF blob
   * @param {string} filename - Filename for download
   */
  downloadPDFBlob(pdfBlob, filename = null) {
    if (!filename) {
      filename = `monte-carlo-results-${new Date().toISOString().split('T')[0]}.pdf`;
    }

    const url = URL.createObjectURL(pdfBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /**
   * Check if PDF service is available
   * @returns {Promise<boolean>}
   */
  async checkServiceStatus() {
    try {
      const response = await fetch(`${this.baseURL}/pdf/status`);
      const status = await response.json();
      return status.status === 'healthy' && status.playwright_available;
    } catch (error) {
      console.warn('PDF service status check failed:', error);
      return false;
    }
  }

  /**
   * Export with fallback to legacy method
   * @param {string} simulationId 
   * @param {Object} resultsData 
   * @param {Function} legacyExportFunction - Fallback function
   */
  async exportWithFallback(simulationId, resultsData, legacyExportFunction) {
    try {
      // Try modern PDF export first
      const serviceAvailable = await this.checkServiceStatus();
      
      if (serviceAvailable) {
        const pdfBlob = await this.exportResultsToPDF(simulationId, resultsData);
        this.downloadPDFBlob(pdfBlob);
        return { success: true, method: 'modern' };
      } else {
        throw new Error('PDF service not available');
      }
    } catch (error) {
      console.warn('Modern PDF export failed, falling back to legacy method:', error);
      
      // Fallback to legacy jsPDF method
      if (legacyExportFunction) {
        try {
          await legacyExportFunction();
          toast.info('📄 PDF generated using fallback method');
          return { success: true, method: 'legacy' };
        } catch (legacyError) {
          console.error('Legacy PDF export also failed:', legacyError);
          toast.error('❌ Both PDF export methods failed');
          return { success: false, error: legacyError };
        }
      } else {
        toast.error('❌ PDF export failed and no fallback available');
        return { success: false, error: error };
      }
    }
  }
}

// Export singleton instance
export const pdfExportService = new PDFExportService();
export default pdfExportService;
