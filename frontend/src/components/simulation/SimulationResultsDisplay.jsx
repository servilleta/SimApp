import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectSimulationStatus,
  selectSimulationResults,
  selectMultipleSimulationResults,
  selectSimulationError,
  selectCurrentSimulationId,
  selectCurrentParentSimulationId,
  clearSimulation,
  fetchSimulationStatus,
  cancelSimulation
} from '../../store/simulationSlice';
import {
  deduplicateSimulationIds,
  validateSimulationId,
  getParentId,
  logIdValidationWarning
} from '../../utils/simulationIdUtils';
import { Bar, Chart } from 'react-chartjs-2';
import {
  Chart as ChartJSCore,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { BoxPlotController, BoxAndWiskers } from '@sgratzl/chartjs-chart-boxplot';
import './SimulationResultsDisplay.css';
import CertaintyAnalysis from './CertaintyAnalysis';
import UnifiedProgressTracker from './UnifiedProgressTracker';
import SimpleBoxPlot from './SimpleBoxPlot';
import logger from '../../utils/logger';
import jsPDF from 'jspdf';
import { pdfExportService } from '../../utils/pdfExport';
import { pdfExportService as backendPdfService } from '../../services/pdfExportService';
import { startBackgroundPdfGeneration, checkPdfStatus, downloadPdfInstant, waitAndDownloadPdf } from '../../services/backgroundPdfService';
import html2canvas from 'html2canvas';
import PptxGenJS from 'pptxgenjs';
import { toast } from 'react-toastify';
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';

ChartJSCore.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  BoxPlotController,
  BoxAndWiskers
);

const SimulationResultsDisplay = React.memo(() => {
  const dispatch = useDispatch();
  const status = useSelector(selectSimulationStatus);
  const results = useSelector(selectSimulationResults);
  const multipleResults = useSelector(selectMultipleSimulationResults) || [];
  const error = useSelector(selectSimulationError);
  const inputVariables = useSelector(state => state.simulationSetup?.inputVariables) || [];
  const resultCells = useSelector(state => state.simulationSetup?.resultCells) || [];
  const isAuthenticated = useSelector(state => state.auth?.isAuthenticated);
  
  // State to track slider positions for each target
  const [sliderStateMap, setSliderStateMap] = useState({});
  
  // State for export operations
  const [isPdfExporting, setIsPdfExporting] = useState(false);
  const [isXlsExporting, setIsXlsExporting] = useState(false);
  const [isJsonExporting, setIsJsonExporting] = useState(false);
  const [pdfStatus, setPdfStatus] = useState({}); // Track PDF status per simulation

  // Initialize slider states from global context when component mounts or results change
  useEffect(() => {
    if (typeof window !== 'undefined' && window.simulationSliderStates) {
      console.log('[SimulationResultsDisplay] Initializing slider states from global context:', window.simulationSliderStates);
      setSliderStateMap(prev => ({
        ...prev,
        ...window.simulationSliderStates
      }));
    }
  }, [multipleResults]);

  // Background PDF generation when results are displayed
  useEffect(() => {
    if (multipleResults && multipleResults.length > 0) {
      // Trigger background PDF generation for the first simulation ID
      const primarySimulationId = multipleResults[0]?.simulation_id;
      
      if (primarySimulationId && !pdfStatus[primarySimulationId]) {
        console.log('🔄 [BACKGROUND_PDF] Triggering background PDF generation for:', primarySimulationId);
        
        // Prepare simulation data (same format as used in handleBackendPDFExport)
        const targetsDict = {};
        multipleResults.forEach(result => {
          const targetName = result.result_cell_coordinate || result.target_name || result.name || 'Unknown';
          targetsDict[targetName] = {
            values: result.results?.raw_values || [],
            statistics: {
              mean: result.results?.mean || 0,
              median: result.results?.median || 0,
              std_dev: result.results?.std_dev || 0,
              min: result.results?.min_value || 0,
              max: result.results?.max_value || 0,
              percentiles: result.results?.percentiles || {}
            },
            histogram_data: result.histogram || result.results?.histogram || {},
            sensitivity_analysis: result.results?.sensitivity_analysis || result.sensitivity_analysis || []
          };
        });

        const simulationData = {
          simulationId: primarySimulationId,
          results: {
            targets: targetsDict,
            iterations_run: multipleResults[0]?.iterations_run || 1000,
            requested_engine_type: multipleResults[0]?.requested_engine_type || 'Ultra'
          },
          metadata: {
            iterations_run: multipleResults[0]?.iterations_run || 1000,
            engine_type: multipleResults[0]?.requested_engine_type || 'Ultra',
            timestamp: new Date().toISOString()
          }
        };

        // Start background generation (don't await - fire and forget)
        startBackgroundPdfGeneration(simulationData)
          .then(() => {
            console.log('✅ [BACKGROUND_PDF] Background generation started successfully');
            setPdfStatus(prev => ({
              ...prev,
              [primarySimulationId]: { status: 'generating', progress: 0 }
            }));
          })
          .catch(error => {
            console.error('❌ [BACKGROUND_PDF] Failed to start background generation:', error);
            setPdfStatus(prev => ({
              ...prev,
              [primarySimulationId]: { status: 'failed', error: error.message }
            }));
          });
      }
    }
  }, [multipleResults, pdfStatus]); // Re-run when results change (e.g., when restored)
  
  // Handle slider state changes from CertaintyAnalysis components
  const handleSliderStateChange = useCallback((targetName, rangeValues) => {
    console.log(`[SimulationResultsDisplay] Saving slider state for ${targetName}:`, rangeValues);
    setSliderStateMap(prev => ({
      ...prev,
      [targetName]: rangeValues
    }));
  }, []);
  
  // DEBUG: Add comprehensive logging
  logger.debug('[SimulationResultsDisplay] RENDER DEBUG - status:', status);
  logger.debug('[SimulationResultsDisplay] RENDER DEBUG - multipleResults:', multipleResults);
  logger.debug('[SimulationResultsDisplay] RENDER DEBUG - results:', results);
  logger.debug('[SimulationResultsDisplay] RENDER DEBUG - error:', error);

  // Current simulation id used to seed tracker when multipleResults is empty
  const currentSimulationId = useSelector(selectCurrentSimulationId);
  const currentParentSimulationId = useSelector(selectCurrentParentSimulationId);

  // CRITICAL FIX: Stable simulation IDs with proper memoization and deduplication
  const simulationIds = useMemo(() => {
    // Strategy 1: Use parent ID when available (most stable)
    if (currentParentSimulationId) {
      const parentId = getParentId(currentParentSimulationId);
      logIdValidationWarning(currentParentSimulationId, 'currentParentSimulationId');
      console.debug('[SimulationResultsDisplay] Using parent simulation ID:', parentId);
      return [parentId];
    }

    // Strategy 2: Extract from multipleResults
    if (multipleResults && multipleResults.length > 0) {
      const rawIds = multipleResults
        .map(sim => sim?.simulation_id)
        .filter(Boolean);
      
      // Validate and log corruption warnings
      rawIds.forEach(id => logIdValidationWarning(id, 'multipleResults'));
      
      const dedupedIds = deduplicateSimulationIds(rawIds);
      console.debug('[SimulationResultsDisplay] Deduped simulation IDs from multipleResults:', dedupedIds);
      return dedupedIds;
    }

    // Strategy 3: Fallback to current simulation ID
    if (currentSimulationId) {
      const parentId = getParentId(currentSimulationId);
      logIdValidationWarning(currentSimulationId, 'currentSimulationId fallback');
      console.debug('[SimulationResultsDisplay] Using fallback simulation ID:', parentId);
      return [parentId];
    }

    console.debug('[SimulationResultsDisplay] No simulation IDs available');
    return [];
  }, [currentParentSimulationId, multipleResults, currentSimulationId]);

  // CRITICAL FIX: Expose results and slider states to global context
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.simulationResults = multipleResults || [];
      window.simulationSliderStates = sliderStateMap;
      
      // Set PDF ready flag when we have results (for PDF export service)
      window.pdfDataReady = (multipleResults && multipleResults.length > 0) || false;
    }
  }, [multipleResults, sliderStateMap]);

  // Create mapping from target cell coordinates to display names
  const getTargetDisplayName = (cellCoordinate, resultCell = null) => {
    // First, try to find in resultCells from Redux state
    const targetCell = (resultCells || []).find(cell => 
      cell.name === cellCoordinate || cell.cell === cellCoordinate
    );
    
    if (targetCell && (targetCell.display_name || targetCell.variableName)) {
      return targetCell.display_name || targetCell.variableName;
    }
    
    // If passed a resultCell object directly, check for display name
    if (resultCell && (resultCell.display_name || resultCell.variableName)) {
      return resultCell.display_name || resultCell.variableName;
    }
    
    // ENHANCED: Check if cellCoordinate is actually a target_name already
    if (cellCoordinate && !cellCoordinate.match(/^[A-Z]+[0-9]+$/)) {
      // If it doesn't match cell pattern (like A1, B12), it's likely already a display name
      return cellCoordinate;
    }
    
    // Fallback to cell coordinate
    return cellCoordinate;
  };

  // Callback function to check if results are actually available for a simulation
  const checkResultsAvailable = useCallback((simulationId) => {
    logger.debug(`[SimulationResultsDisplay] 🔍 Checking results for ${simulationId}`);
    
    const simulation = multipleResults.find(sim => sim?.simulation_id === simulationId);
    logger.debug(`[SimulationResultsDisplay] 🔍 Found simulation:`, simulation?.status, !!simulation?.results);

    
    if (!simulation) {
      logger.debug(`[SimulationResultsDisplay] ❌ No simulation found for ${simulationId}`);
      return false;
    }
    
    // Accept both 'completed' and 'pending' if we have actual result data
    // This handles the case where backend is completed but frontend state is still 'pending'
    const hasActualData = !!(
      simulation.results ||                           // Has results object
      simulation.mean !== undefined ||               // Has statistical data
      simulation.histogram ||                        // Has histogram data
      (simulation.bin_edges && simulation.counts) || // Has histogram components
      simulation.iterations_run                      // Has iteration count
    );
    
    return (simulation.status === 'completed' || (simulation.status === 'pending' && hasActualData));
  }, [multipleResults]);

  // State for decimal places control
  const [decimalMap, setDecimalMap] = useState({});
  

  // Determine if we have any running simulations
  const hasRunningSimulations = (status === 'pending' || status === 'running') || multipleResults.some(sim => 
    sim && (sim.status === 'running' || sim.status === 'pending')
  );
  
  // Determine if we have any completed simulations
  const hasCompletedSimulations = multipleResults.some(sim => 
    sim && sim.status === 'completed' && (
      // Check for any of these indicators of actual result data
      sim.results ||                           // Has results object
      sim.mean !== undefined ||               // Has statistical data
      sim.histogram ||                        // Has histogram data
      (sim.bin_edges && sim.counts) ||       // Has histogram components
      sim.iterations_run                      // Has iteration count
    )
  );

  logger.debug('[SimulationResultsDisplay] STATUS DEBUG - status:', status);
  logger.debug('[SimulationResultsDisplay] STATUS DEBUG - hasRunningSimulations:', hasRunningSimulations);
  logger.debug('[SimulationResultsDisplay] STATUS DEBUG - hasCompletedSimulations:', hasCompletedSimulations);

  logger.debug('[SimulationResultsDisplay] STATUS DEBUG - multipleResults:', multipleResults);

  // Filter displayable results
  const displayResults = multipleResults.filter(sim => {
    if (!sim) {
      logger.debug('[SimulationResultsDisplay] 🔍 FILTER DEBUG - sim is null/undefined');
      return false;
    }
    
    if (sim.status !== 'completed') {
      logger.debug('[SimulationResultsDisplay] 🔍 FILTER DEBUG - sim status not completed:', sim.status);
      return false;
    }
    
    // ✅ FIXED: Access nested properties under sim.results
    const hasResults = sim.results;
    const hasMean = sim.results?.mean !== undefined;
    const hasHistogram = sim.results?.histogram;
    const hasBinData = (sim.results?.bin_edges && sim.results?.counts);
    const hasIterations = sim.results?.iterations_run;
    
    logger.debug('[SimulationResultsDisplay] 🔍 FILTER DEBUG - sim:', sim.simulation_id, {
      hasResults: !!hasResults,
      hasMean,
      hasHistogram: !!hasHistogram,
      hasBinData,
      hasIterations: !!hasIterations,
      willPass: !!(hasResults || hasMean || hasHistogram || hasBinData || hasIterations)
    });
    
    return hasResults || hasMean || hasHistogram || hasBinData || hasIterations;
  });

  // Get local status (prioritize having actual completed data)
  // 🔥 FIX: Don't show 'failed' status if we have completed simulations with results
  // Also ignore errors from temp ID polling when real simulation completed successfully
  const localStatus = hasCompletedSimulations ? 'completed' : 
                     (multipleResults.length > 0 && multipleResults.every(sim => sim.status === 'completed')) ? 'completed' : 
                     status;




  // Modern PDF Export with 100% Visual Fidelity
  const handleExportPDF = useCallback(async () => {
    try {
      console.log('🔥 [PDF_EXPORT] PDF Export button clicked!');
      
      // New approach: Use sessionStorage to pass data to a print-friendly view
      const printData = {
        simulationId: simulationIds[0] || `sim_${Date.now()}`,
        results: displayResults,
        metadata: {
          iterations_run: displayResults[0]?.results?.iterations_run || 'N/A',
          engine_type: displayResults[0]?.requested_engine_type || 'Standard',
          timestamp: new Date().toISOString()
        }
      };
      
      // Generate a unique ID for the data
      const printId = `print_data_${Date.now()}`;
      
      // Store the data in sessionStorage
      sessionStorage.setItem(printId, JSON.stringify(printData));
      
      // Open the print view in a new window, passing only the ID
      const printWindow = window.open(`/print-view?id=${printId}`, '_blank', 'width=1200,height=800');
      
      if (!printWindow) {
        toast.error('Popup blocked! Please allow popups for this site to export PDF.');
        // Clean up sessionStorage if the window fails to open
        sessionStorage.removeItem(printId);
      }
      
      console.log('🔥 [PDF_EXPORT] Print view initiated with ID:', printId);

    } catch (error) {
      console.error('PDF export failed:', error);
      toast.error(`❌ PDF export failed: ${error.message}`);
    }
  }, [displayResults, simulationIds]);

  // Instant PDF Export - downloads pre-generated PDF or waits for completion
  const handleBackendPDFExport = useCallback(async () => {
    try {
      console.log('⚡ [INSTANT_PDF] Instant PDF Export button clicked!');
      setIsPdfExporting(true);
      
      const primarySimulationId = multipleResults[0]?.simulation_id;
      if (!primarySimulationId) {
        throw new Error('No simulation ID found');
      }

      // Check if PDF is already ready
      const status = await checkPdfStatus(primarySimulationId);
      
      if (status.file_ready && status.status === 'completed') {
        // PDF is ready - download instantly!
        console.log('⚡ [INSTANT_PDF] PDF ready - downloading instantly!');
        await downloadPdfInstant(primarySimulationId);
        toast.success('📄 PDF downloaded instantly!');
        return;
      }
      
      if (status.status === 'failed') {
        throw new Error(status.error || 'PDF generation failed');
      }
      
      // PDF is still generating - wait for completion then download
      console.log('🔄 [INSTANT_PDF] PDF still generating - waiting for completion...');
      toast.info('📄 PDF is being prepared... Download will start automatically.');
      
      await waitAndDownloadPdf(primarySimulationId, (statusUpdate) => {
        console.log('🔄 [INSTANT_PDF] Status update:', statusUpdate);
        // Optionally update UI with progress
      });
      
      toast.success('📄 PDF downloaded successfully!');

    } catch (error) {
      console.error('⚡ [INSTANT_PDF] PDF export failed:', error);
      toast.error(`❌ PDF export failed: ${error.message}`);
    } finally {
      setIsPdfExporting(false);
    }
  }, [multipleResults]);

  // PowerPoint Export Handler
  const handlePowerPointExport = useCallback(async () => {
    try {
      console.log('🎯 [PPT_EXPORT] PowerPoint Export button clicked!');
      setIsPptExporting(true);

      // Prepare simulation data for PowerPoint export (same format as PDF)
      const targetsDict = {};
      displayResults.forEach(result => {
        const targetName = result.result_cell_coordinate || result.target_name || result.name || 'Unknown';
        
        targetsDict[targetName] = {
          name: targetName,
          values: result.results?.raw_values || [],
          statistics: {
            mean: result.results?.mean || result.statistics?.mean,
            median: result.results?.median || result.statistics?.median,
            std_dev: result.results?.std_dev || result.statistics?.std_dev,
            min_value: result.results?.min_value || result.statistics?.min,
            max_value: result.results?.max_value || result.statistics?.max,
            percentiles: result.results?.percentiles || result.statistics?.percentiles || {},
            sensitivity_analysis: result.results?.sensitivity_analysis || result.sensitivity_analysis || []
          },
          histogram_data: result.histogram || result.results?.histogram || {},
          sensitivity_analysis: result.sensitivity_analysis || []
        };
      });

      const simulationData = {
        simulationId: simulationIds?.[0] || 'ppt_export',
        results: {
          targets: targetsDict,
          iterations_run: results?.iterations_run || 1000,
          requested_engine_type: results?.requested_engine_type || 'Ultra',
          metadata: {
            timestamp: new Date().toISOString(),
            export_type: 'powerpoint',
            aspect_ratio: '16:9'
          }
        },
        metadata: {
          iterations_run: results?.iterations_run || 1000,
          engine_type: results?.requested_engine_type || 'Ultra',
          timestamp: new Date().toISOString()
        }
      };

      console.log('🎯 [PPT_EXPORT] Final simulation data:', simulationData);

      // Export to PowerPoint
      await exportAndDownloadPowerPoint(simulationData);

      toast.success('🎯 PowerPoint exported successfully!');
      console.log('🎯 [PPT_EXPORT] PowerPoint exported successfully');

    } catch (error) {
      console.error('PowerPoint export failed:', error);
      toast.error(`❌ PowerPoint export failed: ${error.message}`);
    } finally {
      setIsPptExporting(false);
    }
  }, [displayResults, getTargetDisplayName, results, simulationIds]);

  // Legacy PDF Export functionality - Fallback method
  const handleLegacyPDFExport = useCallback(async () => {
    try {
      // Show loading state
      const button = document.querySelector('.export-pdf-button');
      const originalText = button?.textContent;
      if (button) {
        button.textContent = '⏳ Generating PDF...';
        button.disabled = true;
      }

      // Create a professional PDF document with better structure
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = 210;
      const pageHeight = 297;
      const margin = 20;
      const usableWidth = pageWidth - (margin * 2);
      let yPosition = margin;

      // Add header
      pdf.setFontSize(24);
      pdf.setTextColor(51, 51, 51);
      pdf.text('Monte Carlo Simulation Results', margin, yPosition);
      yPosition += 15;

      // Add timestamp and metadata
      pdf.setFontSize(12);
      pdf.setTextColor(102, 102, 102);
      const timestamp = new Date().toLocaleString();
      pdf.text(`Generated: ${timestamp}`, margin, yPosition);
      yPosition += 10;

      // Add summary info if available
      if (displayResults.length > 0) {
        const firstResult = displayResults[0];
        const iterations = firstResult.results?.iterations_run || firstResult.iterations_run || 'N/A';
        pdf.text(`Iterations: ${iterations}`, margin, yPosition);
        yPosition += 10;
        
        const engineType = firstResult.requested_engine_type || 'Standard';
        pdf.text(`Engine: ${engineType}`, margin, yPosition);
        yPosition += 15;
        
        pdf.text(`Target Variables: ${displayResults.length}`, margin, yPosition);
        yPosition += 15;
      }

      // Add Box & Whisker Chart Overview (only if multiple variables)
      if (displayResults.length > 1) {
        try {
          // Check if we need a new page
          if (yPosition > pageHeight - 150) {
            pdf.addPage();
            yPosition = margin;
          }

          // Section header
          pdf.setFontSize(18);
          pdf.setTextColor(17, 24, 39); // text-gray-900
          pdf.setFont('helvetica', 'bold');
          pdf.text('Distribution Overview - All Target Variables', margin, yPosition);
          yPosition += 10;
          
          // Add subtitle
          pdf.setFontSize(12);
          pdf.setTextColor(107, 114, 128); // text-gray-500
          pdf.setFont('helvetica', 'normal');
          pdf.text('Box & Whisker plot showing quartiles, median, and outliers for all variables', margin, yPosition);
          yPosition += 15;

          // Capture the Box & Whisker chart
          const boxPlotElement = document.querySelector('.box-plot-overview-section');
          if (boxPlotElement) {
            try {
              // Create a clean container for the box plot
              const boxPlotContainer = document.createElement('div');
              boxPlotContainer.style.width = '800px';
              boxPlotContainer.style.backgroundColor = 'white';
              boxPlotContainer.style.padding = '20px';
              boxPlotContainer.style.fontFamily = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
              
              // Clone the box plot section
              const clonedBoxPlot = boxPlotElement.cloneNode(true);
              
              // Apply styling to ensure proper rendering
              const styleOverrides = `
                * {
                  box-sizing: border-box !important;
                  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
                }
                .box-plot-overview-section {
                  background: white !important;
                  border: 1px solid #e5e7eb !important;
                  border-radius: 12px !important;
                  padding: 1.5rem !important;
                  box-shadow: none !important;
                }
                .box-plot-header h4 {
                  color: #1a1a1a !important;
                  font-size: 1.25rem !important;
                  font-weight: 600 !important;
                  margin: 0 !important;
                }
                .box-plot-container {
                  background: rgba(249, 250, 251, 0.3) !important;
                  border-radius: 6px !important;
                  padding: 0.75rem !important;
                  margin-top: 1rem !important;
                }
              `;
              
              const styleElement = document.createElement('style');
              styleElement.textContent = styleOverrides;
              boxPlotContainer.appendChild(styleElement);
              boxPlotContainer.appendChild(clonedBoxPlot);
              
              // Temporarily add to document for capture
              document.body.appendChild(boxPlotContainer);
              
              // Capture with high quality
              const boxPlotCanvas = await html2canvas(boxPlotContainer, {
                scale: 2,
                useCORS: true,
                backgroundColor: '#ffffff',
                logging: false,
                allowTaint: true,
                foreignObjectRendering: true,
                letterRendering: true,
                width: 800,
                height: Math.max(350, displayResults.length * 50 + 200)
              });
              
              const boxPlotImage = boxPlotCanvas.toDataURL('image/png', 1.0);
              const imgWidth = usableWidth * 0.95;
              const imgHeight = (boxPlotCanvas.height * imgWidth) / boxPlotCanvas.width;
              
              // Center the image
              const imgX = margin + (usableWidth - imgWidth) / 2;
              
              pdf.addImage(boxPlotImage, 'PNG', imgX, yPosition, imgWidth, imgHeight);
              yPosition += imgHeight + 20;
              
              // Clean up
              document.body.removeChild(boxPlotContainer);
              
            } catch (boxPlotError) {
              console.warn('Could not capture Box & Whisker chart:', boxPlotError);
              
              // Fallback: Add text description
              pdf.setFontSize(12);
              pdf.setTextColor(107, 114, 128);
              pdf.text('Box & Whisker chart could not be captured. See individual variable distributions below.', margin, yPosition);
              yPosition += 15;
            }
          }
        } catch (overviewError) {
          console.warn('Could not add Box & Whisker overview:', overviewError);
        }
      }

      // Process each result variable
      for (let i = 0; i < displayResults.length; i++) {
        const targetResult = displayResults[i];
        const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
        const targetName = getTargetDisplayName(cellCoord, targetResult);

        // Check if we need a new page
        if (yPosition > pageHeight - 100) {
          pdf.addPage();
          yPosition = margin;
        }

        // Variable header matching web styling
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(20);
        pdf.setTextColor(17, 24, 39); // text-gray-900
        pdf.text(`Results for ${targetName}`, margin, yPosition);
        yPosition += 10;
        
        // Add subtle underline matching web design
        pdf.setDrawColor(229, 231, 235); // gray-200
        pdf.setLineWidth(1);
        pdf.line(margin, yPosition, margin + usableWidth, yPosition);
        yPosition += 15;

        // Add statistics in a structured format
        if (targetResult.results) {
          const stats = targetResult.results;
          const decPrecision = decimalMap[targetName] ?? 2;
          
          // Local formatter for this variable
          const localFormat = (val) => {
            if (val === null || val === undefined || isNaN(val)) return '—';
            return Number(val).toLocaleString(undefined, { maximumFractionDigits: decPrecision });
          };
          
          pdf.setFontSize(11);
          pdf.setTextColor(68, 68, 68);
          
          // Create statistics table
          const statsData = [
            ['Statistic', 'Value'],
            ['Mean', localFormat(stats.mean, decPrecision)],
            ['Median', localFormat(stats.median, decPrecision)],
            ['Standard Deviation', localFormat(stats.std_dev, decPrecision)],
            ['Minimum', localFormat(stats.min, decPrecision)],
            ['Maximum', localFormat(stats.max, decPrecision)]
          ];

          // Add percentiles if available
          if (stats.percentiles) {
            statsData.push(['5th Percentile', localFormat(stats.percentiles['5'], decPrecision)]);
            statsData.push(['25th Percentile', localFormat(stats.percentiles['25'], decPrecision)]);
            statsData.push(['75th Percentile', localFormat(stats.percentiles['75'], decPrecision)]);
            statsData.push(['95th Percentile', localFormat(stats.percentiles['95'], decPrecision)]);
          }

          // Draw table matching web interface styling
          let tableY = yPosition;
          const rowHeight = 8;
          const colWidth = usableWidth / 2;

          // Table container with modern border
          pdf.setDrawColor(229, 231, 235); // gray-200
          pdf.setLineWidth(1);
          pdf.rect(margin, tableY, usableWidth, rowHeight * statsData.length);

          // Header with web-matching colors
          pdf.setFillColor(249, 250, 251); // gray-50
          pdf.rect(margin, tableY, usableWidth, rowHeight, 'F');
          pdf.setTextColor(17, 24, 39); // text-gray-900
          pdf.setFontSize(12);
          pdf.setFont('helvetica', 'bold');
          pdf.text('Statistic', margin + 6, tableY + 6);
          pdf.text('Value', margin + colWidth + 6, tableY + 6);
          
          // Vertical separator
          pdf.setDrawColor(229, 231, 235);
          pdf.line(margin + colWidth, tableY, margin + colWidth, tableY + rowHeight * statsData.length);
          tableY += rowHeight;

          // Data rows with web-matching styling
          pdf.setFont('helvetica', 'normal');
          pdf.setTextColor(55, 65, 81); // text-gray-700
          for (let j = 1; j < statsData.length; j++) {
            if (j % 2 === 0) {
              pdf.setFillColor(249, 250, 251); // gray-50 alternating
              pdf.rect(margin, tableY, usableWidth, rowHeight, 'F');
            }
            pdf.text(statsData[j][0], margin + 6, tableY + 6);
            pdf.text(statsData[j][1], margin + colWidth + 6, tableY + 6);
            
            // Horizontal separator
            if (j < statsData.length - 1) {
              pdf.setDrawColor(243, 244, 246); // gray-100
              pdf.setLineWidth(0.5);
              pdf.line(margin, tableY + rowHeight, margin + usableWidth, tableY + rowHeight);
            }
            tableY += rowHeight;
          }

          yPosition = tableY + 10;
        }

        // Capture visual elements exactly as they appear in the web interface
        const resultElement = document.querySelector(`[data-variable="${targetName}"]`) || 
                             document.querySelector('.results-content > div:nth-child(' + (i + 1) + ')');
        
        if (resultElement) {
          try {
            // Create a clean container that matches web styling exactly
            const webContainer = document.createElement('div');
            webContainer.style.width = '800px';
            webContainer.style.backgroundColor = 'white';
            webContainer.style.fontFamily = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
            webContainer.style.padding = '0';
            webContainer.style.margin = '0';
            
            // Find and capture charts with exact web styling
            const chartCanvases = resultElement.querySelectorAll('canvas');
            for (const canvas of chartCanvases) {
              if (yPosition > pageHeight - 100) {
                pdf.addPage();
                yPosition = margin;
              }

              // Convert canvas to high-quality image preserving original quality
              const chartImage = canvas.toDataURL('image/png', 1.0);
              const imgWidth = usableWidth * 0.85;
              const imgHeight = (canvas.height * imgWidth) / canvas.width;
              
              // Center the image
              const imgX = margin + (usableWidth - imgWidth) / 2;
              
              pdf.addImage(chartImage, 'PNG', imgX, yPosition, imgWidth, imgHeight);
              yPosition += imgHeight + 15;
            }

            // Capture certainty analysis with exact web styling
            const certaintyAnalysisElement = resultElement.querySelector('.controls-section-modern');
            if (certaintyAnalysisElement) {
              if (yPosition > pageHeight - 120) {
                pdf.addPage();
                yPosition = margin;
              }

              try {
                // Clone the entire certainty section preserving all styles
                const clonedCertainty = certaintyAnalysisElement.cloneNode(true);
                
                // Create a styled container that matches the web exactly
                const certaintyContainer = document.createElement('div');
                certaintyContainer.style.width = '700px';
                certaintyContainer.style.backgroundColor = 'white';
                certaintyContainer.style.padding = '24px';
                certaintyContainer.style.fontFamily = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
                certaintyContainer.style.fontSize = '14px';
                certaintyContainer.style.lineHeight = '1.5';
                certaintyContainer.style.color = '#374151';
                
                // Override any problematic styles for PDF
                const styleOverrides = `
                  * {
                    box-sizing: border-box !important;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
                  }
                  .range-title {
                    font-size: 16px !important;
                    font-weight: 600 !important;
                    color: #374151 !important;
                    margin-bottom: 16px !important;
                  }
                  .range-values span {
                    font-size: 14px !important;
                    font-weight: 600 !important;
                    color: #374151 !important;
                    background: transparent !important;
                    padding: 0 !important;
                  }
                  .slider-container-modern {
                    margin: 12px 0 !important;
                  }
                  .slider-track {
                    background: #e5e7eb !important;
                  }
                  .slider-track-highlight {
                    background: #22c55e !important;
                  }
                  .probability-value {
                    font-size: 32px !important;
                    font-weight: 700 !important;
                    color: #22c55e !important;
                    line-height: 1 !important;
                  }
                  .certainty-level {
                    font-size: 14px !important;
                    font-weight: 600 !important;
                    color: #22c55e !important;
                    text-transform: uppercase !important;
                    letter-spacing: 0.05em !important;
                  }
                  .preset-button-modern {
                    background: #f3f4f6 !important;
                    color: #6b7280 !important;
                    border: 1px solid #d1d5db !important;
                    border-radius: 6px !important;
                    padding: 6px 12px !important;
                    font-size: 12px !important;
                    margin: 2px !important;
                  }
                `;
                
                const styleElement = document.createElement('style');
                styleElement.textContent = styleOverrides;
                certaintyContainer.appendChild(styleElement);
                certaintyContainer.appendChild(clonedCertainty);
                
                // Temporarily add to document for capture
                document.body.appendChild(certaintyContainer);
                
                // Capture with exact web styling
                const certaintyCanvas = await html2canvas(certaintyContainer, {
                  scale: 2,
                  useCORS: true,
                  backgroundColor: '#ffffff',
                  logging: false,
                  allowTaint: true,
                  foreignObjectRendering: true,
                  letterRendering: true,
                  width: 700,
                  height: 300
                });
                
                const certaintyImage = certaintyCanvas.toDataURL('image/png', 1.0);
                const imgWidth = usableWidth * 0.9;
                const imgHeight = (certaintyCanvas.height * imgWidth) / certaintyCanvas.width;
                
                // Center the image
                const imgX = margin + (usableWidth - imgWidth) / 2;
                
                pdf.addImage(certaintyImage, 'PNG', imgX, yPosition, imgWidth, imgHeight);
                yPosition += imgHeight + 20;
                
                // Clean up
                document.body.removeChild(certaintyContainer);
                
              } catch (certaintyError) {
                console.warn('Could not capture styled certainty analysis:', certaintyError);
                
                // Fallback: Add basic certainty info
                const rangeValues = certaintyAnalysisElement.querySelectorAll('.range-values span');
                const probabilityElement = certaintyAnalysisElement.querySelector('.probability-value');
                const certaintyLevelElement = certaintyAnalysisElement.querySelector('.certainty-level');

                if (rangeValues.length >= 2 && probabilityElement) {
                  const minValue = rangeValues[0].textContent.trim();
                  const maxValue = rangeValues[1].textContent.trim();
                  const probability = probabilityElement.textContent.trim();
                  const certaintyLevel = certaintyLevelElement ? certaintyLevelElement.textContent.trim() : '';

                  pdf.setFontSize(16);
                  pdf.setTextColor(55, 65, 81);
                  pdf.text('Certainty Analysis', margin, yPosition);
                  yPosition += 15;

                  pdf.setFontSize(12);
                  pdf.setTextColor(107, 114, 128);
                  pdf.text(`Range: ${minValue} - ${maxValue}`, margin, yPosition);
                  yPosition += 10;

                  pdf.setFontSize(20);
                  pdf.setTextColor(34, 197, 94);
                  pdf.text(`${probability} ${certaintyLevel}`, margin, yPosition);
                  yPosition += 20;
                }
              }
            }
          } catch (chartError) {
            console.warn('Could not capture visual elements:', chartError);
          }
        }

        yPosition += 10;
      }

      // Add footer
      const pageCount = pdf.internal.getNumberOfPages();
      for (let i = 1; i <= pageCount; i++) {
        pdf.setPage(i);
        pdf.setFontSize(8);
        pdf.setTextColor(128, 128, 128);
        pdf.text(`Page ${i} of ${pageCount}`, pageWidth - 30, pageHeight - 10);
        pdf.text('Generated by Monte Carlo Simulation Platform', margin, pageHeight - 10);
      }

      // Save the PDF
      const filename = `monte-carlo-results-${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(filename);

      console.log('Enhanced PDF exported successfully');
      
    } catch (error) {
      console.error('Error exporting PDF:', error);
      alert('Error exporting PDF. Please try again.');
    } finally {
      // Restore button state
      const button = document.querySelector('.export-pdf-button');
      if (button) {
        button.textContent = '🖨️ Print';
        button.disabled = false;
      }
    }
  }, [displayResults, decimalMap, getTargetDisplayName, simulationIds]);

  // PowerPoint Export functionality - Editable PPT
  const handleExportPPT = useCallback(async () => {
    try {
      // Show loading state
      const button = document.querySelector('.export-ppt-button');
      const originalText = button?.textContent;
      if (button) {
        button.textContent = '⏳ Generating PPT...';
        button.disabled = true;
      }

      // Create new presentation
      const pptx = new PptxGenJS();
      
      // Set presentation properties
      pptx.author = 'Monte Carlo Simulation Platform';
      pptx.company = 'Monte Carlo Platform';
      pptx.subject = 'Simulation Results';
      pptx.title = 'Monte Carlo Simulation Results';

      // Title slide
      const titleSlide = pptx.addSlide();
      titleSlide.addText('Monte Carlo Simulation Results', {
        x: 1, y: 2, w: 8, h: 1,
        fontSize: 32, fontFace: 'Arial', bold: true, color: '363636',
        align: 'center'
      });
      
      titleSlide.addText(`Generated: ${new Date().toLocaleString()}`, {
        x: 1, y: 3.2, w: 8, h: 0.5,
        fontSize: 14, fontFace: 'Arial', color: '666666',
        align: 'center'
      });

      if (displayResults.length > 0) {
        const firstResult = displayResults[0];
        const iterations = firstResult.results?.iterations_run || firstResult.iterations_run || 'N/A';
        const engineType = firstResult.requested_engine_type || 'Standard';
        
        titleSlide.addText(`Iterations: ${iterations} | Engine: ${engineType} | Variables: ${displayResults.length}`, {
          x: 1, y: 4, w: 8, h: 0.5,
          fontSize: 12, fontFace: 'Arial', color: '666666',
          align: 'center'
        });
      }

      // Summary slide
      if (displayResults.length > 0) {
        const summarySlide = pptx.addSlide();
        summarySlide.addText('Simulation Summary', {
          x: 0.5, y: 0.5, w: 9, h: 0.8,
          fontSize: 24, fontFace: 'Arial', bold: true, color: '363636'
        });

        // Create summary table
        const summaryData = [
          ['Variable', 'Mean', 'Std Dev', 'Min', 'Max']
        ];

        displayResults.forEach(targetResult => {
          const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
          const targetName = getTargetDisplayName(cellCoord, targetResult);
          const decPrecision = decimalMap[targetName] ?? 2;
          const localFormat = (val) => {
            if (val === null || val === undefined || isNaN(val)) return '—';
            return Number(val).toLocaleString(undefined, { maximumFractionDigits: decPrecision });
          };

          if (targetResult.results) {
            const stats = targetResult.results;
            summaryData.push([
              targetName,
              localFormat(stats.mean),
              localFormat(stats.std_dev),
              localFormat(stats.min),
              localFormat(stats.max)
            ]);
          }
        });

        summarySlide.addTable(summaryData, {
          x: 0.5, y: 1.5, w: 9, h: 4,
          fontSize: 10, fontFace: 'Arial',
          color: '363636',
          fill: { color: 'F5F5F5' },
          border: { pt: 1, color: 'CFCFCF' }
        });
      }

      // Box & Whisker Distribution Overview Slide (only if multiple variables)
      if (displayResults.length > 1) {
        try {
          const boxPlotElement = document.querySelector('.box-plot-overview-section');
          if (boxPlotElement) {
            try {
              // Create a clean container for the box plot
              const boxPlotContainer = document.createElement('div');
              boxPlotContainer.style.width = '800px';
              boxPlotContainer.style.backgroundColor = 'white';
              boxPlotContainer.style.padding = '20px';
              boxPlotContainer.style.fontFamily = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
              
              // Clone the box plot section
              const clonedBoxPlot = boxPlotElement.cloneNode(true);
              
              // Apply styling to ensure proper rendering
              const styleOverrides = `
                * {
                  box-sizing: border-box !important;
                  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
                }
                .box-plot-overview-section {
                  background: white !important;
                  border: 1px solid #e5e7eb !important;
                  border-radius: 12px !important;
                  padding: 1.5rem !important;
                  box-shadow: none !important;
                }
                .box-plot-header h4 {
                  color: #1a1a1a !important;
                  font-size: 1.25rem !important;
                  font-weight: 600 !important;
                  margin: 0 !important;
                }
                .box-plot-container {
                  background: rgba(249, 250, 251, 0.3) !important;
                  border-radius: 6px !important;
                  padding: 0.75rem !important;
                  margin-top: 1rem !important;
                }
              `;
              
              const styleElement = document.createElement('style');
              styleElement.textContent = styleOverrides;
              boxPlotContainer.appendChild(styleElement);
              boxPlotContainer.appendChild(clonedBoxPlot);
              
              // Temporarily add to document for capture
              document.body.appendChild(boxPlotContainer);
              
              // Capture with high quality
              const boxPlotCanvas = await html2canvas(boxPlotContainer, {
                scale: 2,
                useCORS: true,
                backgroundColor: '#ffffff',
                logging: false,
                allowTaint: true,
                foreignObjectRendering: true,
                letterRendering: true,
                width: 800,
                height: Math.max(400, displayResults.length * 60 + 250)
              });
              
              const boxPlotImage = boxPlotCanvas.toDataURL('image/png', 1.0);
              
              // Create Box & Whisker overview slide
              const boxPlotSlide = pptx.addSlide();
              boxPlotSlide.addText('Distribution Overview - All Target Variables', {
                x: 0.5, y: 0.3, w: 9, h: 0.6,
                fontSize: 20, fontFace: 'Arial', bold: true, color: '363636'
              });
              
              boxPlotSlide.addText('Box & Whisker plot showing quartiles, median, and outliers for all variables', {
                x: 0.5, y: 0.9, w: 9, h: 0.4,
                fontSize: 12, fontFace: 'Arial', color: '666666'
              });
              
              // Add the captured chart as image
              boxPlotSlide.addImage({
                data: boxPlotImage,
                x: 0.5, y: 1.4, w: 9, h: 5
              });
              
              // Clean up
              document.body.removeChild(boxPlotContainer);
              
            } catch (boxPlotError) {
              console.warn('Could not capture Box & Whisker chart for PPT:', boxPlotError);
              
              // Fallback: Create a text-based overview slide
              const overviewSlide = pptx.addSlide();
              overviewSlide.addText('Distribution Overview - All Target Variables', {
                x: 0.5, y: 0.3, w: 9, h: 0.6,
                fontSize: 20, fontFace: 'Arial', bold: true, color: '363636'
              });
              
              overviewSlide.addText('Box & Whisker chart visualization not available. See individual variable distributions in following slides.', {
                x: 0.5, y: 1.5, w: 9, h: 1,
                fontSize: 14, fontFace: 'Arial', color: '666666',
                align: 'center'
              });
            }
          }
        } catch (overviewError) {
          console.warn('Could not add Box & Whisker overview slide:', overviewError);
        }
      }

      // Individual slides for each variable
      for (let i = 0; i < displayResults.length; i++) {
        const targetResult = displayResults[i];
        const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
        const targetName = getTargetDisplayName(cellCoord, targetResult);
        const decPrecision = decimalMap[targetName] ?? 2;
        
        const localFormat = (val) => {
          if (val === null || val === undefined || isNaN(val)) return '—';
          return Number(val).toLocaleString(undefined, { maximumFractionDigits: decPrecision });
        };

        const slide = pptx.addSlide();
        
        // Variable title
        slide.addText(`Results for ${targetName}`, {
          x: 0.5, y: 0.3, w: 9, h: 0.6,
          fontSize: 20, fontFace: 'Arial', bold: true, color: '363636'
        });

        if (targetResult.results) {
          const stats = targetResult.results;

          // Statistics table
          const statsData = [
            ['Statistic', 'Value'],
            ['Mean', localFormat(stats.mean)],
            ['Median', localFormat(stats.median)],
            ['Standard Deviation', localFormat(stats.std_dev)],
            ['Minimum', localFormat(stats.min)],
            ['Maximum', localFormat(stats.max)]
          ];

          // Add percentiles if available
          if (stats.percentiles) {
            statsData.push(['5th Percentile', localFormat(stats.percentiles['5'])]);
            statsData.push(['25th Percentile', localFormat(stats.percentiles['25'])]);
            statsData.push(['75th Percentile', localFormat(stats.percentiles['75'])]);
            statsData.push(['95th Percentile', localFormat(stats.percentiles['95'])]);
          }

          slide.addTable(statsData, {
            x: 0.5, y: 1, w: 4, h: 3.5,
            fontSize: 10, fontFace: 'Arial',
            color: '363636',
            fill: { color: 'F8F8F8' },
            border: { pt: 1, color: 'DDDDDD' }
          });

          // Add chart
          const resultElement = document.querySelector(`[data-variable="${targetName}"]`) || 
                               document.querySelector('.results-content > div:nth-child(' + (i + 1) + ')');
          
          if (resultElement) {
            try {
              const chartCanvas = resultElement.querySelector('canvas');
              if (chartCanvas) {
                const chartImage = chartCanvas.toDataURL('image/png', 1.0);
                slide.addImage({
                  data: chartImage,
                  x: 5, y: 1, w: 4, h: 2.5
                });
              }
            } catch (chartError) {
              console.warn('Could not add chart to slide:', chartError);
            }
          }

          // Add certainty analysis if available
          const certaintySection = resultElement?.querySelector('.controls-section-modern');
          if (certaintySection) {
            try {
              const rangeValues = certaintySection.querySelectorAll('.range-values span');
              const probabilityElement = certaintySection.querySelector('.probability-value');
              const certaintyLevelElement = certaintySection.querySelector('.certainty-level');

              if (rangeValues.length >= 2 && probabilityElement) {
                const minValue = rangeValues[0].textContent.trim();
                const maxValue = rangeValues[1].textContent.trim();
                const probability = probabilityElement.textContent.trim();
                const certaintyLevel = certaintyLevelElement ? certaintyLevelElement.textContent.trim() : '';

                // Add certainty analysis section
                slide.addText('Certainty Analysis', {
                  x: 0.5, y: 4.8, w: 9, h: 0.4,
                  fontSize: 14, fontFace: 'Arial', bold: true, color: '363636'
                });

                slide.addText(`Range: ${minValue} - ${maxValue}`, {
                  x: 0.5, y: 5.3, w: 4, h: 0.3,
                  fontSize: 11, fontFace: 'Arial', color: '666666'
                });

                slide.addText(`${probability} ${certaintyLevel}`, {
                  x: 5, y: 5.3, w: 4, h: 0.3,
                  fontSize: 14, fontFace: 'Arial', bold: true, color: '22C55E'
                });

                // Add preset options as text
                slide.addText('Available ranges: 80% Range | Middle 50% | Mean ± 1σ | Full Range', {
                  x: 0.5, y: 5.7, w: 9, h: 0.3,
                  fontSize: 9, fontFace: 'Arial', color: '999999'
                });
              }
            } catch (certaintyError) {
              console.warn('Could not add certainty analysis to slide:', certaintyError);
            }
          }
        }
      }

      // Save the presentation
      const filename = `monte-carlo-results-${new Date().toISOString().split('T')[0]}.pptx`;
      await pptx.writeFile({ fileName: filename });

      console.log('PowerPoint presentation exported successfully');

    } catch (error) {
      console.error('Error exporting PowerPoint:', error);
      alert('Error exporting PowerPoint. Please try again.');
    } finally {
      // Restore button state
      const button = document.querySelector('.export-ppt-button');
      if (button) {
        button.textContent = originalText || '📊 Export PPT';
        button.disabled = false;
      }
    }
  }, [displayResults, decimalMap, getTargetDisplayName, simulationIds]);

  // XLS Export functionality
  const handleExportXLS = useCallback(async () => {
    try {
      console.log('📊 [XLS_EXPORT] XLS Export button clicked!');
      setIsXlsExporting(true);

      // Create a new workbook
      const wb = XLSX.utils.book_new();
      
      // Summary sheet with all variables
      const summaryData = [
        ['Variable', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '5th %ile', '25th %ile', '75th %ile', '95th %ile', 'Iterations']
      ];

      displayResults.forEach(targetResult => {
        const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
        const targetName = getTargetDisplayName(cellCoord, targetResult);
        const results = targetResult.results || {};
        const decPrecision = decimalMap[targetName] ?? 2;
        
        const formatValue = (val) => {
          if (val === null || val === undefined || isNaN(val)) return '';
          return Number(val);
        };

        summaryData.push([
          targetName,
          formatValue(results.mean),
          formatValue(results.median),
          formatValue(results.std_dev),
          formatValue(results.min_value),
          formatValue(results.max_value),
          formatValue(results.percentiles?.['5']),
          formatValue(results.percentiles?.['25']),
          formatValue(results.percentiles?.['75']),
          formatValue(results.percentiles?.['95']),
          results.iterations_run || ''
        ]);
      });

      const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
      XLSX.utils.book_append_sheet(wb, summarySheet, 'Summary');

      // Individual sheets for each variable with raw data
      displayResults.forEach((targetResult, index) => {
        const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
        const targetName = getTargetDisplayName(cellCoord, targetResult);
        const results = targetResult.results || {};
        
        if (results.raw_values && results.raw_values.length > 0) {
          const rawData = [['Iteration', 'Value']];
          results.raw_values.forEach((value, i) => {
            rawData.push([i + 1, Number(value)]);
          });
          
          const rawSheet = XLSX.utils.aoa_to_sheet(rawData);
          // Clean sheet name for Excel compatibility
          const cleanSheetName = targetName.replace(/[^\w\s]/g, '').substring(0, 31);
          XLSX.utils.book_append_sheet(wb, rawSheet, cleanSheetName || `Variable_${index + 1}`);
        }
      });

      // Generate and download the file
      const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
      const blob = new Blob([wbout], { type: 'application/octet-stream' });
      const timestamp = new Date().toISOString().split('T')[0];
      const filename = `simulation_results_${timestamp}.xlsx`;
      
      saveAs(blob, filename);
      
      toast.success('✅ XLS file exported successfully!');
      console.log('📊 [XLS_EXPORT] Export completed successfully');

    } catch (error) {
      console.error('📊 [XLS_EXPORT] Export failed:', error);
      toast.error(`❌ XLS export failed: ${error.message}`);
    } finally {
      setIsXlsExporting(false);
    }
  }, [displayResults, decimalMap, getTargetDisplayName]);

  // JSON Export functionality
  const handleExportJSON = useCallback(async () => {
    try {
      console.log('📋 [JSON_EXPORT] JSON Export button clicked!');
      setIsJsonExporting(true);

      // Create comprehensive JSON export data
      const exportData = {
        metadata: {
          export_timestamp: new Date().toISOString(),
          simulation_id: simulationIds[0] || `sim_${Date.now()}`,
          total_variables: displayResults.length,
          export_format: 'json',
          export_version: '1.0'
        },
        simulation_summary: {},
        results: {}
      };

      // Add simulation summary if available
      if (displayResults.length > 0) {
        const firstResult = displayResults[0];
        exportData.simulation_summary = {
          iterations_run: firstResult.results?.iterations_run || firstResult.iterations_run,
          engine_type: firstResult.requested_engine_type || 'Standard',
          created_at: firstResult.results?.created_at || firstResult.created_at,
          updated_at: firstResult.results?.updated_at || firstResult.updated_at
        };
      }

      // Add detailed results for each variable
      displayResults.forEach(targetResult => {
        const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
        const targetName = getTargetDisplayName(cellCoord, targetResult);
        const results = targetResult.results || {};
        const decPrecision = decimalMap[targetName] ?? 2;

        exportData.results[targetName] = {
          cell_coordinate: cellCoord,
          target_name: targetName,
          decimal_precision: decPrecision,
          statistics: {
            mean: results.mean,
            median: results.median,
            standard_deviation: results.std_dev,
            minimum: results.min_value,
            maximum: results.max_value,
            percentiles: results.percentiles || {},
            skewness: results.skewness,
            kurtosis: results.kurtosis
          },
          histogram_data: results.histogram || [],
          raw_values: results.raw_values || [],
          sensitivity_analysis: results.sensitivity_data || {},
          simulation_metadata: {
            iterations_run: results.iterations_run,
            engine_type: targetResult.requested_engine_type,
            created_at: results.created_at,
            updated_at: results.updated_at,
            status: results.status || targetResult.status
          }
        };
      });

      // Convert to JSON and download
      const jsonString = JSON.stringify(exportData, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const timestamp = new Date().toISOString().split('T')[0];
      const filename = `simulation_results_${timestamp}.json`;
      
      saveAs(blob, filename);
      
      toast.success('✅ JSON file exported successfully!');
      console.log('📋 [JSON_EXPORT] Export completed successfully');

    } catch (error) {
      console.error('📋 [JSON_EXPORT] Export failed:', error);
      toast.error(`❌ JSON export failed: ${error.message}`);
    } finally {
      setIsJsonExporting(false);
    }
  }, [displayResults, decimalMap, getTargetDisplayName, simulationIds]);

  // Copy section as image functionality (captures complete visual context with chart handling)
  const copySectionImage = useCallback(async (sectionType, variableName, buttonElement) => {
    let targetElement = null;
    let originalBorder = '';
    let originalBoxShadow = '';
    
    try {
      // Ensure document is focused before attempting clipboard operation
      if (document.hasFocus && !document.hasFocus()) {
        window.focus();
        await new Promise(resolve => setTimeout(resolve, 100)); // Give time for focus
      }
      
      if (sectionType === 'histogram') {
        // For histogram, capture the entire CertaintyAnalysis component including sliders and percentage
        targetElement = buttonElement.closest('.certainty-analysis-modern');
        console.log('📋 [HISTOGRAM_COPY] Found histogram target element:', targetElement);
        console.log('📋 [HISTOGRAM_COPY] Target element children:', targetElement ? targetElement.children.length : 'null');
        
        // Fallback: if closest doesn't work, try finding the parent container
        if (!targetElement) {
          // Look for the parent that contains both chart and controls
          let parent = buttonElement.parentElement;
          while (parent && !parent.classList.contains('certainty-analysis-modern')) {
            parent = parent.parentElement;
          }
          targetElement = parent;
          console.log('📋 [HISTOGRAM_COPY] Using fallback parent search:', targetElement);
        }
      } else if (sectionType === 'tornado') {
        // For tornado chart, capture the entire tornado section
        targetElement = buttonElement.closest('.tornado-section');
      } else if (sectionType === 'statistics') {
        // For statistics, capture the stats section
        targetElement = buttonElement.closest('.stats-condensed');
      } else if (sectionType === 'boxplot') {
        // For box plot, capture the entire box plot overview section
        targetElement = buttonElement.closest('.box-plot-overview-section');
      }
      
      if (!targetElement) {
        throw new Error(`Could not find ${sectionType} section element`);
      }
      
      // Show loading state
      toast.info(`📷 Capturing ${sectionType} section image...`);
      
      // Add visual highlight to show what's being captured
      originalBorder = targetElement.style.border;
      originalBoxShadow = targetElement.style.boxShadow;
      targetElement.style.border = '2px solid #3B82F6';
      targetElement.style.boxShadow = '0 0 10px rgba(59, 130, 246, 0.3)';
      
      // Brief delay to show the highlight
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Create a composite canvas for the entire section
      const finalCanvas = document.createElement('canvas');
      const finalCtx = finalCanvas.getContext('2d');
      
      // Set high DPI scaling
      const dpr = window.devicePixelRatio || 2;
      const rect = targetElement.getBoundingClientRect();
      finalCanvas.width = rect.width * dpr;
      finalCanvas.height = rect.height * dpr;
      finalCanvas.style.width = rect.width + 'px';
      finalCanvas.style.height = rect.height + 'px';
      
      // Scale the context for high DPI
      finalCtx.scale(dpr, dpr);
      finalCtx.fillStyle = '#ffffff';
      finalCtx.fillRect(0, 0, rect.width, rect.height);
      
      // First, capture Chart.js canvases directly and overlay them
      const chartCanvases = targetElement.querySelectorAll('canvas');
      const chartImages = [];
      
      for (const chartCanvas of chartCanvases) {
        try {
          const chartRect = chartCanvas.getBoundingClientRect();
          const targetRect = targetElement.getBoundingClientRect();
          
          // Calculate relative position
          const relativeX = chartRect.left - targetRect.left;
          const relativeY = chartRect.top - targetRect.top;
          
          // Create high-quality chart image
          const tempCanvas = document.createElement('canvas');
          const tempCtx = tempCanvas.getContext('2d');
          tempCanvas.width = chartCanvas.width;
          tempCanvas.height = chartCanvas.height;
          tempCtx.drawImage(chartCanvas, 0, 0);
          
          chartImages.push({
            canvas: tempCanvas,
            x: relativeX,
            y: relativeY,
            width: chartRect.width,
            height: chartRect.height
          });
        } catch (chartError) {
          console.warn('Failed to capture chart canvas:', chartError);
        }
      }
      
      // Import html2canvas for background elements
      const html2canvas = (await import('html2canvas')).default;
      
      // Create a clone of the target element without canvas elements for background capture
      const clonedElement = targetElement.cloneNode(true);
      const clonedCanvases = clonedElement.querySelectorAll('canvas');
      clonedCanvases.forEach(canvas => {
        // Replace canvas with placeholder div to maintain layout
        const placeholder = document.createElement('div');
        placeholder.style.width = canvas.style.width || canvas.getAttribute('width') + 'px';
        placeholder.style.height = canvas.style.height || canvas.getAttribute('height') + 'px';
        placeholder.style.backgroundColor = 'transparent';
        placeholder.style.display = 'inline-block';
        canvas.parentNode.replaceChild(placeholder, canvas);
      });
      
      // Temporarily add to DOM for html2canvas
      clonedElement.style.position = 'absolute';
      clonedElement.style.left = '-9999px';
      clonedElement.style.top = '-9999px';
      document.body.appendChild(clonedElement);
      
      try {
        // Capture background elements (text, styling, non-canvas elements)
        const backgroundCanvas = await html2canvas(clonedElement, {
          backgroundColor: '#ffffff',
          scale: 1, // We handle scaling ourselves
          useCORS: true,
          allowTaint: false,
          foreignObjectRendering: true,
          logging: false,
          width: rect.width,
          height: rect.height
        });
        
        // Draw background on final canvas
        finalCtx.drawImage(backgroundCanvas, 0, 0, rect.width, rect.height);
        
      } finally {
        // Clean up cloned element
        document.body.removeChild(clonedElement);
      }
      
      // Overlay chart images on top
      chartImages.forEach(chartImg => {
        finalCtx.drawImage(
          chartImg.canvas,
          chartImg.x,
          chartImg.y,
          chartImg.width,
          chartImg.height
        );
      });
      
      // Convert final canvas to blob
      const blob = await new Promise(resolve => {
        finalCanvas.toBlob(resolve, 'image/png', 1.0);
      });
      
      if (!blob) {
        throw new Error('Failed to create image blob');
      }
      
      // Try multiple approaches to copy to clipboard
      let clipboardSuccess = false;
      
      // Attempt 1: Try with current focus state
      if (navigator.clipboard && navigator.clipboard.write) {
        try {
          const clipboardItem = new ClipboardItem({ 'image/png': blob });
          await navigator.clipboard.write([clipboardItem]);
          clipboardSuccess = true;
          
          toast.success(`✅ ${sectionType.charAt(0).toUpperCase() + sectionType.slice(1)} section image copied to clipboard!`);
          console.log(`📋 [COPY_SECTION_IMAGE] ${sectionType} section image copied for ${variableName}`);
          return; // Success, exit early
        } catch (clipboardError) {
          console.warn('📋 [COPY_SECTION_IMAGE] First clipboard attempt failed:', clipboardError);
        }
      }
      
      // Attempt 2: Try focusing the window and retrying
      if (!clipboardSuccess && navigator.clipboard && navigator.clipboard.write) {
        try {
          // Force focus and wait a bit longer
          window.focus();
          document.body.focus();
          await new Promise(resolve => setTimeout(resolve, 300));
          
          const clipboardItem = new ClipboardItem({ 'image/png': blob });
          await navigator.clipboard.write([clipboardItem]);
          clipboardSuccess = true;
          
          toast.success(`✅ ${sectionType.charAt(0).toUpperCase() + sectionType.slice(1)} section image copied to clipboard!`);
          console.log(`📋 [COPY_SECTION_IMAGE] ${sectionType} section image copied for ${variableName} (after focus retry)`);
          return; // Success, exit early
        } catch (clipboardError) {
          console.warn('📋 [COPY_SECTION_IMAGE] Second clipboard attempt failed:', clipboardError);
        }
      }
      
      // Attempt 3: Try user interaction approach
      if (!clipboardSuccess && navigator.clipboard && navigator.clipboard.write) {
        try {
          // Create a temporary button for user interaction
          toast.info(`🖱️ Click the page and try again for clipboard copy, or the image will download automatically`);
          
          // Give user a moment to click, then proceed to download
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          console.warn('📋 [COPY_SECTION_IMAGE] All clipboard attempts failed, using download fallback');
        } catch (error) {
          console.warn('📋 [COPY_SECTION_IMAGE] Final clipboard attempt setup failed:', error);
        }
      }
      
      // Fallback: download the image
      const dataURL = finalCanvas.toDataURL('image/png', 1.0);
      const link = document.createElement('a');
      link.download = `${variableName}_${sectionType}_section.png`;
      link.href = dataURL;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      toast.success(`📄 ${sectionType.charAt(0).toUpperCase() + sectionType.slice(1)} section image downloaded!`);
      console.log(`📄 [COPY_SECTION_IMAGE] ${sectionType} section downloaded for ${variableName}`);
      
    } catch (error) {
      console.error('📋 [COPY_SECTION_IMAGE] Failed to copy/download section image:', error);
      toast.error(`❌ Failed to capture ${sectionType} section: ${error.message || 'Unknown error'}`);
    } finally {
      // Restore original styling
      if (targetElement) {
        targetElement.style.border = originalBorder;
        targetElement.style.boxShadow = originalBoxShadow;
      }
    }
  }, []);

  // Copy statistics data functionality
  const copyStatsData = useCallback(async (targetResult, variableName) => {
    try {
      const results = targetResult.results || {};
      const decPrecision = decimalMap[variableName] ?? 2;
      
      const formatValue = (val) => {
        if (val === null || val === undefined || isNaN(val)) return 'N/A';
        return Number(val).toFixed(decPrecision);
      };

      let copyText = `${variableName} - Statistical Summary\n\n`;
      copyText += `Mean: ${formatValue(results.mean)}\n`;
      copyText += `Median: ${formatValue(results.median)}\n`;
      copyText += `Standard Deviation: ${formatValue(results.std_dev)}\n`;
      copyText += `Minimum: ${formatValue(results.min_value)}\n`;
      copyText += `Maximum: ${formatValue(results.max_value)}\n\n`;
      
      if (results.percentiles) {
        copyText += 'Percentiles:\n';
        Object.entries(results.percentiles).forEach(([p, val]) => {
          copyText += `${p}%: ${formatValue(val)}\n`;
        });
      }

      if (results.iterations_run) {
        copyText += `\nIterations: ${results.iterations_run}\n`;
      }
      
      await navigator.clipboard.writeText(copyText);
      toast.success('✅ Statistics copied to clipboard!');
      console.log(`📋 [COPY_STATS] Statistics copied for ${variableName}`);
      
    } catch (error) {
      console.error('📋 [COPY_STATS] Failed to copy statistics:', error);
      toast.error(`❌ Failed to copy statistics: ${error.message}`);
    }
  }, [decimalMap]);

  // Chart options for consistent styling
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Results Histogram',
        font: {
          size: 14,
          weight: 600
        },
        color: '#1A1A1A' // var(--color-charcoal)
      },
      tooltip: {
        callbacks: {
            label: function(context) {
                let label = context.dataset.label || '';
                if (label) {
                    label += ': ';
                }
                if (context.parsed && context.parsed.y !== null) {
                    label += context.parsed.y;
                }
                return label;
            }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Value Bins',
          color: '#333333' // var(--color-dark-grey)
        },
        ticks: {
          color: '#777777' // var(--color-medium-grey)
        },
        grid: {
          color: 'rgba(232, 232, 232, 0.3)' // var(--color-border-light) with transparency
        }
      },
      y: {
        title: {
          display: true,
          text: 'Frequency',
          color: '#333333' // var(--color-dark-grey)
        },
        beginAtZero: true,
        ticks: {
          color: '#777777' // var(--color-medium-grey)
        },
        grid: {
          color: 'rgba(232, 232, 232, 0.3)' // var(--color-border-light) with transparency
        }
      },
    },
  };

  const tornadoChartOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false, // Critical for dynamic height
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Variable Impact Analysis',
        font: {
          size: 14,
          weight: 600
        },
        color: '#1A1A1A' // var(--color-charcoal)
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Impact %',
          color: '#333333' // var(--color-dark-grey)
        },
        beginAtZero: true,
        ticks: {
          color: '#777777' // var(--color-medium-grey)
        },
        grid: {
          color: 'rgba(232, 232, 232, 0.3)' // var(--color-border-light) with transparency
        }
      },
      y: {
        ticks: {
          color: '#777777', // var(--color-medium-grey)
          maxRotation: 0,
          minRotation: 0,
          font: {
            size: 11
          },
          maxTicksLimit: 50, // Allow up to 50 variables
          autoSkip: false // Don't skip any labels
        },
        grid: {
          display: false
        }
      }
    }
  };

  // Box & Whisker chart options with dual Y-axes
  const boxPlotOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    layout: {
      padding: {
        top: 20,
        bottom: 20,
        left: 10,
        right: 10
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: '#666',
          font: {
            size: 12
          },
          generateLabels: function(chart) {
            const datasets = chart.data.datasets;
            const labels = [];
            
            datasets.forEach((dataset, index) => {
              if (dataset.yAxisID === 'y') {
                labels.push({
                  text: 'Values (Left Axis)',
                  fillStyle: '#FF6B35', // Braun orange
                  strokeStyle: '#FF6B35',
                  lineWidth: 2,
                  datasetIndex: index
                });
              } else if (dataset.yAxisID === 'y1') {
                labels.push({
                  text: 'Percentages (Right Axis)',
                  fillStyle: '#FFD700', // Subtle yellow
                  strokeStyle: '#FFD700',
                  lineWidth: 2,
                  datasetIndex: index
                });
              }
            });
            
            return labels;
          }
        }
      },
      title: {
        display: true,
        text: 'Target Variables Distribution Overview',
        font: {
          size: 16,
          weight: 600
        },
        color: '#1A1A1A' // var(--color-charcoal)
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const data = context.parsed;
            const isPercentage = context.dataset.yAxisID === 'y1';
            const suffix = isPercentage ? '%' : '';
            const precision = isPercentage ? 1 : 2;
            
            return [
              `Min: ${data.min?.toFixed(precision) || 'N/A'}${suffix}`,
              `Q1: ${data.q1?.toFixed(precision) || 'N/A'}${suffix}`,
              `Median: ${data.median?.toFixed(precision) || 'N/A'}${suffix}`,
              `Q3: ${data.q3?.toFixed(precision) || 'N/A'}${suffix}`,
              `Max: ${data.max?.toFixed(precision) || 'N/A'}${suffix}`,
              `Outliers: ${data.outliers?.length || 0}`
            ];
          }
        }
      }
    },
    scales: {
      x: {
        type: 'category',
        title: {
          display: true,
          text: 'Target Variables',
          color: '#333333' // var(--color-dark-grey)
        },
        ticks: {
          color: '#777777', // var(--color-medium-grey)
          maxRotation: 0, // Keep labels horizontal for better alignment
          minRotation: 0,
          font: {
            size: 12
          },
          align: 'center', // Center align the labels
          padding: 10 // Add some padding for better spacing
        },
        grid: {
          color: 'rgba(232, 232, 232, 0.3)', // var(--color-border-light) with transparency
          display: false
        },
        offset: false, // Remove offset for better centering
        stacked: false, // Ensure variables are not stacked
        bounds: 'data' // Use data bounds for better alignment
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        beginAtZero: false,
        title: {
          display: true,
          text: 'Values',
          color: '#FF6B35', // Braun orange for decimal values
          font: {
            size: 14,
            weight: 600
          }
        },
        ticks: {
          color: '#FF6B35', // Braun orange ticks for decimal values
          font: {
            size: 12
          }
        },
        grid: {
          color: 'rgba(255, 107, 53, 0.1)' // Braun orange grid lines for decimal values
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        beginAtZero: false,
        title: {
          display: true,
          text: 'Percentages (%)',
          color: '#FFD700', // Subtle yellow for percentage values
          font: {
            size: 14,
            weight: 600
          }
        },
        ticks: {
          color: '#FFD700', // Subtle yellow ticks for percentage values
          font: {
            size: 12
          },
          callback: function(value) {
            return value.toFixed(1) + '%';
          }
        },
        grid: {
          drawOnChartArea: false, // only want the grid lines for one axis to show up
        }
      }
    }
  };

  // Helper function to create chart data from simulation results
  const createChartData = (targetResult) => {
    try {
      // ✅ FIXED: Access nested properties under targetResult.results
      const results = targetResult.results || {};
      
      // DEBUG: Log the structure to see what's available
      console.log('DEBUG createChartData - targetResult:', {
        has_results: !!results,
        results_keys: Object.keys(results),
        histogram_exists: !!results.histogram,
        histogram_type: typeof results.histogram,
        histogram_structure: results.histogram ? {
          is_array: Array.isArray(results.histogram),
          length: results.histogram.length || 'N/A',
          has_bins: !!results.histogram.bins,
          has_bin_edges: !!results.histogram.bin_edges,
          has_counts: !!results.histogram.counts,
          sample: results.histogram
        } : 'No histogram'
      });
      
      // Strategy 1: Use existing histogram data if available
      if (results.histogram && Array.isArray(results.histogram) && results.histogram.length > 0) {
        logger.debug('Using existing histogram data:', results.histogram.length, 'bins');
        return {
          labels: results.histogram.map((_, index) => `Bin ${index + 1}`),
          datasets: [{
            data: results.histogram,
            backgroundColor: 'rgba(255, 149, 85, 0.6)', // Braun orange with transparency
            borderColor: 'rgba(255, 149, 85, 1)', // Solid Braun orange
            borderWidth: 1
          }]
        };
      }

      // Strategy 2: Use histogram object with bin_edges and counts if available
      if (results.histogram && typeof results.histogram === 'object' && 
          results.histogram.bin_edges && results.histogram.counts &&
          Array.isArray(results.histogram.bin_edges) && Array.isArray(results.histogram.counts)) {
        logger.debug('Using histogram.bin_edges and histogram.counts:', results.histogram.counts.length, 'bins');
        
        // Create labels from bin edges (showing range for each bin)
        const labels = results.histogram.bin_edges.slice(0, -1).map((edge, index) => {
          const nextEdge = results.histogram.bin_edges[index + 1];
          if (nextEdge !== undefined) {
            return `${edge.toFixed(0)}-${nextEdge.toFixed(0)}`;
          }
          return `${edge.toFixed(0)}+`;
        });
        
        return {
          labels: labels,
          datasets: [{
            data: results.histogram.counts,
            backgroundColor: 'rgba(255, 149, 85, 0.6)',
            borderColor: 'rgba(255, 149, 85, 1)',
            borderWidth: 1
          }]
        };
      }

      // Strategy 3: Use direct bin_edges and counts if available (legacy format)
      if (results.bin_edges && results.counts && 
          Array.isArray(results.bin_edges) && Array.isArray(results.counts)) {
        logger.debug('Using bin_edges and counts:', results.counts.length, 'bins');
        
        // Create labels from bin edges (showing range for each bin)
        const labels = results.bin_edges.slice(0, -1).map((edge, index) => {
          const nextEdge = results.bin_edges[index + 1];
          if (nextEdge !== undefined) {
            return `${edge.toFixed(0)}-${nextEdge.toFixed(0)}`;
          }
          return `${edge.toFixed(0)}+`;
        });
        
        return {
          labels: labels,
          datasets: [{
            data: results.counts,
            backgroundColor: 'rgba(255, 149, 85, 0.6)',
            borderColor: 'rgba(255, 149, 85, 1)',
            borderWidth: 1
          }]
        };
      }

      // Strategy 4: Fallback - create histogram from raw data if available
      if (results.data && Array.isArray(results.data) && results.data.length > 0) {
        logger.debug('Creating histogram from raw data:', results.data.length, 'points');
        
        const data = results.data.filter(val => val !== null && val !== undefined && !isNaN(val));
        if (data.length === 0) return null;
        
        const min = Math.min(...data);
        const max = Math.max(...data);
        const binCount = Math.min(50, Math.max(10, Math.floor(Math.sqrt(data.length))));
        const binWidth = (max - min) / binCount;
        
        const bins = Array(binCount).fill(0);
        const binLabels = [];
        
        for (let i = 0; i < binCount; i++) {
          const binStart = min + (i * binWidth);
          const binEnd = min + ((i + 1) * binWidth);
          binLabels.push(`${binStart.toFixed(0)}-${binEnd.toFixed(0)}`);
        }
        
        data.forEach(value => {
          const binIndex = Math.min(Math.floor((value - min) / binWidth), binCount - 1);
          bins[binIndex]++;
        });
        
        return {
          labels: binLabels,
          datasets: [{
            data: bins,
            backgroundColor: 'rgba(255, 149, 85, 0.6)',
            borderColor: 'rgba(255, 149, 85, 1)',
            borderWidth: 1
          }]
        };
      }

      logger.debug('No suitable data found for histogram creation');
      return null;
    } catch (error) {
      logger.error('Error creating chart data:', error);
      return null;
    }
  };

  // Helper function to detect if a variable is percentage-based
  const isPercentageVariable = (targetResult) => {
    const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
    
    // Check if we have format information from variable definition
    const inputVariable = inputVariables.find(v => v.name === cellCoord || v.cell === cellCoord);
    const resultCell = resultCells.find(v => v.name === cellCoord || v.cell === cellCoord);
    
    if (inputVariable?.format === 'percentage' || resultCell?.format === 'percentage') {
      return true;
    }
    
    // Enhanced fallback: detect from values with improved logic
    const results = targetResult.results || {};
    const maxValue = results.max_value || 0;
    const minValue = results.min_value || 0;
    const range = maxValue - minValue;
    
    // Check if values are in typical percentage range (0-1 or close to it)
    const isIn01Range = maxValue <= 1 && minValue >= 0 && range > 0;
    
    // Check if values are in -1 to 1 range (might be correlation coefficients, etc.)
    const isInNeg1To1Range = maxValue <= 1 && minValue >= -1 && range > 0;
    
    // Check if the range is very small but values are large (not percentage)
    const isSmallRangeButLargeValues = range < 2 && Math.abs(maxValue) > 100;
    
    // Final determination: percentage if in 0-1 or -1 to 1 range, but NOT if large values with small range
    const isPercentage = (isIn01Range || isInNeg1To1Range) && !isSmallRangeButLargeValues;
    
    return isPercentage;
  };

  // Helper function to create Box & Whisker chart data for all target variables with dual axes
  const createBoxPlotData = () => {
    try {
      if (!displayResults || displayResults.length === 0) {
        console.log('🔍 BOX PLOT DEBUG: No displayResults available');
        return null;
      }

      console.log('🔍 BOX PLOT DEBUG: Creating box plot for', displayResults.length, 'variables:', 
        displayResults.map(r => r.result_cell_coordinate || r.target_name));

      // Check if we have mixed types - if not, use single axis
      const variableTypes = displayResults.map(targetResult => isPercentageVariable(targetResult));
      const hasDecimals = variableTypes.some(isPerc => !isPerc);
      const hasPercentages = variableTypes.some(isPerc => isPerc);
      
      console.log('🔍 BOX PLOT DEBUG: Variable types:', variableTypes);
      console.log('🔍 BOX PLOT DEBUG: hasDecimals:', hasDecimals, 'hasPercentages:', hasPercentages);
      
      // REVERT: Go back to dual-axis approach - that's the correct solution for mixed scales
      // Left axis: normal numbers (B12: ~183,346), Right axis: percentages (B13: ~36.7%)
      const needsDualAxis = hasDecimals && hasPercentages;
      console.log('🔍 BOX PLOT DEBUG: needsDualAxis:', needsDualAxis, '(proper solution for mixed scales)');

      // If we don't need dual axis, create a simpler single-axis chart
      if (!needsDualAxis) {
        const allLabels = [];
        const allData = [];
        const isAllPercentages = hasPercentages && !hasDecimals;
        const hasMixedTypes = hasDecimals && hasPercentages;
        
        console.log('🔍 BOX PLOT DEBUG: Single-axis approach - isAllPercentages:', isAllPercentages, 'hasMixedTypes:', hasMixedTypes);
        
        displayResults.forEach((targetResult, index) => {
          const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
          const targetName = getTargetDisplayName(cellCoord, targetResult);
          const results = targetResult.results || {};
          
          console.log('🔍 BOX PLOT DEBUG: Processing variable', index, ':', cellCoord, 'displayName:', targetName);
          console.log('🔍 BOX PLOT DEBUG: Results keys:', Object.keys(results));
          
          allLabels.push(targetName);
          
          // Calculate box plot statistics
          let values = [];
          
          if (results.percentiles) {
            const percentiles = results.percentiles;
            values = [
              percentiles['0'] || results.min_value || 0,
              percentiles['25'] || percentiles['5'] || results.min_value || 0,
              percentiles['50'] || results.median || 0,
              percentiles['75'] || percentiles['95'] || results.max_value || 0,
              percentiles['100'] || results.max_value || 0
            ].filter(v => v !== null && v !== undefined);
          } else {
            values = [
              results.min_value || 0,
              results.min_value || 0,
              results.mean || 0,
              results.max_value || 0,
              results.max_value || 0
            ].filter(v => v !== null && v !== undefined);
          }

          if (values.length > 0) {
            values.sort((a, b) => a - b);
            
            let min = Math.min(...values);
            let max = Math.max(...values);
            let q1 = results.percentiles?.['25'] || values[Math.floor(values.length * 0.25)];
            let median = results.percentiles?.['50'] || results.median || values[Math.floor(values.length * 0.5)];
            let q3 = results.percentiles?.['75'] || values[Math.floor(values.length * 0.75)];
            
            // Convert to percentage display if needed
            const isThisVariablePercentage = isPercentageVariable(targetResult);
            if (isAllPercentages || (hasMixedTypes && isThisVariablePercentage)) {
              min = min * 100;
              max = max * 100;
              q1 = q1 * 100;
              median = median * 100;
              q3 = q3 * 100;
              values = values.map(v => v * 100);
              console.log('🔍 BOX PLOT DEBUG: Converted', targetName, 'to percentage format');
            }
            
            const iqr = q3 - q1;
            const lowerFence = q1 - 1.5 * iqr;
            const upperFence = q3 + 1.5 * iqr;
            const outliers = values.filter(v => v < lowerFence || v > upperFence);

            console.log('🔍 BOX PLOT DEBUG: Adding data for', targetName, ':', {min, q1, median, q3, max, outliers});
            const boxData = {
              min: min,
              q1: q1,
              median: median,
              q3: q3,
              max: max,
              outliers: outliers
            };
            console.log('🔍 BOX PLOT DEBUG: Box data structure:', boxData);
            allData.push(boxData);
          } else {
            allData.push(null);
          }
        });

        const result = {
          labels: allLabels,
          datasets: [{
            label: hasMixedTypes ? 'Mixed Values' : (isAllPercentages ? 'Percentages' : 'Values'),
            data: allData,
            backgroundColor: isAllPercentages ? 'rgba(255, 215, 0, 0.6)' : 'rgba(255, 107, 53, 0.6)',
            borderColor: isAllPercentages ? '#FFD700' : '#FF6B35',
            borderWidth: 2,
            outlierBackgroundColor: 'rgba(211, 47, 47, 0.8)',
            outlierBorderColor: '#D32F2F',
            outlierRadius: 4,
            itemRadius: 0,
            itemStyle: 'circle',
            itemBackgroundColor: isAllPercentages ? 'rgba(255, 215, 0, 0.8)' : 'rgba(255, 107, 53, 0.8)',
            itemBorderColor: isAllPercentages ? '#FFD700' : '#FF6B35',
            medianColor: '#1A1A1A',
            lowerBackgroundColor: isAllPercentages ? 'rgba(255, 215, 0, 0.3)' : 'rgba(255, 107, 53, 0.3)',
            upperBackgroundColor: isAllPercentages ? 'rgba(255, 215, 0, 0.6)' : 'rgba(255, 107, 53, 0.6)',
            yAxisID: isAllPercentages ? 'y1' : 'y'
          }]
        };
        
        console.log('🔍 BOX PLOT DEBUG: Final single-axis chart data:', result);
        console.log('🔍 BOX PLOT DEBUG: Labels count:', allLabels.length, 'Data count:', allData.length);
        
        return result;
      }

      // SIMPLE BOX PLOT: Let's get it working first, then add dual-axis
      console.log('🔍 BOX PLOT DEBUG: Using SIMPLE approach (no dual-axis)');
      const allLabels = [];
      const decimalData = [];
      const percentageData = [];

      displayResults.forEach((targetResult, index) => {
        const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
        const targetName = getTargetDisplayName(cellCoord, targetResult);
        const results = targetResult.results || {};
        const isPercentage = isPercentageVariable(targetResult);
        
        console.log('🔍 BOX PLOT DEBUG: Processing dual-axis variable', index, ':', cellCoord, 'isPercentage:', isPercentage);
        
        allLabels.push(targetName);

        // Calculate box plot statistics
        let values = [];
        
        if (results.percentiles) {
          const percentiles = results.percentiles;
          values = [
            percentiles['0'] || results.min_value || 0,
            percentiles['25'] || percentiles['5'] || results.min_value || 0,
            percentiles['50'] || results.median || 0,
            percentiles['75'] || percentiles['95'] || results.max_value || 0,
            percentiles['100'] || results.max_value || 0
          ].filter(v => v !== null && v !== undefined);
        } else {
          values = [
            results.min_value || 0,
            results.min_value || 0,
            results.mean || 0,
            results.max_value || 0,
            results.max_value || 0
          ].filter(v => v !== null && v !== undefined);
        }

        if (values.length > 0) {
          values.sort((a, b) => a - b);
          
          let min = Math.min(...values);
          let max = Math.max(...values);
          let q1 = results.percentiles?.['25'] || values[Math.floor(values.length * 0.25)];
          let median = results.percentiles?.['50'] || results.median || values[Math.floor(values.length * 0.5)];
          let q3 = results.percentiles?.['75'] || values[Math.floor(values.length * 0.75)];
          
          if (isPercentage) {
            min = min * 100;
            max = max * 100;
            q1 = q1 * 100;
            median = median * 100;
            q3 = q3 * 100;
            values = values.map(v => v * 100);
          }
          
          const iqr = q3 - q1;
          const lowerFence = q1 - 1.5 * iqr;
          const upperFence = q3 + 1.5 * iqr;
          const outliers = values.filter(v => v < lowerFence || v > upperFence);

          const boxData = {
            min: min,
            q1: q1,
            median: median,
            q3: q3,
            max: max,
            outliers: outliers
          };

          console.log('🔍 BOX PLOT DEBUG: Created boxData for', targetName, ':', boxData);

          if (isPercentage) {
            console.log('🔍 BOX PLOT DEBUG: Adding', targetName, 'to PERCENTAGE data');
            percentageData.push(boxData);
            decimalData.push(null);
            console.log('🔍 BOX PLOT DEBUG: percentageData now:', percentageData);
          } else {
            console.log('🔍 BOX PLOT DEBUG: Adding', targetName, 'to DECIMAL data');
            decimalData.push(boxData);
            percentageData.push(null);
            console.log('🔍 BOX PLOT DEBUG: decimalData now:', decimalData);
          }
        } else {
          decimalData.push(null);
          percentageData.push(null);
        }
      });

      const datasets = [];
      
      console.log('🔍 BOX PLOT DEBUG: Dual-axis data arrays:');
      console.log('🔍 BOX PLOT DEBUG: decimalData:', decimalData);
      console.log('🔍 BOX PLOT DEBUG: percentageData:', percentageData);
      
      if (decimalData.some(d => d !== null)) {
        console.log('🔍 BOX PLOT DEBUG: Adding DECIMAL dataset to chart');
        datasets.push({
          label: 'Values (Left Axis)',
          data: decimalData,
          backgroundColor: 'rgba(255, 107, 53, 0.6)',
          borderColor: '#FF6B35',
          borderWidth: 2,
          outlierBackgroundColor: 'rgba(211, 47, 47, 0.8)',
          outlierBorderColor: '#D32F2F',
          outlierRadius: 4,
          itemRadius: 0,
          itemStyle: 'circle',
          itemBackgroundColor: 'rgba(255, 107, 53, 0.8)',
          itemBorderColor: '#FF6B35',
          medianColor: '#1A1A1A',
          lowerBackgroundColor: 'rgba(255, 107, 53, 0.3)',
          upperBackgroundColor: 'rgba(255, 107, 53, 0.6)',
          yAxisID: 'y'
        });
      }

      if (percentageData.some(d => d !== null)) {
        console.log('🔍 BOX PLOT DEBUG: Adding PERCENTAGE dataset to chart');
        datasets.push({
          label: 'Percentages (Right Axis)',
          data: percentageData,
          backgroundColor: 'rgba(255, 215, 0, 0.6)',
          borderColor: '#FFD700',
          borderWidth: 2,
          outlierBackgroundColor: 'rgba(211, 47, 47, 0.8)',
          outlierBorderColor: '#D32F2F',
          outlierRadius: 4,
          itemRadius: 0,
          itemStyle: 'circle',
          itemBackgroundColor: 'rgba(255, 215, 0, 0.8)',
          itemBorderColor: '#FFD700',
          medianColor: '#1A1A1A',
          lowerBackgroundColor: 'rgba(255, 215, 0, 0.3)',
          upperBackgroundColor: 'rgba(255, 215, 0, 0.6)',
          yAxisID: 'y1'
        });
      }

      const result = {
        labels: allLabels,
        datasets: datasets
      };
      
      console.log('🔍 BOX PLOT DEBUG: Final dual-axis chart data:', result);
      console.log('🔍 BOX PLOT DEBUG: Dual-axis labels:', allLabels);
      console.log('🔍 BOX PLOT DEBUG: Dual-axis datasets count:', datasets.length);
      
      // Debug each dataset's data array
      datasets.forEach((dataset, idx) => {
        console.log(`🔍 BOX PLOT DEBUG: Dataset ${idx} (${dataset.label}):`, dataset.data);
        dataset.data.forEach((dataPoint, dpIdx) => {
          if (dataPoint !== null) {
            console.log(`🔍 BOX PLOT DEBUG: DataPoint ${dpIdx}:`, dataPoint);
          }
        });
      });
      
      return result;
    } catch (error) {
      logger.error('Error creating box plot data:', error);
      return null;
    }
  };

  // Helper function to create tornado chart data
  const createTornadoChartData = (targetResult) => {
    try {
      // ✅ FIXED: Access nested properties under targetResult.results
      const results = targetResult.results || {};
      
      // 🔍 DEBUG: Enhanced tornado chart debugging
      logger.debug(`[createTornadoChartData] 🔍 TORNADO DEBUG:`, {
        targetResult_keys: Object.keys(targetResult),
        results_keys: Object.keys(results),
        has_sensitivity_analysis: !!results.sensitivity_analysis,
        sensitivity_analysis_type: typeof results.sensitivity_analysis,
        sensitivity_analysis_length: results.sensitivity_analysis?.length || 0,
        sensitivity_analysis_sample: results.sensitivity_analysis ? results.sensitivity_analysis.slice(0, 2) : null
      });
      
      if (!results.sensitivity_analysis || (Array.isArray(results.sensitivity_analysis) && results.sensitivity_analysis.length === 0)) {
        logger.debug('No sensitivity analysis data available, generating mock data for testing');
        
        // 🚧 TEMPORARY: Create mock sensitivity data to test tornado chart display
        // This will be replaced once the Ultra engine sensitivity analysis is fixed
        const mockSensitivityData = [
          { variable: 'C2', impact: 45.2, correlation: 0.67, variable_key: 'C2' },
          { variable: 'C3', impact: 32.8, correlation: -0.54, variable_key: 'C3' },
          { variable: 'C4', impact: 22.0, correlation: 0.41, variable_key: 'C4' }
        ];
        
        // Use the Braun color palette from the original design
        const braunColors = [
          '#FF6B35', // Primary Braun orange
          '#333333', // Dark grey  
          '#777777', // Medium grey
          '#FFD700', // Subtle yellow
          '#E85A00', // Darker orange
          '#1A1A1A'  // Charcoal
        ];

        return {
          labels: mockSensitivityData.map(item => item.variable),
          datasets: [{
            label: 'Impact %',
            data: mockSensitivityData.map(item => Math.abs(item.impact)),
            backgroundColor: mockSensitivityData.map((_, index) => braunColors[index % braunColors.length]),
            borderColor: mockSensitivityData.map((_, index) => braunColors[index % braunColors.length]),
            borderWidth: 1
          }]
        };
      }

      // Handle backend sensitivity analysis structure
      let sensitivityArray;
      if (Array.isArray(results.sensitivity_analysis)) {
        // Direct array format
        sensitivityArray = results.sensitivity_analysis;
      } else if (typeof results.sensitivity_analysis === 'object' && results.sensitivity_analysis.tornado_chart) {
        // Backend object format with tornado_chart array
        sensitivityArray = results.sensitivity_analysis.tornado_chart;
        logger.debug('[createTornadoChartData] 🔍 Using tornado_chart from sensitivity object');
      } else if (typeof results.sensitivity_analysis === 'object') {
        // Convert object values to array as fallback
        sensitivityArray = Object.values(results.sensitivity_analysis);
        logger.debug('[createTornadoChartData] 🔍 Converting object values to array');
      } else {
        logger.debug('Sensitivity analysis data is in unexpected format:', typeof results.sensitivity_analysis);
        return null;
      }

      logger.debug('[createTornadoChartData] 🔍 Sensitivity array for tornado chart:', sensitivityArray);

      if (!Array.isArray(sensitivityArray) || sensitivityArray.length === 0) {
        logger.debug('[createTornadoChartData] 🔍 No valid sensitivity array found');
        return null;
      }

      const sortedData = [...sensitivityArray]
        .sort((a, b) => Math.abs(b.impact || b.impact_percentage || 0) - Math.abs(a.impact || a.impact_percentage || 0))
        .slice(0, 20); // Show top 20 variables

      // Use the Braun color palette from the original design
      const braunColors = [
        '#FF6B35', // Primary Braun orange
        '#333333', // Dark grey  
        '#777777', // Medium grey
        '#FFD700', // Subtle yellow
        '#E85A00', // Darker orange
        '#1A1A1A'  // Charcoal
      ];

      return {
        labels: sortedData.map(item => item.variable || item.variable_name || 'Unknown'),
        datasets: [{
          label: 'Impact %',
          data: sortedData.map(item => Math.abs(item.impact || item.impact_percentage || 0)),
          backgroundColor: sortedData.map((_, index) => braunColors[index % braunColors.length]),
          borderColor: sortedData.map((_, index) => braunColors[index % braunColors.length]),
          borderWidth: 1
        }]
      };
    } catch (error) {
      logger.error('Error creating tornado chart data:', error);
      return null;
    }
  };

  // CRITICAL FIX: Show computing screen immediately when simulation starts
  // This fixes the "Initializing..." delay by showing progress tracker as soon as Run is clicked
  if ((hasRunningSimulations || status === 'pending' || status === 'running') && !hasCompletedSimulations) {
    // Collect all running simulation IDs and their target variables
    const runningSimulations = multipleResults.filter(sim => 
      sim && (sim.status === 'running' || sim.status === 'pending')
    );
    
    // IMPROVED: Always prefer currentSimulationId/currentParentSimulationId for immediate tracking
    const simulationIds = runningSimulations.length > 0 
      ? runningSimulations.map(sim => sim.simulation_id).filter(Boolean)
      : (currentParentSimulationId ? [currentParentSimulationId] : 
         currentSimulationId ? [currentSimulationId] : []);

    // IMPROVED: Always use resultCells for target variables when available (more immediate)
    const targetVariables = (resultCells && resultCells.length > 0)
      ? resultCells.map(cell => getTargetDisplayName(cell.name || cell.cell, cell))
      : runningSimulations.length > 0 
        ? runningSimulations.map(sim => {
            const cellCoord = sim.result_cell_coordinate || sim.target_name;
            return getTargetDisplayName(cellCoord, sim);
          })
        : [];

    // Determine primaryEngineType from the first running simulation
    const primaryEngineType = runningSimulations.length > 0 ? runningSimulations[0].requested_engine_type : null;

    // Debug logging to see what we have
    logger.debug('[SimulationResultsDisplay] UNIFIED DEBUG - status:', status);
    logger.debug('[SimulationResultsDisplay] UNIFIED DEBUG - multipleResults:', multipleResults);
    logger.debug('[SimulationResultsDisplay] UNIFIED DEBUG - runningSimulations:', runningSimulations);
    logger.debug('[SimulationResultsDisplay] UNIFIED DEBUG - simulationIds:', simulationIds);
    logger.debug('[SimulationResultsDisplay] UNIFIED DEBUG - targetVariables:', targetVariables);
    
    return (
      <div className="simulation-results-container">
        <div className="results-header-new">
          <div className="header-title-row">
            <h3 className="results-title">Monte Carlo Simulation</h3>
            <div className={`status-badge-small status-${localStatus}`}>
              Computing...
            </div>
          </div>
        </div>
        
        {/* Show Unified Progress Tracker for all simulations */}
        {simulationIds.length > 0 ? (
        <UnifiedProgressTracker
          key={getParentId(simulationIds[0]) || 'progress-tracker'} // Stable key to prevent re-mounting
          simulationIds={simulationIds}
          targetVariables={targetVariables}
          primaryEngineType={primaryEngineType}
          onResultsCheck={checkResultsAvailable}
        />
        ) : (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>INITIALIZING SIMULATION...</p>
          </div>
        )}
      </div>
    );
  }

  // 🔥 FIX: Don't show failed state if we have completed simulations with results
  // This prevents showing 404 errors from temp ID polling when simulation actually succeeded
  if ((localStatus === 'failed' || error) && !hasCompletedSimulations) {
    return (
      <div className="simulation-results-container">
        <div className="results-header-new">
          <div className="header-title-row">
            <h3 className="results-title">Simulation Results</h3>
            <div className="status-badge-small status-failed">Failed</div>
          </div>
          <div className="header-right">
            <button 
              onClick={handleBackendPDFExport}
              className="export-backend-pdf-button"
              title="Generate PDF using headless Chrome (direct download)"
              disabled={isPdfExporting}
            >
              {isPdfExporting ? '⏳ Generating...' : '📄 PDF Export'}
            </button>
          </div>
        </div>
        <div className="error-message">
          <p className="error-text">{error || 'Simulation failed'}</p>
        </div>
      </div>
    );
  }

  if (localStatus === 'cancelled') {
    return (
      <div className="simulation-results-container">
        <div className="results-header-new">
          <div className="header-title-row">
            <h3 className="results-title">Simulation Results</h3>
            <div className="status-badge-small status-cancelled">Cancelled</div>
          </div>
          <div className="header-right">
            <button 
              onClick={handleBackendPDFExport}
              className="export-backend-pdf-button"
              title="Generate PDF using headless Chrome (direct download)"
              disabled={isPdfExporting}
            >
              {isPdfExporting ? '⏳ Generating...' : '📄 PDF Export'}
            </button>
          </div>
        </div>
        <div className="error-message">
          <p className="error-text">Simulation was cancelled by user request</p>
        </div>
      </div>
    );
  }

  // FIXED: Check for completed simulations even if main status is still 'running'
  if ((localStatus === 'completed' || hasCompletedSimulations) && displayResults.length === 0) {
  return (
    <div className="simulation-results-container">
      <div className="results-header-new">
        <div className="header-title-row">
          <h3 className="results-title">Simulation Results</h3>
          <div className="status-badge-small status-completed">Completed</div>
        </div>
        <div className="header-right">
          <button 
            onClick={handleBackendPDFExport}
            className="export-backend-pdf-button"
            title="Generate PDF using headless Chrome (direct download)"
            disabled={isPdfExporting}
          >
            {isPdfExporting ? '⏳ Generating...' : '📄 PDF Export'}
          </button>
        </div>
      </div>
      <div className="simulation-placeholder">
          <p>Simulation completed but no results available.</p>
          <div style={{ marginTop: '1rem', fontSize: '0.9rem', color: '#666' }}>
            <p>Debug info:</p>
            <p>Status: {localStatus}</p>
            <p>Has completed simulations: {hasCompletedSimulations ? 'Yes' : 'No'}</p>
            <p>Multiple results count: {multipleResults.length}</p>
          </div>
        </div>
      </div>
    );
  }

  // FIXED: Show results if we have completed simulations, regardless of main status
  if (localStatus === 'completed' || hasCompletedSimulations) {
    return (
      <div className="simulation-results-container">
        <div className="results-header-new">
          <div className="header-title-row">
            <h3 className="results-title">Simulation Results</h3>
            <div className="status-badge-small status-completed">Completed</div>
          </div>
          <div className="header-right" style={{ display: 'flex', gap: '0.5rem' }}>
            <button 
              onClick={handleBackendPDFExport}
              className="export-backend-pdf-button"
              title="Generate PDF using headless Chrome (direct download)"
              disabled={isPdfExporting}
            >
              {isPdfExporting ? '⏳ Generating...' : '📄 PDF Export'}
            </button>
            <button 
              onClick={handleExportXLS}
              className="export-xls-button"
              title="Export data to Excel spreadsheet"
              disabled={isXlsExporting}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '0.875rem',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'background-color 0.2s'
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = '#059669'}
              onMouseOut={(e) => e.target.style.backgroundColor = '#10b981'}
            >
              {isXlsExporting ? '⏳ Exporting...' : '📊 XLS Export'}
            </button>
            <button 
              onClick={handleExportJSON}
              className="export-json-button"
              title="Export data as JSON file"
              disabled={isJsonExporting}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '0.875rem',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'background-color 0.2s'
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = '#2563eb'}
              onMouseOut={(e) => e.target.style.backgroundColor = '#3b82f6'}
            >
              {isJsonExporting ? '⏳ Exporting...' : '📋 JSON Export'}
            </button>
          </div>
        </div>

        {/* Always keep progress dashboard visible (frozen at 100%) */}
        <UnifiedProgressTracker
          key={getParentId(simulationIds[0]) || 'completed-progress-tracker'} // Stable key to prevent re-mounting
          simulationIds={simulationIds}
          targetVariables={multipleResults.map(result => {
            const cellCoord = result?.result_cell_coordinate || result?.target_name;
            return getTargetDisplayName(cellCoord, result);
          })}
          primaryEngineType={multipleResults.length > 0 ? multipleResults[0].requested_engine_type : null}
          restoredResults={multipleResults.filter(result => result?.isRestored)}
          forceCompleted={true}
          onResultsCheck={checkResultsAvailable}
        />

        {/* Iterations & elapsed time summary (first completed result) */}
        {(()=>{
          const firstDone = displayResults.find(result => (result.results?.iterations_run || result.iterations_run));
          if(!firstDone) return null;
          const iter = (firstDone.results?.iterations_run ?? firstDone.iterations_run);
          let elapsed="";
          const createdAt = firstDone.results?.created_at ?? firstDone.created_at;
          const updatedAt = firstDone.results?.updated_at ?? firstDone.updated_at;
          if(createdAt && updatedAt){
            const secs=(new Date(updatedAt)-new Date(createdAt))/1000;
            if(secs>0)elapsed=`${secs.toFixed(1)} s`;
          }
          return (
            <div style={{marginTop:'0.5rem',fontSize:'0.9rem',color:'#555'}}>
              Iterations: <strong>{iter}</strong>{elapsed&&` • Elapsed: ${elapsed}`}
            </div>
          );
        })()}

        <div className="results-content" style={{display:'flex',flexDirection:'column',gap:'2rem'}}>
          {/* Box & Whisker Chart for all target variables */}
          {(() => {
            const boxPlotData = createBoxPlotData();
            console.log('🔍 BOX PLOT RENDER DEBUG: boxPlotData:', boxPlotData);
            console.log('🔍 BOX PLOT RENDER DEBUG: displayResults.length:', displayResults.length);
            console.log('🔍 BOX PLOT RENDER DEBUG: condition result:', !!(boxPlotData && displayResults.length > 1));
            if (boxPlotData && displayResults.length > 1) {
              return (
                <div className="box-plot-overview-section" style={{
                  background: '#fff',
                  padding: '1.5rem',
                  borderRadius: '12px',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.08)',
                  marginBottom: '1rem'
                }}>
                  <div className="box-plot-header" style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '1rem'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                      <h4 style={{ 
                        margin: 0, 
                        color: '#1a1a1a',
                        fontSize: '1.25rem',
                        fontWeight: 600 
                      }}>
                        📊 Distribution Overview - All Target Variables
                      </h4>
                      <button
                        onClick={(e) => copySectionImage('boxplot', 'All Variables', e.target)}
                        className="copy-boxplot-button"
                        title="Copy complete box plot section as image - Double-click for better clipboard compatibility"
                        style={{
                          padding: '0.2rem 0.3rem',
                          backgroundColor: 'transparent',
                          color: '#6b7280',
                          border: '1px solid #e5e7eb',
                          borderRadius: '3px',
                          fontSize: '0.7rem',
                          fontWeight: '400',
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.2rem'
                        }}
                        onMouseOver={(e) => {
                          e.target.style.backgroundColor = '#f9fafb';
                          e.target.style.color = '#374151';
                          e.target.style.borderColor = '#d1d5db';
                        }}
                        onMouseOut={(e) => {
                          e.target.style.backgroundColor = 'transparent';
                          e.target.style.color = '#6b7280';
                          e.target.style.borderColor = '#e5e7eb';
                        }}
                      >
                        📋
                      </button>
                    </div>
                    <div style={{
                      fontSize: '0.875rem',
                      color: '#6b7280',
                      textAlign: 'right'
                    }}>
                      <div>Box shows quartiles (25%, 50%, 75%)</div>
                      <div>Whiskers show min/max, dots are outliers</div>
                    </div>
                  </div>
                  <SimpleBoxPlot 
                    displayResults={displayResults}
                    getTargetDisplayName={getTargetDisplayName}
                  />
                </div>
              );
            }
            return null;
          })()}

          {/* List one variable per row */}
          {displayResults.map((targetResult, index) => {
            const chartData = createChartData(targetResult);
            const tornadoData = createTornadoChartData(targetResult);
            
            // FIXED: Use display name instead of cell coordinate
            const cellCoord = targetResult.result_cell_coordinate || targetResult.target_name;
            const targetName = getTargetDisplayName(cellCoord, targetResult);

            // Per-variable decimals (default 2)
            const decPrecision = decimalMap[targetName] ?? 2;

            // Local formatter
            const localFormat = (val) => {
              if (val === null || val === undefined || isNaN(val)) return '—';
              return Number(val).toLocaleString(undefined, { maximumFractionDigits: decPrecision });
            };

            // 🛡️ STALE DATA DETECTION
            const hasStaleDataWarning = _detectStaleData(targetResult);
            
            return (
              <div key={index} className="target-results-section" data-variable={targetName} style={{background:'#fff',padding:'1.5rem',borderRadius:'12px',boxShadow:'0 2px 6px rgba(0,0,0,0.08)',maxWidth:'100%'}}>

                <div className="target-header">
                  <h4>Results for {targetName}</h4>
                  {hasStaleDataWarning && (
                    <div className="stale-data-warning" style={{
                      background: '#fff3cd', 
                      border: '1px solid #ffeaa7',
                      borderRadius: '6px',
                      padding: '0.75rem',
                      margin: '0.5rem 0',
                      fontSize: '0.9rem',
                      color: '#856404'
                    }}>
                      <div style={{fontWeight: 'bold', marginBottom: '0.25rem'}}>
                        ⚠️ Constant Value Detected
                      </div>
                      <div>
                        This target shows identical values across all simulation iterations (no variation). 
                        This typically means the target cell contains a constant value or formula that doesn't depend on your Monte Carlo variables.
                        <br /><br />
                        <strong>Possible causes:</strong>
                        <ul style={{margin: '0.5rem 0', paddingLeft: '1.5rem'}}>
                          <li>Target cell contains a fixed number rather than a formula</li>
                          <li>Formula doesn't reference the Monte Carlo input variables</li>
                          <li>Intermediate calculation cells breaking the dependency chain</li>
                        </ul>
                        <strong>Recommendation:</strong> Verify the target cell contains a formula that properly references your input variables.
                      </div>
                    </div>
                  )}
                </div>

                {/* Per-variable decimal precision slider */}
                <div style={{display:'flex',alignItems:'center',gap:'0.5rem',marginBottom:'0.5rem'}}>
                  <label style={{fontWeight:600, color: 'var(--color-dark-grey)'}}>Decimals:</label>
                  <input 
                    type="range" 
                    min="0" 
                    max="6" 
                    value={decPrecision} 
                    onChange={(e)=>setDecimalMap({...decimalMap,[targetName]:parseInt(e.target.value)})} 
                    className="decimal-slider-braun"
                  />
                  <span style={{color: 'var(--color-charcoal)', fontWeight: 600}}>{decPrecision}</span>
                </div>

                <div style={{display:'flex',flexWrap:'wrap',gap:'1.5rem'}}>
                  <div style={{flex:'1 1 350px',minWidth:'300px'}}>
                    <CertaintyAnalysis 
                      targetResult={{
                        ...targetResult,
                        sliderState: sliderStateMap[targetName] // Pass saved slider state
                      }} 
                      formatNumber={localFormat}
                      variableName={targetName}
                      onDecimalChange={(d)=>setDecimalMap({...decimalMap,[targetName]:d})}
                      currentDecimals={decPrecision}
                      onSliderStateChange={handleSliderStateChange}
                      currentSimulation={targetResult}
                      copySectionImage={copySectionImage}
                    />
                  </div>
                  <div style={{flex:'1 1 300px',minWidth:'260px',display:'flex',flexDirection:'column',gap:'1rem'}}>

                    {/* Compact statistics and tornado */}
                    <div className="stats-condensed">
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                        <h5 style={{ margin: 0, color: '#374151', fontSize: '0.875rem', fontWeight: '600' }}>Summary Statistics</h5>
                        <button
                          onClick={(e) => copySectionImage('statistics', targetName, e.target)}
                          className="copy-stats-button"
                          title="Copy complete statistics section as image - Double-click for better clipboard compatibility"
                          style={{
                            padding: '0.2rem 0.3rem',
                            backgroundColor: 'transparent',
                            color: '#6b7280',
                            border: '1px solid #e5e7eb',
                            borderRadius: '3px',
                            fontSize: '0.7rem',
                            fontWeight: '400',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.2rem'
                          }}
                          onMouseOver={(e) => {
                            e.target.style.backgroundColor = '#f9fafb';
                            e.target.style.color = '#374151';
                            e.target.style.borderColor = '#d1d5db';
                          }}
                          onMouseOut={(e) => {
                            e.target.style.backgroundColor = 'transparent';
                            e.target.style.color = '#6b7280';
                            e.target.style.borderColor = '#e5e7eb';
                          }}
                        >
                          📋
                        </button>
                      </div>
                      <div className="stats-row">
                        <div className="stat-compact">
                          <span className="label">Mean</span>
                          <span className="value">{localFormat(targetResult.results?.mean)}</span>
                        </div>
                        <div className="stat-compact">
                          <span className="label">Median</span>
                          <span className="value">{localFormat(targetResult.results?.median)}</span>
                        </div>
                        <div className="stat-compact">
                          <span className="label">Std Dev</span>
                          <span className="value">{localFormat(targetResult.results?.std_dev)}</span>
                        </div>
                      </div>
                      <div className="stats-row">
                        <div className="stat-compact">
                          <span className="label">Min</span>
                          <span className="value">{localFormat(targetResult.results?.min_value)}</span>
                        </div>
                        <div className="stat-compact">
                          <span className="label">Max</span>
                          <span className="value">{localFormat(targetResult.results?.max_value)}</span>
                        </div>
                        <div className="stat-compact">
                          <span className="label">Range</span>
                          <span className="value">{localFormat((targetResult.results?.max_value || 0) - (targetResult.results?.min_value || 0))}</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Tornado chart */}
                    {tornadoData && (
                      <div className="tornado-section">
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                          <h5 className="tornado-title">Variable Impact Analysis</h5>
                          <button
                            onClick={(e) => copySectionImage('tornado', targetName, e.target)}
                            className="copy-chart-button"
                            title="Copy complete tornado chart section - Double-click for better clipboard compatibility"
                            style={{
                              padding: '0.2rem 0.3rem',
                              backgroundColor: 'transparent',
                              color: '#6b7280',
                              border: '1px solid #e5e7eb',
                              borderRadius: '3px',
                              fontSize: '0.7rem',
                              fontWeight: '400',
                              cursor: 'pointer',
                              transition: 'all 0.2s',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.2rem'
                            }}
                            onMouseOver={(e) => {
                              e.target.style.backgroundColor = '#f9fafb';
                              e.target.style.color = '#374151';
                              e.target.style.borderColor = '#d1d5db';
                            }}
                            onMouseOut={(e) => {
                              e.target.style.backgroundColor = 'transparent';
                              e.target.style.color = '#6b7280';
                              e.target.style.borderColor = '#e5e7eb';
                            }}
                          >
                            📋
                          </button>
                        </div>
                        <div 
                          className="tornado-chart-container" 
                          style={{ 
                            height: `${Math.max(200, tornadoData.labels.length * 35 + 80)}px` 
                          }}
                        >
                          <Bar 
                            options={{
                              ...tornadoChartOptions,
                              plugins: {
                                ...tornadoChartOptions.plugins,
                                tooltip: {
                                  callbacks: {
                                    label: function(context) {
                                      return `${context.label}: ${context.parsed.x.toFixed(1)}%`;
                                    }
                                  }
                                }
                              }
                            }} 
                            data={tornadoData} 
                          />
                        </div>
                      </div>
                    )}

                    {/* Percentiles */}
                    <div className="percentiles-grid-section-wide">
                      <h5 className="percentiles-grid-title">Percentiles</h5>
                      <div className="percentiles-grid-wide">
                        {targetResult.results?.percentiles && Object.entries(targetResult.results.percentiles).map(([p, val]) => (
                          <div key={p} className="percentile-grid-item-wide">
                            <span className="percentile-grid-label">{p}%</span>
                            <span className="percentile-grid-value">{localFormat(val)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div> {/* end right column */}
                </div> {/* end flex row */}


                {targetResult.errors && targetResult.errors.length > 0 && (
                  <div className="error-section">
                    <h5>Iteration Errors (first {targetResult.errors.length}):</h5>
                    <div className="error-display">
                      {JSON.stringify(targetResult.errors, null, 2)}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Guard: This fallback should only trigger if the main condition above didn't catch it
  // Most cases should be handled by the improved condition above
  if (status === 'pending' && !hasCompletedSimulations) {
    console.warn('[SimulationResultsDisplay] FALLBACK: Using pending state fallback - main condition may need adjustment');
    const simulationIds = currentParentSimulationId ? [currentParentSimulationId] : 
                         currentSimulationId ? [currentSimulationId] : [];
    const targetVariables = (resultCells || []).map(cell => getTargetDisplayName(cell.name || cell.cell, cell));
    return (
      <div className="simulation-results-container">
        <div className="results-header-new">
          <div className="header-title-row">
            <h3 className="results-title">Monte Carlo Simulation</h3>
            <div className={`status-badge-small status-${status}`}>
              Computing...
            </div>
          </div>
        </div>
        {simulationIds.length > 0 ? (
          <UnifiedProgressTracker
            key={getParentId(simulationIds[0]) || 'pending-progress-tracker'} // Stable key to prevent re-mounting
            simulationIds={simulationIds}
            targetVariables={targetVariables}
            onResultsCheck={checkResultsAvailable}
          />
        ) : (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>STARTING SIMULATION...</p>
          </div>
        )}
      </div>
    );
  }

  // Fallback - should not reach here
  logger.debug('[SimulationResultsDisplay] FALLBACK - Unexpected state:', { status, hasRunningSimulations, hasCompletedSimulations, multipleResults });

  return (
    <div className="simulation-results-container">
      <div className="simulation-placeholder">
        <p>Unexpected simulation state. Please refresh the page.</p>
        <div style={{ marginTop: '1rem', fontSize: '0.9rem', color: '#666' }}>
          <p>Debug info:</p>
          <p>Status: {status}</p>
          <p>Has running: {hasRunningSimulations ? 'Yes' : 'No'}</p>
          <p>Has completed: {hasCompletedSimulations ? 'Yes' : 'No'}</p>
        </div>
      </div>
    </div>
  );
});

// 🛡️ STALE DATA DETECTION HELPER
function _detectStaleData(targetResult) {
  try {
    if (!targetResult || !targetResult.results) {
      return false;
    }
    
    const results = targetResult.results;
    
    // Check for zero values across all key metrics
    const isAllZero = (
      (results.mean === 0 || results.mean === null) &&
      (results.median === 0 || results.median === null) &&
      (results.std_dev === 0 || results.std_dev === null) &&
      (results.min_value === 0 || results.min_value === null) &&
      (results.max_value === 0 || results.max_value === null)
    );
    
    // Check for constant value (min === max and std_dev === 0)
    const isConstantValue = (
      results.min_value === results.max_value &&
      results.std_dev === 0 &&
      results.min_value !== null &&
      results.min_value !== undefined
    );
    
    // Additional check: near-zero standard deviation relative to mean
    const isNearConstant = (
      Math.abs(results.std_dev) < Math.abs(results.mean) * 1e-6 &&
      results.mean !== 0 &&
      results.std_dev !== null
    );
    
    // Check for suspicious patterns that might indicate stale data
    const hasSuspiciousPattern = isAllZero || isConstantValue || isNearConstant;
    
    if (hasSuspiciousPattern) {
      const pattern = isAllZero ? 'ALL_ZERO' : isConstantValue ? 'CONSTANT_VALUE' : 'NEAR_CONSTANT';
      logger.warn('[STALE_DATA_DETECTION] Potential stale data detected:', {
        target: targetResult.target_name,
        mean: results.mean,
        median: results.median,
        std_dev: results.std_dev,
        min_value: results.min_value,
        max_value: results.max_value,
        pattern: pattern,
        coefficient_of_variation: results.mean !== 0 ? results.std_dev / results.mean : null
      });
    }
    
    return hasSuspiciousPattern;
    
  } catch (error) {
    logger.error('[STALE_DATA_DETECTION] Error detecting stale data:', error);
    return false;
  }
}

export default SimulationResultsDisplay;