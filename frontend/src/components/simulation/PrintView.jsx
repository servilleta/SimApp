import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Provider, useDispatch } from 'react-redux';
import { store } from '../../store';
import { restoreSimulationResults } from '../../store/simulationSlice';
import SimulationResultsDisplay from './SimulationResultsDisplay';
// import ErrorBoundary from '../common/ErrorBoundary'; // Temporarily removed for raw error diagnosis
import './PrintView.css';

/**
 * PrintView Component
 * 
 * This component renders a print-optimized version of simulation results.
 * It's designed to be opened in a new window/tab and printed to PDF.
 * 
 * Usage:
 * - Navigate to /print-view?data=<base64-encoded-results>
 * - The component will decode the results and display them
 * - Window.print() is automatically triggered after rendering
 */
const PrintViewContent = () => {
  const dispatch = useDispatch();
  const [searchParams] = useSearchParams();
  const [isReadyForPrint, setIsReadyForPrint] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const dataId = searchParams.get('id');
    console.log('[PrintView] Attempting to load data with ID:', dataId);
    
    if (dataId) {
      // Retry mechanism for data loading (handles race condition with PDF export)
      const tryLoadData = (attempt = 1, maxAttempts = 10) => {
        console.log(`[PrintView] Loading attempt ${attempt}/${maxAttempts}`);
        
        const storedData = sessionStorage.getItem(dataId);
        if (storedData) {
        console.log('[PrintView] Successfully retrieved data.');
        sessionStorage.removeItem(dataId); // Clean up immediately
        try {
          const printData = JSON.parse(storedData);
          console.log('[PrintView] Parsed data:', printData);

          if (!printData || !Array.isArray(printData.results) || printData.results.length === 0) {
            console.error('[PrintView] Invalid or empty results array in printData.');
            setError('No valid simulation results found to print.');
            return;
          }

          // Dispatch the results to the Redux store to create a valid state
          dispatch(restoreSimulationResults({ multipleResults: printData.results }));
          
          setIsReadyForPrint(true);

        } catch (e) {
          console.error('[PrintView] Failed to parse stored JSON data:', e);
          setError('Could not parse simulation data.');
        }
          return; // Success, stop retrying
        }
        
        // Fallback: Check for direct window injection (used by headless PDF export)
        console.log('[PrintView] Data not found in sessionStorage, checking window injection...');
        if (window.__PDF_DATA__ && window.__PDF_DATA_ID__ === dataId) {
          console.log('[PrintView] Found injected data in window.__PDF_DATA__');
          try {
            const printData = window.__PDF_DATA__;
            console.log('[PrintView] Using injected data:', printData);

            if (!printData || !Array.isArray(printData.results) || printData.results.length === 0) {
              console.error('[PrintView] Invalid or empty results array in injected data.');
              setError('No valid simulation results found to print.');
              return;
            }

            // Dispatch the results to the Redux store to create a valid state
            dispatch(restoreSimulationResults({ multipleResults: printData.results }));
            
            setIsReadyForPrint(true);

          } catch (e) {
            console.error('[PrintView] Failed to process injected data:', e);
            setError('Could not process injected simulation data.');
          }
          return; // Success, stop retrying
        }
        
        // No data found, retry after delay
        if (attempt < maxAttempts) {
          console.log(`[PrintView] Data not found, retrying in 500ms... (attempt ${attempt}/${maxAttempts})`);
          setTimeout(() => tryLoadData(attempt + 1, maxAttempts), 500);
        } else {
          console.error('[PrintView] Data not found after', maxAttempts, 'attempts for ID:', dataId);
          setError('Could not load simulation data for printing.');
        }
      };
      
      // Start the retry process
      tryLoadData();
    } else {
      console.error('[PrintView] No data ID found in URL.');
      setError('No simulation data ID provided.');
    }
  }, [searchParams, dispatch]);

  useEffect(() => {
    if (isReadyForPrint) {
      const printTimeout = setTimeout(() => {
        window.print();
      }, 3000); // Wait 3s for charts to render

      return () => clearTimeout(printTimeout);
    }
  }, [isReadyForPrint]);

  if (error) {
    return <div className="print-loading">{error}</div>;
  }

  if (!isReadyForPrint) {
    return <div className="print-loading">Loading results for printing...</div>;
  }

  return (
    <div className="print-view-container">
      <div className="print-header no-print">
        <h1>Monte Carlo Simulation Results</h1>
        <p>Generated: {new Date().toLocaleString()}</p>
      </div>
      <div className="print-content">
        {/* Render the component without props; it will now get state from Redux */}
        <SimulationResultsDisplay isPrintView={true} />
      </div>
      <div className="print-footer no-print">
        <p>© Monte Carlo Simulation Platform</p>
      </div>
    </div>
  );
};

// Top-level component that provides all necessary contexts
const PrintView = () => (
  <Provider store={store}>
    <PrintViewContent />
  </Provider>
);

export default PrintView;
