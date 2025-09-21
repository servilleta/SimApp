"""
Screenshot-based PDF Export Service following the successful print solution pattern.
"""
import os
import logging
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from pathlib import Path
import tempfile
import uuid
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

class PDFExportScreenshotService:
    """Service for generating PDFs using the same approach as the working print solution."""
    
    def __init__(self):
        self.logger = logger
    
    async def generate_pdf_from_frontend(
        self, 
        simulation_id: str, 
        results_data: Dict[str, Any],
        auth_token: Optional[str] = None,
        frontend_url: str = "http://frontend:3000"
    ) -> str:
        """
        Generate a PDF using the EXACT same approach as the working print solution.
        This follows the pattern: sessionStorage -> /print-view -> wait for ready -> capture
        """
        try:
            logger.info(f"Starting PDF generation for simulation {simulation_id}")
            logger.info(f"Using frontend URL: {frontend_url}")
            
            # Create temporary file for PDF output (using same dir as main PDF service)
            temp_dir = Path(tempfile.gettempdir()) / "monte_carlo_pdfs"
            temp_dir.mkdir(exist_ok=True)
            pdf_path = temp_dir / f"simulation_pdf_{simulation_id}_{int(time.time())}.pdf"
            
            async with async_playwright() as p:
                # Launch browser (same config as before)
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox']
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    device_scale_factor=1.0
                )
                
                # Enable console logging capture
                page = await context.new_page()
                
                def log_console_message(msg):
                    logger.info(f"🌐 BROWSER CONSOLE [{msg.type}]: {msg.text}")
                
                page.on("console", log_console_message)
                
                # Set authentication if provided
                if auth_token:
                    await page.set_extra_http_headers({
                        'Authorization': f'Bearer {auth_token}'
                    })
                    
                    # Set auth in localStorage (same as before)
                    await page.add_init_script(f"""
                        localStorage.setItem('authToken', '{auth_token}');
                        localStorage.setItem('auth0Token', '{auth_token}');
                        window.authToken = '{auth_token}';
                    """)
                
                # STEP 1: Prepare data in the EXACT same format as print solution
                prepared_targets = []
                
                # DEBUG: Log input data structure
                logger.info(f"📋 PDF Export Input Data Structure Debug:")
                logger.info(f"  results_data keys: {list(results_data.keys()) if isinstance(results_data, dict) else 'Not a dict'}")
                logger.info(f"  targets type: {type(results_data.get('targets'))}")
                if 'targets' in results_data:
                    targets_data = results_data['targets']
                    logger.info(f"  targets keys: {list(targets_data.keys()) if isinstance(targets_data, dict) else f'Not a dict: {targets_data}'}")
                    for i, (key, value) in enumerate(targets_data.items()):
                        logger.info(f"    Target {i}: key='{key}', value_type={type(value)}, value_keys={list(value.keys()) if isinstance(value, dict) else 'not dict'}")
                
                targets = results_data.get('targets', {})
                
                # Convert targets data to the format expected by SimulationResultsDisplay
                for target_name, target_data in targets.items():
                    logger.info(f"Processing target: {target_name}")
                    logger.info(f"Raw target_data keys: {list(target_data.keys()) if isinstance(target_data, dict) else 'Not a dict'}")
                    
                    # Convert from backend format to frontend format
                    if 'values' in target_data and 'statistics' in target_data:
                        # Backend sends: {values: [...], statistics: {...}, histogram_data: {...}}
                        # Frontend expects: {mean, median, std_dev, histogram: {bin_edges, counts}, ...}
                        statistics = target_data.get('statistics', {})
                        histogram_data = target_data.get('histogram_data', {})
                        
                        # DEBUG: Log actual statistics field names
                        logger.info(f"📊 STATISTICS DEBUG for {target_name}:")
                        logger.info(f"  Available statistics keys: {list(statistics.keys()) if isinstance(statistics, dict) else 'Not a dict'}")
                        if isinstance(statistics, dict):
                            for key, value in statistics.items():
                                logger.info(f"    {key}: {value}")
                        values = target_data.get('values', [])
                        
                        # Create proper frontend results structure
                        # Try different possible field names for min/max
                        min_value = statistics.get('min', statistics.get('min_value', statistics.get('minimum', 0)))
                        max_value = statistics.get('max', statistics.get('max_value', statistics.get('maximum', 0)))
                        
                        # Use pre-calculated percentiles from statistics if available, otherwise calculate from raw values
                        percentiles = {}
                        if statistics.get('percentiles'):
                            # Use pre-calculated percentiles from backend (SimApp API format)
                            backend_percentiles = statistics['percentiles']
                            percentiles = {
                                'p5': backend_percentiles.get('5', 0),
                                'p10': backend_percentiles.get('10', backend_percentiles.get('5', 0)),
                                'p25': backend_percentiles.get('25', 0),
                                'p50': backend_percentiles.get('50', statistics.get('median', 0)),  # median
                                'p75': backend_percentiles.get('75', 0),
                                'p90': backend_percentiles.get('90', backend_percentiles.get('95', 0)),
                                'p95': backend_percentiles.get('95', 0)
                            }
                            logger.info(f"Using pre-calculated percentiles for {target_name}: {percentiles}")
                        elif values and len(values) > 0:
                            # Calculate from raw values if no pre-calculated percentiles
                            import numpy as np
                            values_array = np.array(values)
                            percentiles = {
                                'p5': float(np.percentile(values_array, 5)),
                                'p10': float(np.percentile(values_array, 10)),
                                'p25': float(np.percentile(values_array, 25)),
                                'p50': float(np.percentile(values_array, 50)),  # median
                                'p75': float(np.percentile(values_array, 75)),
                                'p90': float(np.percentile(values_array, 90)),
                                'p95': float(np.percentile(values_array, 95))
                            }
                            # Calculate actual min/max from values if not in statistics
                            if min_value == 0:
                                min_value = float(np.min(values_array))
                            if max_value == 0:
                                max_value = float(np.max(values_array))
                            logger.info(f"Calculated percentiles from raw values for {target_name}: {percentiles}")
                        else:
                            logger.warning(f"No percentiles or raw values available for {target_name}")
                                
                        frontend_results = {
                            'mean': statistics.get('mean', 0),
                            'median': statistics.get('median', 0), 
                            'std_dev': statistics.get('std_dev', 0),
                            'min_value': min_value,
                            'max_value': max_value,
                            'iterations_run': len(values) if values else 1000,
                            'raw_values': values[:1000] if values else [],  # Limit to prevent huge payloads
                            'percentiles': percentiles  # Add percentiles for the certainty range
                        }
                        
                        # Process histogram data - handle both formats
                        if histogram_data and isinstance(histogram_data, dict):
                            if 'bin_edges' in histogram_data and 'counts' in histogram_data:
                                # Frontend format: {bin_edges: [...], counts: [...]}
                                frontend_results['histogram'] = {
                                    'bin_edges': histogram_data['bin_edges'],
                                    'counts': histogram_data['counts'],
                                    'bins': histogram_data.get('bins', []),
                                    'values': histogram_data.get('values', [])
                                }
                                logger.info(f"✅ Converted histogram (frontend format) for {target_name}: bin_edges={len(histogram_data['bin_edges'])}, counts={len(histogram_data['counts'])}")
                            elif 'bin_edges' in histogram_data and 'histogram' in histogram_data:
                                # SimApp format: {bin_edges: [...], histogram: [...]}
                                frontend_results['histogram'] = {
                                    'bin_edges': histogram_data['bin_edges'],
                                    'counts': histogram_data['histogram'],  # histogram array becomes counts
                                    'bins': histogram_data.get('bins', []),
                                    'values': histogram_data.get('values', [])
                                }
                                logger.info(f"✅ Converted histogram (SimApp format) for {target_name}: bin_edges={len(histogram_data['bin_edges'])}, histogram={len(histogram_data['histogram'])}")
                            else:
                                logger.warning(f"⚠️ Unknown histogram format for {target_name}: keys={list(histogram_data.keys())}")
                                frontend_results['histogram'] = {}
                        else:
                            logger.warning(f"❌ No valid histogram data for {target_name}: {type(histogram_data)}")
                            frontend_results['histogram'] = {}
                        
                        prepared_targets.append({
                            'simulation_id': f'pdf_export_{target_name}',
                            'temp_id': f'pdf_temp_{target_name}', 
                            'status': 'completed',
                            'target_name': target_name,
                            'result_cell_coordinate': target_name,
                            'target_cell': target_name,
                            'iterations_run': results_data.get('iterations_run', 1000),
                            'requested_engine_type': results_data.get('requested_engine_type', 'Ultra'),
                            'results': frontend_results,
                            'histogram': frontend_results['histogram'],  # Also at top level for compatibility
                            'sensitivity_analysis': target_data.get('sensitivity_analysis', [])
                        })
                        
                        # DEBUG: Log what we're sending to frontend
                        logger.info(f"💾 FRONTEND DATA for {target_name}:")
                        logger.info(f"  min_value: {frontend_results['min_value']}")
                        logger.info(f"  max_value: {frontend_results['max_value']}")
                        logger.info(f"  percentiles: {list(frontend_results['percentiles'].keys()) if frontend_results['percentiles'] else 'None'}")
                        logger.info(f"  sensitivity_analysis items: {len(target_data.get('sensitivity_analysis', []))}")
                        
                        logger.info(f"✅ Converted target {target_name} to frontend format")
                    else:
                        # Fallback for unknown format
                        logger.warning(f"Unknown target data format for {target_name}, using as-is")
                        prepared_targets.append({
                            'simulation_id': f'pdf_export_{target_name}',
                            'temp_id': f'pdf_temp_{target_name}', 
                            'status': 'completed',
                            'target_name': target_name,
                            'result_cell_coordinate': target_name,
                            'target_cell': target_name,
                            'iterations_run': results_data.get('iterations_run', 1000),
                            'requested_engine_type': results_data.get('requested_engine_type', 'Ultra'),
                            'results': target_data,
                            'histogram': target_data.get('histogram', {}),
                            'sensitivity_analysis': target_data.get('sensitivity_analysis', [])
                        })
                
                # STEP 2: Create print data in EXACT same format as print solution
                print_data = {
                    'simulationId': 'pdf_export',
                    'results': prepared_targets,
                    'metadata': {
                        'iterations_run': results_data.get('iterations_run', 1000),
                        'engine_type': results_data.get('requested_engine_type', 'Ultra'),
                        'timestamp': results_data.get('metadata', {}).get('timestamp', 'N/A')
                    }
                }
                
                # STEP 3: Generate unique ID (EXACT same as print solution)
                pdf_data_id = f"pdf_data_{int(time.time() * 1000)}"
                
                logger.info(f"Prepared PDF data with ID: {pdf_data_id}")
                logger.info(f"Prepared {len(prepared_targets)} targets for PDF generation")
                # Debug: Log the data structure being sent
                for i, target in enumerate(prepared_targets):
                    logger.info(f"Target {i}: {target.get('target_name')} - Has histogram: {'histogram' in target} - Histogram keys: {list(target.get('histogram', {}).keys())}")
                    if target.get('histogram'):
                        hist = target['histogram']
                        logger.info(f"  Histogram structure: {type(hist)} - Has bin_edges: {'bin_edges' in hist if isinstance(hist, dict) else False} - Has counts: {'counts' in hist if isinstance(hist, dict) else False}")
                        if isinstance(hist, dict) and 'bin_edges' in hist and 'counts' in hist:
                            logger.info(f"  ✅ Valid histogram: {len(hist['bin_edges'])} bin_edges, {len(hist['counts'])} counts")
                        else:
                            logger.warning(f"  ❌ Invalid histogram structure")
                    logger.info(f"  Results keys: {list(target.get('results', {}).keys()) if isinstance(target.get('results'), dict) else 'Not a dict'}")
                    logger.info(f"  Has mean: {'mean' in target.get('results', {})}")
                
                # STEP 4: First go to any page and inject data BEFORE navigating to print-view
                # This prevents the race condition where PrintView loads before data is available
                await page.goto(f"{frontend_url}/", wait_until='networkidle')
                logger.info("Navigated to homepage first for data injection")
                
                # STEP 5: Inject data while NOT on print-view page (prevents race condition)
                await page.evaluate(f"""
                    try {{
                        // Try sessionStorage first (same as working print solution)
                        sessionStorage.setItem('{pdf_data_id}', JSON.stringify({json.dumps(print_data)}));
                        console.log('PDF data pre-stored in sessionStorage with ID: {pdf_data_id}');
                    }} catch (e) {{
                        console.log('SessionStorage failed, using direct injection:', e.message);
                        // Fallback: inject data directly into window
                        window.__PDF_DATA__ = {json.dumps(print_data)};
                        window.__PDF_DATA_ID__ = '{pdf_data_id}';
                        console.log('PDF data pre-injected directly into window');
                    }}
                """)
                logger.info("Data pre-injection completed")
                
                # STEP 6: NOW navigate to print-view (data is already available!)
                await page.goto(f"{frontend_url}/print-view?id={pdf_data_id}", wait_until='networkidle')
                logger.info("Navigated to print-view page with data already available")
                
                # STEP 7: Wait for print view to be ready with enhanced retry mechanism
                logger.info("Waiting for print view to load and process data...")
                
                # Enhanced waiting with multiple retry strategies
                max_retries = 6
                retry_delay = 5000  # 5 seconds between retries
                
                for retry in range(max_retries):
                    try:
                        logger.info(f"Print view loading attempt {retry + 1}/{max_retries}")
                        
                        # Check page status
                        page_title = await page.title()
                        logger.info(f"Page title: {page_title}")
                        
                        # Comprehensive page status check
                        page_status = await page.evaluate(f"""
                            (() => {{
                                // Check for React app root
                                const root = document.querySelector('#root');
                                const printContent = document.querySelector('.print-content');
                                const printLoading = document.querySelector('.print-loading');
                                const printView = document.querySelector('.print-view');
                                const simulationContainer = document.querySelector('.simulation-results-container');
                                
                                // Check for React errors
                                const errorBoundary = document.querySelector('.error-boundary');
                                
                                // Check data availability 
                                const sessionData = sessionStorage.getItem('{pdf_data_id}');
                                const windowData = window.__PDF_DATA__;
                                
                                // Get any error messages
                                const errorElements = document.querySelectorAll('.print-loading');
                                const errorText = Array.from(errorElements).map(el => el.textContent).join('; ');
                                
                                return {{
                                    rootExists: !!root,
                                    printContent: !!printContent,
                                    printLoading: !!printLoading,
                                    printView: !!printView,
                                    simulationContainer: !!simulationContainer,
                                    errorBoundary: !!errorBoundary,
                                    sessionDataExists: !!sessionData,
                                    windowDataExists: !!windowData,
                                    errorText: errorText,
                                    location: window.location.href,
                                    bodyContent: document.body ? document.body.innerText.substring(0, 500) : 'No body'
                                }};
                            }})();
                        """)
                        
                        logger.info(f"Page status: {page_status}")
                        
                        # Check if print content is ready
                        if page_status['printContent']:
                            logger.info("✅ Print content found! Proceeding with PDF generation")
                            break
                        elif page_status['errorText'] and 'error' in page_status['errorText'].lower():
                            logger.error(f"Print view error detected: {page_status['errorText']}")
                            # Re-inject data if there's an error
                            await page.evaluate(f"""
                                console.log('Re-injecting data due to error...');
                                sessionStorage.setItem('{pdf_data_id}', JSON.stringify({json.dumps(print_data)}));
                                window.__PDF_DATA__ = {json.dumps(print_data)};
                                window.__PDF_DATA_ID__ = '{pdf_data_id}';
                            """)
                            # Refresh the page to retry
                            await page.reload(wait_until='networkidle')
                        else:
                            # Wait for React app to load and process data
                            if retry < max_retries - 1:
                                logger.info(f"Print content not ready yet, waiting {retry_delay}ms...")
                                await page.wait_for_timeout(retry_delay)
                            else:
                                # Final attempt - try to wait for selector with shorter timeout
                                logger.warning("Final attempt - waiting for .print-content with 10s timeout")
                                try:
                                    await page.wait_for_selector('.print-content, .print-loading', timeout=10000)
                                    # Check what we found
                                    final_check = await page.evaluate("""
                                        (() => {
                                            const content = document.querySelector('.print-content');
                                            const loading = document.querySelector('.print-loading');
                                            return {
                                                foundContent: !!content,
                                                foundLoading: !!loading,
                                                loadingText: loading ? loading.textContent : null
                                            };
                                        })();
                                    """)
                                    if final_check['foundContent']:
                                        logger.info("✅ Print content found on final attempt!")
                                        break
                                    elif final_check['foundLoading']:
                                        logger.warning(f"Still loading: {final_check['loadingText']}")
                                        raise Exception(f"Print view still loading: {final_check['loadingText']}")
                                    else:
                                        raise Exception("Print content never appeared")
                                except Exception as final_error:
                                    logger.error(f"Final wait failed: {final_error}")
                                    raise Exception(f"PDF generation failed: Print view did not load properly after {max_retries} retries. Last status: {page_status}")
                        
                    except Exception as retry_error:
                        if retry == max_retries - 1:
                            logger.error(f"All retries exhausted. Final error: {retry_error}")
                            raise retry_error
                        else:
                            logger.warning(f"Retry {retry + 1} failed: {retry_error}. Continuing...")
                            continue
                
                logger.info("Print view successfully loaded and ready for PDF generation")
                
                # STEP 7: Hide cookie banner and other non-essential elements for PDF
                await page.evaluate("""
                    // Hide cookie consent banner
                    const cookieBanner = document.querySelector('[role="dialog"][aria-live="polite"]');
                    if (cookieBanner) {
                        cookieBanner.style.display = 'none';
                        console.log('Hidden cookie banner for PDF export');
                    }
                    
                    // Hide any other toast notifications
                    const toastContainer = document.querySelector('.Toastify__toast-container');
                    if (toastContainer) {
                        toastContainer.style.display = 'none';
                        console.log('Hidden toast notifications for PDF export');
                    }
                """)
                
                # STEP 8: Wait for charts (EXACT same as print solution - 3 seconds)
                await page.wait_for_timeout(3000)
                logger.info("Waited for charts and dynamic components to render")
                
                # STEP 9: Add comprehensive debugging 
                await page.evaluate("""
                    console.log('=== PDF DEBUG SESSION START ===');
                    console.log('Current URL:', window.location.href);
                    console.log('Print content exists:', !!document.querySelector('.print-content'));
                    console.log('Simulation container exists:', !!document.querySelector('.simulation-results-container'));
                    console.log('Canvas elements:', document.querySelectorAll('canvas').length);
                    console.log('=== PDF DEBUG SESSION END ===');
                """)
                
                # STEP 10: Take debug screenshot
                debug_path = str(pdf_path).replace('.pdf', '_debug.png')
                await page.screenshot(path=debug_path, full_page=True)
                logger.info(f"Debug screenshot taken: {debug_path}")
                
                # STEP 11: Generate PDF
                logger.info("Generating PDF...")
                await page.pdf(
                    path=str(pdf_path),
                    format='A4',
                    print_background=True,
                    margin={'top': '20mm', 'right': '20mm', 'bottom': '20mm', 'left': '20mm'}
                )
                
                await browser.close()
                
            logger.info(f"PDF generated successfully: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise e
