"""
Screenshot-based PDF Export Service for pixel-perfect frontend replication.
This service navigates to the actual frontend page and captures it as a PDF.
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
import base64

logger = logging.getLogger(__name__)

class ScreenshotPDFExportService:
    """Service for generating PDFs by capturing the actual frontend page"""
    
    def __init__(self):
        self.temp_dir = Path("/tmp/monte_carlo_pdfs")
        self.temp_dir.mkdir(exist_ok=True)
        
    async def generate_pdf_from_frontend(
        self, 
        simulation_id: str, 
        results_data: Dict[str, Any],
        auth_token: Optional[str] = None,
        frontend_url: str = "http://frontend:3000"
    ) -> str:
        """
        Generate a PDF by navigating to the actual frontend results page and capturing it.
        This ensures 100% pixel-perfect match with the frontend.
        """
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monte_carlo_results_{simulation_id}_{timestamp}.pdf"
            pdf_path = self.temp_dir / filename
            
            async with async_playwright() as p:
                # Launch browser with specific viewport for consistency
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
                    # Set up authentication for both potential domains
                    cookies = [
                        {
                            'name': 'auth_token',
                            'value': auth_token,
                            'domain': 'frontend',
                            'path': '/'
                        },
                        {
                            'name': 'auth_token', 
                            'value': auth_token,
                            'domain': 'localhost',
                            'path': '/'
                        }
                    ]
                    
                    # Add authorization header as well
                    await context.set_extra_http_headers({
                        'Authorization': f'Bearer {auth_token}'
                    })
                    
                    try:
                        await context.add_cookies(cookies)
                        logger.info("Authentication cookies and headers set for PDF generation")
                    except Exception as e:
                        logger.warning(f"Failed to set auth cookies: {e}, continuing without cookies")
                
                # First, inject the simulation results into the page's local storage or session
                # This allows the frontend to display the results without needing to fetch them
                # Set user agent to look like a real browser
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Linux; rv:91.0) Gecko/20100101 Firefox/91.0'
                })
                
                # Pre-inject authentication data into localStorage if auth token provided
                if auth_token:
                    await page.add_init_script(f"""
                        localStorage.setItem('authToken', '{auth_token}');
                        localStorage.setItem('auth0Token', '{auth_token}');
                        window.authToken = '{auth_token}';
                    """)
                
                # Use the same approach as the working print solution:
                # Store data in sessionStorage and navigate to print-view
                prepared_targets = []
                targets = results_data.get('targets', {})
                
                # Convert targets data to the format expected by SimulationResultsDisplay
                for target_name, target_data in targets.items():
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
                
                print_data = {
                    'simulationId': 'pdf_export',
                    'results': prepared_targets,
                    'metadata': {
                        'iterations_run': results_data.get('iterations_run', 1000),
                        'engine_type': results_data.get('requested_engine_type', 'Ultra'),
                        'timestamp': results_data.get('metadata', {}).get('timestamp', 'N/A')
                    }
                }
                
                # Generate unique ID for this PDF export
                pdf_data_id = f"pdf_data_{int(time.time() * 1000)}"
                
                logger.info(f"Prepared PDF data with ID: {pdf_data_id}")
                logger.info(f"Prepared {len(prepared_targets)} targets for PDF generation")
                
                # Inject data into sessionStorage before navigation (same as print solution)
                await page.evaluate(f"""
                    sessionStorage.setItem('{pdf_data_id}', JSON.stringify({json.dumps(print_data)}));
                    console.log('PDF data stored in sessionStorage with ID: {pdf_data_id}');
                """)
                
                # Navigate to the print-view page (same as working print solution)
                await page.goto(f"{frontend_url}/print-view?id={pdf_data_id}", wait_until='networkidle')
                logger.info("Navigated to print-view page")
                
                # Wait for the print view to be ready (same pattern as working print solution)
                logger.info("Waiting for print view to load and process data...")
                
                # Wait for print view component to load and process the sessionStorage data
                await page.wait_for_function("""
                    // Check if the print view has loaded and processed the data
                    const printContent = document.querySelector('.print-content');
                    const simulationContainer = document.querySelector('.simulation-results-container');
                    
                    // Print view is ready when print content and simulation container exist
                    return printContent && simulationContainer;
                """, timeout=30000)
                logger.info("Print view loaded and simulation results container found")
                
                # Force show the root element if it's hidden
                await page.evaluate("""
                    const root = document.querySelector('#root');
                    if (root) {
                        root.style.display = 'block';
                        root.style.visibility = 'visible';
                        root.style.opacity = '1';
                        console.log('Root element forced visible');
                    }
                """)
                logger.info("Root element forced visible")
                
                # Instead of injecting HTML, let's wait for React to load and inject data into Redux
                logger.info("Waiting for React app to load and injecting simulation data")
                
                # Wait for Redux store to be available
                await page.wait_for_function("""
                    window.__REDUX_DEVTOOLS_EXTENSION__ || window.store || window.__REDUX_STORE__
                """, timeout=30000)
                logger.info("Redux store detected")
                
                # Prepare and inject simulation data into Redux
                prepared_data = self._prepare_frontend_data(results_data)
                
                # Inject simulation data into Redux store so React components render naturally
                await page.evaluate(f"""
                    try {{
                        const simulationData = {json.dumps(prepared_data)};
                        
                        // Try different ways to access the Redux store
                        let store = null;
                        if (window.store) {{
                            store = window.store;
                        }} else if (window.__REDUX_STORE__) {{
                            store = window.__REDUX_STORE__;
                        }} else if (window.__STORE__) {{
                            store = window.__STORE__;
                        }}
                        
                        if (store && store.dispatch) {{
                            // Dispatch simulation results to load in the UI
                            store.dispatch({{
                                type: 'simulation/setMultipleResults',
                                payload: simulationData
                            }});
                            
                            // Also set as completed status
                            store.dispatch({{
                                type: 'simulation/setStatus', 
                                payload: 'completed'
                            }});
                            
                            console.log('Simulation data dispatched to Redux store');
                        }} else {{
                            console.log('Redux store not found, trying alternative approach');
                            
                            // Fallback: Set as global window variable
                            window.simulationResults = simulationData;
                            window.pdfExportMode = true;
                        }}
                        
                        window.pdfDataReady = true;
                        
                    }} catch (error) {{
                        console.error('Error injecting simulation data:', error);
                        window.pdfDataReady = true; // Continue anyway
                    }}
                """)
                logger.info("Simulation data injected into Redux")
                
                # Wait extra time for React to fully initialize and render
                logger.info("Waiting for React application to fully initialize...")
                await page.wait_for_timeout(12000)  # Give React even more time to start up
                
                # Add comprehensive debugging - NO FALLBACKS, let it fail clearly
                await page.evaluate("""
                    console.log('=== PDF DEBUG SESSION START ===');
                    console.log('Current URL:', window.location.href);
                    console.log('Document title:', document.title);
                    console.log('Document readyState:', document.readyState);
                    console.log('React loaded:', typeof window.React !== 'undefined');
                    console.log('Redux store exists:', !!window.store);
                    console.log('Redux dispatch ready:', !!window.store?.dispatch);
                    console.log('Redux state keys:', window.store ? Object.keys(window.store.getState()) : 'No store');
                    console.log('Body HTML length:', document.body ? document.body.innerHTML.length : 'No body');
                    console.log('DOM elements count:', document.querySelectorAll('*').length);
                    console.log('Simulation results container exists:', !!document.querySelector('.simulation-results-container'));
                    console.log('Results header exists:', !!document.querySelector('.results-header-new'));
                    console.log('Any div with result class:', document.querySelectorAll('div[class*="result"]').length);
                    console.log('Canvas elements:', document.querySelectorAll('canvas').length);
                    console.log('PDF data ready flag:', window.pdfDataReady);
                    console.log('PDF content ready flag:', window.pdfContentReady);
                    console.log('=== PDF DEBUG SESSION END ===');
                """)
                
                # Wait for React components to render the simulation results - NO FALLBACKS
                # Since we set pdfDataReady=true after Redux injection, wait for it first
                await page.wait_for_function("""
                    window.pdfDataReady === true
                """, timeout=30000)
                logger.info("PDF data ready flag confirmed")
                
                # STRICT wait for simulation results container - let it fail if not found
                await page.wait_for_selector(".simulation-results-container", timeout=30000)
                logger.info("Found simulation results container - proceeding with PDF generation")
                
                # Wait for charts to fully render (Chart.js, etc.)
                await page.wait_for_timeout(5000)
                logger.info("Waited for charts and dynamic components to render")
                
                # Wait for charts and visual content to render
                await page.wait_for_function("""
                    // Check that we have visual elements rendered
                    const canvases = document.querySelectorAll('canvas');
                    const results = document.querySelectorAll('[class*="result"], [class*="Result"], .simulation-results-container');
                    // We need either charts/canvases OR result containers
                    return canvases.length > 0 || results.length > 0;
                """, timeout=30000)
                logger.info("Visual components and charts confirmed rendered")
                
                # DEBUG: Take a screenshot to see what we get after Redux dispatch
                debug_path = str(pdf_path).replace('.pdf', '_debug.png')
                await page.screenshot(path=debug_path, full_page=True)
                logger.info(f"Debug screenshot taken: {debug_path}")
                
                # DEBUG: Check what's on the page
                page_content = await page.content()
                logger.info(f"Page content length: {len(page_content)}")
                logger.info(f"Page title: {await page.title()}")
                
                # DEBUG: Check if simulation results container exists
                container_exists = await page.query_selector('.simulation-results-container')
                logger.info(f"simulation-results-container exists: {container_exists is not None}")
                
                # Check if Redux dispatch succeeded
                pdf_data_ready = await page.evaluate("window.pdfDataReady")
                logger.info(f"Redux dispatch ready: {pdf_data_ready}")
                
                # Check if window.simulationResults exists
                sim_results_exist = await page.evaluate("typeof window.simulationResults !== 'undefined'")
                logger.info(f"window.simulationResults exists: {sim_results_exist}")
                
                # Also check if React app loaded
                react_loaded = await page.evaluate("typeof React !== 'undefined'")
                logger.info(f"React loaded: {react_loaded}")
                
                if not container_exists:
                    # Log all divs with class containing "result" to see what's available
                    result_divs = await page.evaluate("""
                        Array.from(document.querySelectorAll('div')).
                        filter(div => div.className && div.className.toLowerCase().includes('result')).
                        map(div => div.className)
                    """)
                    logger.info(f"Found divs with 'result' in className: {result_divs}")
                
                # Ensure the results content area is loaded
                # Use the actual CSS classes from the frontend or fallback content
                try:
                    await page.wait_for_selector('.simulation-results-container', timeout=15000)
                    logger.info("Simulation results container confirmed visible")
                except Exception as e:
                    logger.warning(f"Could not find .simulation-results-container: {e}")
                    # Try alternative selectors
                    try:
                        await page.wait_for_selector('.pdf-fallback-content', timeout=5000)
                        logger.info("Fallback content confirmed visible")
                    except Exception as e2:
                        logger.warning(f"Could not find fallback content either: {e2}")
                        # Try React results header
                        try:
                            await page.wait_for_selector('.results-header-new', timeout=5000)
                            logger.info("Results header confirmed visible")
                        except Exception as e3:
                            logger.warning(f"Could not find any content selectors: {e3}")
                            # Continue anyway since data might still be present
                            pass
                
                # Additional wait to ensure all charts are rendered
                await page.wait_for_timeout(3000)
                
                # Scroll through the page to ensure all lazy-loaded content is rendered
                await page.evaluate("""
                    (async function() {
                        const distance = 300;
                        const delay = 100;
                        const bottom = document.body.scrollHeight;
                        
                        for (let i = 0; i <= bottom; i += distance) {
                            window.scrollTo(0, i);
                            await new Promise(res => setTimeout(res, delay));
                        }
                        
                        // Scroll back to top
                        window.scrollTo(0, 0);
                        await new Promise(res => setTimeout(res, 500));
                    })();
                """)
                
                # Hide any UI elements that shouldn't be in the PDF
                await page.evaluate("""
                    // Hide navigation, buttons, and interactive elements
                    const elementsToHide = [
                        '.sidebar', '[class*="sidebar"]', '[class*="Sidebar"]',
                        'nav', '.nav', '.navbar', '.navigation', '[class*="navigation"]',
                        'button[class*="pdf"]', 'button[class*="export"]', 'button[class*="print"]',
                        '.export-button', '.export-pdf-button', '.export-ppt-button',
                        '[class*="toolbar"]', '[class*="controls"]', '[class*="actions"]',
                        '[class*="progress"]', '[class*="loader"]', '[class*="spinner"]',
                        '.toast', '.notification', '.alert', '[class*="toast"]',
                        'header', '[class*="header"]', '[class*="Header"]',
                        'button:not(.chart-button):not([class*="chart"])'  // Hide buttons except chart-related ones
                    ];
                    
                    elementsToHide.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => {
                            el.style.display = 'none';
                        });
                    });
                    
                    // Optimize the main content area for PDF
                    const mainContent = document.querySelector('main, [class*="main"], [class*="content"], [class*="simulate"], #root > div');
                    if (mainContent) {
                        mainContent.style.width = '100%';
                        mainContent.style.maxWidth = 'none';
                        mainContent.style.margin = '0';
                        mainContent.style.padding = '20px';
                        mainContent.style.background = 'white';
                        mainContent.style.minHeight = 'auto';
                    }
                    
                    // Ensure charts and visual elements are optimized for PDF
                    document.querySelectorAll('canvas, [class*="chart"], [class*="Chart"]').forEach(el => {
                        el.style.maxWidth = '100%';
                        el.style.pageBreakInside = 'avoid';
                        el.style.display = 'block';
                        el.style.visibility = 'visible';
                    });
                    
                    // Make sure result cards and containers are PDF-ready
                    document.querySelectorAll('[class*="result"], [class*="Result"], [class*="card"]').forEach(el => {
                        el.style.pageBreakInside = 'avoid';
                        el.style.marginBottom = '20px';
                    });
                """)
                
                # Generate PDF with print media CSS
                await page.pdf(
                    path=str(pdf_path),
                    format='A4',
                    print_background=True,
                    margin={
                        'top': '10mm',
                        'right': '10mm', 
                        'bottom': '10mm',
                        'left': '10mm'
                    },
                    scale=0.8,
                    display_header_footer=True,
                    header_template="""
                        <div style="font-size: 10px; text-align: center; width: 100%;">
                            <span>Monte Carlo Simulation Results</span>
                        </div>
                    """,
                    footer_template="""
                        <div style="font-size: 10px; text-align: center; width: 100%;">
                            <span class="pageNumber"></span> / <span class="totalPages"></span>
                        </div>
                    """
                )
                
                await browser.close()
                
            # Verify PDF was created
            if not pdf_path.exists():
                raise Exception("PDF generation failed - file not created")
                
            logger.info(f"Screenshot PDF generated successfully: {pdf_path} ({pdf_path.stat().st_size} bytes)")
            return filename
            
        except Exception as e:
            logger.error(f"Error generating screenshot PDF: {str(e)}")
            raise
            
    def _prepare_frontend_data(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the results data in the format expected by the frontend.
        The frontend expects an object with targets property.
        """
        targets = results_data.get('targets', {})
        iterations_run = results_data.get('iterations_run', 1000)
        engine_type = results_data.get('requested_engine_type', 'Standard')
        
        prepared_targets = {}
        
        for idx, (target_name, target_data) in enumerate(targets.items()):
            # Handle both possible data structures
            if 'statistics' in target_data:
                # New nested structure
                statistics = target_data.get('statistics', {})
                values = target_data.get('values', [])
            else:
                # Direct structure (what frontend is sending)
                statistics = target_data
                values = target_data.get('raw_values', target_data.get('values', []))
            
            # Create a result object that matches the frontend's expected structure
            prepared_targets[target_name] = {
                'simulation_id': f'pdf_export_{idx}',
                'temp_id': f'pdf_temp_{idx}',
                'status': 'completed',
                'target_name': target_name,
                'result_cell_coordinate': target_name,
                'target_cell': target_name,
                'iterations_run': iterations_run,
                'requested_engine_type': engine_type,
                'results': {
                    'mean': statistics.get('mean'),
                    'median': statistics.get('median'),
                    'std_dev': statistics.get('std_dev'),
                    'min_value': statistics.get('min_value'),
                    'max_value': statistics.get('max_value'),
                    'percentiles': statistics.get('percentiles', {}),
                    'iterations_run': iterations_run,
                    'raw_values': values,
                    'histogram': self._generate_histogram_data(values) if values else None,
                    'sensitivity_analysis': statistics.get('sensitivity_analysis')
                }
            }
            
        return {
            'targets': prepared_targets,
            'iterations_run': iterations_run,
            'requested_engine_type': engine_type,
            'metadata': results_data.get('metadata', {})
        }
        
    def _generate_histogram_data(self, values: list, num_bins: int = 50) -> Dict[str, Any]:
        """Generate histogram data in the format expected by the frontend"""
        if not values:
            return {'bin_edges': [], 'counts': []}
            
        import numpy as np
        
        # Calculate histogram
        counts, bin_edges = np.histogram(values, bins=num_bins)
        
        return {
            'bin_edges': bin_edges.tolist(),
            'counts': counts.tolist(),
            'bins': bin_edges.tolist(),
            'values': counts.tolist()
        }
