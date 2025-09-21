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
            
            # Create temporary file for PDF output
            temp_dir = Path(tempfile.gettempdir())
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
                
                # STEP 4: Store in sessionStorage (EXACT same as print solution)
                await page.evaluate(f"""
                    sessionStorage.setItem('{pdf_data_id}', JSON.stringify({json.dumps(print_data)}));
                    console.log('PDF data stored in sessionStorage with ID: {pdf_data_id}');
                """)
                logger.info("Data stored in sessionStorage")
                
                # STEP 5: Navigate to print-view (EXACT same as print solution)
                await page.goto(f"{frontend_url}/print-view?id={pdf_data_id}", wait_until='networkidle')
                logger.info("Navigated to print-view page")
                
                # STEP 6: Wait for print view to be ready (EXACT same pattern as print solution)
                logger.info("Waiting for print view to load and process data...")
                
                await page.wait_for_function("""
                    // Same check as print solution - look for print content and simulation container
                    const printContent = document.querySelector('.print-content');
                    const simulationContainer = document.querySelector('.simulation-results-container');
                    
                    return printContent && simulationContainer;
                """, timeout=30000)
                logger.info("Print view loaded and simulation results container found")
                
                # STEP 7: Wait for charts (EXACT same as print solution - 3 seconds)
                await page.wait_for_timeout(3000)
                logger.info("Waited for charts and dynamic components to render")
                
                # STEP 8: Add comprehensive debugging 
                await page.evaluate("""
                    console.log('=== PDF DEBUG SESSION START ===');
                    console.log('Current URL:', window.location.href);
                    console.log('Print content exists:', !!document.querySelector('.print-content'));
                    console.log('Simulation container exists:', !!document.querySelector('.simulation-results-container'));
                    console.log('Canvas elements:', document.querySelectorAll('canvas').length);
                    console.log('=== PDF DEBUG SESSION END ===');
                """)
                
                # STEP 9: Take debug screenshot
                debug_path = str(pdf_path).replace('.pdf', '_debug.png')
                await page.screenshot(path=debug_path, full_page=True)
                logger.info(f"Debug screenshot taken: {debug_path}")
                
                # STEP 10: Generate PDF
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
