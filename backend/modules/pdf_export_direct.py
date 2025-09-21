"""
Service for direct, pixel-perfect PDF export using a backend headless browser.
"""
import asyncio
import logging
from pathlib import Path
import tempfile
import uuid
from playwright.async_api import async_playwright
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# A simple in-memory cache to hold data for the headless browser to access.
# In a multi-server/multi-worker setup, this should be replaced with a shared cache like Redis.
data_cache = {}

class DirectPDFExportService:
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "monte_carlo_direct_pdfs"
        self.temp_dir.mkdir(exist_ok=True)

    async def generate_pdf_from_view(self, simulation_id: str, results_data: dict, frontend_url: str) -> Path:
        """
        Generates a PDF by having a headless browser visit the print view and save the file.

        Args:
            simulation_id: The ID of the simulation for naming.
            results_data: The dictionary containing simulation results.
            frontend_url: The base URL of the frontend application (e.g., http://frontend:3000).

        Returns:
            The path to the generated PDF file.
        """
        # The data passed to the print view needs to be in the same format as the sessionStorage method
        print_data_payload = {
            "simulationId": simulation_id,
            "results": results_data.get("targets", {}),
            "metadata": {
                "iterations_run": results_data.get("iterations_run"),
                "engine_type": results_data.get("requested_engine_type"),
                "timestamp": results_data.get("timestamp")
            }
        }
        
        data_id = f"print_data_{uuid.uuid4()}"
        
        # Store data in the cache for the browser to fetch.
        data_cache[data_id] = print_data_payload
        asyncio.create_task(self._schedule_cache_cleanup(data_id, delay=300)) # 5-minute cleanup

        pdf_path = self.temp_dir / f"direct_export_{simulation_id}_{uuid.uuid4()}.pdf"
        
        # The headless browser needs to access the frontend service.
        # Within Docker Compose, this is done via the service name, e.g., 'http://frontend:3000'
        print_url = f"{frontend_url}/print-view?id={data_id}&source=backend"
        
        logger.info(f"Initiating direct PDF export. Rendering URL: {print_url}")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
                page = await browser.new_page()
                
                # It's crucial to handle console logs from the headless browser for debugging
                page.on("console", lambda msg: logger.info(f"Playwright Console ({msg.type}): {msg.text}"))

                await page.goto(print_url, wait_until='networkidle', timeout=60000)

                # Wait for the results to be ready by checking for a canvas element,
                # which indicates a chart has rendered.
                await page.wait_for_selector('canvas', state='visible', timeout=30000)
                
                # Give charts an extra moment to complete their animations
                await asyncio.sleep(3)

                await page.pdf(
                    path=str(pdf_path),
                    format='A4',
                    print_background=True,
                    margin={'top': '15mm', 'right': '15mm', 'bottom': '15mm', 'left': '15mm'}
                )
                await browser.close()
                
            logger.info(f"Direct PDF generated successfully at {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"Playwright PDF generation failed for URL {print_url}: {e}", exc_info=True)
            # Try to capture a screenshot for debugging
            try:
                debug_path = self.temp_dir / f"failed_export_{simulation_id}.png"
                await page.screenshot(path=str(debug_path))
                logger.info(f"Saved debug screenshot to {debug_path}")
            except Exception as screenshot_error:
                logger.error(f"Failed to capture debug screenshot: {screenshot_error}")
            
            raise HTTPException(status_code=500, detail=f"Failed to generate PDF. Check server logs for details.")

    async def _schedule_cache_cleanup(self, data_id: str, delay: int):
        """Removes an item from the cache after a delay."""
        await asyncio.sleep(delay)
        if data_id in data_cache:
            del data_cache[data_id]
            logger.info(f"Cleaned up cached data for {data_id}")

# This function is not an endpoint dependency, but a utility for the new data endpoint
def get_cached_print_data(data_id: str):
    """Utility to retrieve cached data for the print view."""
    logger.info(f"Headless browser requesting data for ID: {data_id}")
    data = data_cache.get(data_id)
    if not data:
        logger.warning(f"Cache miss for data ID: {data_id}")
        return None
    logger.info(f"Cache hit for data ID: {data_id}")
    return data
