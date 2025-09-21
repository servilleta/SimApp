"""
Modern PDF Export Service using Playwright for 100% visual fidelity.
This ensures the PDF looks exactly like the webpage results.
"""
import asyncio
import base64
import tempfile
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from playwright.async_api import async_playwright
from fastapi import HTTPException

try:
    from .pdf_export_enhanced import EnhancedPDFExportService
except ImportError:
    EnhancedPDFExportService = None

try:
    from .pdf_export_screenshot import PDFExportScreenshotService
except ImportError:
    PDFExportScreenshotService = None

logger = logging.getLogger(__name__)

class PDFExportService:
    """Service for generating PDFs with perfect visual fidelity using Playwright"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "monte_carlo_pdfs"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def generate_pdf_from_frontend(
        self, 
        simulation_id: str,
        results_data: Dict[str, Any],
        auth_token: Optional[str] = None,
        frontend_url: str = "http://frontend:3000"
    ) -> str:
        """
        Generate PDF using the screenshot service (main entry point for background generation).
        This method delegates to the screenshot service for pixel-perfect frontend match.
        """
        # Use ONLY screenshot service for pixel-perfect frontend match
        if not PDFExportScreenshotService:
            logger.error("Screenshot PDF service not available")
            raise HTTPException(status_code=500, detail="Screenshot PDF service not available")
            
        logger.info("Using screenshot PDF export service for pixel-perfect frontend match")
        screenshot_service = PDFExportScreenshotService()
        
        try:
            pdf_path = await screenshot_service.generate_pdf_from_frontend(
                simulation_id, 
                results_data,
                auth_token=auth_token,
                frontend_url=frontend_url
            )
            return pdf_path  # Already returns full path
        except Exception as e:
            logger.error(f"Screenshot PDF service failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Screenshot PDF generation failed: {str(e)}")
        
    async def generate_pdf_from_results_page(
        self, 
        simulation_id: str,
        results_data: Dict[str, Any],
        auth_token: Optional[str] = None,
        base_url: str = "http://localhost:3000"
    ) -> str:
        """
        Generate a PDF by rendering the actual results page in a headless browser.
        This ensures 100% visual fidelity with the webpage.
        """
        # Use ONLY screenshot service for pixel-perfect frontend match
        if not PDFExportScreenshotService:
            logger.error("Screenshot PDF service not available")
            raise HTTPException(status_code=500, detail="Screenshot PDF service not available")
            
        logger.info("Using screenshot PDF export service for pixel-perfect frontend match")
        screenshot_service = PDFExportScreenshotService()
        
        try:
            pdf_path = await screenshot_service.generate_pdf_from_frontend(
                simulation_id, 
                results_data,
                auth_token=auth_token,
                frontend_url="http://frontend:3000"  # Docker service name
            )
            return pdf_path  # Already returns full path
        except Exception as e:
            logger.error(f"Screenshot PDF service failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Screenshot PDF generation failed: {str(e)}")
    
    async def _create_standalone_results_html(self, results_data: Dict[str, Any], simulation_id: str) -> str:
        """Create a standalone HTML page with all necessary CSS and data embedded"""
        
        # Debug: Log the data structure being received
        logger.info(f"PDF Generation - Received results_data keys: {list(results_data.keys()) if results_data else 'None'}")
        logger.info(f"PDF Generation - Simulation ID: {simulation_id}")
        
        # Debug: Check the targets structure
        targets = results_data.get('targets', {}) if results_data else {}
        logger.info(f"PDF Generation - Targets keys: {list(targets.keys()) if targets else 'No targets'}")
        logger.info(f"PDF Generation - Number of targets: {len(targets) if targets else 0}")
        
        # Debug: Log first target structure if available
        if targets:
            first_target_name = list(targets.keys())[0]
            first_target_data = targets[first_target_name]
            logger.info(f"PDF Generation - First target '{first_target_name}' keys: {list(first_target_data.keys()) if isinstance(first_target_data, dict) else 'Not a dict'}")
            logger.info(f"PDF Generation - First target data type: {type(first_target_data)}")
            
            # Debug: Check what's actually in the statistics and values
            if isinstance(first_target_data, dict):
                statistics = first_target_data.get('statistics', {})
                values = first_target_data.get('values', [])
                logger.info(f"PDF Generation - Statistics keys: {list(statistics.keys()) if isinstance(statistics, dict) else 'Not a dict'}")
                logger.info(f"PDF Generation - Statistics content: {statistics}")
                logger.info(f"PDF Generation - Values count: {len(values) if isinstance(values, list) else 'Not a list'}")
                logger.info(f"PDF Generation - Values type: {type(values)}")
                if isinstance(values, list) and len(values) > 0:
                    logger.info(f"PDF Generation - First few values: {values[:5]}")
        
        # Get the timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monte Carlo Simulation Results - {simulation_id}</title>
    
    <!-- Tailwind CSS CDN for styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Chart.js for visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-box-and-violin-plot"></script>
    
    <!-- Custom styles for print -->
    <style>
        @media print {{
            body {{ margin: 0; padding: 20px; }}
            .no-print {{ display: none !important; }}
            .page-break {{ page-break-before: always; }}
        }}
        
        /* Ensure charts render properly */
        canvas {{
            max-width: 100% !important;
            height: auto !important;
        }}
        
        /* Professional styling */
        .results-container {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #374151;
        }}
        
        .statistics-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        
        .statistics-table th,
        .statistics-table td {{
            border: 1px solid #e5e7eb;
            padding: 12px;
            text-align: left;
        }}
        
        .statistics-table th {{
            background-color: #f9fafb;
            font-weight: 600;
        }}
        
        .header-section {{
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .variable-section {{
            margin-bottom: 40px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 24px;
        }}
        
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        
        .certainty-analysis {{
            background: #f0f9ff;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body class="bg-white">
    <div class="results-container max-w-6xl mx-auto p-8">
        <!-- Header Section -->
        <div class="header-section">
            <h1 class="text-4xl font-bold text-gray-900 mb-4">Monte Carlo Simulation Results</h1>
            <div class="grid grid-cols-2 gap-8 text-sm text-gray-600">
                <div>
                    <p><strong>Generated:</strong> {timestamp}</p>
                    <p><strong>Simulation ID:</strong> {simulation_id}</p>
                </div>
                <div>
                    <p><strong>Iterations:</strong> {results_data.get('iterations_run', 'N/A')}</p>
                    <p><strong>Engine:</strong> {results_data.get('requested_engine_type', 'Standard')}</p>
                </div>
            </div>
        </div>
        
        <!-- Results will be inserted here -->
        <div id="results-content">
            {await self._generate_results_html(results_data)}
        </div>
        
        <!-- Footer -->
        <div class="text-center text-sm text-gray-500 mt-12 pt-8 border-t">
            <p>Generated by Monte Carlo Simulation Platform</p>
            <p>© {datetime.now().year} Monte Carlo Analytics. All rights reserved.</p>
        </div>
    </div>
    
    <script>
        // Initialize charts after DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {{
            {await self._generate_chart_scripts(results_data)}
        }});
    </script>
</body>
</html>
"""
        return html_content
    
    async def _generate_results_html(self, results_data: Dict[str, Any]) -> str:
        """Generate the HTML content for simulation results"""
        
        html_parts = []
        
        # Process each target variable
        targets = results_data.get('targets', {})
        logger.info(f"PDF HTML Generation - Found {len(targets) if targets else 0} targets")
        
        if not targets:
            logger.warning("PDF HTML Generation - No targets found, returning 'No results data available'")
            return "<p class='text-gray-500'>No results data available.</p>"
        
        for idx, (target_name, target_data) in enumerate(targets.items()):
            if idx > 0:
                html_parts.append('<div class="page-break"></div>')
                
            html_parts.append(f"""
            <div class="variable-section">
                <h2 class="text-2xl font-bold text-gray-900 mb-6">{target_name}</h2>
                
                <!-- Statistics Table -->
                <div class="mb-8">
                    <h3 class="text-lg font-semibold text-gray-800 mb-4">Statistical Summary</h3>
                    <table class="statistics-table">
                        <thead>
                            <tr>
                                <th>Statistic</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_statistics_rows(target_data)}
                        </tbody>
                    </table>
                </div>
                
                <!-- Distribution Chart -->
                <div class="chart-container">
                    <h3 class="text-lg font-semibold text-gray-800 mb-4">Distribution</h3>
                    <canvas id="chart-{idx}" width="800" height="400"></canvas>
                </div>
                
                <!-- Certainty Analysis if available -->
                {self._generate_certainty_analysis(target_data)}
            </div>
            """)
        
        return "\n".join(html_parts)
    
    def _generate_statistics_rows(self, target_data: Dict[str, Any]) -> str:
        """Generate HTML rows for statistics table"""
        stats = target_data.get('statistics', {})
        
        # Standard statistics to display
        stat_labels = {
            'mean': 'Mean',
            'median': 'Median',
            'std': 'Standard Deviation',
            'min': 'Minimum',
            'max': 'Maximum',
            'p5': '5th Percentile',
            'p25': '25th Percentile (Q1)',
            'p75': '75th Percentile (Q3)',
            'p95': '95th Percentile',
            'variance': 'Variance',
            'skewness': 'Skewness',
            'kurtosis': 'Kurtosis'
        }
        
        rows = []
        for key, label in stat_labels.items():
            if key in stats:
                value = stats[key]
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.2f}"
                else:
                    formatted_value = str(value)
                rows.append(f"<tr><td>{label}</td><td>{formatted_value}</td></tr>")
        
        return "\n".join(rows)
    
    def _generate_certainty_analysis(self, target_data: Dict[str, Any]) -> str:
        """Generate certainty analysis section if data is available"""
        # This would be populated based on your certainty analysis data structure
        return ""
    
    async def _generate_chart_scripts(self, results_data: Dict[str, Any]) -> str:
        """Generate JavaScript for rendering charts"""
        
        scripts = []
        targets = results_data.get('targets', {})
        
        for idx, (target_name, target_data) in enumerate(targets.items()):
            values = target_data.get('values', [])
            if not values:
                continue
                
            # Create histogram data
            script = f"""
            // Chart for {target_name}
            const ctx{idx} = document.getElementById('chart-{idx}').getContext('2d');
            const values{idx} = {values};
            
            // Create histogram bins
            const bins{idx} = createHistogramBins(values{idx}, 50);
            
            new Chart(ctx{idx}, {{
                type: 'bar',
                data: {{
                    labels: bins{idx}.labels,
                    datasets: [{{
                        label: 'Frequency',
                        data: bins{idx}.counts,
                        backgroundColor: 'rgba(59, 130, 246, 0.6)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Distribution of {target_name}'
                        }},
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Value'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Frequency'
                            }}
                        }}
                    }}
                }}
            }});
            """
            scripts.append(script)
        
        # Add utility function for creating histogram bins
        utility_script = """
        function createHistogramBins(data, numBins) {
            if (data.length === 0) return { labels: [], counts: [] };
            
            const min = Math.min(...data);
            const max = Math.max(...data);
            const binWidth = (max - min) / numBins;
            
            const bins = new Array(numBins).fill(0);
            const labels = [];
            
            // Create bin labels
            for (let i = 0; i < numBins; i++) {
                const binStart = min + i * binWidth;
                const binEnd = min + (i + 1) * binWidth;
                labels.push(`${binStart.toFixed(2)}`);
            }
            
            // Count values in each bin
            data.forEach(value => {
                let binIndex = Math.floor((value - min) / binWidth);
                if (binIndex >= numBins) binIndex = numBins - 1;
                if (binIndex < 0) binIndex = 0;
                bins[binIndex]++;
            });
            
            return { labels, counts: bins };
        }
        """
        
        return utility_script + "\n" + "\n".join(scripts)
    
    async def _render_pdf_with_playwright(self, html_content: str, simulation_id: str) -> str:
        """Use Playwright to render HTML to PDF with perfect fidelity"""
        
        # Create temporary HTML file
        html_file = self.temp_dir / f"results_{simulation_id}.html"
        pdf_file = self.temp_dir / f"monte_carlo_results_{simulation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Write HTML content
        html_file.write_text(html_content, encoding='utf-8')
        
        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate to the HTML file
                await page.goto(f"file://{html_file.absolute()}")
                
                # Wait for scripts to load
                await page.wait_for_load_state('networkidle')
                
                # Wait for charts to render - increased timeout
                await page.wait_for_timeout(8000)  # Give more time for charts to load
                
                # Try to wait for Chart.js to be available
                try:
                    await page.wait_for_function('typeof Chart !== "undefined"', timeout=5000)
                    logger.info("Chart.js loaded successfully")
                except:
                    logger.warning("Chart.js may not have loaded properly")
                
                # Wait for any charts to be rendered
                try:
                    await page.wait_for_selector('canvas', timeout=5000)
                    logger.info("Canvas elements found on page")
                except:
                    logger.warning("No canvas elements found - charts may not be rendering")
                
                # Additional wait for chart rendering
                await page.wait_for_timeout(2000)
                
                # Generate PDF with optimal settings for visual fidelity
                await page.pdf(
                    path=str(pdf_file),
                    format='A4',
                    margin={
                        'top': '0.5in',
                        'right': '0.5in',
                        'bottom': '0.5in',
                        'left': '0.5in'
                    },
                    print_background=True,  # Include background colors/images
                    prefer_css_page_size=True,
                    display_header_footer=False
                )
                
                await browser.close()
                
        except Exception as e:
            logger.error(f"Playwright PDF generation failed: {str(e)}")
            raise
        finally:
            # Clean up temporary HTML file
            if html_file.exists():
                html_file.unlink()
        
        logger.info(f"PDF generated successfully: {pdf_file}")
        return str(pdf_file)
    
    async def generate_pdf_from_url(
        self, 
        url: str, 
        simulation_id: str,
        wait_for_selector: Optional[str] = None
    ) -> str:
        """
        Generate PDF directly from a URL (alternative method).
        Useful when you want to capture the live webpage.
        """
        pdf_file = self.temp_dir / f"monte_carlo_results_url_{simulation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set viewport for consistent rendering
                await page.set_viewport_size({"width": 1920, "height": 1080})
                
                # Navigate to URL
                await page.goto(url, wait_until='networkidle')
                
                # Wait for specific element if provided
                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, timeout=10000)
                
                # Additional wait for dynamic content
                await page.wait_for_timeout(2000)
                
                # Generate PDF
                await page.pdf(
                    path=str(pdf_file),
                    format='A4',
                    margin={
                        'top': '0.5in',
                        'right': '0.5in',
                        'bottom': '0.5in',
                        'left': '0.5in'
                    },
                    print_background=True,
                    prefer_css_page_size=True
                )
                
                await browser.close()
                
        except Exception as e:
            logger.error(f"URL PDF generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF generation from URL failed: {str(e)}")
        
        return str(pdf_file)
    
    def cleanup_old_pdfs(self, max_age_hours: int = 24):
        """Clean up old PDF files to prevent disk space issues"""
        import time
        
        current_time = time.time()
        for pdf_file in self.temp_dir.glob("*.pdf"):
            file_age_hours = (current_time - pdf_file.stat().st_mtime) / 3600
            if file_age_hours > max_age_hours:
                try:
                    pdf_file.unlink()
                    logger.info(f"Cleaned up old PDF: {pdf_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up PDF {pdf_file}: {e}")

# Global instance
pdf_export_service = PDFExportService()
