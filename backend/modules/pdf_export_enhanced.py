import os
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from pathlib import Path
import subprocess
import tempfile
import uuid

logger = logging.getLogger(__name__)

class EnhancedPDFExportService:
    """Enhanced PDF export service that replicates the frontend's complete look and feel"""
    
    def __init__(self):
        self.temp_dir = Path("/tmp/monte_carlo_pdfs")
        self.temp_dir.mkdir(exist_ok=True)
        
    async def generate_pdf(self, simulation_id: str, results_data: Dict[str, Any]) -> str:
        """Generate a PDF that exactly replicates the frontend visualization"""
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monte_carlo_results_{simulation_id}_{timestamp}.pdf"
            pdf_path = self.temp_dir / filename
            
            # Generate the enhanced HTML content with all visualizations
            html_content = await self._generate_enhanced_html(simulation_id, results_data)
            
            # Save HTML to temporary file
            html_file = self.temp_dir / f"temp_{uuid.uuid4()}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Convert HTML to PDF using Playwright for best rendering
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Load the HTML file
                await page.goto(f'file://{html_file}', wait_until='networkidle')
                
                # Wait for charts to be fully rendered
                # First wait for Chart.js to be loaded
                await page.wait_for_function('typeof Chart !== "undefined"', timeout=10000)
                
                # Wait for all chart canvases to be rendered
                await page.wait_for_selector('canvas', state='visible')
                
                # Execute chart rendering if needed
                await page.evaluate('''
                    // Force chart updates
                    if (window.Chart && Chart.instances) {
                        Object.values(Chart.instances).forEach(chart => {
                            if (chart) {
                                chart.update();
                            }
                        });
                    }
                ''')
                
                # Additional wait for rendering to complete
                await page.wait_for_timeout(3000)
                
                # Generate PDF with print CSS
                await page.pdf(
                    path=str(pdf_path),
                    format='A4',
                    print_background=True,
                    margin={'top': '10mm', 'bottom': '10mm', 'left': '10mm', 'right': '10mm'}
                )
                
                await browser.close()
            
            # Clean up HTML file
            html_file.unlink()
            
            # Verify PDF was created
            if not pdf_path.exists():
                raise Exception("PDF generation failed")
                
            logger.info(f"Enhanced PDF generated successfully: {pdf_path} ({pdf_path.stat().st_size} bytes)")
            return filename
            
        except Exception as e:
            logger.error(f"Error generating enhanced PDF: {str(e)}")
            raise
            
    async def _generate_enhanced_html(self, simulation_id: str, results_data: Dict[str, Any]) -> str:
        """Generate enhanced HTML content that replicates the frontend exactly"""
        
        # Extract data
        targets = results_data.get('targets', {})
        iterations_run = results_data.get('iterations_run', 'N/A')
        engine_type = results_data.get('requested_engine_type', 'Standard')
        timestamp = results_data.get('timestamp', datetime.now().isoformat())
        
        # Debug logging
        logger.info(f"PDF Generation - Number of targets: {len(targets)}")
        
        # Generate the complete HTML with all visualizations
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monte Carlo Simulation Results</title>
    
    <!-- Chart.js with plugins -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <script src="https://unpkg.com/@sgratzl/chartjs-chart-boxplot@4.2.0/build/index.umd.js"></script>
    
    <!-- Ensure Chart.js is available globally -->
    <script>
        window.addEventListener('load', function() {{
            console.log('Chart.js loaded:', typeof Chart !== 'undefined');
            console.log('BoxPlot plugin loaded:', typeof window.ChartjsBoxPlot !== 'undefined');
        }});
    </script>
    
    <style>
        /* Frontend-matching styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            color: #1a1a1a;
            background: #f9fafb;
            padding: 2rem;
            line-height: 1.6;
        }}
        
        /* Braun color palette from frontend */
        :root {{
            --color-braun-orange: #FF6B35;
            --color-charcoal: #333333;
            --color-medium-grey: #777777;
            --color-light-grey: #CCCCCC;
            --color-white: #FFFFFF;
            --color-border-light: #e5e7eb;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 2px solid var(--color-border-light);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--color-charcoal);
            margin-bottom: 1rem;
        }}
        
        .metadata {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            font-size: 0.875rem;
            color: var(--color-medium-grey);
        }}
        
        .metadata-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .metadata-item strong {{
            color: var(--color-charcoal);
        }}
        
        /* Box plot section matching frontend */
        .box-plot-overview-section {{
            background: white;
            border: 1px solid var(--color-border-light);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 3rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        .box-plot-header {{
            border-bottom: 1px solid rgba(229, 231, 235, 0.5);
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
        }}
        
        .box-plot-header h2 {{
            color: var(--color-charcoal);
            font-size: 1.25rem;
            font-weight: 600;
        }}
        
        .box-plot-container {{
            background: rgba(249, 250, 251, 0.3);
            border-radius: 6px;
            padding: 1rem;
            height: 400px;
            position: relative;
        }}
        
        /* Target variable sections */
        .target-section {{
            background: white;
            border: 1px solid var(--color-border-light);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            page-break-inside: avoid;
        }}
        
        .target-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--color-border-light);
        }}
        
        .target-title {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--color-charcoal);
        }}
        
        .content-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        /* Statistics table matching frontend */
        .stats-table {{
            background: rgba(249, 250, 251, 0.5);
            border-radius: 8px;
            padding: 1rem;
        }}
        
        .stats-table h3 {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--color-charcoal);
            margin-bottom: 0.75rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }}
        
        .stat-item {{
            background: white;
            border: 1px solid var(--color-border-light);
            border-radius: 4px;
            padding: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .stat-label {{
            font-size: 0.75rem;
            color: var(--color-medium-grey);
        }}
        
        .stat-value {{
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--color-charcoal);
        }}
        
        /* Histogram section */
        .histogram-section {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }}
        
        .histogram-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--color-charcoal);
            text-align: center;
            margin-bottom: 1rem;
        }}
        
        .histogram-container {{
            height: 300px;
            position: relative;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 6px;
            padding: 0.5rem;
        }}
        
        /* Slider representation */
        .slider-container {{
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(249, 250, 251, 0.5);
            border-radius: 6px;
        }}
        
        .slider-track {{
            height: 6px;
            background: var(--color-light-grey);
            border-radius: 3px;
            position: relative;
            margin: 1rem 0;
        }}
        
        .slider-range {{
            position: absolute;
            height: 100%;
            background: var(--color-braun-orange);
            border-radius: 3px;
        }}
        
        .slider-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--color-medium-grey);
        }}
        
        /* Percentiles grid matching frontend */
        .percentiles-section {{
            background: rgba(249, 250, 251, 0.5);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1.5rem;
        }}
        
        .percentiles-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--color-charcoal);
            margin-bottom: 0.75rem;
            text-align: center;
        }}
        
        .percentiles-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.5rem;
        }}
        
        .percentile-item {{
            background: white;
            border: 1px solid var(--color-border-light);
            border-radius: 4px;
            padding: 0.5rem;
            text-align: center;
            box-shadow: 0.5px 0.5px 1px rgba(0, 0, 0, 0.04);
        }}
        
        .percentile-label {{
            display: block;
            font-size: 0.6rem;
            font-weight: 600;
            color: var(--color-braun-orange);
            margin-bottom: 0.25rem;
        }}
        
        .percentile-value {{
            display: block;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--color-charcoal);
        }}
        
        /* Tornado chart section */
        .tornado-section {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1.5rem;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }}
        
        .tornado-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--color-charcoal);
            margin-bottom: 1rem;
            text-align: center;
        }}
        
        .tornado-container {{
            position: relative;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 6px;
            padding: 0.5rem;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .target-section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Monte Carlo Simulation Results</h1>
        <div class="metadata">
            <div class="metadata-item">
                <strong>Generated:</strong> {datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            <div class="metadata-item">
                <strong>Iterations:</strong> {iterations_run}
            </div>
            <div class="metadata-item">
                <strong>Engine:</strong> {engine_type}
            </div>
            <div class="metadata-item">
                <strong>Simulation ID:</strong> {simulation_id}
            </div>
        </div>
    </div>
    
    {await self._generate_enhanced_results_html(targets)}
    
    {self._generate_visualization_scripts(targets)}
</body>
</html>
"""
        return html_content
        
    async def _generate_enhanced_results_html(self, targets: Dict[str, Any]) -> str:
        """Generate enhanced HTML for results with all visualizations"""
        html_parts = []
        
        # Add box plot overview if multiple targets
        if len(targets) > 1:
            html_parts.append(self._generate_box_plot_overview(targets))
        
        # Generate detailed section for each target
        for idx, (target_name, target_data) in enumerate(targets.items()):
            html_parts.append(self._generate_target_section(target_name, target_data, idx))
        
        return '\n'.join(html_parts)
        
    def _generate_box_plot_overview(self, targets: Dict[str, Any]) -> str:
        """Generate box plot overview section"""
        return f"""
    <div class="box-plot-overview-section">
        <div class="box-plot-header">
            <h2>Distribution Overview - All Target Variables</h2>
            <p style="color: var(--color-medium-grey); font-size: 0.875rem; margin-top: 0.5rem;">
                Box plots show median (line), quartiles (box), and min/max values (whiskers)
            </p>
        </div>
        <div class="box-plot-container">
            <canvas id="boxplot-overview"></canvas>
        </div>
    </div>
"""
        
    def _generate_target_section(self, target_name: str, target_data: Dict[str, Any], idx: int) -> str:
        """Generate complete section for a target variable"""
        statistics = target_data.get('statistics', {})
        values = target_data.get('values', [])
        
        # Format numbers
        def format_number(val):
            if val is None:
                return 'N/A'
            if abs(val) >= 1000000:
                return f"{val/1000000:.2f}M"
            elif abs(val) >= 1000:
                return f"{val/1000:.2f}K"
            else:
                return f"{val:.2f}"
        
        # Calculate additional statistics
        if values:
            sorted_values = sorted(values)
            range_val = max(values) - min(values)
        else:
            range_val = 0
            
        return f"""
    <div class="target-section">
        <div class="target-header">
            <h2 class="target-title">{target_name}</h2>
        </div>
        
        <div class="content-grid">
            <!-- Statistics Table -->
            <div class="stats-table">
                <h3>Statistical Summary</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-label">Mean</span>
                        <span class="stat-value">{format_number(statistics.get('mean'))}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Median</span>
                        <span class="stat-value">{format_number(statistics.get('median'))}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Std Dev</span>
                        <span class="stat-value">{format_number(statistics.get('std_dev'))}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Range</span>
                        <span class="stat-value">{format_number(range_val)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Min</span>
                        <span class="stat-value">{format_number(statistics.get('min_value'))}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Max</span>
                        <span class="stat-value">{format_number(statistics.get('max_value'))}</span>
                    </div>
                </div>
            </div>
            
            <!-- Histogram with Certainty Analysis -->
            <div class="histogram-section">
                <h3 class="histogram-title">Distribution</h3>
                <div class="histogram-container">
                    <canvas id="histogram-{idx}"></canvas>
                </div>
                <!-- Slider representation -->
                <div class="slider-container">
                    <div class="slider-labels">
                        <span>{format_number(statistics.get('min_value', 0))}</span>
                        <span style="color: var(--color-braun-orange); font-weight: 600;">
                            Certainty Range: {format_number(statistics.get('percentiles', {}).get('25', 0))} - {format_number(statistics.get('percentiles', {}).get('75', 0))}
                        </span>
                        <span>{format_number(statistics.get('max_value', 0))}</span>
                    </div>
                    <div class="slider-track">
                        <div class="slider-range" style="left: 25%; width: 50%;"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Percentiles Grid -->
        <div class="percentiles-section">
            <h3 class="percentiles-title">Percentiles</h3>
            <div class="percentiles-grid">
                {self._generate_percentile_items(statistics.get('percentiles', {}))}
            </div>
        </div>
        
        <!-- Tornado Chart for Sensitivity Analysis -->
        {self._generate_tornado_section(target_data, idx)}
    </div>
"""
        
    def _generate_percentile_items(self, percentiles: Dict[str, float]) -> str:
        """Generate percentile grid items"""
        items = []
        percentile_labels = {
            '5': 'P5', '10': 'P10', '25': 'Q1', '50': 'Median', '75': 'Q3', '90': 'P90', '95': 'P95'
        }
        
        for p in ['5', '10', '25', '50', '75', '90', '95']:
            if p in percentiles:
                label = percentile_labels.get(p, f'P{p}')
                value = percentiles[p]
                formatted_value = f"{value/1000000:.2f}M" if abs(value) >= 1000000 else f"{value/1000:.2f}K" if abs(value) >= 1000 else f"{value:.2f}"
                items.append(f"""
                <div class="percentile-item">
                    <span class="percentile-label">{label}</span>
                    <span class="percentile-value">{formatted_value}</span>
                </div>""")
        
        return '\n'.join(items)
        
    def _generate_tornado_section(self, target_data: Dict[str, Any], idx: int) -> str:
        """Generate tornado chart section if sensitivity analysis is available"""
        sensitivity = target_data.get('statistics', {}).get('sensitivity_analysis')
        if not sensitivity:
            return ""
            
        return f"""
        <div class="tornado-section">
            <h3 class="tornado-title">Variable Impact Analysis</h3>
            <div class="tornado-container" style="height: 300px;">
                <canvas id="tornado-{idx}"></canvas>
            </div>
        </div>
"""
        
    def _generate_visualization_scripts(self, targets: Dict[str, Any]) -> str:
        """Generate all Chart.js scripts for visualizations"""
        scripts = []
        
        # Box plot overview script
        if len(targets) > 1:
            scripts.append(self._generate_box_plot_script(targets))
        
        # Individual target scripts
        for idx, (target_name, target_data) in enumerate(targets.items()):
            scripts.append(self._generate_histogram_script(target_name, target_data, idx))
            
            # Add tornado chart if sensitivity analysis exists
            sensitivity = target_data.get('statistics', {}).get('sensitivity_analysis')
            if sensitivity:
                scripts.append(self._generate_tornado_script(target_name, sensitivity, idx))
        
        return f"""
    <script>
        // Wait for DOM and Chart.js to load
        function initializeCharts() {{
            if (typeof Chart === 'undefined' || typeof window.ChartjsBoxPlot === 'undefined') {{
                console.log('Waiting for Chart.js to load...');
                setTimeout(initializeCharts, 100);
                return;
            }}
            
            console.log('Initializing charts...');
            
            // Register Chart.js boxplot plugin
            Chart.register(window.ChartjsBoxPlot.BoxPlotController, window.ChartjsBoxPlot.BoxAndWiskers);
            
            {' '.join(scripts)}
        }}
        
        // Utility function for creating histogram bins
        function createHistogramBins(data, numBins) {{
            if (data.length === 0) return {{ labels: [], counts: [] }};
            
            const min = Math.min(...data);
            const max = Math.max(...data);
            const binWidth = (max - min) / numBins;
            
            const bins = Array(numBins).fill(0);
            const labels = [];
            
            for (let i = 0; i < numBins; i++) {{
                const binStart = min + i * binWidth;
                const binEnd = binStart + binWidth;
                labels.push(`${{binStart.toFixed(0)}} - ${{binEnd.toFixed(0)}}`);
            }}
            
            data.forEach(value => {{
                const binIndex = Math.min(Math.floor((value - min) / binWidth), numBins - 1);
                bins[binIndex]++;
            }});
            
            return {{ labels, counts: bins }};
        }}
        
        // Initialize charts when page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initializeCharts);
        }} else {{
            initializeCharts();
        }}
    </script>
"""
        
    def _generate_box_plot_script(self, targets: Dict[str, Any]) -> str:
        """Generate script for box plot overview"""
        box_data = []
        labels = []
        
        for target_name, target_data in targets.items():
            labels.append(target_name)
            values = target_data.get('values', [])
            if values:
                sorted_values = sorted(values)
                n = len(sorted_values)
                box_data.append({
                    'min': sorted_values[0],
                    'q1': sorted_values[int(n * 0.25)],
                    'median': sorted_values[int(n * 0.5)],
                    'q3': sorted_values[int(n * 0.75)],
                    'max': sorted_values[-1],
                    'outliers': []  # Simplified - not calculating outliers for PDF
                })
            else:
                box_data.append({'min': 0, 'q1': 0, 'median': 0, 'q3': 0, 'max': 0})
        
        return f"""
        // Box plot overview
        const boxCtx = document.getElementById('boxplot-overview').getContext('2d');
        new Chart(boxCtx, {{
            type: 'boxplot',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Distribution',
                    data: {json.dumps(box_data)},
                    backgroundColor: 'rgba(255, 107, 53, 0.1)',
                    borderColor: '#FF6B35',
                    borderWidth: 2,
                    outlierBackgroundColor: '#FF6B35',
                    outlierBorderColor: '#FF6B35',
                    medianColor: '#333333',
                    lowerBackgroundColor: 'rgba(255, 107, 53, 0.1)',
                    upperBackgroundColor: 'rgba(255, 107, 53, 0.1)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            boxplotLabel: function(item) {{
                                return [
                                    `Max: ${{item.max.toLocaleString()}}`,
                                    `Q3: ${{item.q3.toLocaleString()}}`,
                                    `Median: ${{item.median.toLocaleString()}}`,
                                    `Q1: ${{item.q1.toLocaleString()}}`,
                                    `Min: ${{item.min.toLocaleString()}}`
                                ];
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        ticks: {{
                            callback: function(value) {{
                                if (Math.abs(value) >= 1000000) {{
                                    return (value / 1000000).toFixed(1) + 'M';
                                }} else if (Math.abs(value) >= 1000) {{
                                    return (value / 1000).toFixed(1) + 'K';
                                }}
                                return value.toFixed(0);
                            }}
                        }}
                    }}
                }}
            }}
        }});
"""
        
    def _generate_histogram_script(self, target_name: str, target_data: Dict[str, Any], idx: int) -> str:
        """Generate script for histogram with certainty coloring"""
        values = target_data.get('values', [])
        statistics = target_data.get('statistics', {})
        
        # Use percentiles for certainty range
        p25 = statistics.get('percentiles', {}).get('25', 0)
        p75 = statistics.get('percentiles', {}).get('75', 0)
        
        return f"""
        // Histogram for {target_name}
        const histCtx{idx} = document.getElementById('histogram-{idx}').getContext('2d');
        const values{idx} = {json.dumps(values[:1000])};  // Limit to 1000 for performance
        const bins{idx} = createHistogramBins(values{idx}, 50);
        
        // Calculate colors based on certainty range
        const p25_{idx} = {p25};
        const p75_{idx} = {p75};
        const min_{idx} = Math.min(...values{idx});
        const max_{idx} = Math.max(...values{idx});
        const binWidth_{idx} = (max_{idx} - min_{idx}) / 50;
        
        const backgroundColors{idx} = bins{idx}.labels.map((label, index) => {{
            const binCenter = min_{idx} + (index + 0.5) * binWidth_{idx};
            return (binCenter >= p25_{idx} && binCenter <= p75_{idx}) 
                ? 'rgba(255, 107, 53, 0.7)'  // Braun orange for certainty
                : 'rgba(119, 119, 119, 0.4)'; // Grey for outside
        }});
        
        new Chart(histCtx{idx}, {{
            type: 'bar',
            data: {{
                labels: bins{idx}.labels,
                datasets: [{{
                    label: 'Frequency',
                    data: bins{idx}.counts,
                    backgroundColor: backgroundColors{idx},
                    borderColor: backgroundColors{idx}.map(c => c.replace('0.7', '1').replace('0.4', '0.7')),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        display: false  // Hide x-axis labels for cleaner look
                    }},
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Frequency'
                        }}
                    }}
                }}
            }}
        }});
"""
        
    def _generate_tornado_script(self, target_name: str, sensitivity_data: Any, idx: int) -> str:
        """Generate script for tornado chart"""
        # Process sensitivity data
        if isinstance(sensitivity_data, list):
            sorted_data = sorted(sensitivity_data, key=lambda x: abs(x.get('impact', 0)), reverse=True)[:10]
        else:
            # Handle object format
            sorted_data = []
            
        labels = [item.get('variable', f'Var{i}') for i, item in enumerate(sorted_data)]
        impacts = [item.get('impact', 0) for item in sorted_data]
        
        # Assign colors based on impact direction
        colors = ['#FF6B35' if impact > 0 else '#333333' for impact in impacts]
        
        return f"""
        // Tornado chart for {target_name}
        const tornadoCtx{idx} = document.getElementById('tornado-{idx}').getContext('2d');
        new Chart(tornadoCtx{idx}, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Impact %',
                    data: {json.dumps(impacts)},
                    backgroundColor: {json.dumps(colors)},
                    borderColor: {json.dumps(colors)},
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.label + ': ' + context.parsed.x.toFixed(1) + '%';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Impact (%)'
                        }},
                        grid: {{
                            drawBorder: false
                        }}
                    }},
                    y: {{
                        grid: {{
                            display: false
                        }}
                    }}
                }}
            }}
        }});
"""
