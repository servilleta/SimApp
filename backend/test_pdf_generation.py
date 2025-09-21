#!/usr/bin/env python3
"""
Direct test script for PDF generation in B2B API.
This bypasses simulation and directly tests PDF generation.
"""

import sys
import os
sys.path.append('/home/paperspace/PROJECT/backend')

# Mock simulation result data
mock_simulation_result = {
    "simulation_id": "test_pdf_sim_123",
    "status": "completed",
    "execution_time": "45.32s",
    "iterations_completed": 10000,
    "results": {
        "B10": {
            "cell_name": "Portfolio Value",
            "statistics": {
                "mean": 1250.67,
                "std": 234.89,
                "min": 890.23,
                "max": 1650.12,
                "var_95": 890.23,
                "var_99": 645.12
            },
            "values": [1200 + i*0.1 for i in range(100)]  # Mock values for chart
        }
    },
    "download_links": {
        "pdf": "/simapp-api/simulations/test_pdf_sim_123/download/pdf",
        "xlsx": "/simapp-api/simulations/test_pdf_sim_123/download/xlsx",
        "json": "/simapp-api/simulations/test_pdf_sim_123/download/json"
    },
    "created_at": "2025-09-17T08:42:26.622856Z"
}

def test_reportlab_pdf():
    """Test reportlab-based PDF generation (current B2B API approach)"""
    print("🔧 Testing reportlab-based PDF generation...")
    
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        # Build PDF content
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f2937')
        )
        story.append(Paragraph("Monte Carlo Simulation Results", title_style))
        story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        pdf_content = buffer.getvalue()
        print(f"✅ Reportlab PDF generation successful! ({len(pdf_content)} bytes)")
        
        # Save test file
        with open('test_reportlab.pdf', 'wb') as f:
            f.write(pdf_content)
        print("📄 Saved as test_reportlab.pdf")
        
        return True
        
    except ImportError as e:
        print(f"❌ Reportlab dependencies missing: {e}")
        return False
    except Exception as e:
        print(f"❌ Reportlab PDF generation failed: {e}")
        return False

def test_playwright_pdf():
    """Test Playwright-based PDF generation (frontend approach)"""
    print("🔧 Testing Playwright-based PDF generation...")
    
    try:
        from modules.pdf_export import PDFExportService
        import asyncio
        
        async def run_test():
            pdf_service = PDFExportService()
            
            pdf_path = await pdf_service.generate_pdf_from_frontend(
                simulation_id="test_pdf_sim_123",
                results_data=mock_simulation_result,
                auth_token=None,
                frontend_url="http://frontend:3000"
            )
            
            # Check if PDF was created
            from pathlib import Path
            pdf_file = Path(pdf_path)
            if pdf_file.exists():
                file_size = pdf_file.stat().st_size
                print(f"✅ Playwright PDF generation successful! ({file_size} bytes)")
                print(f"📄 PDF saved at: {pdf_path}")
                
                # Copy to local directory for inspection
                import shutil
                shutil.copy(pdf_path, 'test_playwright.pdf')
                print("📄 Copied as test_playwright.pdf")
                
                return True
            else:
                print("❌ Playwright PDF file not found")
                return False
        
        return asyncio.run(run_test())
        
    except ImportError as e:
        print(f"❌ Playwright dependencies missing: {e}")
        return False
    except Exception as e:
        print(f"❌ Playwright PDF generation failed: {e}")
        return False

def main():
    """Run both PDF generation tests"""
    print("🚀 PDF Generation Test Suite")
    print("=" * 50)
    
    # Test reportlab (current B2B API)
    reportlab_success = test_reportlab_pdf()
    print()
    
    # Test Playwright (frontend approach)  
    playwright_success = test_playwright_pdf()
    print()
    
    # Summary
    print("📊 Test Results:")
    print(f"   Reportlab PDF: {'✅ PASS' if reportlab_success else '❌ FAIL'}")
    print(f"   Playwright PDF: {'✅ PASS' if playwright_success else '❌ FAIL'}")
    
    if reportlab_success and not playwright_success:
        print("\n💡 Recommendation: Use reportlab (dependencies are working)")
    elif playwright_success and not reportlab_success:
        print("\n💡 Recommendation: Use Playwright (frontend approach working)")
    elif reportlab_success and playwright_success:
        print("\n💡 Both methods work! Frontend consistency suggests using Playwright")
    else:
        print("\n❌ Both methods failed - need to debug dependencies")

if __name__ == "__main__":
    main()
