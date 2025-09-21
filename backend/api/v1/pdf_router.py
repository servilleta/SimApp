"""
PDF Export Router - Modern PDF generation with 100% visual fidelity with background generation
"""
import os
import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from modules.pdf_export import pdf_export_service
from auth.auth0_dependencies import get_current_auth0_user
from models import User
from database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdf", tags=["PDF Export"])

class PDFExportRequest(BaseModel):
    simulation_id: str
    results_data: Dict[str, Any]
    export_type: str = "results"  # "results", "url"
    url: Optional[str] = None
    wait_for_selector: Optional[str] = None

class PDFExportResponse(BaseModel):
    success: bool
    message: str
    download_url: Optional[str] = None
    file_size: Optional[int] = None

class BackgroundPDFRequest(BaseModel):
    simulation_id: str
    results_data: Dict[str, Any]
    user_id: int

# In-memory storage for PDF status (in production, use Redis or database)
pdf_generation_status = {}
pdf_file_storage = {}

# Create persistent PDF storage directory
PERSISTENT_PDF_DIR = Path("/tmp/persistent_pdfs")
PERSISTENT_PDF_DIR.mkdir(exist_ok=True)

async def generate_pdf_background(simulation_id: str, results_data: Dict[str, Any], user_id: int, auth_token: str):
    """Background task to generate PDF without blocking the UI."""
    try:
        logger.info(f"🔄 Starting background PDF generation for simulation {simulation_id}")
        pdf_generation_status[simulation_id] = {"status": "generating", "progress": 0}
        
        # Use the EXACT same working method as SimApp API
        pdf_path = await pdf_export_service.generate_pdf_from_results_page(
            simulation_id=simulation_id,
            results_data=results_data,
            auth_token=auth_token
        )
        
        if pdf_path and os.path.exists(pdf_path):
            # Move PDF to persistent storage
            persistent_filename = f"simulation_{simulation_id}_{int(datetime.now().timestamp())}.pdf"
            persistent_path = PERSISTENT_PDF_DIR / persistent_filename
            
            # Copy file to persistent location
            import shutil
            shutil.copy2(pdf_path, persistent_path)
            
            # Store metadata
            pdf_file_storage[simulation_id] = {
                "file_path": str(persistent_path),
                "filename": persistent_filename,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(persistent_path)
            }
            
            pdf_generation_status[simulation_id] = {
                "status": "completed", 
                "progress": 100,
                "file_ready": True,
                "file_path": str(persistent_path)
            }
            
            logger.info(f"✅ Background PDF generation completed for simulation {simulation_id}")
            
            # Clean up temporary file
            try:
                os.remove(pdf_path)
            except:
                pass
                
        else:
            raise Exception("PDF generation failed - no file created")
            
    except Exception as e:
        logger.error(f"❌ Background PDF generation failed for simulation {simulation_id}: {e}")
        pdf_generation_status[simulation_id] = {
            "status": "failed", 
            "progress": 0,
            "error": str(e)
        }

@router.post("/generate-background")
async def generate_pdf_background_endpoint(
    request: BackgroundPDFRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    current_user: User = Depends(get_current_auth0_user)
):
    """
    Start background PDF generation for a simulation.
    This doesn't block the UI and generates PDF for later instant download.
    """
    try:
        logger.info(f"Starting background PDF generation for simulation {request.simulation_id}")
        
        # Extract auth token from request headers
        auth_header = http_request.headers.get("authorization", "")
        auth_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None
        
        if not auth_token:
            raise HTTPException(status_code=401, detail="Authentication token required for PDF generation")
        
        # Start background task
        background_tasks.add_task(
            generate_pdf_background,
            request.simulation_id,
            request.results_data,
            current_user.id,
            auth_token
        )
        
        # Initialize status
        pdf_generation_status[request.simulation_id] = {"status": "queued", "progress": 0}
        
        return {
            "success": True,
            "message": "PDF generation started in background",
            "simulation_id": request.simulation_id
        }
        
    except Exception as e:
        logger.error(f"Error starting background PDF generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start PDF generation: {str(e)}")

@router.get("/status/{simulation_id}")
async def get_pdf_status(
    simulation_id: str,
    current_user: User = Depends(get_current_auth0_user)
):
    """Check the status of PDF generation for a simulation."""
    try:
        status = pdf_generation_status.get(simulation_id, {"status": "not_found", "progress": 0})
        file_info = pdf_file_storage.get(simulation_id, {})
        
        return {
            "simulation_id": simulation_id,
            "status": status.get("status", "not_found"),
            "progress": status.get("progress", 0),
            "file_ready": status.get("file_ready", False),
            "error": status.get("error"),
            "file_size": file_info.get("file_size"),
            "created_at": file_info.get("created_at")
        }
        
    except Exception as e:
        logger.error(f"Error checking PDF status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check PDF status: {str(e)}")

@router.get("/download/{simulation_id}")
async def download_pdf_instant(
    simulation_id: str,
    current_user: User = Depends(get_current_auth0_user)
):
    """
    Instantly download a pre-generated PDF.
    This provides instant download since PDF was generated in background.
    """
    try:
        logger.info(f"Instant PDF download requested for simulation {simulation_id} by user {current_user.id}")
        
        # Check if PDF exists
        file_info = pdf_file_storage.get(simulation_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="PDF not found. Generate PDF first.")
        
        # Verify user access (users can only download their own PDFs)
        if file_info.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied to this PDF")
        
        pdf_path = file_info["file_path"]
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF file not found on disk")
        
        logger.info(f"✅ Serving instant PDF download: {pdf_path}")
        
        return FileResponse(
            path=pdf_path,
            filename=file_info["filename"],
            media_type='application/pdf',
            headers={
                "Content-Disposition": f"attachment; filename=\"{file_info['filename']}\""
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving instant PDF download: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")

@router.post("/export", response_model=PDFExportResponse)
async def export_simulation_pdf(
    request: PDFExportRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    current_user: User = Depends(get_current_auth0_user)
):
    """
    Generate a PDF export of simulation results with perfect visual fidelity.
    
    This endpoint uses Playwright to render the results page in a headless browser,
    ensuring the PDF looks exactly like the webpage.
    """
    try:
        logger.info(f"Starting PDF export for simulation {request.simulation_id} by user {current_user.id}")
        
        # Extract auth token from request headers
        auth_header = http_request.headers.get("authorization", "")
        auth_token = None
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Generate PDF based on export type
        if request.export_type == "url" and request.url:
            pdf_path = await pdf_export_service.generate_pdf_from_url(
                url=request.url,
                simulation_id=request.simulation_id,
                wait_for_selector=request.wait_for_selector
            )
        else:
            # Generate from results data (default)
            pdf_path = await pdf_export_service.generate_pdf_from_results_page(
                simulation_id=request.simulation_id,
                results_data=request.results_data,
                auth_token=auth_token
            )
        
        # Get file info
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise HTTPException(status_code=500, detail="PDF file was not created successfully")
        
        file_size = pdf_file.stat().st_size
        
        # Schedule cleanup in background
        background_tasks.add_task(
            pdf_export_service.cleanup_old_pdfs,
            max_age_hours=24
        )
        
        # Create download URL (relative to the temp directory)
        download_url = f"/pdf/download/{pdf_file.name}"
        
        logger.info(f"PDF export completed successfully: {pdf_path} ({file_size} bytes)")
        
        return PDFExportResponse(
            success=True,
            message="PDF generated successfully",
            download_url=download_url,
            file_size=file_size
        )
        
    except Exception as e:
        logger.error(f"PDF export failed for simulation {request.simulation_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"PDF export failed: {str(e)}"
        )

@router.get("/download/{filename}")
async def download_pdf(
    filename: str,
    current_user: User = Depends(get_current_auth0_user)
):
    """
    Download a generated PDF file.
    """
    try:
        # Validate filename to prevent directory traversal
        if not filename.endswith('.pdf') or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        pdf_path = pdf_export_service.temp_dir / filename
        
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        logger.info(f"Serving PDF download: {filename} to user {current_user.id}")
        
        return FileResponse(
            path=str(pdf_path),
            filename=filename,
            media_type='application/pdf',
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF file not found")
    except Exception as e:
        logger.error(f"Error serving PDF download {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve PDF file")

@router.post("/export-url", response_model=PDFExportResponse)
async def export_url_to_pdf(
    url: str,
    simulation_id: str,
    wait_for_selector: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_auth0_user)
):
    """
    Generate PDF directly from a URL.
    Useful for capturing the live webpage with all interactions.
    """
    try:
        logger.info(f"Starting URL PDF export for {url} by user {current_user.id}")
        
        pdf_path = await pdf_export_service.generate_pdf_from_url(
            url=url,
            simulation_id=simulation_id,
            wait_for_selector=wait_for_selector
        )
        
        pdf_file = Path(pdf_path)
        file_size = pdf_file.stat().st_size
        
        # Schedule cleanup
        background_tasks.add_task(
            pdf_export_service.cleanup_old_pdfs,
            max_age_hours=24
        )
        
        download_url = f"/pdf/download/{pdf_file.name}"
        
        return PDFExportResponse(
            success=True,
            message="PDF generated from URL successfully",
            download_url=download_url,
            file_size=file_size
        )
        
    except Exception as e:
        logger.error(f"URL PDF export failed for {url}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"URL PDF export failed: {str(e)}"
        )

@router.get("/status")
async def pdf_service_status():
    """
    Check the status of the PDF service.
    """
    try:
        # Check if playwright is available
        from playwright.async_api import async_playwright
        
        return {
            "status": "healthy",
            "service": "PDF Export Service",
            "playwright_available": True,
            "temp_directory": str(pdf_export_service.temp_dir),
            "temp_dir_exists": pdf_export_service.temp_dir.exists()
        }
    except ImportError:
        return {
            "status": "error",
            "service": "PDF Export Service",
            "playwright_available": False,
            "error": "Playwright not installed"
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "PDF Export Service",
            "error": str(e)
        }
