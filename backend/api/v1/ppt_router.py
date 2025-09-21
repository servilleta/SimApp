import os
import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio

from auth.auth0_dependencies import get_current_auth0_user
from modules.ppt_export import PowerPointExportService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize PowerPoint export service
try:
    ppt_service = PowerPointExportService()
    logger.info("✅ PowerPoint export service initialized successfully")
except ImportError as e:
    logger.warning(f"⚠️ PowerPoint export service unavailable - missing dependencies: {e}")
    ppt_service = None
except Exception as e:
    logger.error(f"❌ Failed to initialize PowerPoint export service: {e}")
    ppt_service = None

@router.post("/export")
async def export_powerpoint(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_auth0_user)
):
    """
    Export simulation results as an editable PowerPoint presentation.
    
    This endpoint generates a 16:9 PowerPoint presentation with:
    - Editable charts (histograms, tornado charts)
    - Editable text and statistics
    - Professional layout and formatting
    - Multiple slides with different views of the data
    """
    
    if not ppt_service:
        raise HTTPException(
            status_code=503, 
            detail="PowerPoint export service is not available. Missing python-pptx dependency."
        )
    
    try:
        # Extract request data
        simulation_id = request_data.get('simulationId', 'unknown')
        results_data = request_data.get('results', {})
        metadata = request_data.get('metadata', {})
        
        logger.info(f"Starting PowerPoint export for simulation {simulation_id} by user {current_user.id}")
        
        if not results_data:
            raise HTTPException(status_code=400, detail="No simulation results provided")
        
        # Generate PowerPoint presentation with auth token for pixel-perfect screenshots
        ppt_path = await ppt_service.generate_powerpoint_presentation(
            simulation_id=simulation_id,
            results_data=results_data,
            metadata=metadata,
            auth_token=getattr(current_user, 'token', None),  # Pass auth token for frontend access
            frontend_url="http://frontend:3000"
        )
        
        if not os.path.exists(ppt_path):
            raise HTTPException(status_code=500, detail="PowerPoint generation failed - file not created")
        
        # Get file size for logging
        file_size = os.path.getsize(ppt_path)
        logger.info(f"PowerPoint export completed successfully: {ppt_path} ({file_size} bytes)")
        
        # Schedule file cleanup after response
        background_tasks.add_task(cleanup_temp_file, ppt_path)
        
        # Return file
        filename = f"simulation_presentation_{simulation_id}.pptx"
        return FileResponse(
            path=ppt_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Export-Type": "powerpoint",
                "X-Simulation-ID": simulation_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PowerPoint export failed for simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"PowerPoint export failed: {str(e)}")

@router.get("/download/{filename}")
async def download_powerpoint(
    filename: str,
    current_user = Depends(get_current_auth0_user)
):
    """
    Download a previously generated PowerPoint presentation.
    """
    
    try:
        # Validate filename for security
        if not filename.endswith('.pptx') or '..' in filename or '/' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Look for file in temp directory
        temp_dir = Path("/tmp/monte_carlo_presentations")
        file_path = temp_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="PowerPoint file not found or expired")
        
        logger.info(f"Serving PowerPoint download: {filename} to user {current_user.id}")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PowerPoint download failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail="Download failed")

@router.get("/status")
async def get_powerpoint_export_status():
    """
    Get the status of PowerPoint export service.
    """
    
    is_available = ppt_service is not None
    
    status = {
        "service_available": is_available,
        "service_type": "python-pptx" if is_available else "unavailable",
        "features": [
            "16:9 aspect ratio slides",
            "Editable charts and graphs", 
            "Professional layouts",
            "Statistical summary tables",
            "Variable impact analysis",
            "Methodology documentation"
        ] if is_available else [],
        "supported_formats": ["pptx"] if is_available else []
    }
    
    return status

async def cleanup_temp_file(file_path: str):
    """
    Clean up temporary PowerPoint files after a delay.
    """
    try:
        # Wait 5 minutes before cleanup to allow download
        await asyncio.sleep(300)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary PowerPoint file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup PowerPoint file {file_path}: {e}")
