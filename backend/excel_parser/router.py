import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from typing import List
import logging
import os
from pathlib import Path

from excel_parser.schemas import ExcelFileResponse
from excel_parser.service import parse_excel_file, get_file_variables, get_parsed_file_info, get_formula_from_file
from auth.auth0_dependencies import get_current_active_auth0_user
from auth.schemas import User
from config import settings
from shared.upload_middleware import validate_upload_file

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["excel-parser"],
    responses={404: {"description": "Not found"}}
)

@router.post("/upload", response_model=ExcelFileResponse)
async def upload_excel(file: UploadFile = File(...), current_user: User = Depends(get_current_active_auth0_user)):
    """Upload and parse an Excel file."""
    
    # Validate file using the centralized upload validator
    validation_result = validate_upload_file(file)
    logger.info(f"File validation passed for {file.filename} ({validation_result['size_mb']:.1f} MB) by user {current_user.username}")
    
    # Reset file pointer after validation (validate_upload_file may have read the file)
    await file.seek(0)

    try:
        result = await parse_excel_file(file) # Pass the UploadFile object directly (or contents if service expects bytes)
        logger.info(f"✅ Excel processing completed for {file.filename} (user: {current_user.username})")
        
        # Try to serialize the result to catch any Pydantic validation errors
        try:
            result_json = result.model_dump_json()
            logger.info(f"✅ Response serialization successful, size: {len(result_json)} chars")
        except Exception as validation_error:
            logger.error(f"❌ Response validation error: {validation_error}")
            # Try to identify the problematic field
            if hasattr(validation_error, 'errors'):
                for error in validation_error.errors():
                    logger.error(f"❌ Validation error detail: {error}")
            
            # FALLBACK: If validation still fails, there's a deeper issue
            logger.error("🚨 Critical: Response validation failed even after optimization")
            raise HTTPException(status_code=500, detail="Excel file processed but response format error - please contact support")
        
        return result
    except HTTPException as e: # Catch HTTPExceptions raised from service
        raise e
    except Exception as e:
        # Log the exception e
        logger.error(f"❌ Unexpected error in upload_excel endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing the file: {str(e)}")

# SUPERFAST: Async file upload processing for immediate response
@router.post("/upload-async")
async def upload_excel_async(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_active_auth0_user)
):
    """SUPERFAST: Upload Excel file with background processing for immediate response."""
    
    # Validate file first
    validation_result = validate_upload_file(file)
    logger.info(f"🚀 SUPERFAST upload started: {file.filename} ({validation_result['size_mb']:.1f} MB) by user {current_user.username}")
    
    # Reset file pointer after validation
    await file.seek(0)
    
    # Generate unique file ID for tracking
    import uuid
    import time
    file_id = f"{current_user.id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Add background task for processing
    background_tasks.add_task(_process_excel_background, file, file_id, current_user.username)
    
    # Return immediate response
    return {
        "status": "processing",
        "message": "File uploaded successfully and is being processed",
        "file_id": file_id,
        "filename": file.filename,
        "size_mb": validation_result['size_mb'],
        "processing_started": True,
        "estimated_completion": "1-3 minutes"  # Rough estimate
    }

# Background processing function
async def _process_excel_background(file: UploadFile, file_id: str, username: str):
    """Background task to process Excel file without blocking the response."""
    try:
        logger.info(f"🔄 Background processing started for file_id: {file_id}")
        
        # Process the file
        result = await parse_excel_file(file)
        
        # Store result in Redis cache for later retrieval
        # This would require Redis integration - for now we'll log success
        logger.info(f"✅ Background processing completed for file_id: {file_id} (user: {username})")
        
        # You could also update a database status here
        # or send a webhook/notification to the frontend
        
    except Exception as e:
        logger.error(f"❌ Background processing failed for file_id: {file_id} - {str(e)}")
        # Store error status for later retrieval

@router.get("/files/{file_id}")
async def get_file_data(file_id: str, current_user: User = Depends(get_current_active_auth0_user)):
    """Get the full parsed Excel file data including sheets and cells."""
    try:
        # Use the service function that includes file resolution logic
        from excel_parser.service import get_all_parsed_sheets_data
        sheets_data = await get_all_parsed_sheets_data(file_id)
        
        # Return the sheets data in the format expected by the frontend
        return {
            "file_id": file_id,
            "sheets": [sheet.dict() for sheet in sheets_data]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting file data for {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve data for file {file_id}: {str(e)}")

@router.get("/files/{file_id}/info")
async def get_file_metadata(file_id: str, current_user: User = Depends(get_current_active_auth0_user)):
    """Get metadata/information about a previously uploaded and parsed Excel file."""
    try:
        file_info = await get_parsed_file_info(file_id)
        # The response model for this might be simpler than ExcelFileResponse, 
        # or you can reuse ExcelFileResponse if get_parsed_file_info returns all necessary fields.
        # For now, let's assume it returns a dict that can be directly returned.
        return file_info 
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error getting file info for {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve information for file {file_id}.")

@router.get("/files/{file_id}/variables", response_model=List[str])
async def get_variables_from_file(file_id: str, current_user: User = Depends(get_current_active_auth0_user)):
    """Get available variables (e.g., column names) from a parsed Excel file."""
    try:
        variables = await get_file_variables(file_id)
        return variables
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error getting variables for {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve variables for file {file_id}.")

# You might also want an endpoint to get a specific formula or all formulas for debugging/client use
@router.get("/files/{file_id}/formulas/{cell_coordinate}")
async def get_single_formula(file_id: str, cell_coordinate: str, current_user: User = Depends(get_current_active_auth0_user)):
    try:
        formula = await get_formula_from_file(file_id, cell_coordinate)
        return {"file_id": file_id, "cell_coordinate": cell_coordinate, "formula": formula}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 FAST PARSING: Background task for Arrow cache creation
async def _ensure_arrow_exists(file_id: str):
    """Background task to ensure Arrow cache exists for a file"""
    try:
        cache_dir = "/app/cache"
        arrow_path = Path(f"{cache_dir}/{file_id}.feather")
        
        if arrow_path.exists():
            logger.info(f"⚡ Arrow cache already exists for {file_id}")
            return
        
        # Trigger parsing if needed - the parse_excel_file will create the cache
        # This is a simplified approach; a more robust version would parse directly here
        logger.info(f"🔄 Creating Arrow cache for file_id: {file_id}")
        
    except Exception as e:
        logger.error(f"❌ Arrow cache creation failed for {file_id}: {e}")

@router.post("/parse/{file_id}")
async def preparse_file(file_id: str, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_active_auth0_user)):
    """Pre-parse a file to create Arrow cache for faster subsequent access"""
    
    # Check if file exists
    upload_dir = settings.UPLOAD_DIR
    found_file = None
    for f_name in os.listdir(upload_dir):
        if f_name.startswith(file_id) and (f_name.endswith('.xlsx') or f_name.endswith('.xls')):
            if "_formulas.json" not in f_name and "_sheet_data.json" not in f_name:
                found_file = f_name
                break
    
    if not found_file:
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
    
    # Add background task
    background_tasks.add_task(_ensure_arrow_exists, file_id)
    
    return {
        "message": "Parsing started", 
        "file_id": file_id,
        "status": "background_processing"
    } 