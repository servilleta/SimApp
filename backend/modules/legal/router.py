"""
Legal documents router for serving privacy policy, terms of service, etc.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Legal documents directory
LEGAL_DIR = Path(__file__).parent.parent.parent / "legal"

@router.get("/privacy")
async def get_privacy_policy():
    """Serve the privacy policy document"""
    try:
        privacy_file = LEGAL_DIR / "PRIVACY_POLICY.md"
        if not privacy_file.exists():
            raise HTTPException(status_code=404, detail="Privacy policy not found")
        
        with open(privacy_file, 'r') as f:
            content = f.read()
            
        return PlainTextResponse(content=content, media_type="text/markdown")
    except Exception as e:
        logger.error(f"Error serving privacy policy: {e}")
        raise HTTPException(status_code=500, detail="Failed to load privacy policy")

@router.get("/terms")
async def get_terms_of_service():
    """Serve the terms of service document"""
    try:
        terms_file = LEGAL_DIR / "TERMS_OF_SERVICE.md"
        if not terms_file.exists():
            raise HTTPException(status_code=404, detail="Terms of service not found")
        
        with open(terms_file, 'r') as f:
            content = f.read()
            
        return PlainTextResponse(content=content, media_type="text/markdown")
    except Exception as e:
        logger.error(f"Error serving terms of service: {e}")
        raise HTTPException(status_code=500, detail="Failed to load terms of service")

@router.get("/cookie-policy")
async def get_cookie_policy():
    """Serve the cookie policy document"""
    try:
        cookie_file = LEGAL_DIR / "COOKIE_POLICY.md"
        if not cookie_file.exists():
            raise HTTPException(status_code=404, detail="Cookie policy not found")
        
        with open(cookie_file, 'r') as f:
            content = f.read()
            
        return PlainTextResponse(content=content, media_type="text/markdown")
    except Exception as e:
        logger.error(f"Error serving cookie policy: {e}")
        raise HTTPException(status_code=500, detail="Failed to load cookie policy")

@router.get("/acceptable-use")
async def get_acceptable_use_policy():
    """Serve the acceptable use policy document"""
    try:
        aup_file = LEGAL_DIR / "ACCEPTABLE_USE_POLICY.md"
        if not aup_file.exists():
            raise HTTPException(status_code=404, detail="Acceptable use policy not found")
        
        with open(aup_file, 'r') as f:
            content = f.read()
            
        return PlainTextResponse(content=content, media_type="text/markdown")
    except Exception as e:
        logger.error(f"Error serving acceptable use policy: {e}")
        raise HTTPException(status_code=500, detail="Failed to load acceptable use policy")

@router.get("/open-source-licenses")
async def get_open_source_licenses():
    """Serve the open source licenses document"""
    try:
        licenses_file = LEGAL_DIR / "OPEN_SOURCE_LICENSES.md"
        if not licenses_file.exists():
            raise HTTPException(status_code=404, detail="Open source licenses not found")
        
        with open(licenses_file, 'r') as f:
            content = f.read()
            
        return PlainTextResponse(content=content, media_type="text/markdown")
    except Exception as e:
        logger.error(f"Error serving open source licenses: {e}")
        raise HTTPException(status_code=500, detail="Failed to load open source licenses")

@router.get("/data-processing-agreement")
async def get_data_processing_agreement():
    """Serve the data processing agreement template"""
    try:
        dpa_file = LEGAL_DIR / "DATA_PROCESSING_AGREEMENT_TEMPLATE.md"
        if not dpa_file.exists():
            raise HTTPException(status_code=404, detail="Data processing agreement not found")
        
        with open(dpa_file, 'r') as f:
            content = f.read()
            
        return PlainTextResponse(content=content, media_type="text/markdown")
    except Exception as e:
        logger.error(f"Error serving data processing agreement: {e}")
        raise HTTPException(status_code=500, detail="Failed to load data processing agreement")

@router.get("/all")
async def get_all_legal_documents():
    """Get a list of all available legal documents"""
    try:
        documents = []
        if LEGAL_DIR.exists():
            for file_path in LEGAL_DIR.glob("*.md"):
                documents.append({
                    "name": file_path.stem.replace("_", " ").title(),
                    "filename": file_path.name,
                    "endpoint": f"/legal/{file_path.stem.lower().replace('_', '-')}"
                })
        
        return {
            "documents": documents,
            "total": len(documents),
            "message": "All legal documents available via API endpoints"
        }
    except Exception as e:
        logger.error(f"Error listing legal documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list legal documents")