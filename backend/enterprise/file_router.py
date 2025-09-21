"""
🏢 ENTERPRISE FILE ROUTER
Secure, user-isolated file upload and management endpoints.

This router provides enterprise-grade file operations with:
- Complete user isolation
- File encryption at rest
- Upload quota management
- Secure file access verification
- Comprehensive audit logging
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response
from sqlalchemy.orm import Session

from database import get_db
from models import User as UserModel
from auth.auth0_dependencies import get_current_active_auth0_user
from enterprise.file_service import enterprise_file_service

logger = logging.getLogger(__name__)

# Create enterprise file router
router = APIRouter(prefix="/api/enterprise/files", tags=["🏢 Enterprise Files"])

@router.post("/upload")
async def upload_enterprise_file(
    file: UploadFile = File(...),
    category: str = "upload",
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    🏢 Upload file with enterprise security and encryption.
    
    🔒 SECURITY: 
    - Files are automatically associated with authenticated user
    - Stored in user-isolated directories
    - Encrypted at rest using Fernet encryption
    - Upload quotas enforced based on user tier
    """
    try:
        logger.info(f"🏢 [ENTERPRISE_FILES] File upload started: {file.filename} by user {current_user.id}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Check file size (basic validation)
        if file.size and file.size > 500 * 1024 * 1024:  # 500MB limit
            raise HTTPException(status_code=413, detail="File too large (max 500MB)")
        
        # Save file with enterprise security
        metadata = await enterprise_file_service.save_user_file(
            user_id=current_user.id,
            file=file,
            file_category=category
        )
        
        logger.info(f"✅ [ENTERPRISE_FILES] File uploaded successfully: {metadata['file_id']}")
        
        return {
            "file_id": metadata["file_id"],
            "filename": metadata["original_filename"],
            "size": metadata["file_size"],
            "category": metadata["category"],
            "uploaded_at": metadata["uploaded_at"],
            "message": "File uploaded and encrypted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_FILES] Upload failed for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

@router.get("/list")
async def list_enterprise_files(
    category: Optional[str] = None,
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    🏢 List user's files with optional category filtering.
    
    🔒 SECURITY: Only returns files owned by the authenticated user.
    """
    try:
        files = await enterprise_file_service.list_user_files(
            user_id=current_user.id,
            category=category
        )
        
        # Return safe metadata (exclude sensitive file paths)
        safe_files = []
        for file_meta in files:
            safe_files.append({
                "file_id": file_meta["file_id"],
                "filename": file_meta["original_filename"],
                "size": file_meta["file_size"],
                "category": file_meta["category"],
                "uploaded_at": file_meta["uploaded_at"],
                "content_type": file_meta.get("content_type"),
                "file_hash": file_meta["file_hash"][:16] + "..."  # Truncated hash
            })
        
        logger.info(f"✅ [ENTERPRISE_FILES] Listed {len(safe_files)} files for user {current_user.id}")
        
        return {
            "files": safe_files,
            "total_count": len(safe_files),
            "category_filter": category
        }
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_FILES] List failed for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")

@router.get("/{file_id}/download")
async def download_enterprise_file(
    file_id: str,
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    🏢 Download file with user verification and decryption.
    
    🔒 SECURITY: 
    - Verifies user owns the file before allowing download
    - Automatically decrypts file content
    - Complete audit trail of file access
    """
    try:
        logger.info(f"🏢 [ENTERPRISE_FILES] Download requested: {file_id} by user {current_user.id}")
        
        # Get and decrypt file
        content, metadata = await enterprise_file_service.get_user_file(
            user_id=current_user.id,
            file_id=file_id,
            verify_ownership=True
        )
        
        # Return file with appropriate headers
        filename = metadata["original_filename"]
        content_type = metadata.get("content_type", "application/octet-stream")
        
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_FILES] Download failed: {file_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="File download failed")

@router.get("/{file_id}/info")
async def get_enterprise_file_info(
    file_id: str,
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    🏢 Get file information with user verification.
    
    🔒 SECURITY: Only returns info for files owned by the authenticated user.
    """
    try:
        # Get file metadata (without content)
        _, metadata = await enterprise_file_service.get_user_file(
            user_id=current_user.id,
            file_id=file_id,
            verify_ownership=True
        )
        
        # Return safe metadata
        return {
            "file_id": metadata["file_id"],
            "filename": metadata["original_filename"],
            "size": metadata["file_size"],
            "encrypted_size": metadata["encrypted_size"],
            "category": metadata["category"],
            "uploaded_at": metadata["uploaded_at"],
            "content_type": metadata.get("content_type"),
            "file_hash": metadata["file_hash"][:16] + "..."  # Truncated hash
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_FILES] Info failed: {file_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get file info")

@router.delete("/{file_id}")
async def delete_enterprise_file(
    file_id: str,
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    🏢 Delete file with user verification.
    
    🔒 SECURITY: Only allows deletion of files owned by the authenticated user.
    """
    try:
        logger.info(f"🏢 [ENTERPRISE_FILES] Delete requested: {file_id} by user {current_user.id}")
        
        # Delete file with ownership verification
        success = await enterprise_file_service.delete_user_file(
            user_id=current_user.id,
            file_id=file_id
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="File deletion failed")
        
        return {
            "message": "File deleted successfully",
            "file_id": file_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_FILES] Delete failed: {file_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="File deletion failed")

@router.get("/storage/usage")
async def get_storage_usage(
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    🏢 Get user's storage usage statistics.
    
    Returns detailed information about storage consumption and quotas.
    """
    try:
        usage_stats = await enterprise_file_service.get_user_storage_usage(
            user_id=current_user.id
        )
        
        return usage_stats
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_FILES] Storage usage failed for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get storage usage")

@router.post("/{file_id}/migrate-from-legacy")
async def migrate_legacy_file(
    file_id: str,
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    🔄 Migrate a file from legacy uploads/ directory to enterprise storage.
    
    This endpoint helps transition existing files to the new secure storage system.
    """
    try:
        from enterprise.file_service import get_legacy_file_path, migrate_legacy_file_to_enterprise
        
        logger.info(f"🔄 [MIGRATION] Legacy file migration requested: {file_id} by user {current_user.id}")
        
        # Check if file exists in legacy storage
        legacy_path = await get_legacy_file_path(file_id)
        if not legacy_path:
            raise HTTPException(status_code=404, detail="Legacy file not found")
        
        # Migrate to enterprise storage
        new_file_id = await migrate_legacy_file_to_enterprise(
            file_id=file_id,
            user_id=current_user.id,
            legacy_path=legacy_path
        )
        
        if not new_file_id:
            raise HTTPException(status_code=500, detail="Migration failed")
        
        return {
            "message": "File migrated successfully",
            "legacy_file_id": file_id,
            "new_file_id": new_file_id,
            "legacy_path": legacy_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [MIGRATION] Migration failed: {file_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="File migration failed")

# Health check endpoint
@router.get("/health")
async def enterprise_files_health_check():
    """Enterprise file service health check."""
    try:
        # Basic health checks
        from pathlib import Path
        
        base_path = enterprise_file_service.base_path
        encryption_available = enterprise_file_service.cipher is not None
        
        return {
            "status": "healthy",
            "service": "enterprise-file-service",
            "version": "1.0.0",
            "base_path_exists": base_path.exists(),
            "encryption_available": encryption_available,
            "capabilities": [
                "user-isolation",
                "file-encryption",
                "quota-management",
                "audit-logging",
                "secure-access-verification"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_FILES] Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
