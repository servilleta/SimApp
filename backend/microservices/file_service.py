"""
📁 FILE SERVICE - Microservice Architecture

Handles all file-related operations:
- File upload and storage
- File encryption and decryption
- File metadata management
- Storage quotas and usage tracking
- File sharing and permissions

This service is part of the microservices decomposition from the monolithic application.
"""

import logging
import uuid
import os
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Response, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from sqlalchemy import and_
from pydantic import BaseModel
from cryptography.fernet import Fernet

# Import from monolith (during transition)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from models import User
from auth.auth0_dependencies import get_current_active_auth0_user

logger = logging.getLogger(__name__)

# FastAPI app for File Service
app = FastAPI(
    title="File Service",
    description="Microservice for file management, storage, and encryption",
    version="1.0.0"
)

security = HTTPBearer()

# ===============================
# PYDANTIC MODELS (API SCHEMAS)
# ===============================

class FileMetadata(BaseModel):
    file_id: str
    original_filename: str
    file_size: int
    encrypted_size: int
    content_type: str
    file_hash: str
    category: str
    uploaded_at: datetime
    
    class Config:
        from_attributes = True

class StorageUsage(BaseModel):
    user_id: int
    total_files: int
    total_size_mb: float
    quota_mb: int
    quota_used_percent: float
    files_by_category: Dict[str, Dict[str, Any]]

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    category: str
    uploaded_at: str
    message: str

class FileListResponse(BaseModel):
    files: List[FileMetadata]
    total_count: int
    category_filter: Optional[str] = None

# ===============================
# FILE SERVICE CLASS
# ===============================

class FileService:
    """
    Core file service handling all file-related operations.
    Designed for microservices architecture with encryption and isolation.
    """
    
    def __init__(self, base_storage_path: str = "./microservices-file-storage"):
        self.base_path = Path(base_storage_path)
        self.logger = logging.getLogger(__name__)
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # User tier quotas (in MB)
        self.tier_quotas = {
            "starter": 100,      # 100 MB
            "professional": 500,  # 500 MB  
            "enterprise": 2000,  # 2 GB
            "unlimited": -1      # No limit
        }
        
        # Ensure base directory exists
        self._ensure_base_directories()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for file encryption at rest."""
        try:
            key_file = self.base_path / "encryption.key"
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
                self.logger.info("🔑 [FILE_SERVICE] Loaded existing encryption key")
                return key
            else:
                # Create new key
                key = Fernet.generate_key()
                
                # Ensure directory exists
                key_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save key with secure permissions
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)  # Read/write owner only
                
                self.logger.info("🔑 [FILE_SERVICE] Created new encryption key")
                return key
                
        except Exception as e:
            self.logger.warning(f"⚠️ [FILE_SERVICE] Failed to handle encryption key: {e}")
            self.logger.info("🔑 [FILE_SERVICE] Using temporary encryption key")
            return Fernet.generate_key()
    
    def _ensure_base_directories(self):
        """Create base directory structure with secure permissions."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            os.chmod(self.base_path, 0o750)  # rwxr-x--- (owner + group)
            
            # Create users directory
            users_dir = self.base_path / "users"
            users_dir.mkdir(exist_ok=True)
            os.chmod(users_dir, 0o750)
            
            self.logger.info(f"✅ [FILE_SERVICE] Base directories created: {self.base_path}")
            
        except Exception as e:
            self.logger.error(f"❌ [FILE_SERVICE] Failed to create base directories: {e}")
            # Continue without failing - directories will be created on first use
    
    def _get_user_directory(self, user_id: int) -> Path:
        """Get user-specific directory with secure permissions."""
        user_dir = self.base_path / "users" / str(user_id)
        
        if not user_dir.exists():
            user_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(user_dir, 0o700)  # rwx------ (owner only)
            
            # Create subdirectories
            for subdir in ["uploads", "temp", "results"]:
                sub_path = user_dir / subdir
                sub_path.mkdir(exist_ok=True)
                os.chmod(sub_path, 0o700)
            
            self.logger.info(f"📁 [FILE_SERVICE] Created user directory: {user_dir}")
        
        return user_dir
    
    async def upload_file(self, user_id: int, file: UploadFile, category: str = "uploads") -> FileMetadata:
        """Upload and encrypt a file for a user."""
        try:
            # Check upload quota first
            await self._check_user_quota(user_id, file)
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Get user directory
            user_dir = self._get_user_directory(user_id)
            category_dir = user_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Create secure filename
            safe_filename = self._sanitize_filename(file.filename)
            encrypted_filename = f"{file_id}_{safe_filename}.enc"
            file_path = category_dir / encrypted_filename
            
            # Read and encrypt file content
            content = await file.read()
            encrypted_content = self.cipher.encrypt(content)
            
            # Save encrypted file
            with open(file_path, 'wb') as f:
                f.write(encrypted_content)
            
            # Set secure permissions
            os.chmod(file_path, 0o600)  # rw------- (owner only)
            
            # Create metadata
            metadata = FileMetadata(
                file_id=file_id,
                original_filename=file.filename,
                file_size=len(content),
                encrypted_size=len(encrypted_content),
                content_type=file.content_type or "application/octet-stream",
                file_hash=hashlib.sha256(content).hexdigest(),
                category=category,
                uploaded_at=datetime.now(timezone.utc)
            )
            
            # Save metadata
            metadata_path = category_dir / f"{file_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                f.write(metadata.json(indent=2))
            os.chmod(metadata_path, 0o600)
            
            self.logger.info(f"✅ [FILE_SERVICE] Uploaded file {file_id} for user {user_id}: {safe_filename}")
            return metadata
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ [FILE_SERVICE] Failed to upload file for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    
    async def get_file(self, user_id: int, file_id: str) -> tuple[bytes, FileMetadata]:
        """Get and decrypt a file for a user."""
        try:
            # Find file in user's directories
            user_dir = self._get_user_directory(user_id)
            metadata_file = None
            
            # Search in all categories
            for category in ["uploads", "temp", "results"]:
                category_dir = user_dir / category
                potential_metadata = category_dir / f"{file_id}_metadata.json"
                if potential_metadata.exists():
                    metadata_file = potential_metadata
                    break
            
            if not metadata_file:
                raise HTTPException(status_code=404, detail="File not found")
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_dict = f.read()
                metadata = FileMetadata.parse_raw(metadata_dict)
            
            # Load and decrypt file
            file_path = metadata_file.parent / f"{file_id}_{self._sanitize_filename(metadata.original_filename)}.enc"
            
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File content not found")
            
            with open(file_path, 'rb') as f:
                encrypted_content = f.read()
            
            # Decrypt content
            try:
                content = self.cipher.decrypt(encrypted_content)
            except Exception as e:
                self.logger.error(f"❌ [FILE_SERVICE] Decryption failed for file {file_id}: {e}")
                raise HTTPException(status_code=500, detail="File decryption failed")
            
            self.logger.info(f"✅ [FILE_SERVICE] Retrieved file {file_id} for user {user_id}")
            return content, metadata
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ [FILE_SERVICE] Failed to get file {file_id} for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve file")
    
    async def list_user_files(self, user_id: int, category: Optional[str] = None) -> List[FileMetadata]:
        """List all files for a user with optional category filtering."""
        try:
            user_dir = self._get_user_directory(user_id)
            files = []
            
            # Determine categories to search
            categories = [category] if category else ["uploads", "temp", "results"]
            
            for cat in categories:
                category_dir = user_dir / cat
                if not category_dir.exists():
                    continue
                
                # Find all metadata files
                for metadata_file in category_dir.glob("*_metadata.json"):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = f.read()
                            metadata = FileMetadata.parse_raw(metadata_dict)
                            files.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"⚠️ [FILE_SERVICE] Failed to read metadata {metadata_file}: {e}")
            
            self.logger.info(f"✅ [FILE_SERVICE] Listed {len(files)} files for user {user_id}")
            return files
            
        except Exception as e:
            self.logger.error(f"❌ [FILE_SERVICE] Failed to list files for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to list files")
    
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """Delete a file for a user."""
        try:
            # Get file first to verify ownership
            content, metadata = await self.get_file(user_id, file_id)
            
            # Delete file and metadata
            user_dir = self._get_user_directory(user_id)
            
            # Find and delete in all categories
            for category in ["uploads", "temp", "results"]:
                category_dir = user_dir / category
                
                file_path = category_dir / f"{file_id}_{self._sanitize_filename(metadata.original_filename)}.enc"
                metadata_path = category_dir / f"{file_id}_metadata.json"
                
                if file_path.exists():
                    file_path.unlink()
                
                if metadata_path.exists():
                    metadata_path.unlink()
            
            self.logger.info(f"✅ [FILE_SERVICE] Deleted file {file_id} for user {user_id}")
            return True
            
        except HTTPException as e:
            if e.status_code == 404:
                return True  # Already deleted
            raise
        except Exception as e:
            self.logger.error(f"❌ [FILE_SERVICE] Failed to delete file {file_id} for user {user_id}: {e}")
            return False
    
    async def get_storage_usage(self, user_id: int) -> StorageUsage:
        """Get storage usage statistics for a user."""
        try:
            files = await self.list_user_files(user_id)
            
            total_size = sum(f.file_size for f in files)
            
            # Get user tier and quota
            user_tier = await self._get_user_tier(user_id)
            quota_mb = self.tier_quotas.get(user_tier, 100)
            quota_bytes = quota_mb * 1024 * 1024 if quota_mb > 0 else -1
            
            # Group by category
            files_by_category = {}
            for file in files:
                category = file.category
                if category not in files_by_category:
                    files_by_category[category] = {"count": 0, "size": 0}
                
                files_by_category[category]["count"] += 1
                files_by_category[category]["size"] += file.file_size
            
            return StorageUsage(
                user_id=user_id,
                total_files=len(files),
                total_size_mb=round(total_size / (1024 * 1024), 2),
                quota_mb=quota_mb,
                quota_used_percent=round((total_size / quota_bytes) * 100, 2) if quota_bytes > 0 else 0,
                files_by_category=files_by_category
            )
            
        except Exception as e:
            self.logger.error(f"❌ [FILE_SERVICE] Failed to get storage usage for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get storage usage")
    
    async def _check_user_quota(self, user_id: int, file: UploadFile):
        """Check if user has sufficient quota for file upload."""
        try:
            # Get current usage
            usage = await self.get_storage_usage(user_id)
            
            # Check if quota is unlimited
            if usage.quota_mb == -1:
                return  # Unlimited quota
            
            # Read file size
            content = await file.read()
            file_size = len(content)
            await file.seek(0)  # Reset file pointer
            
            # Check if adding this file would exceed quota
            new_total_mb = usage.total_size_mb + (file_size / (1024 * 1024))
            
            if new_total_mb > usage.quota_mb:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload would exceed quota. Used: {usage.total_size_mb}MB, File: {file_size/(1024*1024):.2f}MB, Quota: {usage.quota_mb}MB"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ [FILE_SERVICE] Quota check failed for user {user_id}: {e}")
            # Allow upload if quota check fails (fail open for availability)
            self.logger.warning("⚠️ [FILE_SERVICE] Quota check failed, allowing upload")
    
    async def _get_user_tier(self, user_id: int) -> str:
        """Get user subscription tier (placeholder - integrate with User Service)."""
        # TODO: Call User Service API to get subscription tier
        return "professional"
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for secure storage."""
        import re
        
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove/replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 100:
            name, ext = os.path.splitext(filename)
            filename = name[:90] + ext
        
        return filename

# Global service instance
file_service = FileService()

# ===============================
# API ENDPOINTS
# ===============================

@app.get("/health")
async def health_check():
    """Service health check."""
    return {"status": "healthy", "service": "file-service", "version": "1.0.0"}

@app.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    category: str = "uploads",
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Upload a file with encryption and user isolation."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    metadata = await file_service.upload_file(current_user.id, file, category)
    
    return FileUploadResponse(
        file_id=metadata.file_id,
        filename=metadata.original_filename,
        size=metadata.file_size,
        category=metadata.category,
        uploaded_at=metadata.uploaded_at.isoformat(),
        message="File uploaded and encrypted successfully"
    )

@app.get("/list", response_model=FileListResponse)
async def list_files(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """List user's files with optional category filtering."""
    files = await file_service.list_user_files(current_user.id, category)
    
    return FileListResponse(
        files=files,
        total_count=len(files),
        category_filter=category
    )

@app.get("/{file_id}/download")
async def download_file(
    file_id: str,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Download a file with decryption."""
    content, metadata = await file_service.get_file(current_user.id, file_id)
    
    return Response(
        content=content,
        media_type=metadata.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{metadata.original_filename}"',
            "Content-Length": str(len(content))
        }
    )

@app.get("/{file_id}/info", response_model=FileMetadata)
async def get_file_info(
    file_id: str,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get file metadata."""
    _, metadata = await file_service.get_file(current_user.id, file_id)
    return metadata

@app.delete("/{file_id}")
async def delete_file(
    file_id: str,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Delete a file."""
    success = await file_service.delete_file(current_user.id, file_id)
    if not success:
        raise HTTPException(status_code=500, detail="File deletion failed")
    
    return {"message": "File deleted successfully", "file_id": file_id}

@app.get("/storage/usage", response_model=StorageUsage)
async def get_storage_usage(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get user's storage usage statistics."""
    return await file_service.get_storage_usage(current_user.id)

# ===============================
# SERVICE DISCOVERY ENDPOINTS
# ===============================

@app.get("/service-info")
async def get_service_info():
    """Service discovery information."""
    return {
        "service_name": "file-service",
        "version": "1.0.0",
        "description": "File management, storage, and encryption",
        "endpoints": {
            "upload": "/upload",
            "list": "/list",
            "download": "/{file_id}/download",
            "info": "/{file_id}/info",
            "delete": "/{file_id}",
            "usage": "/storage/usage"
        },
        "dependencies": ["database", "encryption"],
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
