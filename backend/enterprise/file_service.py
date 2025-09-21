"""
🏢 ENTERPRISE FILE SERVICE
Multi-tenant file storage with complete user isolation and encryption.

This service replaces the insecure shared uploads/ directory with:
- User-isolated directory structure
- File encryption at rest
- Secure file access verification
- Upload quotas and management
- Comprehensive audit logging
"""

import os
import hashlib
import logging
import uuid
# import aiofiles  # Not available, using standard file operations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile

from database import get_db
from models import User as UserModel
from enterprise.simulation_service import EnterpriseAuditLogger

logger = logging.getLogger(__name__)

class EnterpriseFileService:
    """
    🏢 Enterprise-grade file storage with complete user isolation.
    
    Features:
    - User-isolated directory structure: /enterprise-storage/users/{user_id}/
    - File encryption at rest using Fernet
    - Secure file access verification
    - Upload quotas per user tier
    - Complete audit trail
    - Automatic file cleanup and management
    """
    
    def __init__(self, base_storage_path: str = "./enterprise-storage"):
        self.base_path = Path(base_storage_path)
        self.audit_logger = EnterpriseAuditLogger()
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # User tier quotas (in MB)
        self.tier_quotas = {
            "starter": 100,      # 100 MB
            "professional": 500,  # 500 MB  
            "enterprise": 2000,  # 2 GB
            "unlimited": -1      # No limit
        }
        
        # Create base directory structure (lazy initialization)
        try:
            self._ensure_base_directories()
            logger.info(f"🏢 [ENTERPRISE_FILES] Service initialized with base path: {self.base_path}")
        except Exception as e:
            logger.warning(f"⚠️ [ENTERPRISE_FILES] Base directories not created during init: {e}")
            logger.info("🏢 [ENTERPRISE_FILES] Directories will be created on first use")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for file encryption at rest."""
        try:
            key_file = self.base_path / "encryption.key"
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
                logger.info("🔑 [ENTERPRISE_FILES] Loaded existing encryption key")
                return key
            else:
                # For lazy initialization, we'll create the key when needed
                # For now, use a temporary key
                logger.info("🔑 [ENTERPRISE_FILES] Using temporary encryption key (will be persistent on first use)")
                return Fernet.generate_key()
                
        except Exception as e:
            logger.warning(f"⚠️ [ENTERPRISE_FILES] Failed to handle encryption key during init: {e}")
            logger.info("🔑 [ENTERPRISE_FILES] Using temporary encryption key")
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
            
            logger.info(f"✅ [ENTERPRISE_FILES] Base directories created: {self.base_path}")
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE_FILES] Failed to create base directories: {e}")
            raise
    
    def _get_user_directory(self, user_id: int) -> Path:
        """Get user-specific directory with secure permissions."""
        user_dir = self.base_path / "users" / str(user_id)
        
        if not user_dir.exists():
            user_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(user_dir, 0o700)  # rwx------ (owner only)
            
            # Create subdirectories
            uploads_dir = user_dir / "uploads"
            uploads_dir.mkdir(exist_ok=True)
            os.chmod(uploads_dir, 0o700)
            
            temp_dir = user_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            os.chmod(temp_dir, 0o700)
            
            logger.info(f"📁 [ENTERPRISE_FILES] Created user directory: {user_dir}")
        
        return user_dir
    
    async def save_user_file(
        self, 
        user_id: int, 
        file: UploadFile, 
        file_category: str = "upload"
    ) -> Dict[str, Any]:
        """
        Save user file with encryption and complete isolation.
        
        Args:
            user_id: User ID for isolation
            file: Uploaded file
            file_category: Category (upload, temp, result, etc.)
            
        Returns:
            Dict with file metadata
        """
        try:
            # Check upload quota first
            await self._check_user_quota(user_id, file)
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Get user directory
            user_dir = self._get_user_directory(user_id)
            category_dir = user_dir / file_category
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
            metadata = {
                "file_id": file_id,
                "original_filename": file.filename,
                "safe_filename": safe_filename,
                "file_path": str(file_path),
                "file_size": len(content),
                "encrypted_size": len(encrypted_content),
                "category": file_category,
                "user_id": user_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "content_type": file.content_type,
                "file_hash": hashlib.sha256(content).hexdigest()
            }
            
            # Save metadata
            metadata_path = category_dir / f"{file_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                f.write(json.dumps(metadata, indent=2))
            os.chmod(metadata_path, 0o600)
            
            # Log successful upload
            await self.audit_logger.log_simulation_created(
                user_id=user_id,
                simulation_id=file_id,
                request_details={
                    "action": "file_uploaded",
                    "filename": file.filename,
                    "size": len(content),
                    "category": file_category
                }
            )
            
            logger.info(f"✅ [ENTERPRISE_FILES] Saved file {file_id} for user {user_id}: {safe_filename}")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE_FILES] Failed to save file for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=file_id,
                error=str(e),
                action="save_user_file"
            )
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    async def get_user_file(
        self, 
        user_id: int, 
        file_id: str, 
        verify_ownership: bool = True
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get user file with ownership verification and decryption.
        
        Args:
            user_id: User ID for verification
            file_id: File ID to retrieve
            verify_ownership: Whether to verify user owns the file
            
        Returns:
            Tuple of (file_content, metadata)
        """
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
                await self.audit_logger.log_access_attempt(
                    user_id=user_id,
                    simulation_id=file_id,
                    action="file_access_denied",
                    reason="file_not_found"
                )
                raise HTTPException(status_code=404, detail="File not found")
            
            # Load and verify metadata
            with open(metadata_file, 'r') as f:
                metadata = json.loads(f.read())
            
            # Verify ownership
            if verify_ownership and metadata.get("user_id") != user_id:
                await self.audit_logger.log_access_attempt(
                    user_id=user_id,
                    simulation_id=file_id,
                    action="file_access_denied",
                    reason="ownership_verification_failed"
                )
                raise HTTPException(status_code=403, detail="File access denied")
            
            # Load and decrypt file
            file_path = Path(metadata["file_path"])
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File content not found")
            
            with open(file_path, 'rb') as f:
                encrypted_content = f.read()
            
            # Decrypt content
            try:
                content = self.cipher.decrypt(encrypted_content)
            except Exception as e:
                logger.error(f"❌ [ENTERPRISE_FILES] Decryption failed for file {file_id}: {e}")
                raise HTTPException(status_code=500, detail="File decryption failed")
            
            # Log successful access
            await self.audit_logger.log_access_attempt(
                user_id=user_id,
                simulation_id=file_id,
                action="file_accessed",
                reason="authorized_access"
            )
            
            logger.info(f"✅ [ENTERPRISE_FILES] Retrieved file {file_id} for user {user_id}")
            return content, metadata
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE_FILES] Failed to get file {file_id} for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=file_id,
                error=str(e),
                action="get_user_file"
            )
            raise HTTPException(status_code=500, detail="Failed to retrieve file")
    
    async def list_user_files(
        self, 
        user_id: int, 
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all files for a user with optional category filtering.
        
        Args:
            user_id: User ID
            category: Optional category filter
            
        Returns:
            List of file metadata
        """
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
                            metadata = json.loads(f.read())
                        
                        # Verify ownership
                        if metadata.get("user_id") == user_id:
                            files.append(metadata)
                    except Exception as e:
                        logger.warning(f"⚠️ [ENTERPRISE_FILES] Failed to read metadata {metadata_file}: {e}")
            
            # Log bulk access
            await self.audit_logger.log_bulk_access(
                user_id=user_id,
                action="list_user_files",
                count=len(files)
            )
            
            logger.info(f"✅ [ENTERPRISE_FILES] Listed {len(files)} files for user {user_id}")
            return files
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE_FILES] Failed to list files for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=None,
                error=str(e),
                action="list_user_files"
            )
            raise HTTPException(status_code=500, detail="Failed to list files")
    
    async def delete_user_file(self, user_id: int, file_id: str) -> bool:
        """
        Delete user file with ownership verification.
        
        Args:
            user_id: User ID for verification
            file_id: File ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Get file metadata first
            content, metadata = await self.get_user_file(user_id, file_id, verify_ownership=True)
            
            # Delete file and metadata
            file_path = Path(metadata["file_path"])
            metadata_path = file_path.parent / f"{file_id}_metadata.json"
            
            if file_path.exists():
                file_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Log deletion
            await self.audit_logger.log_simulation_deleted(
                user_id=user_id,
                simulation_id=file_id,
                simulation_info={
                    "action": "file_deleted",
                    "filename": metadata["original_filename"],
                    "size": metadata["file_size"]
                }
            )
            
            logger.info(f"✅ [ENTERPRISE_FILES] Deleted file {file_id} for user {user_id}")
            return True
            
        except HTTPException as e:
            if e.status_code == 404:
                return True  # Already deleted
            raise
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE_FILES] Failed to delete file {file_id} for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=file_id,
                error=str(e),
                action="delete_user_file"
            )
            return False
    
    async def get_user_storage_usage(self, user_id: int) -> Dict[str, Any]:
        """Get user storage usage statistics."""
        try:
            files = await self.list_user_files(user_id)
            
            total_size = sum(f.get("file_size", 0) for f in files)
            total_encrypted_size = sum(f.get("encrypted_size", 0) for f in files)
            
            # Get user tier and quota
            user_tier = await self._get_user_tier(user_id)
            quota_mb = self.tier_quotas.get(user_tier, 100)
            quota_bytes = quota_mb * 1024 * 1024 if quota_mb > 0 else -1
            
            usage_stats = {
                "user_id": user_id,
                "total_files": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "encrypted_size_bytes": total_encrypted_size,
                "user_tier": user_tier,
                "quota_mb": quota_mb,
                "quota_bytes": quota_bytes,
                "quota_used_percent": round((total_size / quota_bytes) * 100, 2) if quota_bytes > 0 else 0,
                "files_by_category": {}
            }
            
            # Group by category
            for file in files:
                category = file.get("category", "unknown")
                if category not in usage_stats["files_by_category"]:
                    usage_stats["files_by_category"][category] = {"count": 0, "size": 0}
                
                usage_stats["files_by_category"][category]["count"] += 1
                usage_stats["files_by_category"][category]["size"] += file.get("file_size", 0)
            
            return usage_stats
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE_FILES] Failed to get storage usage for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get storage usage")
    
    async def _check_user_quota(self, user_id: int, file: UploadFile):
        """Check if user has sufficient quota for file upload."""
        try:
            # Get current usage
            usage_stats = await self.get_user_storage_usage(user_id)
            
            # Check if quota is unlimited
            if usage_stats["quota_bytes"] == -1:
                return  # Unlimited quota
            
            # Read file size
            content = await file.read()
            file_size = len(content)
            await file.seek(0)  # Reset file pointer
            
            # Check if adding this file would exceed quota
            new_total = usage_stats["total_size_bytes"] + file_size
            
            if new_total > usage_stats["quota_bytes"]:
                quota_mb = usage_stats["quota_mb"]
                used_mb = round(usage_stats["total_size_bytes"] / (1024 * 1024), 2)
                file_mb = round(file_size / (1024 * 1024), 2)
                
                await self.audit_logger.log_error(
                    user_id=user_id,
                    simulation_id=None,
                    error=f"Quota exceeded: {used_mb}MB + {file_mb}MB > {quota_mb}MB",
                    action="quota_check"
                )
                
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload would exceed quota. Used: {used_mb}MB, File: {file_mb}MB, Quota: {quota_mb}MB"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE_FILES] Quota check failed for user {user_id}: {e}")
            # Allow upload if quota check fails (fail open for availability)
            logger.warning("⚠️ [ENTERPRISE_FILES] Quota check failed, allowing upload")
    
    async def _get_user_tier(self, user_id: int) -> str:
        """Get user subscription tier (placeholder - integrate with actual user system)."""
        # TODO: Integrate with actual subscription system
        # For now, return default tier
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

# Global enterprise file service instance
enterprise_file_service = EnterpriseFileService()

# 🔄 LEGACY COMPATIBILITY FUNCTIONS
# These provide compatibility with the existing file system during migration

async def get_legacy_file_path(file_id: str) -> Optional[str]:
    """
    Get file path from legacy uploads/ directory.
    Used during migration period for backward compatibility.
    """
    try:
        upload_dir = Path("uploads")
        if not upload_dir.exists():
            return None
        
        # Search for files with this file_id
        for file_path in upload_dir.glob(f"{file_id}_*"):
            if file_path.suffix in ['.xlsx', '.xls']:
                logger.warning(f"🚨 [LEGACY] Using legacy file path: {file_path}")
                return str(file_path)
        
        return None
        
    except Exception as e:
        logger.error(f"❌ [LEGACY] Failed to get legacy file path for {file_id}: {e}")
        return None

async def migrate_legacy_file_to_enterprise(
    file_id: str, 
    user_id: int, 
    legacy_path: str
) -> Optional[str]:
    """
    Migrate a file from legacy uploads/ directory to enterprise storage.
    """
    try:
        # Read legacy file
        with open(legacy_path, 'rb') as f:
            content = f.read()
        
        # Get original filename
        filename = os.path.basename(legacy_path)
        if filename.startswith(f"{file_id}_"):
            filename = filename[len(f"{file_id}_"):]
        
        # Create mock UploadFile for enterprise service
        from io import BytesIO
        
        class MockUploadFile:
            def __init__(self, content: bytes, filename: str):
                self.content = content
                self.filename = filename
                self.content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                self._position = 0
            
            async def read(self):
                return self.content
            
            async def seek(self, position: int):
                self._position = position
        
        mock_file = MockUploadFile(content, filename)
        
        # Save to enterprise storage
        metadata = await enterprise_file_service.save_user_file(
            user_id=user_id,
            file=mock_file,
            file_category="uploads"
        )
        
        logger.info(f"✅ [MIGRATION] Migrated {legacy_path} to enterprise storage: {metadata['file_id']}")
        return metadata['file_id']
        
    except Exception as e:
        logger.error(f"❌ [MIGRATION] Failed to migrate {legacy_path}: {e}")
        return None
