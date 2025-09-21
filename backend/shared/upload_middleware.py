"""
Upload Middleware

Handles file upload validation, size limits, and security checks.
"""

import os
import logging
from typing import Optional
from fastapi import HTTPException, UploadFile
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class UploadValidator:
    """Validator for file uploads with security and size checks."""
    
    def __init__(self):
        self.max_size = settings.MAX_UPLOAD_SIZE
        self.allowed_mime_types = settings.ALLOWED_EXCEL_MIME_TYPES
        self.upload_dir = Path(settings.UPLOAD_DIR)
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(exist_ok=True)
        
    def validate_file(self, file: UploadFile) -> dict:
        """
        Validate uploaded file for size, type, and security.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Dictionary with validation results
            
        Raises:
            HTTPException: If validation fails
        """
        validation_result = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": 0,
            "size_mb": 0.0,
            "is_valid": True,
            "errors": []
        }
        
        try:
            # Check if file is provided
            if not file or not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="No file provided"
                )
            
            # Validate filename
            self._validate_filename(file.filename)
            
            # Validate content type
            self._validate_content_type(file.content_type)
            
            # Validate file size
            file_size = self._get_file_size(file)
            validation_result["size_bytes"] = file_size
            validation_result["size_mb"] = round(file_size / (1024 * 1024), 2)
            
            self._validate_file_size(file_size)
            
            # Security checks
            self._security_checks(file.filename)
            
            logger.info(f"File validation passed: {file.filename} ({validation_result['size_mb']:.1f} MB)")
            
        except HTTPException:
            validation_result["is_valid"] = False
            raise
        except Exception as e:
            logger.error(f"Unexpected error in file validation: {e}")
            validation_result["is_valid"] = False
            validation_result["errors"].append(str(e))
            raise HTTPException(
                status_code=500,
                detail="File validation failed due to server error"
            )
            
        return validation_result
    
    def _validate_filename(self, filename: str):
        """Validate filename for security and format."""
        if not filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )
        
        # Check file extension
        allowed_extensions = ['.xlsx', '.xls']
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed extensions: {', '.join(allowed_extensions)}"
            )
        
        # Security: Check for dangerous characters
        dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            if char in filename:
                raise HTTPException(
                    status_code=400,
                    detail="Filename contains invalid characters"
                )
    
    def _validate_content_type(self, content_type: str):
        """Validate MIME type."""
        if not content_type:
            logger.warning("No content type provided, allowing based on filename")
            return
            
        if content_type not in self.allowed_mime_types:
            # Be lenient with content types as browsers can be inconsistent
            logger.warning(f"Unexpected content type: {content_type}, but allowing based on filename")
    
    def _get_file_size(self, file: UploadFile) -> int:
        """Get the size of uploaded file."""
        # Save current position
        current_pos = file.file.tell()
        
        try:
            # Seek to end to get size
            file.file.seek(0, 2)
            size = file.file.tell()
            
            # Reset to original position
            file.file.seek(current_pos)
            
            return size
        except Exception as e:
            logger.error(f"Error getting file size: {e}")
            # Reset position anyway
            try:
                file.file.seek(current_pos)
            except:
                pass
            raise HTTPException(
                status_code=400,
                detail="Unable to determine file size"
            )
    
    def _validate_file_size(self, file_size: int):
        """Validate file size against limits."""
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        if file_size > self.max_size:
            max_mb = self.max_size / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_mb:.1f} MB, uploaded: {actual_mb:.1f} MB"
            )
    
    def _security_checks(self, filename: str):
        """Additional security checks."""
        # Check filename length
        if len(filename) > 255:
            raise HTTPException(
                status_code=400,
                detail="Filename too long (maximum 255 characters)"
            )
        
        # Check for suspicious patterns
        suspicious_patterns = ['script', 'exec', 'cmd', 'powershell']
        filename_lower = filename.lower()
        
        for pattern in suspicious_patterns:
            if pattern in filename_lower:
                logger.warning(f"Suspicious filename detected: {filename}")
                # Don't block but log for monitoring
                break
    
    def check_disk_space(self) -> dict:
        """Check available disk space in upload directory."""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(self.upload_dir)
            
            # Convert to MB
            total_mb = total / (1024 * 1024)
            used_mb = used / (1024 * 1024)
            free_mb = free / (1024 * 1024)
            
            # Calculate percentages
            used_percent = (used / total) * 100
            
            result = {
                "total_mb": round(total_mb, 2),
                "used_mb": round(used_mb, 2),
                "free_mb": round(free_mb, 2),
                "used_percent": round(used_percent, 2),
                "low_space_warning": used_percent > 90,
                "critical_space_warning": used_percent > 95
            }
            
            # Log warnings
            if result["critical_space_warning"]:
                logger.critical(f"Critical disk space: {used_percent:.1f}% used")
            elif result["low_space_warning"]:
                logger.warning(f"Low disk space: {used_percent:.1f}% used")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return {
                "error": str(e),
                "total_mb": 0,
                "used_mb": 0,
                "free_mb": 0,
                "used_percent": 0
            }
    
    def get_upload_stats(self) -> dict:
        """Get statistics about uploads."""
        try:
            from shared.file_cleanup import file_cleanup_service
            disk_usage = file_cleanup_service.get_disk_usage()
            disk_space = self.check_disk_space()
            
            return {
                "timestamp": disk_usage.get("timestamp"),
                "upload_directory": disk_usage.get("directories", {}).get("uploads", {}),
                "disk_space": disk_space,
                "limits": {
                    "max_upload_size_mb": round(self.max_size / (1024 * 1024), 2),
                    "allowed_mime_types": self.allowed_mime_types,
                    "retention_days": settings.FILE_RETENTION_DAYS
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting upload stats: {e}")
            return {"error": str(e)}


# Global instance
upload_validator = UploadValidator()


def validate_upload_file(file: UploadFile) -> dict:
    """
    Convenience function for validating uploaded files.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Validation result dictionary
        
    Raises:
        HTTPException: If validation fails
    """
    return upload_validator.validate_file(file) 