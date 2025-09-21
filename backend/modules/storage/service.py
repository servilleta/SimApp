"""
Storage service module - implements StorageServiceProtocol

Handles file storage, retrieval, and results storage.
This service can be easily extracted to a microservice later.
"""

import logging
import os
import uuid
import json
from typing import Optional, Dict
from datetime import datetime

from ..base import BaseService, StorageServiceProtocol


logger = logging.getLogger(__name__)


class StorageService(BaseService, StorageServiceProtocol):
    """Storage service implementation"""
    
    def __init__(self, upload_dir: str = "uploads", results_dir: str = "results"):
        super().__init__("storage")
        self.upload_dir = upload_dir
        self.results_dir = results_dir
        
    async def initialize(self) -> None:
        """Initialize the storage service"""
        await super().initialize()
        
        # Create directories if they don't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Storage service initialized - uploads: {self.upload_dir}, results: {self.results_dir}")
    
    async def save_file(self, file_content: bytes, filename: str, user_id: int) -> str:
        """Save uploaded file securely"""
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Create safe filename
        safe_filename = f"{file_id}_{filename}"
        file_path = os.path.join(self.upload_dir, safe_filename)
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Save metadata
        metadata = {
            "file_id": file_id,
            "original_filename": filename,
            "user_id": user_id,
            "uploaded_at": datetime.utcnow().isoformat(),
            "file_path": file_path,
            "size": len(file_content)
        }
        
        metadata_path = os.path.join(self.upload_dir, f"{file_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"File saved: {file_id} for user {user_id}")
        return file_id
    
    async def get_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve file content"""
        # Find file by ID
        metadata_path = os.path.join(self.upload_dir, f"{file_id}_metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            file_path = metadata["file_path"]
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {e}")
            return None
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete file"""
        try:
            metadata_path = os.path.join(self.upload_dir, f"{file_id}_metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Delete actual file
                file_path = metadata["file_path"]
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Delete metadata
                os.remove(metadata_path)
                
                logger.info(f"File deleted: {file_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    async def save_results(self, simulation_id: str, results: Dict) -> bool:
        """Save simulation results"""
        try:
            results_path = os.path.join(self.results_dir, f"{simulation_id}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved for simulation {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results for {simulation_id}: {e}")
            return False
    
    async def get_results(self, simulation_id: str) -> Optional[Dict]:
        """Get simulation results"""
        try:
            results_path = os.path.join(self.results_dir, f"{simulation_id}.json")
            
            if not os.path.exists(results_path):
                return None
            
            with open(results_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to retrieve results for {simulation_id}: {e}")
            return None
    
    async def get_file_metadata(self, file_id: str) -> Optional[Dict]:
        """Get file metadata"""
        try:
            metadata_path = os.path.join(self.upload_dir, f"{file_id}_metadata.json")
            
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to retrieve metadata for {file_id}: {e}")
            return None
    
    async def cleanup_old_files(self, days: int = 30):
        """Cleanup old files"""
        # This would implement cleanup logic based on file age
        pass 