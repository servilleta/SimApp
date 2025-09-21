"""
Persistent Excel File Storage Service

Handles long-term storage of Excel files for completed simulations.
These files are kept separate from temporary uploads and are not subject
to regular cleanup, allowing users to access their historical simulations.
"""

import os
import shutil
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Set
from pathlib import Path

from config import settings
from database import get_db
from models import SimulationResult

logger = logging.getLogger(__name__)


class PersistentExcelStorage:
    """Service for managing long-term Excel file storage for simulations."""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.persistent_dir = Path(settings.PERSISTENT_EXCEL_DIR)
        self.retention_days = settings.SIMULATION_FILE_RETENTION_DAYS
        
        # Ensure persistent directory exists
        self.persistent_dir.mkdir(exist_ok=True)
        
    def move_to_persistent_storage(self, file_id: str, original_filename: str) -> Optional[str]:
        """
        Move an Excel file from temporary uploads to persistent storage.
        
        Args:
            file_id: The unique file identifier
            original_filename: Original filename of the Excel file
            
        Returns:
            Path to the persistent file if successful, None otherwise
        """
        try:
            # Find the source file in uploads directory
            source_pattern = f"{file_id}_*"
            source_files = list(self.upload_dir.glob(source_pattern))
            
            if not source_files:
                logger.warning(f"No source file found for file_id: {file_id}")
                return None
                
            source_file = source_files[0]  # Take the first match
            
            # Create persistent file path
            safe_filename = os.path.basename(original_filename)
            persistent_filename = f"{file_id}_{safe_filename}"
            persistent_path = self.persistent_dir / persistent_filename
            
            # Move file to persistent storage
            shutil.move(str(source_file), str(persistent_path))
            
            logger.info(f"📁 Moved Excel file to persistent storage: {file_id} -> {persistent_path}")
            return str(persistent_path)
            
        except Exception as e:
            logger.error(f"Failed to move file {file_id} to persistent storage: {e}")
            return None
    
    def get_persistent_file_path(self, file_id: str) -> Optional[str]:
        """
        Get the path to a persistent Excel file.
        
        Args:
            file_id: The unique file identifier
            
        Returns:
            Path to the file if it exists, None otherwise
        """
        try:
            # Look for files with this file_id
            pattern = f"{file_id}_*"
            matching_files = list(self.persistent_dir.glob(pattern))
            
            if matching_files:
                return str(matching_files[0])
                
            return None
            
        except Exception as e:
            logger.error(f"Error finding persistent file for {file_id}: {e}")
            return None
    
    def get_simulation_linked_file_ids(self) -> Set[str]:
        """
        Get file_ids that are linked to completed simulations in the database.
        These files should not be deleted by cleanup routines.
        
        Returns:
            Set of file_ids that are linked to simulations
        """
        linked_file_ids = set()
        
        try:
            db = next(get_db())
            
            # Get all simulations that have file_ids
            simulations = db.query(SimulationResult).filter(
                SimulationResult.file_id.isnot(None),
                SimulationResult.file_id != ''
            ).all()
            
            for sim in simulations:
                if sim.file_id:
                    linked_file_ids.add(sim.file_id)
                    
            logger.info(f"Found {len(linked_file_ids)} file_ids linked to simulations")
            
        except Exception as e:
            logger.error(f"Error getting simulation-linked file_ids: {e}")
        finally:
            try:
                db.close()
            except:
                pass
                
        return linked_file_ids
    
    def cleanup_old_persistent_files(self) -> Dict[str, Any]:
        """
        Clean up old persistent files that are no longer linked to simulations
        or are older than the retention period.
        
        Returns:
            Cleanup statistics
        """
        cleanup_stats = {
            "start_time": datetime.now().isoformat(),
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": [],
            "status": "completed"
        }
        
        try:
            # Get file_ids linked to simulations
            linked_file_ids = self.get_simulation_linked_file_ids()
            
            # Calculate cutoff date for unlinked files
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            logger.info(f"Cleaning persistent files older than {cutoff_date}")
            
            for file_path in self.persistent_dir.iterdir():
                if file_path.is_file():
                    try:
                        # Extract file_id from filename
                        file_id = file_path.stem.split('_')[0]
                        
                        # Check if file is linked to a simulation
                        if file_id in linked_file_ids:
                            logger.debug(f"Keeping simulation-linked file: {file_path}")
                            continue
                        
                        # Check file age for unlinked files
                        file_mtime = file_path.stat().st_mtime
                        if file_mtime < cutoff_timestamp:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            cleanup_stats["files_deleted"] += 1
                            cleanup_stats["bytes_freed"] += file_size
                            logger.info(f"Deleted old unlinked persistent file: {file_path}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing persistent file {file_path}: {e}")
                        cleanup_stats["errors"].append(f"Error processing {file_path}: {str(e)}")
            
            cleanup_stats["end_time"] = datetime.now().isoformat()
            
            logger.info(
                f"Persistent file cleanup completed: "
                f"{cleanup_stats['files_deleted']} files deleted, "
                f"{cleanup_stats['bytes_freed'] / (1024*1024):.1f} MB freed"
            )
            
        except Exception as e:
            logger.error(f"Error during persistent file cleanup: {e}")
            cleanup_stats["errors"].append(str(e))
            cleanup_stats["status"] = "error"
            
        return cleanup_stats
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for persistent Excel files."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "persistent_directory": str(self.persistent_dir),
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0,
            "linked_files": 0,
            "unlinked_files": 0
        }
        
        try:
            linked_file_ids = self.get_simulation_linked_file_ids()
            
            for file_path in self.persistent_dir.iterdir():
                if file_path.is_file():
                    stats["total_files"] += 1
                    file_size = file_path.stat().st_size
                    stats["total_size_bytes"] += file_size
                    
                    # Check if file is linked
                    file_id = file_path.stem.split('_')[0]
                    if file_id in linked_file_ids:
                        stats["linked_files"] += 1
                    else:
                        stats["unlinked_files"] += 1
            
            stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            stats["error"] = str(e)
            
        return stats


# Global instance
persistent_excel_storage = PersistentExcelStorage()


def move_simulation_file_to_persistent_storage(file_id: str, original_filename: str) -> Optional[str]:
    """
    Convenience function to move a simulation file to persistent storage.
    
    Args:
        file_id: The unique file identifier
        original_filename: Original filename of the Excel file
        
    Returns:
        Path to the persistent file if successful, None otherwise
    """
    return persistent_excel_storage.move_to_persistent_storage(file_id, original_filename)


def get_persistent_excel_file_path(file_id: str) -> Optional[str]:
    """
    Convenience function to get the path to a persistent Excel file.
    
    Args:
        file_id: The unique file identifier
        
    Returns:
        Path to the file if it exists, None otherwise
    """
    return persistent_excel_storage.get_persistent_file_path(file_id)

