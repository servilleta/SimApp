"""
File Cleanup Service

Handles automatic cleanup of old files, temporary data, and simulation artifacts
to prevent disk space issues in production.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from pathlib import Path

from config import settings
from shared.progress_store import _progress_store
from shared.persistent_excel_storage import persistent_excel_storage

logger = logging.getLogger(__name__)


class FileCleanupService:
    """Service for managing file cleanup operations."""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.saved_simulations_dir = Path(settings.SAVED_SIMULATIONS_DIR)
        self.retention_days = settings.FILE_RETENTION_DAYS
        self.active_simulations = {}
        
    def get_protected_file_ids(self) -> Set[str]:
        """
        Get a set of file_ids that should be protected from cleanup.
        This includes both active simulations and completed simulation files.
        """
        protected_file_ids = set()
        
        # Get active simulation file_ids
        logger.info("CLEANUP_DEBUG: Fetching active file IDs...")
        try:
            progress_keys = _progress_store.get_all_progress_keys()
            logger.info(f"CLEANUP_DEBUG: Found {len(progress_keys)} progress keys in Redis.")
            
            for key in progress_keys:
                logger.debug(f"CLEANUP_DEBUG: Processing key: {key}")
                try:
                    # Extract sim_id from key like "simulation:progress:some-uuid"
                    sim_id = key.split(':')[-1]
                    metadata = _progress_store.get_simulation_metadata(sim_id)
                    
                    if metadata and 'file_id' in metadata:
                        file_id = metadata['file_id']
                        protected_file_ids.add(file_id)
                        logger.info(f"CLEANUP_DEBUG: Found active file_id {file_id} for sim {sim_id}")
                    else:
                        logger.warning(f"CLEANUP_DEBUG: No metadata or file_id found for sim {sim_id}")

                except Exception as e:
                    logger.error(f"CLEANUP_DEBUG: Error processing key {key}: {e}")

        except Exception as e:
            logger.error(f"Could not get active simulation file IDs: {e}")
        
        # Get simulation-linked file_ids from database
        logger.info("CLEANUP_DEBUG: Fetching simulation-linked file IDs...")
        try:
            simulation_linked_ids = persistent_excel_storage.get_simulation_linked_file_ids()
            protected_file_ids.update(simulation_linked_ids)
            logger.info(f"CLEANUP_DEBUG: Added {len(simulation_linked_ids)} simulation-linked file IDs")
        except Exception as e:
            logger.error(f"Could not get simulation-linked file IDs: {e}")
        
        logger.info(f"CLEANUP_DEBUG: Returning {len(protected_file_ids)} protected file IDs.")
        return protected_file_ids
        
    def cleanup_old_files(self) -> Dict[str, Any]:
        """
        Clean up old files from upload and temporary directories.
        
        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {
            "start_time": datetime.now().isoformat(),
            "upload_files_deleted": 0,
            "upload_bytes_freed": 0,
            "temp_files_deleted": 0,
            "temp_bytes_freed": 0,
            "errors": [],
            "total_bytes_freed": 0
        }
        
        if not settings.FILE_CLEANUP_ENABLED:
            logger.info("File cleanup is disabled in settings")
            cleanup_stats["status"] = "disabled"
            return cleanup_stats
            
        try:
            # Get file IDs that should be protected from deletion
            protected_file_ids = self.get_protected_file_ids()
            logger.info(f"Found {len(protected_file_ids)} protected files to exclude from cleanup.")

            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            logger.info(f"Starting file cleanup - removing files older than {cutoff_date}")
            
            # Clean upload directory
            upload_stats = self._cleanup_directory(
                self.upload_dir, 
                cutoff_timestamp,
                "upload files",
                exclude_ids=protected_file_ids
            )
            cleanup_stats["upload_files_deleted"] = upload_stats["files_deleted"]
            cleanup_stats["upload_bytes_freed"] = upload_stats["bytes_freed"]
            
            # Clean temporary files
            temp_stats = self._cleanup_temp_files(cutoff_timestamp)
            cleanup_stats["temp_files_deleted"] = temp_stats["files_deleted"]
            cleanup_stats["temp_bytes_freed"] = temp_stats["bytes_freed"]
            
            # Calculate totals
            cleanup_stats["total_bytes_freed"] = (
                cleanup_stats["upload_bytes_freed"] + 
                cleanup_stats["temp_bytes_freed"]
            )
            
            cleanup_stats["end_time"] = datetime.now().isoformat()
            cleanup_stats["status"] = "completed"
            
            logger.info(
                f"File cleanup completed: "
                f"{cleanup_stats['upload_files_deleted'] + cleanup_stats['temp_files_deleted']} files deleted, "
                f"{cleanup_stats['total_bytes_freed'] / (1024*1024):.1f} MB freed"
            )
            
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")
            cleanup_stats["errors"].append(str(e))
            cleanup_stats["status"] = "error"
            
        return cleanup_stats
    
    def _cleanup_directory(self, directory: Path, cutoff_timestamp: float, description: str, exclude_ids: Set[str] = None) -> Dict[str, int]:
        """Clean up files in a specific directory, excluding active files."""
        stats = {"files_deleted": 0, "bytes_freed": 0}
        exclude_ids = exclude_ids or set()
        
        if not directory.exists():
            logger.debug(f"Directory {directory} does not exist, skipping cleanup")
            return stats
            
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    try:
                        # Extract file_id from filename (e.g., "some-uuid.xlsx" or "some-uuid_filename.xlsx")
                        file_id = file_path.stem.split('_')[0]
                        if file_id in exclude_ids:
                            logger.debug(f"Skipping active file: {file_path}")
                            continue

                        # Check file age
                        file_mtime = file_path.stat().st_mtime
                        if file_mtime < cutoff_timestamp:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            stats["files_deleted"] += 1
                            stats["bytes_freed"] += file_size
                            logger.debug(f"Deleted old {description}: {file_path}")
                            
                    except (OSError, IOError) as e:
                        logger.warning(f"Could not delete file {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")
            
        return stats
    
    def _cleanup_temp_files(self, cutoff_timestamp: float) -> Dict[str, int]:
        """Clean up temporary files and cache data."""
        stats = {"files_deleted": 0, "bytes_freed": 0}
        
        # Common temporary file patterns
        temp_patterns = [
            "*.tmp",
            "*.temp", 
            "*~",
            ".DS_Store",
            "Thumbs.db",
            "*.pyc",
            "__pycache__/*"
        ]
        
        try:
            # Clean temp files in upload directory
            for pattern in temp_patterns:
                for temp_file in self.upload_dir.glob(pattern):
                    if temp_file.is_file():
                        try:
                            file_mtime = temp_file.stat().st_mtime
                            if file_mtime < cutoff_timestamp:
                                file_size = temp_file.stat().st_size
                                temp_file.unlink()
                                stats["files_deleted"] += 1
                                stats["bytes_freed"] += file_size
                                logger.debug(f"Deleted temp file: {temp_file}")
                        except (OSError, IOError) as e:
                            logger.warning(f"Could not delete temp file {temp_file}: {e}")
                            
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
            
        return stats
    
    def cleanup_simulation_results(self, max_results: int = 1000) -> Dict[str, Any]:
        """
        Clean up old simulation results from memory store.
        
        Args:
            max_results: Maximum number of results to keep in memory
            
        Returns:
            Cleanup statistics
        """
        from simulation.service import SIMULATION_RESULTS_STORE
        
        cleanup_stats = {
            "start_time": datetime.now().isoformat(),
            "results_before": len(SIMULATION_RESULTS_STORE),
            "results_deleted": 0,
            "status": "completed"
        }
        
        try:
            if len(SIMULATION_RESULTS_STORE) > max_results:
                # Get all results with timestamps
                results_with_time = []
                for sim_id, result in SIMULATION_RESULTS_STORE.items():
                    created_at = getattr(result, 'created_at', None)
                    if created_at:
                        results_with_time.append((sim_id, created_at))
                
                # Sort by creation time (oldest first)
                results_with_time.sort(key=lambda x: x[1])
                
                # Delete oldest results
                to_delete = len(SIMULATION_RESULTS_STORE) - max_results
                for i in range(to_delete):
                    sim_id = results_with_time[i][0]
                    if sim_id in SIMULATION_RESULTS_STORE:
                        del SIMULATION_RESULTS_STORE[sim_id]
                        cleanup_stats["results_deleted"] += 1
                
                logger.info(f"Cleaned up {cleanup_stats['results_deleted']} old simulation results")
            
            cleanup_stats["results_after"] = len(SIMULATION_RESULTS_STORE)
            cleanup_stats["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error cleaning simulation results: {e}")
            cleanup_stats["status"] = "error"
            cleanup_stats["error"] = str(e)
            
        return cleanup_stats
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics for monitored directories."""
        usage_stats = {
            "timestamp": datetime.now().isoformat(),
            "directories": {}
        }
        
        directories_to_check = [
            ("uploads", self.upload_dir),
            ("saved_simulations", self.saved_simulations_dir)
        ]
        
        for name, directory in directories_to_check:
            if directory.exists():
                total_size = 0
                file_count = 0
                
                try:
                    for file_path in directory.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1
                            
                    usage_stats["directories"][name] = {
                        "path": str(directory),
                        "total_size_bytes": total_size,
                        "total_size_mb": round(total_size / (1024 * 1024), 2),
                        "file_count": file_count
                    }
                    
                except Exception as e:
                    logger.error(f"Error calculating disk usage for {directory}: {e}")
                    usage_stats["directories"][name] = {
                        "path": str(directory),
                        "error": str(e)
                    }
            else:
                usage_stats["directories"][name] = {
                    "path": str(directory),
                    "exists": False
                }
                
        return usage_stats
    
    def force_cleanup_user_files(self, user_id: int) -> Dict[str, Any]:
        """
        Force cleanup of all files for a specific user.
        Used when user account is deleted.
        """
        cleanup_stats = {
            "user_id": user_id,
            "start_time": datetime.now().isoformat(),
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": []
        }
        
        try:
            # This would integrate with user management
            # For now, it's a placeholder for future implementation
            logger.info(f"Force cleanup for user {user_id} - feature to be implemented")
            cleanup_stats["status"] = "not_implemented"
            
        except Exception as e:
            logger.error(f"Error in force cleanup for user {user_id}: {e}")
            cleanup_stats["errors"].append(str(e))
            cleanup_stats["status"] = "error"
            
        return cleanup_stats


# Global instance
file_cleanup_service = FileCleanupService()


def run_scheduled_cleanup():
    """Function to be called by the scheduler."""
    logger.info("Running scheduled file cleanup")
    result = file_cleanup_service.cleanup_old_files()
    
    # Also cleanup simulation results
    sim_result = file_cleanup_service.cleanup_simulation_results()
    
    return {
        "file_cleanup": result,
        "simulation_cleanup": sim_result
    } 