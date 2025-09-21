"""
Production Backup and Disaster Recovery Service
Phase 5: Production Deployment
"""

import os
import logging
import asyncio
import shutil
import tarfile
import gzip
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import subprocess
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BackupType(Enum):
    DATABASE = "database"
    FILES = "files"
    CONFIGURATION = "configuration"
    FULL = "full"

class BackupStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BackupInfo:
    id: str
    type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime]
    file_path: str
    file_size: Optional[int]
    error_message: Optional[str]
    retention_days: int

class BackupService:
    """Comprehensive backup and disaster recovery service"""
    
    def __init__(self):
        self.backup_dir = Path(os.getenv("BACKUP_DIR", "/app/backups"))
        self.retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.active_backups: Dict[str, BackupInfo] = {}
        
    async def create_database_backup(self, backup_id: Optional[str] = None) -> BackupInfo:
        """Create a database backup"""
        if not backup_id:
            backup_id = f"db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        backup_info = BackupInfo(
            id=backup_id,
            type=BackupType.DATABASE,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            file_path="",
            file_size=None,
            error_message=None,
            retention_days=self.retention_days
        )
        
        self.active_backups[backup_id] = backup_info
        
        try:
            backup_info.status = BackupStatus.RUNNING
            
            # Create database dump
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"database_backup_{timestamp}.sql.gz"
            file_path = self.backup_dir / filename
            
            # Get database connection info
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not configured")
                
            # Parse database URL
            import urllib.parse
            parsed = urllib.parse.urlparse(db_url)
            
            # Create pg_dump command
            dump_cmd = [
                "pg_dump",
                "-h", parsed.hostname or "localhost",
                "-p", str(parsed.port or 5432),
                "-U", parsed.username or "postgres",
                "-d", parsed.path.lstrip('/') if parsed.path else "montecarlo_db",
                "--no-password",
                "--verbose",
                "--clean",
                "--create"
            ]
            
            # Set password via environment
            env = os.environ.copy()
            if parsed.password:
                env["PGPASSWORD"] = parsed.password
            
            # Execute dump and compress
            with gzip.open(file_path, 'wt') as f:
                process = subprocess.run(
                    dump_cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=True,
                    text=True
                )
            
            # Get file size
            file_size = file_path.stat().st_size
            
            backup_info.file_path = str(file_path)
            backup_info.file_size = file_size
            backup_info.status = BackupStatus.COMPLETED
            backup_info.completed_at = datetime.now()
            
            logger.info(f"Database backup completed: {backup_id} ({file_size} bytes)")
            return backup_info
            
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            backup_info.completed_at = datetime.now()
            logger.error(f"Database backup failed: {backup_id} - {e}")
            raise
            
    async def create_files_backup(self, backup_id: Optional[str] = None) -> BackupInfo:
        """Create a files backup"""
        if not backup_id:
            backup_id = f"files_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        backup_info = BackupInfo(
            id=backup_id,
            type=BackupType.FILES,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            file_path="",
            file_size=None,
            error_message=None,
            retention_days=self.retention_days
        )
        
        self.active_backups[backup_id] = backup_info
        
        try:
            backup_info.status = BackupStatus.RUNNING
            
            # Create tar archive of important directories
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"files_backup_{timestamp}.tar.gz"
            file_path = self.backup_dir / filename
            
            # Directories to backup
            backup_dirs = [
                "/app/uploads",
                "/app/saved_simulations_files",
                "/app/logs"
            ]
            
            # Filter existing directories
            existing_dirs = [d for d in backup_dirs if os.path.exists(d)]
            
            if not existing_dirs:
                raise ValueError("No directories found to backup")
            
            # Create tar archive
            with tarfile.open(file_path, 'w:gz') as tar:
                for dir_path in existing_dirs:
                    if os.path.exists(dir_path):
                        tar.add(dir_path, arcname=os.path.basename(dir_path))
                        
            # Get file size
            file_size = file_path.stat().st_size
            
            backup_info.file_path = str(file_path)
            backup_info.file_size = file_size
            backup_info.status = BackupStatus.COMPLETED
            backup_info.completed_at = datetime.now()
            
            logger.info(f"Files backup completed: {backup_id} ({file_size} bytes)")
            return backup_info
            
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            backup_info.completed_at = datetime.now()
            logger.error(f"Files backup failed: {backup_id} - {e}")
            raise
            
    async def cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Clean up local files
            for backup_file in self.backup_dir.glob("*"):
                if backup_file.is_file():
                    file_date = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        backup_file.unlink()
                        logger.info(f"Deleted old backup file: {backup_file}")
                        
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            
    async def list_backups(self) -> List[BackupInfo]:
        """List all backups"""
        backups = []
        
        # Local backups
        for backup_file in self.backup_dir.glob("*"):
            if backup_file.is_file():
                backup_info = BackupInfo(
                    id=backup_file.stem,
                    type=self._get_backup_type_from_filename(backup_file.name),
                    status=BackupStatus.COMPLETED,
                    created_at=datetime.fromtimestamp(backup_file.stat().st_mtime),
                    completed_at=datetime.fromtimestamp(backup_file.stat().st_mtime),
                    file_path=str(backup_file),
                    file_size=backup_file.stat().st_size,
                    error_message=None,
                    retention_days=self.retention_days
                )
                backups.append(backup_info)
                
        return sorted(backups, key=lambda x: x.created_at, reverse=True)
        
    def _get_backup_type_from_filename(self, filename: str) -> BackupType:
        """Determine backup type from filename"""
        if "database" in filename:
            return BackupType.DATABASE
        elif "files" in filename:
            return BackupType.FILES
        elif "config" in filename:
            return BackupType.CONFIGURATION
        elif "full" in filename:
            return BackupType.FULL
        else:
            return BackupType.FILES  # Default

# Global backup service instance
backup_service = BackupService() 