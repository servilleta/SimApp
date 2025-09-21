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
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import subprocess
import json
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
    s3_key: Optional[str]
    error_message: Optional[str]
    retention_days: int

class BackupService:
    """Comprehensive backup and disaster recovery service"""
    
    def __init__(self):
        self.backup_dir = Path(os.getenv("BACKUP_DIR", "/app/backups"))
        self.retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        self.s3_enabled = os.getenv("BACKUP_S3_ENABLED", "false").lower() == "true"
        self.s3_bucket = os.getenv("BACKUP_S3_BUCKET")
        self.s3_client = None
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if enabled
        if self.s3_enabled and self.s3_bucket:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=os.getenv("AWS_REGION", "us-east-1")
                )
                logger.info("S3 backup client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self.s3_enabled = False
        
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
            s3_key=None,
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
                "-h", parsed.hostname,
                "-p", str(parsed.port or 5432),
                "-U", parsed.username,
                "-d", parsed.path.lstrip('/'),
                "--no-password",
                "--verbose",
                "--clean",
                "--create"
            ]
            
            # Set password via environment
            env = os.environ.copy()
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
            
            # Upload to S3 if enabled
            if self.s3_enabled:
                await self._upload_to_s3(backup_info)
                
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
            s3_key=None,
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
            
            # Upload to S3 if enabled
            if self.s3_enabled:
                await self._upload_to_s3(backup_info)
                
            return backup_info
            
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            backup_info.completed_at = datetime.now()
            logger.error(f"Files backup failed: {backup_id} - {e}")
            raise
            
    async def create_configuration_backup(self, backup_id: Optional[str] = None) -> BackupInfo:
        """Create a configuration backup"""
        if not backup_id:
            backup_id = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        backup_info = BackupInfo(
            id=backup_id,
            type=BackupType.CONFIGURATION,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            file_path="",
            file_size=None,
            s3_key=None,
            error_message=None,
            retention_days=self.retention_days
        )
        
        self.active_backups[backup_id] = backup_info
        
        try:
            backup_info.status = BackupStatus.RUNNING
            
            # Create configuration backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"config_backup_{timestamp}.tar.gz"
            file_path = self.backup_dir / filename
            
            # Configuration files to backup
            config_files = [
                "/app/docker-compose.deploy.yml",
                "/app/nginx.conf",
                "/app/monitoring/prometheus.yml",
                "/app/alembic.ini",
                "/app/production.env"
            ]
            
            # Filter existing files
            existing_files = [f for f in config_files if os.path.exists(f)]
            
            # Create tar archive
            with tarfile.open(file_path, 'w:gz') as tar:
                for file_path_to_backup in existing_files:
                    if os.path.exists(file_path_to_backup):
                        tar.add(file_path_to_backup, arcname=os.path.basename(file_path_to_backup))
                        
            # Get file size
            file_size = file_path.stat().st_size
            
            backup_info.file_path = str(file_path)
            backup_info.file_size = file_size
            backup_info.status = BackupStatus.COMPLETED
            backup_info.completed_at = datetime.now()
            
            logger.info(f"Configuration backup completed: {backup_id} ({file_size} bytes)")
            
            # Upload to S3 if enabled
            if self.s3_enabled:
                await self._upload_to_s3(backup_info)
                
            return backup_info
            
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            backup_info.completed_at = datetime.now()
            logger.error(f"Configuration backup failed: {backup_id} - {e}")
            raise
            
    async def create_full_backup(self, backup_id: Optional[str] = None) -> BackupInfo:
        """Create a full system backup"""
        if not backup_id:
            backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        backup_info = BackupInfo(
            id=backup_id,
            type=BackupType.FULL,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            file_path="",
            file_size=None,
            s3_key=None,
            error_message=None,
            retention_days=self.retention_days
        )
        
        self.active_backups[backup_id] = backup_info
        
        try:
            backup_info.status = BackupStatus.RUNNING
            
            # Create individual backups
            db_backup = await self.create_database_backup(f"{backup_id}_db")
            files_backup = await self.create_files_backup(f"{backup_id}_files")
            config_backup = await self.create_configuration_backup(f"{backup_id}_config")
            
            # Create combined archive
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"full_backup_{timestamp}.tar.gz"
            file_path = self.backup_dir / filename
            
            # Combine all backups
            with tarfile.open(file_path, 'w:gz') as tar:
                tar.add(db_backup.file_path, arcname=f"database/{os.path.basename(db_backup.file_path)}")
                tar.add(files_backup.file_path, arcname=f"files/{os.path.basename(files_backup.file_path)}")
                tar.add(config_backup.file_path, arcname=f"config/{os.path.basename(config_backup.file_path)}")
                
            # Get file size
            file_size = file_path.stat().st_size
            
            backup_info.file_path = str(file_path)
            backup_info.file_size = file_size
            backup_info.status = BackupStatus.COMPLETED
            backup_info.completed_at = datetime.now()
            
            logger.info(f"Full backup completed: {backup_id} ({file_size} bytes)")
            
            # Upload to S3 if enabled
            if self.s3_enabled:
                await self._upload_to_s3(backup_info)
                
            # Clean up individual backup files
            os.remove(db_backup.file_path)
            os.remove(files_backup.file_path)
            os.remove(config_backup.file_path)
            
            return backup_info
            
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            backup_info.completed_at = datetime.now()
            logger.error(f"Full backup failed: {backup_id} - {e}")
            raise
            
    async def _upload_to_s3(self, backup_info: BackupInfo):
        """Upload backup to S3"""
        if not self.s3_client or not self.s3_bucket:
            return
            
        try:
            s3_key = f"backups/{backup_info.type.value}/{os.path.basename(backup_info.file_path)}"
            
            # Upload file
            self.s3_client.upload_file(
                backup_info.file_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'Metadata': {
                        'backup_id': backup_info.id,
                        'backup_type': backup_info.type.value,
                        'created_at': backup_info.created_at.isoformat()
                    }
                }
            )
            
            backup_info.s3_key = s3_key
            logger.info(f"Backup uploaded to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Failed to upload backup to S3: {e}")
            # Don't fail the backup if S3 upload fails
            
    async def restore_database(self, backup_file: str) -> bool:
        """Restore database from backup"""
        try:
            logger.info(f"Starting database restore from: {backup_file}")
            
            # Get database connection info
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not configured")
                
            # Parse database URL
            import urllib.parse
            parsed = urllib.parse.urlparse(db_url)
            
            # Create psql command
            restore_cmd = [
                "psql",
                "-h", parsed.hostname,
                "-p", str(parsed.port or 5432),
                "-U", parsed.username,
                "-d", parsed.path.lstrip('/'),
                "--no-password"
            ]
            
            # Set password via environment
            env = os.environ.copy()
            env["PGPASSWORD"] = parsed.password
            
            # Execute restore
            with gzip.open(backup_file, 'rt') as f:
                process = subprocess.run(
                    restore_cmd,
                    stdin=f,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=True,
                    text=True
                )
            
            logger.info("Database restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
            
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
                        
            # Clean up S3 files if enabled
            if self.s3_enabled and self.s3_client:
                await self._cleanup_s3_backups(cutoff_date)
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            
    async def _cleanup_s3_backups(self, cutoff_date: datetime):
        """Clean up old S3 backup files"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix="backups/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        self.s3_client.delete_object(
                            Bucket=self.s3_bucket,
                            Key=obj['Key']
                        )
                        logger.info(f"Deleted old S3 backup: {obj['Key']}")
                        
        except Exception as e:
            logger.error(f"S3 backup cleanup failed: {e}")
            
    async def get_backup_status(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup status"""
        return self.active_backups.get(backup_id)
        
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
                    s3_key=None,
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
            
    async def schedule_backups(self):
        """Schedule automatic backups"""
        backup_schedule = os.getenv("BACKUP_SCHEDULE", "0 2 * * *")  # Daily at 2 AM
        
        # This would integrate with a scheduler like APScheduler
        # For now, we'll implement a simple loop
        while True:
            try:
                # Check if it's time for a backup (simplified)
                now = datetime.now()
                if now.hour == 2 and now.minute == 0:  # 2 AM
                    logger.info("Starting scheduled backup")
                    await self.create_full_backup()
                    
                # Check for cleanup
                if now.hour == 3 and now.minute == 0:  # 3 AM
                    logger.info("Starting backup cleanup")
                    await self.cleanup_old_backups()
                    
            except Exception as e:
                logger.error(f"Scheduled backup error: {e}")
                
            # Wait 1 minute before checking again
            await asyncio.sleep(60)

# Global backup service instance
backup_service = BackupService() 