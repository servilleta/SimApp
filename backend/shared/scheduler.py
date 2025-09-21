"""
Background Scheduler Service

Manages background jobs like file cleanup, memory monitoring, and maintenance tasks.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from config import settings
from shared.file_cleanup import run_scheduled_cleanup

logger = logging.getLogger(__name__)


class SchedulerService:
    """Service for managing background scheduled tasks."""
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.jobs_started = False
        
    def start_scheduler(self):
        """Start the background scheduler with all jobs."""
        if self.scheduler is not None:
            logger.warning("Scheduler already started")
            return
            
        try:
            self.scheduler = AsyncIOScheduler()
            
            # Add cleanup job if enabled
            if settings.FILE_CLEANUP_ENABLED:
                self._add_cleanup_job()
            
            # Add memory monitoring job
            self._add_memory_monitoring_job()
            
            # Start the scheduler
            self.scheduler.start()
            self.jobs_started = True
            
            logger.info("Background scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def stop_scheduler(self):
        """Stop the background scheduler."""
        if self.scheduler is not None:
            try:
                self.scheduler.shutdown(wait=False)
                self.scheduler = None
                self.jobs_started = False
                logger.info("Background scheduler stopped")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
    
    def _add_cleanup_job(self):
        """Add file cleanup job to scheduler."""
        try:
            # Run file cleanup every X hours
            self.scheduler.add_job(
                run_scheduled_cleanup,
                trigger=IntervalTrigger(hours=settings.CLEANUP_INTERVAL_HOURS),
                id='file_cleanup',
                name='File Cleanup Job',
                max_instances=1,  # Don't run multiple instances
                coalesce=True,    # Coalesce missed runs
                misfire_grace_time=30 * 60  # 30 minutes grace time
            )
            
            logger.info(f"File cleanup job scheduled every {settings.CLEANUP_INTERVAL_HOURS} hours")
            
        except Exception as e:
            logger.error(f"Failed to add cleanup job: {e}")
    
    def _add_memory_monitoring_job(self):
        """Add memory monitoring job to scheduler."""
        try:
            # Run memory check every 15 minutes
            self.scheduler.add_job(
                self._monitor_memory,
                trigger=IntervalTrigger(minutes=15),
                id='memory_monitor',
                name='Memory Monitoring Job',
                max_instances=1,
                coalesce=True
            )
            
            logger.info("Memory monitoring job scheduled every 15 minutes")
            
        except Exception as e:
            logger.error(f"Failed to add memory monitoring job: {e}")
    
    def _monitor_memory(self):
        """Monitor system memory usage and log warnings."""
        try:
            import psutil
            
            # Get memory info
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage for upload directory
            from shared.file_cleanup import file_cleanup_service
            disk_usage = file_cleanup_service.get_disk_usage()
            
            # Log memory status
            if memory_percent > 90:
                logger.critical(f"Critical memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
            else:
                logger.debug(f"Memory usage: {memory_percent:.1f}%")
            
            # Check disk usage
            for dir_name, dir_info in disk_usage.get("directories", {}).items():
                if "total_size_mb" in dir_info and dir_info["total_size_mb"] > 1000:  # 1GB
                    logger.warning(f"Large disk usage in {dir_name}: {dir_info['total_size_mb']:.1f} MB")
                    
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")
    
    def run_manual_cleanup(self) -> Dict[str, Any]:
        """Run file cleanup manually (not scheduled)."""
        try:
            logger.info("Running manual file cleanup")
            return run_scheduled_cleanup()
        except Exception as e:
            logger.error(f"Error in manual cleanup: {e}")
            return {"error": str(e), "status": "failed"}
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all scheduled jobs."""
        if not self.scheduler:
            return {"status": "not_started", "jobs": []}
        
        try:
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
            
            return {
                "status": "running" if self.jobs_started else "stopped",
                "job_count": len(jobs),
                "jobs": jobs
            }
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {"status": "error", "error": str(e)}
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a specific job."""
        try:
            if self.scheduler:
                self.scheduler.pause_job(job_id)
                logger.info(f"Paused job: {job_id}")
                return True
        except Exception as e:
            logger.error(f"Error pausing job {job_id}: {e}")
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a specific job."""
        try:
            if self.scheduler:
                self.scheduler.resume_job(job_id)
                logger.info(f"Resumed job: {job_id}")
                return True
        except Exception as e:
            logger.error(f"Error resuming job {job_id}: {e}")
        return False
    
    def add_job(self, func, trigger, **kwargs):
        """Add a custom job to the scheduler."""
        try:
            if self.scheduler:
                self.scheduler.add_job(func, trigger, **kwargs)
                logger.info(f"Added custom job: {kwargs.get('id', 'unnamed')}")
                return True
        except Exception as e:
            logger.error(f"Error adding custom job: {e}")
        return False


# Global instance
scheduler_service = SchedulerService() 