import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ProgressQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.last_progress = {}  # Track last progress per simulation
        self._running = True
        self._processor_task = None
        self._cleanup_task = None
        self._cleanup_interval = 300  # 5 minutes
    
    async def start(self):
        """Start the progress queue processor"""
        self._running = True
        self._processor_task = asyncio.create_task(self.process_queue())
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
    async def stop(self):
        """Stop the progress queue processor"""
        self._running = False
        if self._processor_task:
            await self._processor_task
        if self._cleanup_task:
            await self._cleanup_task
            
    async def add_progress(self, simulation_id: str, progress_data: Dict[str, Any]):
        """Add a progress update to the queue"""
        await self.queue.put({
            'simulation_id': simulation_id,
            'progress_data': progress_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    async def _interpolate_progress(self, simulation_id: str, start: float, end: float):
        """Interpolate between two progress values to smooth updates"""
        if simulation_id not in self.last_progress:
            self.last_progress[simulation_id] = 0
            
        current = self.last_progress[simulation_id]
        steps = max(1, int((end - start) / 2))  # 2% increments
        
        for i in range(steps):
            interpolated = start + ((end - start) * (i + 1) / steps)
            await self._update_frontend({
                'simulation_id': simulation_id,
                'progress_percentage': interpolated,
                'interpolated': True
            })
            await asyncio.sleep(0.02)  # 20ms between updates
            
    async def _update_frontend(self, progress_data: Dict[str, Any]):
        """Update the frontend with progress data"""
        try:
            from simulation.service import update_simulation_progress
            simulation_id = progress_data.pop('simulation_id')
            # CRITICAL FIX: Properly await the async update
            await update_simulation_progress(simulation_id, progress_data)
        except Exception as e:
            logger.error(f"Failed to update frontend progress: {e}")
            
    async def process_queue(self):
        """Process progress updates from the queue"""
        while self._running:
            try:
                item = await self.queue.get()
                simulation_id = item['simulation_id']
                progress_data = item['progress_data']
                current_progress = progress_data.get('progress_percentage', 0)
                
                # Get last progress for this simulation
                last_progress = self.last_progress.get(simulation_id, 0)
                
                # If progress jump is too large, interpolate
                if current_progress - last_progress > 10:
                    await self._interpolate_progress(
                        simulation_id, 
                        last_progress,
                        current_progress
                    )
                else:
                    # Update frontend directly
                    await self._update_frontend({
                        'simulation_id': simulation_id,
                        **progress_data
                    })
                
                # Update last progress
                self.last_progress[simulation_id] = current_progress
                
            except Exception as e:
                logger.error(f"Error processing progress update: {e}")
            finally:
                self.queue.task_done()
                
    async def _periodic_cleanup(self):
        """Periodically clean up completed simulations"""
        while self._running:
            try:
                self._cleanup_old_progress()
                await asyncio.sleep(self._cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                
    def _cleanup_old_progress(self):
        """Clean up progress tracking for completed simulations"""
        try:
            from shared.progress_store import get_progress
            to_remove = []
            
            for sim_id in self.last_progress.keys():
                progress = get_progress(sim_id)
                if not progress or progress.get('status') in ['completed', 'failed', 'cancelled']:
                    to_remove.append(sim_id)
                    
            for sim_id in to_remove:
                del self.last_progress[sim_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up old progress: {e}")

# Create a global progress queue instance
progress_queue = ProgressQueue()

# Start the queue processor when the module is imported
asyncio.create_task(progress_queue.start()) 