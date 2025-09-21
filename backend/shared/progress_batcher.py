"""
🚀 PROGRESS BATCHING SYSTEM
Reduces Redis load by batching high-frequency progress updates from ultra engine
"""
import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ProgressBatcher:
    """
    Batches progress updates to reduce Redis connection pressure.
    Particularly useful for ultra engine which sends 200+ progress updates.
    """
    
    def __init__(self, batch_interval: float = 0.1, max_batch_size: int = 20):
        """
        Args:
            batch_interval: How often to flush batches (seconds)
            max_batch_size: Maximum updates per batch before forcing flush
        """
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        
        # Batched updates per simulation
        self._batched_updates: Dict[str, Dict[str, Any]] = {}
        self._batch_timestamps: Dict[str, float] = {}
        self._update_counts: Dict[str, int] = defaultdict(int)
        
        # Background task management
        self._batch_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.total_updates_received = 0
        self.total_batches_sent = 0
        self.total_updates_saved = 0  # Updates saved by batching
        
        logger.info(f"🚀 [BATCHER] Initialized with interval={batch_interval}s, max_size={max_batch_size}")
    
    async def start(self):
        """Start the background batching task"""
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._batch_processor())
            logger.info("🚀 [BATCHER] Background processor started")
    
    async def stop(self):
        """Stop the background batching task and flush remaining updates"""
        self._shutdown_event.set()
        if self._batch_task:
            await self._batch_task
        await self._flush_all_batches()
        logger.info("🚀 [BATCHER] Stopped and flushed all batches")
    
    async def add_update(self, simulation_id: str, progress_data: Dict[str, Any]) -> bool:
        """
        Add a progress update to the batch.
        
        Returns:
            True if update was batched, False if it should be sent immediately
        """
        self.total_updates_received += 1
        current_time = time.time()
        
        # Initialize batch for new simulation
        if simulation_id not in self._batched_updates:
            self._batched_updates[simulation_id] = {}
            self._batch_timestamps[simulation_id] = current_time
        
        # Merge update into batch (new data overwrites old)
        self._batched_updates[simulation_id].update(progress_data)
        self._update_counts[simulation_id] += 1
        
        # Check if we need to force flush this simulation
        should_flush = (
            self._update_counts[simulation_id] >= self.max_batch_size or
            (current_time - self._batch_timestamps[simulation_id]) >= self.batch_interval
        )
        
        if should_flush:
            await self._flush_simulation(simulation_id)
            return False  # Tell caller this was sent immediately
        
        return True  # Update was batched
    
    async def _batch_processor(self):
        """Background task that processes batches periodically"""
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self.batch_interval)
                await self._flush_old_batches()
        except asyncio.CancelledError:
            logger.info("🚀 [BATCHER] Background processor cancelled")
        except Exception as e:
            logger.error(f"❌ [BATCHER] Background processor error: {e}")
    
    async def _flush_old_batches(self):
        """Flush batches that have exceeded the time threshold"""
        current_time = time.time()
        simulation_ids_to_flush = []
        
        for simulation_id, timestamp in self._batch_timestamps.items():
            if (current_time - timestamp) >= self.batch_interval:
                simulation_ids_to_flush.append(simulation_id)
        
        for simulation_id in simulation_ids_to_flush:
            await self._flush_simulation(simulation_id)
    
    async def _flush_simulation(self, simulation_id: str):
        """Flush the batch for a specific simulation"""
        if simulation_id not in self._batched_updates:
            return
        
        batch_data = self._batched_updates.pop(simulation_id)
        batch_count = self._update_counts.pop(simulation_id, 0)
        self._batch_timestamps.pop(simulation_id, None)
        
        if not batch_data:
            return
        
        # Send the batched update
        try:
            from shared.progress_store import _progress_store
            await _progress_store.set_progress_async(
                simulation_id, 
                batch_data, 
                bypass_merge=True  # We already merged in the batch
            )
            
            self.total_batches_sent += 1
            self.total_updates_saved += max(0, batch_count - 1)  # -1 because we still send 1 update
            
            logger.debug(f"🚀 [BATCHER] Flushed batch for {simulation_id}: {batch_count} updates merged")
            
        except Exception as e:
            logger.error(f"❌ [BATCHER] Failed to flush batch for {simulation_id}: {e}")
    
    async def _flush_all_batches(self):
        """Flush all remaining batches"""
        simulation_ids = list(self._batched_updates.keys())
        for simulation_id in simulation_ids:
            await self._flush_simulation(simulation_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        active_batches = len(self._batched_updates)
        efficiency = (self.total_updates_saved / max(1, self.total_updates_received)) * 100
        
        return {
            "total_updates_received": self.total_updates_received,
            "total_batches_sent": self.total_batches_sent,
            "total_updates_saved": self.total_updates_saved,
            "active_batches": active_batches,
            "efficiency_percent": round(efficiency, 1),
            "running": self._batch_task is not None and not self._batch_task.done()
        }

# Global batcher instance
_progress_batcher: Optional[ProgressBatcher] = None

async def get_progress_batcher() -> ProgressBatcher:
    """Get or create the global progress batcher"""
    global _progress_batcher
    if _progress_batcher is None:
        _progress_batcher = ProgressBatcher(batch_interval=0.5, max_batch_size=20)
        await _progress_batcher.start()
    return _progress_batcher

async def batch_progress_update(simulation_id: str, progress_data: Dict[str, Any]) -> bool:
    """
    Add a progress update to the batch.
    
    Returns:
        True if update was batched, False if it should be sent immediately
    """
    batcher = await get_progress_batcher()
    return await batcher.add_update(simulation_id, progress_data)

async def flush_simulation_batch(simulation_id: str):
    """Force flush any pending batch for a specific simulation"""
    global _progress_batcher
    if _progress_batcher:
        await _progress_batcher._flush_simulation(simulation_id)

async def shutdown_progress_batcher():
    """Shutdown the global progress batcher"""
    global _progress_batcher
    if _progress_batcher:
        await _progress_batcher.stop()
        _progress_batcher = None
