# Background Progress Update System
# High-performance non-blocking progress updates

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from collections import deque
import threading

logger = logging.getLogger(__name__)

class BackgroundProgressManager:
    """🚀 High-performance background progress update system"""
    
    def __init__(self):
        self.update_queue = asyncio.Queue(maxsize=1000)  # Large queue for high throughput
        self.batch_queue = deque(maxlen=100)  # Batch updates for efficiency
        self.is_running = False
        self.batch_size = 10  # Process updates in batches
        self.batch_timeout = 0.1  # Max wait time for batch (100ms)
        self.last_batch_time = time.time()
        self.update_count = 0
        self.error_count = 0
        
        # Performance metrics
        self.metrics = {
            'updates_processed': 0,
            'batches_processed': 0,
            'average_batch_size': 0,
            'errors': 0,
            'queue_high_water_mark': 0
        }
    
    async def start(self):
        """Start the background update processor"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🚀 Starting background progress update system")
        
        # Start background tasks
        asyncio.create_task(self._process_updates())
        asyncio.create_task(self._batch_processor())
        asyncio.create_task(self._metrics_reporter())
    
    async def stop(self):
        """Stop the background update processor"""
        self.is_running = False
        logger.info("🛑 Stopping background progress update system")
        
        # Process remaining updates
        await self._flush_remaining_updates()
    
    async def queue_update(self, simulation_id: str, progress_data: Dict[str, Any], priority: str = "normal"):
        """🚀 Queue a progress update for background processing"""
        try:
            update_item = {
                'simulation_id': simulation_id,
                'progress_data': progress_data,
                'timestamp': time.time(),
                'priority': priority,
                'retry_count': 0
            }
            
            # Try to queue update without blocking
            try:
                self.update_queue.put_nowait(update_item)
                self.update_count += 1
                
                # Track queue metrics
                current_size = self.update_queue.qsize()
                if current_size > self.metrics['queue_high_water_mark']:
                    self.metrics['queue_high_water_mark'] = current_size
                    
            except asyncio.QueueFull:
                logger.warning(f"⚠️ Progress update queue full, dropping update for {simulation_id}")
                self.error_count += 1
                
        except Exception as e:
            logger.error(f"❌ Error queuing progress update for {simulation_id}: {e}")
            self.error_count += 1
    
    async def _process_updates(self):
        """🔄 Main background update processor"""
        while self.is_running:
            try:
                # Get update with timeout to allow periodic checks
                try:
                    update_item = await asyncio.wait_for(
                        self.update_queue.get(), 
                        timeout=1.0
                    )
                    self.batch_queue.append(update_item)
                    
                except asyncio.TimeoutError:
                    # No updates received, continue loop
                    continue
                
                # Check if we should process batch
                should_process = (
                    len(self.batch_queue) >= self.batch_size or
                    time.time() - self.last_batch_time > self.batch_timeout
                )
                
                if should_process and self.batch_queue:
                    await self._process_batch()
                    
            except Exception as e:
                logger.error(f"❌ Error in background update processor: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    async def _process_batch(self):
        """📦 Process a batch of updates efficiently"""
        if not self.batch_queue:
            return
        
        batch = list(self.batch_queue)
        self.batch_queue.clear()
        self.last_batch_time = time.time()
        
        try:
            # Import progress store here to avoid circular imports
            from shared.progress_store import _progress_store
            
            # Process batch of updates
            update_tasks = []
            for update_item in batch:
                task = self._process_single_update(
                    _progress_store,
                    update_item['simulation_id'],
                    update_item['progress_data']
                )
                update_tasks.append(task)
            
            # Execute all updates concurrently
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            # Count successful updates
            successful_updates = sum(1 for r in results if not isinstance(r, Exception))
            
            # Update metrics
            self.metrics['updates_processed'] += successful_updates
            self.metrics['batches_processed'] += 1
            self.metrics['average_batch_size'] = (
                self.metrics['updates_processed'] / max(1, self.metrics['batches_processed'])
            )
            
            logger.debug(f"📦 Processed batch: {successful_updates}/{len(batch)} updates successful")
            
        except Exception as e:
            logger.error(f"❌ Error processing update batch: {e}")
            self.metrics['errors'] += 1
    
    async def _process_single_update(self, progress_store, simulation_id: str, progress_data: Dict[str, Any]):
        """Process a single progress update"""
        try:
            await progress_store.set_progress_async(simulation_id, progress_data, bypass_merge=True)
            return True
        except Exception as e:
            logger.error(f"❌ Error updating progress for {simulation_id}: {e}")
            return False
    
    async def _batch_processor(self):
        """⏰ Periodic batch processor for timeout-based batching"""
        while self.is_running:
            try:
                await asyncio.sleep(self.batch_timeout)
                
                # Process batch if timeout reached and we have updates
                if (self.batch_queue and 
                    time.time() - self.last_batch_time > self.batch_timeout):
                    await self._process_batch()
                    
            except Exception as e:
                logger.error(f"❌ Error in batch processor: {e}")
    
    async def _metrics_reporter(self):
        """📊 Periodic metrics reporting"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Report every 30 seconds
                
                if self.metrics['updates_processed'] > 0:
                    logger.info(
                        f"📊 Background Progress Metrics: "
                        f"Updates: {self.metrics['updates_processed']}, "
                        f"Batches: {self.metrics['batches_processed']}, "
                        f"Avg Batch Size: {self.metrics['average_batch_size']:.1f}, "
                        f"Errors: {self.metrics['errors']}, "
                        f"Queue Peak: {self.metrics['queue_high_water_mark']}"
                    )
                    
            except Exception as e:
                logger.error(f"❌ Error in metrics reporter: {e}")
    
    async def _flush_remaining_updates(self):
        """🔄 Flush any remaining updates during shutdown"""
        logger.info("🔄 Flushing remaining progress updates...")
        
        try:
            # Process any remaining batched updates
            if self.batch_queue:
                await self._process_batch()
            
            # Process any remaining queued updates
            remaining_updates = []
            while not self.update_queue.empty():
                try:
                    update_item = self.update_queue.get_nowait()
                    remaining_updates.append(update_item)
                except asyncio.QueueEmpty:
                    break
            
            if remaining_updates:
                self.batch_queue.extend(remaining_updates)
                await self._process_batch()
                
            logger.info(f"✅ Flushed {len(remaining_updates)} remaining updates")
            
        except Exception as e:
            logger.error(f"❌ Error flushing remaining updates: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.metrics,
            'queue_size': self.update_queue.qsize(),
            'batch_queue_size': len(self.batch_queue),
            'is_running': self.is_running,
            'update_rate': self.metrics['updates_processed'] / max(1, time.time() - getattr(self, 'start_time', time.time()))
        }


# Global background progress manager instance
_background_manager = BackgroundProgressManager()

# Convenience functions for easy import
async def start_background_progress():
    """Start the background progress system"""
    await _background_manager.start()

async def stop_background_progress():
    """Stop the background progress system"""
    await _background_manager.stop()

async def queue_progress_update(simulation_id: str, progress_data: Dict[str, Any], priority: str = "normal"):
    """Queue a progress update for background processing"""
    await _background_manager.queue_update(simulation_id, progress_data, priority)

def get_background_metrics() -> Dict[str, Any]:
    """Get background update metrics"""
    return _background_manager.get_metrics()
