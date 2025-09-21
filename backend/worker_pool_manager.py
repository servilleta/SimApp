#!/usr/bin/env python3
"""
🔄 Worker Pool Manager for Parallel Simulation Processing
Manages multiple worker processes for optimal CPU utilization.
"""

import asyncio
import multiprocessing
import concurrent.futures
import queue
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import psutil
import os

from multicore_worker_config import multicore_config

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SimulationTask:
    """
    Represents a simulation task in the worker queue
    """
    task_id: str
    simulation_config: Dict[str, Any]
    priority: int = 1
    created_at: float = None
    started_at: float = None
    completed_at: float = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class WorkerPoolManager:
    """
    Manages a pool of worker processes for parallel simulation execution
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multicore_config.recommended_workers
        self.active_workers = 0
        self.task_queue = asyncio.Queue()
        self.result_store: Dict[str, SimulationTask] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.is_running = False
        
        logger.info(f"🚀 Initializing WorkerPoolManager with {self.max_workers} workers")
    
    async def start(self):
        """
        Start the worker pool
        """
        if self.is_running:
            logger.warning("Worker pool already running")
            return
        
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=multiprocessing.get_context('spawn')
        )
        self.is_running = True
        
        # Start monitoring task
        asyncio.create_task(self._monitor_system_load())
        
        logger.info(f"✅ Worker pool started with {self.max_workers} processes")
    
    async def stop(self):
        """
        Stop the worker pool and cleanup
        """
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
        
        logger.info("🛑 Worker pool stopped")
    
    async def submit_simulation(self, task: SimulationTask) -> str:
        """
        Submit a simulation task to the worker pool
        """
        if not self.is_running:
            await self.start()
        
        # Store task
        self.result_store[task.task_id] = task
        
        # Add to queue
        await self.task_queue.put(task)
        
        # Process task immediately if workers available
        asyncio.create_task(self._process_next_task())
        
        logger.info(f"📝 Task {task.task_id} submitted to queue")
        return task.task_id
    
    async def _process_next_task(self):
        """
        Process the next task in the queue
        """
        if self.active_workers >= self.max_workers:
            return  # All workers busy
        
        try:
            # Get task from queue (non-blocking)
            task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return  # No tasks available
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.active_workers += 1
        
        # Submit to process pool
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.process_pool,
            self._run_simulation_worker,
            task.simulation_config
        )
        
        # Handle completion
        asyncio.create_task(self._handle_task_completion(task, future))
        
        logger.info(f"🔄 Task {task.task_id} started on worker process")
    
    async def _handle_task_completion(self, task: SimulationTask, future: asyncio.Future):
        """
        Handle task completion or failure
        """
        try:
            result = await future
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            duration = task.completed_at - task.started_at
            logger.info(f"✅ Task {task.task_id} completed in {duration:.2f}s")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            
            logger.error(f"❌ Task {task.task_id} failed: {e}")
        
        finally:
            self.active_workers -= 1
            
            # Process next task if available
            if not self.task_queue.empty():
                asyncio.create_task(self._process_next_task())
    
    @staticmethod
    def _run_simulation_worker(simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Worker function that runs in a separate process
        This function should import and run the actual simulation
        """
        import sys
        import os
        
        # Add the app directory to Python path
        sys.path.append('/app')
        sys.path.append('/home/paperspace/SimApp/backend')
        
        try:
            # Import simulation engine
            from simulation.engine import MonteCarloEngine
            
            # Create engine instance
            engine = MonteCarloEngine()
            
            # Run simulation
            result = engine.run_simulation(simulation_config)
            
            return {
                'success': True,
                'result': result,
                'worker_pid': os.getpid(),
                'worker_cpu_count': multiprocessing.cpu_count()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'worker_pid': os.getpid()
            }
    
    async def get_task_status(self, task_id: str) -> Optional[SimulationTask]:
        """
        Get the status of a task
        """
        return self.result_store.get(task_id)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get current queue and worker statistics
        """
        pending_tasks = sum(1 for task in self.result_store.values() 
                          if task.status == TaskStatus.PENDING)
        running_tasks = sum(1 for task in self.result_store.values() 
                          if task.status == TaskStatus.RUNNING)
        completed_tasks = sum(1 for task in self.result_store.values() 
                            if task.status == TaskStatus.COMPLETED)
        
        return {
            'max_workers': self.max_workers,
            'active_workers': self.active_workers,
            'available_workers': self.max_workers - self.active_workers,
            'queue_depth': self.task_queue.qsize(),
            'pending_tasks': pending_tasks,
            'running_tasks': running_tasks,
            'completed_tasks': completed_tasks,
            'total_tasks': len(self.result_store),
            'cpu_utilization': psutil.cpu_percent(interval=1),
            'memory_utilization': psutil.virtual_memory().percent
        }
    
    async def _monitor_system_load(self):
        """
        Monitor system load and adjust worker count if needed
        """
        while self.is_running:
            try:
                # Check if we should scale to Server 2
                scaling_rec = multicore_config.should_scale_to_server2()
                
                if scaling_rec['scale_needed']:
                    logger.warning(f"🚨 High load detected: {scaling_rec['reasons']}")
                    logger.info(f"💡 Consider starting Server 2 for additional capacity")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in load monitoring: {e}")
                await asyncio.sleep(60)  # Longer delay on error

# Global worker pool instance
worker_pool = WorkerPoolManager()

async def get_worker_pool() -> WorkerPoolManager:
    """
    Get the global worker pool instance
    """
    if not worker_pool.is_running:
        await worker_pool.start()
    return worker_pool

async def submit_parallel_simulation(simulation_config: Dict[str, Any]) -> str:
    """
    Convenience function to submit a simulation to the worker pool
    """
    import uuid
    
    task = SimulationTask(
        task_id=str(uuid.uuid4()),
        simulation_config=simulation_config
    )
    
    pool = await get_worker_pool()
    return await pool.submit_simulation(task)

async def get_simulation_result(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the result of a simulation task
    """
    pool = await get_worker_pool()
    task = await pool.get_task_status(task_id)
    
    if not task:
        return None
    
    return {
        'task_id': task_id,
        'status': task.status.value,
        'result': task.result,
        'error': task.error,
        'created_at': task.created_at,
        'started_at': task.started_at,
        'completed_at': task.completed_at,
        'duration': (task.completed_at - task.started_at) if task.completed_at and task.started_at else None
    }

if __name__ == "__main__":
    # Test the worker pool
    async def test_worker_pool():
        pool = WorkerPoolManager(max_workers=4)
        await pool.start()
        
        # Submit test tasks
        for i in range(8):
            task = SimulationTask(
                task_id=f"test_task_{i}",
                simulation_config={
                    'iterations': 1000,
                    'test_mode': True
                }
            )
            await pool.submit_simulation(task)
        
        # Wait a bit and check stats
        await asyncio.sleep(2)
        stats = await pool.get_queue_stats()
        print("📊 Worker Pool Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        await pool.stop()
    
    asyncio.run(test_worker_pool())
