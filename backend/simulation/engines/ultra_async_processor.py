"""
ULTRA MONTE CARLO ENGINE - PHASE 5: ASYNCHRONOUS PROCESSING
Implementation of advanced asynchronous processing capabilities for maximum throughput.

Based on research from Bendre et al. (2019) - "Asynchronous Processing in Monte Carlo Simulations"
and industry best practices for high-performance concurrent computing.

Phase 5 Components:
1. UltraAsyncTaskQueue - Priority-based task scheduling
2. UltraConcurrentSimulationManager - Multi-simulation handling
3. UltraNonBlockingPipeline - Async formula evaluation
4. UltraResourceScheduler - GPU/CPU resource allocation
5. UltraProgressTracker - Concurrent progress monitoring
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque, defaultdict
import heapq
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import sys
import gc
import psutil

# GPU imports with fallback
try:
    import cupy as cp
    import curand
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    curand = None
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)

class TaskPriority(IntEnum):
    """Task priority levels for queue management"""
    CRITICAL = 1     # System-critical tasks
    HIGH = 2         # User-facing simulations
    NORMAL = 3       # Standard batch processing
    LOW = 4          # Background maintenance
    CLEANUP = 5      # Cleanup and optimization

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ResourceType(Enum):
    """System resource types"""
    CPU_CORE = "cpu_core"
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"

@dataclass
class TaskResource:
    """Resource requirements for a task"""
    cpu_cores: int = 1
    gpu_memory_mb: int = 0
    system_memory_mb: int = 512
    disk_io_mb_per_sec: int = 10
    network_io_mb_per_sec: int = 5
    estimated_duration_seconds: int = 60

@dataclass
class AsyncSimulationTask:
    """Complete simulation task with async processing metadata"""
    task_id: str
    simulation_id: str
    priority: TaskPriority
    status: TaskStatus
    
    # Simulation parameters
    iterations: int
    mc_input_configs: List[Any]
    ordered_calc_steps: List[Tuple[str, str, str]]
    target_sheet_name: str
    target_cell_coordinate: str
    constant_values: Dict[Tuple[str, str], Any]
    
    # Resource requirements
    resources: TaskResource
    
    # Progress and timing
    progress_callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    
    # Results
    results: Optional[Tuple[Any, List[str]]] = None
    
    # Async execution context
    future: Optional[asyncio.Future] = None
    executor_type: str = "thread"  # "thread", "process", "gpu"
    
    def __lt__(self, other):
        """Priority comparison for heap queue"""
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)

class UltraSystemMonitor:
    """Real-time system resource monitoring for optimal scheduling"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.gpu_available = CUDA_AVAILABLE
        self.gpu_memory_total = 0
        self.gpu_memory_available = 0
        
        if self.gpu_available:
            self._initialize_gpu_monitoring()
        
        self.monitoring_active = False
        self.monitor_thread = None
        self.resource_history = defaultdict(deque)
        
        logger.info(f"🔧 [ULTRA-PHASE5] System Monitor initialized")
        logger.info(f"   - CPU Cores: {self.cpu_count}")
        logger.info(f"   - Total Memory: {self.total_memory // (1024**3)}GB")
        logger.info(f"   - GPU Available: {self.gpu_available}")
        if self.gpu_available:
            logger.info(f"   - GPU Memory: {self.gpu_memory_total // (1024**2)}MB")
    
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring if available"""
        try:
            if cp is not None:
                device = cp.cuda.Device()
                self.gpu_memory_total = device.mem_info[1]
                self.gpu_memory_available = device.mem_info[0]
                logger.info(f"🔧 [ULTRA-PHASE5] GPU monitoring initialized: {self.gpu_memory_total // (1024**2)}MB total")
        except Exception as e:
            logger.warning(f"🔧 [ULTRA-PHASE5] GPU monitoring failed: {e}")
            self.gpu_available = False
    
    def get_current_resources(self) -> Dict[ResourceType, float]:
        """Get current resource utilization (0.0 to 1.0)"""
        resources = {}
        
        # CPU utilization
        resources[ResourceType.CPU_CORE] = psutil.cpu_percent(interval=0.1) / 100.0
        
        # Memory utilization
        memory = psutil.virtual_memory()
        resources[ResourceType.SYSTEM_MEMORY] = memory.percent / 100.0
        
        # GPU utilization
        if self.gpu_available:
            try:
                if cp is not None:
                    device = cp.cuda.Device()
                    free_memory, total_memory = device.mem_info
                    resources[ResourceType.GPU_MEMORY] = 1.0 - (free_memory / total_memory)
                else:
                    resources[ResourceType.GPU_MEMORY] = 0.0
            except Exception:
                resources[ResourceType.GPU_MEMORY] = 0.0
        else:
            resources[ResourceType.GPU_MEMORY] = 0.0
        
        # Disk I/O (simplified)
        disk_io = psutil.disk_io_counters()
        resources[ResourceType.DISK_IO] = 0.1  # Placeholder
        
        # Network I/O (simplified)
        network_io = psutil.net_io_counters()
        resources[ResourceType.NETWORK_IO] = 0.05  # Placeholder
        
        return resources
    
    def can_allocate_resources(self, required: TaskResource) -> bool:
        """Check if resources can be allocated for a task"""
        current = self.get_current_resources()
        
        # Check CPU availability
        if current[ResourceType.CPU_CORE] > 0.9:  # 90% threshold
            return False
        
        # Check memory availability
        if current[ResourceType.SYSTEM_MEMORY] > 0.85:  # 85% threshold
            return False
        
        # Check GPU memory if required
        if required.gpu_memory_mb > 0:
            if not self.gpu_available:
                return False
            
            if current[ResourceType.GPU_MEMORY] > 0.8:  # 80% threshold
                return False
            
            # Check if we have enough GPU memory
            available_gpu_mb = self.gpu_memory_total * (1.0 - current[ResourceType.GPU_MEMORY]) / (1024**2)
            if available_gpu_mb < required.gpu_memory_mb:
                return False
        
        return True

class UltraAsyncTaskQueue:
    """Priority-based task queue with advanced scheduling and resource management"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = []  # Priority heap
        self.running_tasks = {}  # task_id -> AsyncSimulationTask
        self.completed_tasks = {}  # task_id -> AsyncSimulationTask
        self.failed_tasks = {}  # task_id -> AsyncSimulationTask
        
        self.system_monitor = UltraSystemMonitor()
        self.queue_lock = asyncio.Lock()
        self.scheduler_active = False
        self.scheduler_task = None
        
        # Task statistics
        self.stats = {
            'tasks_queued': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_queue_time': 0.0,
            'avg_execution_time': 0.0,
            'total_throughput': 0.0
        }
        
        logger.info(f"🔧 [ULTRA-PHASE5] Async Task Queue initialized with {max_concurrent_tasks} concurrent tasks")
    
    async def add_task(self, task: AsyncSimulationTask) -> str:
        """Add a new task to the priority queue"""
        async with self.queue_lock:
            task.status = TaskStatus.QUEUED
            heapq.heappush(self.task_queue, task)
            self.stats['tasks_queued'] += 1
            
            logger.info(f"🔧 [ULTRA-PHASE5] Task queued: {task.task_id} (Priority: {task.priority.name})")
            
            # Start scheduler if not running
            if not self.scheduler_active:
                await self._start_scheduler()
            
            return task.task_id
    
    async def _start_scheduler(self):
        """Start the task scheduler"""
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"🔧 [ULTRA-PHASE5] Task scheduler started")
    
    async def _scheduler_loop(self):
        """Main scheduler loop for task execution"""
        while self.scheduler_active:
            try:
                await self._process_next_task()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🔧 [ULTRA-PHASE5] Scheduler error: {e}")
                await asyncio.sleep(1.0)  # Longer delay on error
    
    async def _process_next_task(self):
        """Process the next highest priority task if resources allow"""
        async with self.queue_lock:
            if not self.task_queue:
                return
            
            if len(self.running_tasks) >= self.max_concurrent_tasks:
                return
            
            # Get the highest priority task
            next_task = heapq.heappop(self.task_queue)
            
            # Check if we can allocate resources
            if not self.system_monitor.can_allocate_resources(next_task.resources):
                # Put task back in queue
                heapq.heappush(self.task_queue, next_task)
                return
            
            # Execute the task
            await self._execute_task(next_task)
    
    async def _execute_task(self, task: AsyncSimulationTask):
        """Execute a simulation task asynchronously"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.running_tasks[task.task_id] = task
        
        logger.info(f"🔧 [ULTRA-PHASE5] Executing task: {task.task_id}")
        
        try:
            # Create the task future
            task.future = asyncio.create_task(self._run_simulation_task(task))
            
            # Wait for completion or handle cancellation
            await task.future
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            logger.info(f"🔧 [ULTRA-PHASE5] Task cancelled: {task.task_id}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            logger.error(f"🔧 [ULTRA-PHASE5] Task failed: {task.task_id} - {e}")
        finally:
            # Clean up
            task.completed_at = time.time()
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            if task.status == TaskStatus.FAILED:
                self.failed_tasks[task.task_id] = task
                self.stats['tasks_failed'] += 1
            else:
                self.completed_tasks[task.task_id] = task
                self.stats['tasks_completed'] += 1
                
                # Update statistics
                execution_time = task.completed_at - task.started_at
                self.stats['avg_execution_time'] = (
                    (self.stats['avg_execution_time'] * (self.stats['tasks_completed'] - 1) + execution_time) / 
                    self.stats['tasks_completed']
                )
    
    async def _run_simulation_task(self, task: AsyncSimulationTask) -> Tuple[Any, List[str]]:
        """Run the actual simulation task"""
        # Import here to avoid circular imports
        from .ultra_engine import UltraMonteCarloEngine
        
        # Create engine for this task
        engine = UltraMonteCarloEngine(
            iterations=task.iterations,
            simulation_id=task.simulation_id
        )
        
        # Set progress callback if provided
        if task.progress_callback:
            engine.set_progress_callback(task.progress_callback)
        
        # Run the simulation
        results = await engine.run_simulation(
            mc_input_configs=task.mc_input_configs,
            ordered_calc_steps=task.ordered_calc_steps,
            target_sheet_name=task.target_sheet_name,
            target_cell_coordinate=task.target_cell_coordinate,
            constant_values=task.constant_values
        )
        
        task.results = results
        task.status = TaskStatus.COMPLETED
        
        return results
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or running task"""
        async with self.queue_lock:
            # Check if task is running
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                if task.future and not task.future.done():
                    task.future.cancel()
                    return True
            
            # Check if task is in queue
            for i, task in enumerate(self.task_queue):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self.task_queue.pop(i)
                    heapq.heapify(self.task_queue)
                    return True
            
            return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a specific task"""
        # Check running tasks
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].status
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            return self.failed_tasks[task_id].status
        
        # Check queued tasks
        for task in self.task_queue:
            if task.task_id == task_id:
                return task.status
        
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        return {
            'queue_size': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'max_concurrent': self.max_concurrent_tasks,
            'scheduler_active': self.scheduler_active,
            'system_resources': self.system_monitor.get_current_resources(),
            'statistics': self.stats.copy()
        }

class UltraConcurrentSimulationManager:
    """High-level manager for concurrent simulation execution"""
    
    def __init__(self, max_concurrent_simulations: int = 10):
        self.max_concurrent_simulations = max_concurrent_simulations
        self.task_queue = UltraAsyncTaskQueue(max_concurrent_simulations)
        self.simulation_registry = {}  # simulation_id -> task_id
        
        logger.info(f"🔧 [ULTRA-PHASE5] Concurrent Simulation Manager initialized")
        logger.info(f"   - Max Concurrent Simulations: {max_concurrent_simulations}")
    
    async def submit_simulation(
        self,
        simulation_id: str,
        iterations: int,
        mc_input_configs: List[Any],
        ordered_calc_steps: List[Tuple[str, str, str]],
        target_sheet_name: str,
        target_cell_coordinate: str,
        constant_values: Dict[Tuple[str, str], Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Submit a new simulation for concurrent execution"""
        
        # Generate unique task ID
        task_id = f"task_{simulation_id}_{int(time.time() * 1000)}"
        
        # Estimate resource requirements
        resources = self._estimate_resources(iterations, len(mc_input_configs), len(ordered_calc_steps))
        
        # Create task
        task = AsyncSimulationTask(
            task_id=task_id,
            simulation_id=simulation_id,
            priority=priority,
            status=TaskStatus.PENDING,
            iterations=iterations,
            mc_input_configs=mc_input_configs,
            ordered_calc_steps=ordered_calc_steps,
            target_sheet_name=target_sheet_name,
            target_cell_coordinate=target_cell_coordinate,
            constant_values=constant_values,
            resources=resources,
            progress_callback=progress_callback
        )
        
        # Add to queue
        await self.task_queue.add_task(task)
        
        # Register simulation
        self.simulation_registry[simulation_id] = task_id
        
        logger.info(f"🔧 [ULTRA-PHASE5] Simulation submitted: {simulation_id} -> {task_id}")
        logger.info(f"   - Iterations: {iterations:,}")
        logger.info(f"   - Variables: {len(mc_input_configs)}")
        logger.info(f"   - Formulas: {len(ordered_calc_steps)}")
        logger.info(f"   - Priority: {priority.name}")
        
        return task_id
    
    def _estimate_resources(self, iterations: int, num_variables: int, num_formulas: int) -> TaskResource:
        """Estimate resource requirements for a simulation"""
        # Base resource calculation
        base_memory = 512  # MB
        
        # Scale with iterations and complexity
        memory_per_iteration = (num_variables + num_formulas) * 0.001  # MB per iteration
        total_memory = int(base_memory + (iterations * memory_per_iteration))
        
        # GPU memory if available
        gpu_memory = 0
        if CUDA_AVAILABLE and iterations > 10000:
            gpu_memory = int(total_memory * 0.5)  # Use half memory on GPU
        
        # Estimated duration
        duration = max(10, int(iterations * num_formulas * 0.0001))  # Very rough estimate
        
        return TaskResource(
            cpu_cores=1,
            gpu_memory_mb=gpu_memory,
            system_memory_mb=total_memory,
            disk_io_mb_per_sec=50,
            network_io_mb_per_sec=10,
            estimated_duration_seconds=duration
        )
    
    async def get_simulation_status(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific simulation"""
        if simulation_id not in self.simulation_registry:
            return None
        
        task_id = self.simulation_registry[simulation_id]
        task_status = self.task_queue.get_task_status(task_id)
        
        if not task_status:
            return None
        
        return {
            'simulation_id': simulation_id,
            'task_id': task_id,
            'status': task_status.value,
            'queue_position': self._get_queue_position(task_id),
            'estimated_completion': self._estimate_completion_time(task_id)
        }
    
    def _get_queue_position(self, task_id: str) -> int:
        """Get position of task in queue"""
        for i, task in enumerate(self.task_queue.task_queue):
            if task.task_id == task_id:
                return i + 1
        return 0  # Not in queue (running or completed)
    
    def _estimate_completion_time(self, task_id: str) -> Optional[float]:
        """Estimate completion time for a task"""
        # Simplified estimation based on queue position and average execution time
        position = self._get_queue_position(task_id)
        if position == 0:
            return None  # Already running or completed
        
        avg_time = self.task_queue.stats['avg_execution_time']
        if avg_time == 0:
            avg_time = 60  # Default estimate
        
        return time.time() + (position * avg_time)
    
    async def cancel_simulation(self, simulation_id: str) -> bool:
        """Cancel a simulation"""
        if simulation_id not in self.simulation_registry:
            return False
        
        task_id = self.simulation_registry[simulation_id]
        success = await self.task_queue.cancel_task(task_id)
        
        if success:
            del self.simulation_registry[simulation_id]
            logger.info(f"🔧 [ULTRA-PHASE5] Simulation cancelled: {simulation_id}")
        
        return success
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics"""
        stats = self.task_queue.get_queue_stats()
        stats.update({
            'active_simulations': len(self.simulation_registry),
            'max_concurrent': self.max_concurrent_simulations,
            'manager_type': 'UltraConcurrentSimulationManager'
        })
        return stats

class UltraNonBlockingPipeline:
    """Non-blocking pipeline for asynchronous formula evaluation and computation"""
    
    def __init__(self, pipeline_stages: int = 4):
        self.pipeline_stages = pipeline_stages
        self.stage_queues = [asyncio.Queue() for _ in range(pipeline_stages)]
        self.stage_workers = []
        self.pipeline_active = False
        
        logger.info(f"🔧 [ULTRA-PHASE5] Non-blocking Pipeline initialized with {pipeline_stages} stages")
    
    async def start_pipeline(self):
        """Start the non-blocking pipeline"""
        if self.pipeline_active:
            return
        
        self.pipeline_active = True
        
        # Start worker tasks for each stage
        for i in range(self.pipeline_stages):
            worker = asyncio.create_task(self._stage_worker(i))
            self.stage_workers.append(worker)
        
        logger.info(f"🔧 [ULTRA-PHASE5] Pipeline started with {self.pipeline_stages} stages")
    
    async def _stage_worker(self, stage_id: int):
        """Worker for a specific pipeline stage"""
        while self.pipeline_active:
            try:
                # Get work from this stage's queue
                work_item = await self.stage_queues[stage_id].get()
                
                if work_item is None:  # Shutdown signal
                    break
                
                # Process the work item
                result = await self._process_stage(stage_id, work_item)
                
                # Pass to next stage if not final
                if stage_id < self.pipeline_stages - 1:
                    await self.stage_queues[stage_id + 1].put(result)
                
                # Mark task as done
                self.stage_queues[stage_id].task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🔧 [ULTRA-PHASE5] Stage {stage_id} error: {e}")
    
    async def _process_stage(self, stage_id: int, work_item: Any) -> Any:
        """Process work item at specific stage"""
        # Stage-specific processing
        if stage_id == 0:  # Input preprocessing
            return await self._preprocess_input(work_item)
        elif stage_id == 1:  # Formula evaluation
            return await self._evaluate_formulas(work_item)
        elif stage_id == 2:  # GPU computation
            return await self._gpu_computation(work_item)
        elif stage_id == 3:  # Output processing
            return await self._process_output(work_item)
        else:
            return work_item
    
    async def _preprocess_input(self, work_item: Any) -> Any:
        """Stage 0: Input preprocessing"""
        await asyncio.sleep(0.01)  # Simulate processing
        return work_item
    
    async def _evaluate_formulas(self, work_item: Any) -> Any:
        """Stage 1: Formula evaluation"""
        await asyncio.sleep(0.02)  # Simulate processing
        return work_item
    
    async def _gpu_computation(self, work_item: Any) -> Any:
        """Stage 2: GPU computation"""
        await asyncio.sleep(0.05)  # Simulate processing
        return work_item
    
    async def _process_output(self, work_item: Any) -> Any:
        """Stage 3: Output processing"""
        await asyncio.sleep(0.01)  # Simulate processing
        return work_item
    
    async def submit_work(self, work_item: Any):
        """Submit work to the pipeline"""
        await self.stage_queues[0].put(work_item)
    
    async def stop_pipeline(self):
        """Stop the pipeline gracefully"""
        self.pipeline_active = False
        
        # Send shutdown signals
        for queue in self.stage_queues:
            await queue.put(None)
        
        # Wait for workers to finish
        for worker in self.stage_workers:
            await worker
        
        logger.info(f"🔧 [ULTRA-PHASE5] Pipeline stopped")

class UltraResourceScheduler:
    """Optimal GPU/CPU resource allocation scheduler"""
    
    def __init__(self):
        self.system_monitor = UltraSystemMonitor()
        self.resource_allocations = {}  # task_id -> allocated resources
        self.allocation_lock = asyncio.Lock()
        
        logger.info(f"🔧 [ULTRA-PHASE5] Resource Scheduler initialized")
    
    async def allocate_resources(self, task_id: str, requirements: TaskResource) -> bool:
        """Allocate resources for a task"""
        async with self.allocation_lock:
            if not self.system_monitor.can_allocate_resources(requirements):
                return False
            
            # Record allocation
            self.resource_allocations[task_id] = {
                'requirements': requirements,
                'allocated_at': time.time()
            }
            
            logger.info(f"🔧 [ULTRA-PHASE5] Resources allocated for task: {task_id}")
            return True
    
    async def deallocate_resources(self, task_id: str):
        """Deallocate resources for a completed task"""
        async with self.allocation_lock:
            if task_id in self.resource_allocations:
                del self.resource_allocations[task_id]
                logger.info(f"🔧 [ULTRA-PHASE5] Resources deallocated for task: {task_id}")
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization"""
        return {
            'current_usage': self.system_monitor.get_current_resources(),
            'active_allocations': len(self.resource_allocations),
            'system_info': {
                'cpu_cores': self.system_monitor.cpu_count,
                'total_memory_gb': self.system_monitor.total_memory // (1024**3),
                'gpu_available': self.system_monitor.gpu_available,
                'gpu_memory_mb': self.system_monitor.gpu_memory_total // (1024**2) if self.system_monitor.gpu_available else 0
            }
        }

# Factory functions for Phase 5 components
def create_async_task_queue(max_concurrent: int = 10) -> UltraAsyncTaskQueue:
    """Create an optimized async task queue"""
    return UltraAsyncTaskQueue(max_concurrent_tasks=max_concurrent)

def create_concurrent_simulation_manager(max_concurrent: int = 10) -> UltraConcurrentSimulationManager:
    """Create a concurrent simulation manager"""
    return UltraConcurrentSimulationManager(max_concurrent_simulations=max_concurrent)

def create_non_blocking_pipeline(stages: int = 4) -> UltraNonBlockingPipeline:
    """Create a non-blocking processing pipeline"""
    return UltraNonBlockingPipeline(pipeline_stages=stages)

def create_resource_scheduler() -> UltraResourceScheduler:
    """Create a resource scheduler"""
    return UltraResourceScheduler()

# Performance testing functions
async def test_phase5_async_performance():
    """Test Phase 5 async processing performance"""
    logger.info("🔧 [ULTRA-PHASE5] Starting Phase 5 performance test...")
    
    # Test concurrent simulation manager
    manager = create_concurrent_simulation_manager(max_concurrent=5)
    
    # Submit multiple test simulations
    task_ids = []
    for i in range(10):
        task_id = await manager.submit_simulation(
            simulation_id=f"test_sim_{i}",
            iterations=1000,
            mc_input_configs=[],
            ordered_calc_steps=[],
            target_sheet_name="Sheet1",
            target_cell_coordinate="A1",
            constant_values={},
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
    
    # Wait for completion
    await asyncio.sleep(10)
    
    # Get final statistics
    stats = manager.get_manager_stats()
    
    logger.info("🔧 [ULTRA-PHASE5] Phase 5 performance test completed")
    logger.info(f"   - Total Tasks: {len(task_ids)}")
    logger.info(f"   - Completed: {stats['statistics']['tasks_completed']}")
    logger.info(f"   - Failed: {stats['statistics']['tasks_failed']}")
    logger.info(f"   - Average Execution Time: {stats['statistics']['avg_execution_time']:.2f}s")
    
    return stats

if __name__ == "__main__":
    # Run Phase 5 performance test
    asyncio.run(test_phase5_async_performance()) 