"""
ULTRA MONTE CARLO ENGINE - PHASE 5: ASYNCHRONOUS PROCESSING CORE
Core components for advanced asynchronous processing capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque
import heapq
import multiprocessing
import psutil

# GPU imports with fallback
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
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

@dataclass
class TaskResource:
    """Resource requirements for a task"""
    cpu_cores: int = 1
    gpu_memory_mb: int = 0
    system_memory_mb: int = 512
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
        
        if self.gpu_available:
            self._initialize_gpu_monitoring()
        
        logger.info(f"🔧 [ULTRA-PHASE5] System Monitor initialized")
        logger.info(f"   - CPU Cores: {self.cpu_count}")
        logger.info(f"   - Total Memory: {self.total_memory // (1024**3)}GB")
        logger.info(f"   - GPU Available: {self.gpu_available}")
    
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring if available"""
        try:
            if cp is not None:
                device = cp.cuda.Device()
                self.gpu_memory_total = device.mem_info[1]
                logger.info(f"🔧 [ULTRA-PHASE5] GPU monitoring initialized: {self.gpu_memory_total // (1024**2)}MB total")
        except Exception as e:
            logger.warning(f"🔧 [ULTRA-PHASE5] GPU monitoring failed: {e}")
            self.gpu_available = False
    
    def get_current_resources(self) -> Dict[str, float]:
        """Get current resource utilization (0.0 to 1.0)"""
        resources = {}
        
        # CPU utilization
        resources['cpu'] = psutil.cpu_percent(interval=0.1) / 100.0
        
        # Memory utilization
        memory = psutil.virtual_memory()
        resources['memory'] = memory.percent / 100.0
        
        # GPU utilization
        if self.gpu_available:
            try:
                if cp is not None:
                    device = cp.cuda.Device()
                    free_memory, total_memory = device.mem_info
                    resources['gpu_memory'] = 1.0 - (free_memory / total_memory)
                else:
                    resources['gpu_memory'] = 0.0
            except Exception:
                resources['gpu_memory'] = 0.0
        else:
            resources['gpu_memory'] = 0.0
        
        return resources
    
    def can_allocate_resources(self, required: TaskResource) -> bool:
        """Check if resources can be allocated for a task"""
        current = self.get_current_resources()
        
        # Check CPU availability
        if current['cpu'] > 0.9:  # 90% threshold
            return False
        
        # Check memory availability
        if current['memory'] > 0.85:  # 85% threshold
            return False
        
        # Check GPU memory if required
        if required.gpu_memory_mb > 0:
            if not self.gpu_available:
                return False
            
            if current['gpu_memory'] > 0.8:  # 80% threshold
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
            'avg_execution_time': 0.0
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
            
            # Wait for completion
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
                if task.started_at and task.completed_at:
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