"""
ULTRA MONTE CARLO ENGINE - PHASE 5: RESOURCE SCHEDULER
Optimal GPU/CPU resource allocation scheduler for concurrent simulations.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing
import psutil
from collections import defaultdict, deque

# GPU imports with fallback
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """System resource types"""
    CPU_CORE = "cpu_core"
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"

@dataclass
class ResourceRequest:
    """Resource allocation request"""
    task_id: str
    cpu_cores: int = 1
    gpu_memory_mb: int = 0
    system_memory_mb: int = 512
    disk_io_mb_per_sec: int = 10
    network_io_mb_per_sec: int = 5
    estimated_duration_seconds: int = 60
    priority: int = 5  # Lower is higher priority
    requested_at: float = field(default_factory=time.time)

@dataclass
class ResourceAllocation:
    """Active resource allocation"""
    task_id: str
    allocated_resources: ResourceRequest
    allocated_at: float
    expected_completion: float
    actual_usage: Dict[ResourceType, float] = field(default_factory=dict)

class UltraSystemMonitor:
    """Enhanced system resource monitoring with historical tracking"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.cpu_count = multiprocessing.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.gpu_available = CUDA_AVAILABLE
        self.gpu_memory_total = 0
        
        # Historical tracking
        self.resource_history = defaultdict(lambda: deque(maxlen=60))  # 60 seconds of history
        self.monitoring_active = False
        self.monitor_task = None
        
        if self.gpu_available:
            self._initialize_gpu_monitoring()
        
        logger.info(f"🔧 [ULTRA-PHASE5] Enhanced System Monitor initialized")
        logger.info(f"   - CPU Cores: {self.cpu_count}")
        logger.info(f"   - Total Memory: {self.total_memory // (1024**3)}GB")
        logger.info(f"   - GPU Available: {self.gpu_available}")
        if self.gpu_available:
            logger.info(f"   - GPU Memory: {self.gpu_memory_total // (1024**2)}MB")
    
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities"""
        try:
            if cp is not None:
                device = cp.cuda.Device()
                self.gpu_memory_total = device.mem_info[1]
                logger.info(f"🔧 [ULTRA-PHASE5] GPU monitoring initialized: {self.gpu_memory_total // (1024**2)}MB total")
        except Exception as e:
            logger.warning(f"🔧 [ULTRA-PHASE5] GPU monitoring failed: {e}")
            self.gpu_available = False
    
    async def start_monitoring(self):
        """Start continuous resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"🔧 [ULTRA-PHASE5] Resource monitoring started")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                current_resources = self._get_current_resources()
                timestamp = time.time()
                
                # Store in history
                for resource_type, value in current_resources.items():
                    self.resource_history[resource_type].append((timestamp, value))
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🔧 [ULTRA-PHASE5] Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def _get_current_resources(self) -> Dict[ResourceType, float]:
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
        
        # Disk and Network I/O (simplified)
        resources[ResourceType.DISK_IO] = 0.1  # Placeholder
        resources[ResourceType.NETWORK_IO] = 0.05  # Placeholder
        
        return resources
    
    def get_resource_trends(self, lookback_seconds: int = 30) -> Dict[ResourceType, Dict[str, float]]:
        """Get resource usage trends over time"""
        trends = {}
        current_time = time.time()
        
        for resource_type, history in self.resource_history.items():
            if not history:
                trends[resource_type] = {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'trend': 0.0}
                continue
            
            # Filter recent history
            recent_values = [
                value for timestamp, value in history 
                if current_time - timestamp <= lookback_seconds
            ]
            
            if recent_values:
                trends[resource_type] = {
                    'avg': sum(recent_values) / len(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values),
                    'trend': self._calculate_trend(recent_values)
                }
            else:
                trends[resource_type] = {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'trend': 0.0}
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1.0 to 1.0)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        
        # Normalize to -1.0 to 1.0
        return max(-1.0, min(1.0, slope * n))
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"🔧 [ULTRA-PHASE5] Resource monitoring stopped")

class UltraResourceScheduler:
    """Optimal GPU/CPU resource allocation scheduler"""
    
    def __init__(self, max_cpu_utilization: float = 0.8, max_memory_utilization: float = 0.85, max_gpu_utilization: float = 0.9):
        self.max_cpu_utilization = max_cpu_utilization
        self.max_memory_utilization = max_memory_utilization
        self.max_gpu_utilization = max_gpu_utilization
        
        self.system_monitor = UltraSystemMonitor()
        self.active_allocations = {}  # task_id -> ResourceAllocation
        self.pending_requests = []  # List of ResourceRequest
        self.allocation_lock = asyncio.Lock()
        
        # Scheduling algorithm parameters
        self.scheduling_interval = 1.0  # seconds
        self.scheduler_active = False
        self.scheduler_task = None
        
        # Statistics
        self.allocation_stats = {
            'total_requests': 0,
            'total_allocations': 0,
            'total_rejections': 0,
            'avg_wait_time': 0.0,
            'resource_efficiency': {}
        }
        
        logger.info(f"🔧 [ULTRA-PHASE5] Resource Scheduler initialized")
        logger.info(f"   - Max CPU Utilization: {max_cpu_utilization * 100:.1f}%")
        logger.info(f"   - Max Memory Utilization: {max_memory_utilization * 100:.1f}%")
        logger.info(f"   - Max GPU Utilization: {max_gpu_utilization * 100:.1f}%")
    
    async def start_scheduler(self):
        """Start the resource scheduler"""
        if self.scheduler_active:
            return
        
        await self.system_monitor.start_monitoring()
        
        self.scheduler_active = True
        self.scheduler_task = asyncio.create_task(self._scheduling_loop())
        
        logger.info(f"🔧 [ULTRA-PHASE5] Resource scheduler started")
    
    async def _scheduling_loop(self):
        """Main scheduling loop"""
        while self.scheduler_active:
            try:
                await self._process_pending_requests()
                await self._update_active_allocations()
                await asyncio.sleep(self.scheduling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🔧 [ULTRA-PHASE5] Scheduler error: {e}")
                await asyncio.sleep(self.scheduling_interval)
    
    async def _process_pending_requests(self):
        """Process pending resource requests"""
        async with self.allocation_lock:
            if not self.pending_requests:
                return
            
            # Sort by priority and request time
            self.pending_requests.sort(key=lambda r: (r.priority, r.requested_at))
            
            allocated_requests = []
            
            for request in self.pending_requests:
                if await self._try_allocate_resources(request):
                    allocated_requests.append(request)
            
            # Remove allocated requests from pending
            for request in allocated_requests:
                self.pending_requests.remove(request)
    
    async def _try_allocate_resources(self, request: ResourceRequest) -> bool:
        """Try to allocate resources for a request"""
        current_resources = self.system_monitor._get_current_resources()
        
        # Check CPU availability
        if current_resources[ResourceType.CPU_CORE] > self.max_cpu_utilization:
            return False
        
        # Check memory availability
        if current_resources[ResourceType.SYSTEM_MEMORY] > self.max_memory_utilization:
            return False
        
        # Check GPU memory if required
        if request.gpu_memory_mb > 0:
            if not self.system_monitor.gpu_available:
                return False
            
            if current_resources[ResourceType.GPU_MEMORY] > self.max_gpu_utilization:
                return False
            
            # Check if we have enough GPU memory
            available_gpu_mb = self.system_monitor.gpu_memory_total * (1.0 - current_resources[ResourceType.GPU_MEMORY]) / (1024**2)
            if available_gpu_mb < request.gpu_memory_mb:
                return False
        
        # Allocate resources
        allocation = ResourceAllocation(
            task_id=request.task_id,
            allocated_resources=request,
            allocated_at=time.time(),
            expected_completion=time.time() + request.estimated_duration_seconds
        )
        
        self.active_allocations[request.task_id] = allocation
        
        # Update statistics
        self.allocation_stats['total_allocations'] += 1
        wait_time = time.time() - request.requested_at
        self.allocation_stats['avg_wait_time'] = (
            (self.allocation_stats['avg_wait_time'] * (self.allocation_stats['total_allocations'] - 1) + wait_time) /
            self.allocation_stats['total_allocations']
        )
        
        logger.info(f"🔧 [ULTRA-PHASE5] Resources allocated for task: {request.task_id}")
        logger.info(f"   - CPU Cores: {request.cpu_cores}")
        logger.info(f"   - GPU Memory: {request.gpu_memory_mb}MB")
        logger.info(f"   - System Memory: {request.system_memory_mb}MB")
        logger.info(f"   - Wait Time: {wait_time:.2f}s")
        
        return True
    
    async def _update_active_allocations(self):
        """Update and clean up active allocations"""
        current_time = time.time()
        completed_tasks = []
        
        for task_id, allocation in self.active_allocations.items():
            # Check if task should have completed
            if current_time > allocation.expected_completion:
                completed_tasks.append(task_id)
        
        # Clean up completed tasks
        for task_id in completed_tasks:
            await self.deallocate_resources(task_id, reason="timeout")
    
    async def request_resources(self, request: ResourceRequest) -> bool:
        """Request resource allocation for a task"""
        async with self.allocation_lock:
            self.allocation_stats['total_requests'] += 1
            
            # Try immediate allocation
            if await self._try_allocate_resources(request):
                return True
            
            # Add to pending queue
            self.pending_requests.append(request)
            logger.info(f"🔧 [ULTRA-PHASE5] Resource request queued: {request.task_id}")
            
            return False
    
    async def deallocate_resources(self, task_id: str, reason: str = "completion") -> bool:
        """Deallocate resources for a completed task"""
        async with self.allocation_lock:
            if task_id not in self.active_allocations:
                return False
            
            allocation = self.active_allocations[task_id]
            actual_duration = time.time() - allocation.allocated_at
            
            del self.active_allocations[task_id]
            
            logger.info(f"🔧 [ULTRA-PHASE5] Resources deallocated for task: {task_id} ({reason})")
            logger.info(f"   - Actual Duration: {actual_duration:.2f}s")
            logger.info(f"   - Expected Duration: {allocation.allocated_resources.estimated_duration_seconds}s")
            
            return True
    
    async def get_allocation_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get allocation status for a task"""
        # Check active allocations
        if task_id in self.active_allocations:
            allocation = self.active_allocations[task_id]
            return {
                'status': 'allocated',
                'allocated_at': allocation.allocated_at,
                'expected_completion': allocation.expected_completion,
                'resources': {
                    'cpu_cores': allocation.allocated_resources.cpu_cores,
                    'gpu_memory_mb': allocation.allocated_resources.gpu_memory_mb,
                    'system_memory_mb': allocation.allocated_resources.system_memory_mb
                }
            }
        
        # Check pending requests
        for request in self.pending_requests:
            if request.task_id == task_id:
                queue_position = self.pending_requests.index(request) + 1
                return {
                    'status': 'pending',
                    'queue_position': queue_position,
                    'requested_at': request.requested_at,
                    'estimated_wait_time': self._estimate_wait_time(queue_position)
                }
        
        return None
    
    def _estimate_wait_time(self, queue_position: int) -> float:
        """Estimate wait time based on queue position"""
        avg_allocation_time = self.allocation_stats.get('avg_wait_time', 60.0)
        return queue_position * avg_allocation_time
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization and allocation statistics"""
        current_resources = self.system_monitor._get_current_resources()
        resource_trends = self.system_monitor.get_resource_trends()
        
        return {
            'current_utilization': current_resources,
            'resource_trends': resource_trends,
            'active_allocations': len(self.active_allocations),
            'pending_requests': len(self.pending_requests),
            'allocation_limits': {
                'max_cpu_utilization': self.max_cpu_utilization,
                'max_memory_utilization': self.max_memory_utilization,
                'max_gpu_utilization': self.max_gpu_utilization
            },
            'system_info': {
                'cpu_cores': self.system_monitor.cpu_count,
                'total_memory_gb': self.system_monitor.total_memory // (1024**3),
                'gpu_available': self.system_monitor.gpu_available,
                'gpu_memory_mb': self.system_monitor.gpu_memory_total // (1024**2) if self.system_monitor.gpu_available else 0
            },
            'statistics': self.allocation_stats.copy()
        }
    
    async def shutdown(self):
        """Shutdown the resource scheduler"""
        logger.info(f"🔧 [ULTRA-PHASE5] Shutting down Resource Scheduler...")
        
        self.scheduler_active = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        await self.system_monitor.stop_monitoring()
        
        # Clear all allocations
        self.active_allocations.clear()
        self.pending_requests.clear()
        
        logger.info(f"🔧 [ULTRA-PHASE5] Resource Scheduler shutdown complete")

# Factory function
def create_resource_scheduler(
    max_cpu: float = 0.8, 
    max_memory: float = 0.85, 
    max_gpu: float = 0.9
) -> UltraResourceScheduler:
    """Create a resource scheduler with optimal configuration"""
    return UltraResourceScheduler(
        max_cpu_utilization=max_cpu,
        max_memory_utilization=max_memory,
        max_gpu_utilization=max_gpu
    ) 