"""
ULTRA MONTE CARLO ENGINE - PHASE 5: CONCURRENT SIMULATION MANAGER
High-level manager for concurrent simulation execution with resource optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from .phase5_async_core import (
    UltraAsyncTaskQueue, 
    AsyncSimulationTask, 
    TaskPriority, 
    TaskStatus, 
    TaskResource
)

logger = logging.getLogger(__name__)

class UltraConcurrentSimulationManager:
    """High-level manager for concurrent simulation execution"""
    
    def __init__(self, max_concurrent_simulations: int = 10):
        self.max_concurrent_simulations = max_concurrent_simulations
        self.task_queue = UltraAsyncTaskQueue(max_concurrent_simulations)
        self.simulation_registry = {}  # simulation_id -> task_id
        self.batch_registry = {}  # batch_id -> list of task_ids
        
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
        progress_callback: Optional[Callable] = None,
        batch_id: Optional[str] = None
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
        
        # Register with batch if provided
        if batch_id:
            if batch_id not in self.batch_registry:
                self.batch_registry[batch_id] = []
            self.batch_registry[batch_id].append(task_id)
        
        logger.info(f"🔧 [ULTRA-PHASE5] Simulation submitted: {simulation_id} -> {task_id}")
        logger.info(f"   - Iterations: {iterations:,}")
        logger.info(f"   - Variables: {len(mc_input_configs)}")
        logger.info(f"   - Formulas: {len(ordered_calc_steps)}")
        logger.info(f"   - Priority: {priority.name}")
        if batch_id:
            logger.info(f"   - Batch ID: {batch_id}")
        
        return task_id
    
    def _estimate_resources(self, iterations: int, num_variables: int, num_formulas: int) -> TaskResource:
        """Estimate resource requirements for a simulation"""
        # Base resource calculation
        base_memory = 512  # MB
        
        # Scale with iterations and complexity
        memory_per_iteration = (num_variables + num_formulas) * 0.001  # MB per iteration
        total_memory = int(base_memory + (iterations * memory_per_iteration))
        
        # GPU memory if available and beneficial
        gpu_memory = 0
        try:
            import cupy as cp
            if iterations > 10000:  # Only use GPU for larger simulations
                gpu_memory = int(total_memory * 0.5)  # Use half memory on GPU
        except ImportError:
            gpu_memory = 0
        
        # Estimated duration (very rough estimate)
        base_duration = 30  # seconds
        complexity_factor = (num_formulas * num_variables) / 1000.0
        iteration_factor = iterations / 10000.0
        duration = max(10, int(base_duration + (complexity_factor * iteration_factor * 30)))
        
        return TaskResource(
            cpu_cores=1,
            gpu_memory_mb=gpu_memory,
            system_memory_mb=total_memory,
            estimated_duration_seconds=duration
        )
    
    async def submit_batch_simulation(
        self,
        batch_id: str,
        simulations: List[Dict[str, Any]],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> List[str]:
        """Submit multiple simulations as a batch"""
        task_ids = []
        
        logger.info(f"🔧 [ULTRA-PHASE5] Submitting batch simulation: {batch_id} with {len(simulations)} simulations")
        
        for i, sim_config in enumerate(simulations):
            task_id = await self.submit_simulation(
                simulation_id=sim_config['simulation_id'],
                iterations=sim_config['iterations'],
                mc_input_configs=sim_config['mc_input_configs'],
                ordered_calc_steps=sim_config['ordered_calc_steps'],
                target_sheet_name=sim_config['target_sheet_name'],
                target_cell_coordinate=sim_config['target_cell_coordinate'],
                constant_values=sim_config['constant_values'],
                priority=priority,
                progress_callback=sim_config.get('progress_callback'),
                batch_id=batch_id
            )
            task_ids.append(task_id)
        
        logger.info(f"🔧 [ULTRA-PHASE5] Batch {batch_id} submitted with {len(task_ids)} tasks")
        return task_ids
    
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
    
    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch of simulations"""
        if batch_id not in self.batch_registry:
            return None
        
        task_ids = self.batch_registry[batch_id]
        batch_status = {
            'batch_id': batch_id,
            'total_tasks': len(task_ids),
            'task_statuses': {},
            'summary': {
                'pending': 0,
                'queued': 0,
                'running': 0,
                'completed': 0,
                'failed': 0,
                'cancelled': 0
            }
        }
        
        for task_id in task_ids:
            status = self.task_queue.get_task_status(task_id)
            if status:
                batch_status['task_statuses'][task_id] = status.value
                batch_status['summary'][status.value] += 1
        
        # Calculate completion percentage
        total_tasks = batch_status['total_tasks']
        completed_tasks = batch_status['summary']['completed'] + batch_status['summary']['failed']
        batch_status['completion_percentage'] = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        return batch_status
    
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
    
    async def cancel_batch(self, batch_id: str) -> Dict[str, bool]:
        """Cancel all simulations in a batch"""
        if batch_id not in self.batch_registry:
            return {}
        
        task_ids = self.batch_registry[batch_id].copy()
        results = {}
        
        for task_id in task_ids:
            success = await self.task_queue.cancel_task(task_id)
            results[task_id] = success
        
        # Clean up batch registry
        del self.batch_registry[batch_id]
        
        # Clean up simulation registry
        for sim_id, task_id in list(self.simulation_registry.items()):
            if task_id in task_ids:
                del self.simulation_registry[sim_id]
        
        logger.info(f"🔧 [ULTRA-PHASE5] Batch cancelled: {batch_id} ({len(task_ids)} tasks)")
        return results
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics"""
        stats = self.task_queue.get_queue_stats()
        stats.update({
            'active_simulations': len(self.simulation_registry),
            'active_batches': len(self.batch_registry),
            'max_concurrent': self.max_concurrent_simulations,
            'manager_type': 'UltraConcurrentSimulationManager',
            'phase': 'Phase 5 - Asynchronous Processing'
        })
        return stats
    
    async def wait_for_batch_completion(self, batch_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for a batch to complete with optional timeout"""
        if batch_id not in self.batch_registry:
            return {'error': 'Batch not found'}
        
        start_time = time.time()
        
        while True:
            batch_status = await self.get_batch_status(batch_id)
            if not batch_status:
                return {'error': 'Batch not found'}
            
            # Check if all tasks are completed
            if batch_status['completion_percentage'] >= 100:
                return batch_status
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return {'error': 'Timeout waiting for batch completion', 'batch_status': batch_status}
            
            # Wait before checking again
            await asyncio.sleep(1.0)
    
    async def get_simulation_result(self, simulation_id: str) -> Optional[Tuple[Any, List[str]]]:
        """Get the result of a completed simulation"""
        if simulation_id not in self.simulation_registry:
            return None
        
        task_id = self.simulation_registry[simulation_id]
        
        # Check completed tasks
        if task_id in self.task_queue.completed_tasks:
            task = self.task_queue.completed_tasks[task_id]
            return task.results
        
        return None
    
    async def shutdown(self):
        """Gracefully shutdown the concurrent simulation manager"""
        logger.info(f"🔧 [ULTRA-PHASE5] Shutting down Concurrent Simulation Manager...")
        
        # Cancel all running tasks
        if self.task_queue.scheduler_task:
            self.task_queue.scheduler_active = False
            self.task_queue.scheduler_task.cancel()
            try:
                await self.task_queue.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Clear registries
        self.simulation_registry.clear()
        self.batch_registry.clear()
        
        logger.info(f"🔧 [ULTRA-PHASE5] Concurrent Simulation Manager shutdown complete")

# Factory function
def create_concurrent_simulation_manager(max_concurrent: int = 10) -> UltraConcurrentSimulationManager:
    """Create a concurrent simulation manager with optimal configuration"""
    return UltraConcurrentSimulationManager(max_concurrent_simulations=max_concurrent) 