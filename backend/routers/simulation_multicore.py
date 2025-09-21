#!/usr/bin/env python3
"""
🚀 Multi-Core Simulation Router
Enhanced simulation endpoints with parallel processing capabilities.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import asyncio
import uuid
from datetime import datetime

from auth.auth0_dependencies import get_current_active_auth0_user
from worker_pool_manager import get_worker_pool, submit_parallel_simulation, get_simulation_result
from multicore_worker_config import get_scaling_recommendation

logger = logging.getLogger(__name__)

router = APIRouter()

class SimulationRequest(BaseModel):
    """
    Enhanced simulation request with parallel processing options
    """
    excel_data: Dict[str, Any]
    iterations: int = 10000
    confidence_level: float = 0.95
    priority: int = 1
    parallel_processing: bool = True
    max_workers: Optional[int] = None

class SimulationResponse(BaseModel):
    """
    Response for simulation submission
    """
    task_id: str
    status: str
    message: str
    estimated_completion_time: Optional[float] = None
    worker_info: Dict[str, Any]

class SimulationStatus(BaseModel):
    """
    Status response for simulation tracking
    """
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    worker_id: Optional[str] = None

@router.post("/simulation/parallel", response_model=SimulationResponse)
async def run_parallel_simulation(
    request: SimulationRequest,
    current_user: dict = Depends(get_current_active_auth0_user)
):
    """
    Submit a simulation for parallel processing across multiple CPU cores
    """
    try:
        # Get worker pool
        worker_pool = await get_worker_pool()
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Prepare simulation configuration
        simulation_config = {
            'excel_data': request.excel_data,
            'iterations': request.iterations,
            'confidence_level': request.confidence_level,
            'user_id': current_user.get('sub'),
            'parallel_processing': request.parallel_processing,
            'task_id': task_id
        }
        
        # Submit to worker pool
        submitted_task_id = await submit_parallel_simulation(simulation_config)
        
        # Get current queue stats
        queue_stats = await worker_pool.get_queue_stats()
        
        # Estimate completion time based on queue
        estimated_time = queue_stats['queue_depth'] * 30  # Rough estimate: 30s per simulation
        
        logger.info(f"🚀 Parallel simulation {task_id} submitted for user {current_user.get('email')}")
        
        return SimulationResponse(
            task_id=submitted_task_id,
            status="submitted",
            message=f"Simulation submitted to parallel processing queue",
            estimated_completion_time=estimated_time,
            worker_info={
                "available_workers": queue_stats['available_workers'],
                "queue_depth": queue_stats['queue_depth'],
                "total_workers": queue_stats['max_workers']
            }
        )
        
    except Exception as e:
        logger.error(f"Error submitting parallel simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit simulation: {str(e)}")

@router.get("/simulation/status/{task_id}", response_model=SimulationStatus)
async def get_parallel_simulation_status(
    task_id: str,
    current_user: dict = Depends(get_current_active_auth0_user)
):
    """
    Get the status of a parallel simulation task
    """
    try:
        result = await get_simulation_result(task_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Simulation task not found")
        
        return SimulationStatus(
            task_id=task_id,
            status=result['status'],
            result=result.get('result'),
            error=result.get('error'),
            created_at=datetime.fromtimestamp(result['created_at']) if result.get('created_at') else None,
            started_at=datetime.fromtimestamp(result['started_at']) if result.get('started_at') else None,
            completed_at=datetime.fromtimestamp(result['completed_at']) if result.get('completed_at') else None,
            duration=result.get('duration')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting simulation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get simulation status: {str(e)}")

@router.get("/simulation/queue/stats")
async def get_queue_statistics(
    current_user: dict = Depends(get_current_active_auth0_user)
):
    """
    Get current worker queue statistics and system load
    """
    try:
        worker_pool = await get_worker_pool()
        queue_stats = await worker_pool.get_queue_stats()
        
        # Get scaling recommendation
        scaling_rec = get_scaling_recommendation()
        
        return {
            "queue_stats": queue_stats,
            "scaling_recommendation": scaling_rec,
            "system_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "server_id": "server_1",
                "multicore_enabled": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue statistics: {str(e)}")

@router.post("/simulation/batch")
async def submit_batch_simulations(
    simulations: List[SimulationRequest],
    current_user: dict = Depends(get_current_active_auth0_user)
):
    """
    Submit multiple simulations for parallel processing
    """
    try:
        if len(simulations) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 simulations per batch")
        
        submitted_tasks = []
        
        for i, simulation in enumerate(simulations):
            task_id = str(uuid.uuid4())
            
            simulation_config = {
                'excel_data': simulation.excel_data,
                'iterations': simulation.iterations,
                'confidence_level': simulation.confidence_level,
                'user_id': current_user.get('sub'),
                'parallel_processing': simulation.parallel_processing,
                'task_id': task_id,
                'batch_id': f"batch_{uuid.uuid4()}",
                'batch_index': i
            }
            
            submitted_task_id = await submit_parallel_simulation(simulation_config)
            submitted_tasks.append({
                "task_id": submitted_task_id,
                "batch_index": i,
                "status": "submitted"
            })
        
        logger.info(f"📦 Batch of {len(simulations)} simulations submitted for user {current_user.get('email')}")
        
        return {
            "message": f"Batch of {len(simulations)} simulations submitted",
            "tasks": submitted_tasks,
            "batch_id": f"batch_{uuid.uuid4()}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting batch simulations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit batch simulations: {str(e)}")

@router.get("/simulation/performance/metrics")
async def get_performance_metrics(
    current_user: dict = Depends(get_current_active_auth0_user)
):
    """
    Get detailed performance metrics for the simulation system
    """
    try:
        worker_pool = await get_worker_pool()
        queue_stats = await worker_pool.get_queue_stats()
        scaling_rec = get_scaling_recommendation()
        
        # Calculate efficiency metrics
        total_capacity = queue_stats['max_workers']
        utilized_capacity = queue_stats['active_workers']
        efficiency = (utilized_capacity / total_capacity) * 100 if total_capacity > 0 else 0
        
        return {
            "system_efficiency": {
                "cpu_utilization": queue_stats['cpu_utilization'],
                "memory_utilization": queue_stats['memory_utilization'],
                "worker_efficiency": efficiency,
                "cores_utilized": utilized_capacity,
                "cores_available": total_capacity
            },
            "queue_performance": {
                "pending_tasks": queue_stats['pending_tasks'],
                "running_tasks": queue_stats['running_tasks'],
                "completed_tasks": queue_stats['completed_tasks'],
                "queue_depth": queue_stats['queue_depth'],
                "throughput_estimate": f"{queue_stats['max_workers'] * 2} simulations/hour"
            },
            "scaling_status": {
                "current_server": "server_1",
                "server_2_needed": scaling_rec['scale_needed'],
                "scaling_reasons": scaling_rec.get('reasons', []),
                "estimated_benefit": scaling_rec.get('estimated_benefit', {})
            },
            "recommendations": [
                f"Current setup can handle {total_capacity} parallel simulations",
                f"System is running at {efficiency:.1f}% worker capacity",
                "Consider Server 2 if queue depth consistently > 5" if not scaling_rec['scale_needed'] else "Server 2 scaling recommended now"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# Background task for automatic scaling
async def monitor_and_scale():
    """
    Background task to monitor system load and trigger Server 2 if needed
    """
    while True:
        try:
            scaling_rec = get_scaling_recommendation()
            
            if scaling_rec['scale_needed']:
                logger.warning(f"🚨 Auto-scaling trigger: {scaling_rec['reasons']}")
                # Here you would integrate with the Paperspace API to start Server 2
                # For now, just log the recommendation
                
        except Exception as e:
            logger.error(f"Error in scaling monitor: {e}")
        
        await asyncio.sleep(60)  # Check every minute

# Start the monitoring task when the router loads
@router.on_event("startup")
async def start_monitoring():
    """
    Start background monitoring for auto-scaling
    """
    asyncio.create_task(monitor_and_scale())
    logger.info("🎯 Started automatic scaling monitoring")
