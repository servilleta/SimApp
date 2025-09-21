"""
⚡ SIMULATION SERVICE - Microservice Architecture

Handles all simulation-related operations:
- Monte Carlo simulation execution
- GPU resource management
- Simulation queue and scheduling
- Progress tracking and WebSocket updates
- Results storage and retrieval

This service is part of the microservices decomposition from the monolithic application.
"""

import logging
import uuid
import asyncio
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from sqlalchemy import and_
from pydantic import BaseModel
import redis
import json
import multiprocessing
import concurrent.futures

# Import from monolith (during transition)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from models import User, SimulationResult
from auth.auth0_dependencies import get_current_active_auth0_user
from simulation.schemas import SimulationRequest, SimulationResponse, VariableConfig, ConstantConfig

logger = logging.getLogger(__name__)

# FastAPI app for Simulation Service
app = FastAPI(
    title="Simulation Service",
    description="Microservice for Monte Carlo simulation processing and management",
    version="1.0.0"
)

security = HTTPBearer()

# ===============================
# PYDANTIC MODELS (API SCHEMAS)
# ===============================

class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: str
    message: str
    progress_percentage: float
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

class SimulationQueueStatus(BaseModel):
    position: int
    estimated_wait_time_minutes: int
    queue_length: int
    ahead_of_user: int

class GPUAllocation(BaseModel):
    gpu_id: str
    allocated_at: datetime
    estimated_runtime_minutes: int
    priority: str

class SimulationCreateResponse(BaseModel):
    simulation_id: str
    status: str
    message: str
    created_at: str
    queue_position: Optional[int] = None
    estimated_start_time: Optional[str] = None

# ===============================
# SIMULATION SERVICE CLASS
# ===============================

class SimulationService:
    """
    Core simulation service handling Monte Carlo simulations.
    Designed for microservices architecture with GPU management and queuing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Redis for progress tracking and queuing
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            self.logger.info("✅ [SIMULATION_SERVICE] Redis connected")
        except Exception as e:
            self.logger.warning(f"⚠️ [SIMULATION_SERVICE] Redis not available: {e}")
            self.redis_client = None
        
        # Process pool for CPU-intensive simulations
        self.max_concurrent_simulations = max(1, multiprocessing.cpu_count() // 2)
        self.simulation_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_concurrent_simulations,
            mp_context=multiprocessing.get_context('spawn')
        )
        
        # WebSocket connections for real-time updates
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
        # GPU management
        self.gpu_allocations: Dict[str, GPUAllocation] = {}
        
        self.logger.info(f"✅ [SIMULATION_SERVICE] Initialized with {self.max_concurrent_simulations} workers")
    
    async def create_simulation(self, db: Session, user_id: int, request: SimulationRequest) -> SimulationCreateResponse:
        """Create a new simulation request."""
        try:
            # Generate simulation ID if not provided
            simulation_id = request.simulation_id or str(uuid.uuid4())
            
            # Create simulation record
            simulation_record = SimulationResult(
                simulation_id=simulation_id,
                user_id=user_id,
                file_id=request.file_id,
                original_filename=request.original_filename,
                engine_type=request.engine_type,
                target_cell=", ".join(request.target_cells) if request.target_cells else None,
                variables_config=request.variables,
                constants_config=request.constants,
                iterations_requested=request.iterations,
                status="pending",
                message="Simulation has been queued for processing."
            )
            
            db.add(simulation_record)
            db.commit()
            db.refresh(simulation_record)
            
            # Add to processing queue
            queue_position = await self._add_to_queue(simulation_id, user_id, request)
            
            # Set initial progress in Redis
            if self.redis_client:
                progress_data = {
                    "simulation_id": simulation_id,
                    "status": "pending",
                    "progress": 0,
                    "message": "Simulation queued for processing",
                    "user_id": user_id,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                self.redis_client.setex(f"simulation:{simulation_id}", 3600, json.dumps(progress_data))
            
            # Start async processing
            asyncio.create_task(self._process_simulation_async(simulation_id, user_id, request, db))
            
            self.logger.info(f"✅ [SIMULATION_SERVICE] Created simulation {simulation_id} for user {user_id}")
            
            return SimulationCreateResponse(
                simulation_id=simulation_id,
                status="pending",
                message="Simulation created and queued for processing",
                created_at=simulation_record.created_at.isoformat(),
                queue_position=queue_position
            )
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"❌ [SIMULATION_SERVICE] Failed to create simulation for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to create simulation")
    
    async def get_simulation_status(self, db: Session, user_id: int, simulation_id: str) -> SimulationStatusResponse:
        """Get simulation status and progress."""
        try:
            # Get from database
            simulation = db.query(SimulationResult).filter(
                and_(
                    SimulationResult.user_id == user_id,
                    SimulationResult.simulation_id == simulation_id
                )
            ).first()
            
            if not simulation:
                raise HTTPException(status_code=404, detail="Simulation not found")
            
            # Get real-time progress from Redis
            progress_percentage = 0.0
            if self.redis_client:
                progress_data = self.redis_client.get(f"simulation:{simulation_id}")
                if progress_data:
                    progress_info = json.loads(progress_data)
                    progress_percentage = progress_info.get("progress", 0)
            
            return SimulationStatusResponse(
                simulation_id=simulation_id,
                status=simulation.status,
                message=simulation.message or "",
                progress_percentage=progress_percentage,
                created_at=simulation.created_at,
                started_at=simulation.started_at,
                completed_at=simulation.completed_at
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ [SIMULATION_SERVICE] Failed to get status for simulation {simulation_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get simulation status")
    
    async def get_simulation_results(self, db: Session, user_id: int, simulation_id: str) -> SimulationResponse:
        """Get completed simulation results."""
        try:
            simulation = db.query(SimulationResult).filter(
                and_(
                    SimulationResult.user_id == user_id,
                    SimulationResult.simulation_id == simulation_id
                )
            ).first()
            
            if not simulation:
                raise HTTPException(status_code=404, detail="Simulation not found")
            
            if simulation.status != "completed":
                raise HTTPException(status_code=400, detail="Simulation is not completed yet")
            
            # Convert to response format
            return simulation.to_simulation_response()
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ [SIMULATION_SERVICE] Failed to get results for simulation {simulation_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get simulation results")
    
    async def list_user_simulations(self, db: Session, user_id: int, status_filter: Optional[str] = None) -> List[SimulationStatusResponse]:
        """List all simulations for a user."""
        try:
            query = db.query(SimulationResult).filter(SimulationResult.user_id == user_id)
            
            if status_filter:
                query = query.filter(SimulationResult.status == status_filter)
            
            simulations = query.order_by(SimulationResult.created_at.desc()).all()
            
            # Convert to status responses
            responses = []
            for sim in simulations:
                # Get real-time progress from Redis
                progress_percentage = 0.0
                if self.redis_client:
                    progress_data = self.redis_client.get(f"simulation:{sim.simulation_id}")
                    if progress_data:
                        progress_info = json.loads(progress_data)
                        progress_percentage = progress_info.get("progress", 0)
                
                responses.append(SimulationStatusResponse(
                    simulation_id=sim.simulation_id,
                    status=sim.status,
                    message=sim.message or "",
                    progress_percentage=progress_percentage,
                    created_at=sim.created_at,
                    started_at=sim.started_at,
                    completed_at=sim.completed_at
                ))
            
            self.logger.info(f"✅ [SIMULATION_SERVICE] Listed {len(responses)} simulations for user {user_id}")
            return responses
            
        except Exception as e:
            self.logger.error(f"❌ [SIMULATION_SERVICE] Failed to list simulations for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to list simulations")
    
    async def cancel_simulation(self, db: Session, user_id: int, simulation_id: str) -> bool:
        """Cancel a running simulation."""
        try:
            simulation = db.query(SimulationResult).filter(
                and_(
                    SimulationResult.user_id == user_id,
                    SimulationResult.simulation_id == simulation_id
                )
            ).first()
            
            if not simulation:
                raise HTTPException(status_code=404, detail="Simulation not found")
            
            if simulation.status in ["completed", "failed", "cancelled"]:
                raise HTTPException(status_code=400, detail="Simulation cannot be cancelled")
            
            # Update status
            simulation.status = "cancelled"
            simulation.message = "Simulation cancelled by user"
            simulation.completed_at = datetime.now(timezone.utc)
            
            db.commit()
            
            # Update Redis progress
            if self.redis_client:
                progress_data = {
                    "simulation_id": simulation_id,
                    "status": "cancelled",
                    "progress": 0,
                    "message": "Simulation cancelled by user",
                    "user_id": user_id
                }
                self.redis_client.setex(f"simulation:{simulation_id}", 3600, json.dumps(progress_data))
            
            # Notify WebSocket clients
            await self._broadcast_to_simulation(simulation_id, {
                "type": "status_update",
                "simulation_id": simulation_id,
                "status": "cancelled",
                "message": "Simulation cancelled by user"
            })
            
            self.logger.info(f"✅ [SIMULATION_SERVICE] Cancelled simulation {simulation_id} for user {user_id}")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"❌ [SIMULATION_SERVICE] Failed to cancel simulation {simulation_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to cancel simulation")
    
    async def get_queue_status(self, user_id: int) -> SimulationQueueStatus:
        """Get queue status for user's simulations."""
        try:
            # This is a simplified implementation
            # In production, you'd track actual queue positions
            
            if self.redis_client:
                queue_length = self.redis_client.llen("simulation_queue")
                # Simplified queue position calculation
                position = min(queue_length, 5)  # Mock position
            else:
                queue_length = 0
                position = 0
            
            return SimulationQueueStatus(
                position=position,
                estimated_wait_time_minutes=position * 2,  # 2 minutes per position
                queue_length=queue_length,
                ahead_of_user=max(0, position - 1)
            )
            
        except Exception as e:
            self.logger.error(f"❌ [SIMULATION_SERVICE] Failed to get queue status for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get queue status")
    
    async def _add_to_queue(self, simulation_id: str, user_id: int, request: SimulationRequest) -> int:
        """Add simulation to processing queue."""
        try:
            if self.redis_client:
                queue_item = {
                    "simulation_id": simulation_id,
                    "user_id": user_id,
                    "request": request.dict(),
                    "queued_at": datetime.now(timezone.utc).isoformat()
                }
                
                # Add to queue
                queue_position = self.redis_client.rpush("simulation_queue", json.dumps(queue_item))
                self.logger.info(f"✅ [SIMULATION_SERVICE] Added simulation {simulation_id} to queue position {queue_position}")
                return queue_position
            else:
                return 1  # Mock position when Redis not available
                
        except Exception as e:
            self.logger.error(f"❌ [SIMULATION_SERVICE] Failed to add simulation {simulation_id} to queue: {e}")
            return 1  # Fallback position
    
    async def _process_simulation_async(self, simulation_id: str, user_id: int, request: SimulationRequest, db: Session):
        """Process simulation asynchronously."""
        try:
            # Update status to running
            simulation = db.query(SimulationResult).filter(
                SimulationResult.simulation_id == simulation_id
            ).first()
            
            if simulation:
                simulation.status = "running"
                simulation.started_at = datetime.now(timezone.utc)
                simulation.message = "Simulation is now running"
                db.commit()
            
            # Update Redis progress
            if self.redis_client:
                progress_data = {
                    "simulation_id": simulation_id,
                    "status": "running",
                    "progress": 5,
                    "message": "Simulation started",
                    "user_id": user_id
                }
                self.redis_client.setex(f"simulation:{simulation_id}", 3600, json.dumps(progress_data))
            
            # Notify WebSocket clients
            await self._broadcast_to_simulation(simulation_id, {
                "type": "status_update",
                "simulation_id": simulation_id,
                "status": "running",
                "message": "Simulation started"
            })
            
            # Simulate processing with progress updates
            for progress in range(10, 101, 10):
                await asyncio.sleep(2)  # Simulate work
                
                # Update progress in Redis
                if self.redis_client:
                    progress_data["progress"] = progress
                    progress_data["message"] = f"Processing... {progress}%"
                    self.redis_client.setex(f"simulation:{simulation_id}", 3600, json.dumps(progress_data))
                
                # Notify WebSocket clients
                await self._broadcast_to_simulation(simulation_id, {
                    "type": "progress_update",
                    "simulation_id": simulation_id,
                    "progress": progress,
                    "message": f"Processing... {progress}%"
                })
            
            # Complete simulation
            if simulation:
                simulation.status = "completed"
                simulation.completed_at = datetime.now(timezone.utc)
                simulation.message = "Simulation completed successfully"
                simulation.iterations_run = request.iterations
                
                # Mock results
                simulation.mean = 100.0
                simulation.median = 98.5
                simulation.std_dev = 15.2
                simulation.min_value = 50.0
                simulation.max_value = 150.0
                
                db.commit()
            
            # Final progress update
            if self.redis_client:
                progress_data["status"] = "completed"
                progress_data["progress"] = 100
                progress_data["message"] = "Simulation completed successfully"
                self.redis_client.setex(f"simulation:{simulation_id}", 3600, json.dumps(progress_data))
            
            # Final WebSocket notification
            await self._broadcast_to_simulation(simulation_id, {
                "type": "completion",
                "simulation_id": simulation_id,
                "status": "completed",
                "message": "Simulation completed successfully"
            })
            
            self.logger.info(f"✅ [SIMULATION_SERVICE] Completed simulation {simulation_id}")
            
        except Exception as e:
            self.logger.error(f"❌ [SIMULATION_SERVICE] Processing failed for simulation {simulation_id}: {e}")
            
            # Update simulation as failed
            if simulation:
                simulation.status = "failed"
                simulation.completed_at = datetime.now(timezone.utc)
                simulation.message = f"Simulation failed: {str(e)}"
                db.commit()
    
    async def _broadcast_to_simulation(self, simulation_id: str, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections for a simulation."""
        if simulation_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[simulation_id]:
                try:
                    await websocket.send_json(message)
                except WebSocketDisconnect:
                    disconnected.append(websocket)
                except Exception as e:
                    self.logger.warning(f"⚠️ [SIMULATION_SERVICE] WebSocket send failed: {e}")
                    disconnected.append(websocket)
            
            # Remove disconnected clients
            for ws in disconnected:
                if ws in self.active_connections[simulation_id]:
                    self.active_connections[simulation_id].remove(ws)
            
            # Clean up empty connection lists
            if not self.active_connections[simulation_id]:
                del self.active_connections[simulation_id]

# Global service instance
simulation_service = SimulationService()

# ===============================
# API ENDPOINTS
# ===============================

@app.get("/health")
async def health_check():
    """Service health check."""
    return {"status": "healthy", "service": "simulation-service", "version": "1.0.0"}

@app.post("/simulations", response_model=SimulationCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation(
    request: SimulationRequest,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Create a new Monte Carlo simulation."""
    return await simulation_service.create_simulation(db, current_user.id, request)

@app.get("/simulations/{simulation_id}/status", response_model=SimulationStatusResponse)
async def get_simulation_status(
    simulation_id: str,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Get simulation status and progress."""
    return await simulation_service.get_simulation_status(db, current_user.id, simulation_id)

@app.get("/simulations/{simulation_id}/results", response_model=SimulationResponse)
async def get_simulation_results(
    simulation_id: str,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Get completed simulation results."""
    return await simulation_service.get_simulation_results(db, current_user.id, simulation_id)

@app.get("/simulations", response_model=List[SimulationStatusResponse])
async def list_simulations(
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """List user's simulations."""
    return await simulation_service.list_user_simulations(db, current_user.id, status_filter)

@app.put("/simulations/{simulation_id}/cancel")
async def cancel_simulation(
    simulation_id: str,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Cancel a running simulation."""
    success = await simulation_service.cancel_simulation(db, current_user.id, simulation_id)
    return {"message": "Simulation cancelled successfully", "simulation_id": simulation_id}

@app.get("/queue/status", response_model=SimulationQueueStatus)
async def get_queue_status(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get simulation queue status."""
    return await simulation_service.get_queue_status(current_user.id)

@app.websocket("/ws/progress/{simulation_id}")
async def websocket_endpoint(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time simulation progress."""
    await websocket.accept()
    
    # Add to active connections
    if simulation_id not in simulation_service.active_connections:
        simulation_service.active_connections[simulation_id] = []
    simulation_service.active_connections[simulation_id].append(websocket)
    
    try:
        # Send initial status
        if simulation_service.redis_client:
            progress_data = simulation_service.redis_client.get(f"simulation:{simulation_id}")
            if progress_data:
                await websocket.send_json(json.loads(progress_data))
        
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()  # Keep connection open
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        # Remove from active connections
        if simulation_id in simulation_service.active_connections:
            if websocket in simulation_service.active_connections[simulation_id]:
                simulation_service.active_connections[simulation_id].remove(websocket)
            
            if not simulation_service.active_connections[simulation_id]:
                del simulation_service.active_connections[simulation_id]

# ===============================
# SERVICE DISCOVERY ENDPOINTS
# ===============================

@app.get("/service-info")
async def get_service_info():
    """Service discovery information."""
    return {
        "service_name": "simulation-service",
        "version": "1.0.0",
        "description": "Monte Carlo simulation processing and management",
        "endpoints": {
            "create": "/simulations",
            "status": "/simulations/{id}/status",
            "results": "/simulations/{id}/results",
            "list": "/simulations",
            "cancel": "/simulations/{id}/cancel",
            "queue": "/queue/status",
            "websocket": "/ws/progress/{id}"
        },
        "dependencies": ["database", "redis", "file-service"],
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
