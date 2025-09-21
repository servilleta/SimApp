"""
🏢 ENTERPRISE ROUTER
Secure, user-isolated API endpoints for enterprise deployment.

This router replaces the existing simulation endpoints with user-aware versions.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from database import get_db
from models import User as UserModel
from auth.auth0_dependencies import get_current_active_auth0_user
from simulation.schemas import SimulationRequest, SimulationResponse
from enterprise.simulation_service import enterprise_simulation_service

logger = logging.getLogger(__name__)

# Create enterprise router
router = APIRouter(prefix="/api/enterprise", tags=["enterprise"])

@router.post("/simulations", response_model=SimulationResponse)
async def create_enterprise_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    🏢 Create a new simulation with enterprise security.
    
    🔒 SECURITY: Automatically associates simulation with authenticated user.
    Users can only create simulations under their own account.
    """
    try:
        logger.info(f"🏢 [ENTERPRISE] Creating simulation for user {current_user.id} ({current_user.email})")
        
        # Create simulation with user association
        simulation_response = await enterprise_simulation_service.create_user_simulation(
            user_id=current_user.id,
            request=request,
            db=db
        )
        
        # Queue the actual simulation execution
        background_tasks.add_task(
            _execute_enterprise_simulation,
            user_id=current_user.id,
            simulation_id=simulation_response.simulation_id,
            request=request
        )
        
        logger.info(f"✅ [ENTERPRISE] Simulation {simulation_response.simulation_id} created and queued for user {current_user.id}")
        return simulation_response
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE] Failed to create simulation for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create simulation")

@router.get("/simulations/{simulation_id}", response_model=SimulationResponse)
async def get_enterprise_simulation(
    simulation_id: str,
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    🏢 Get a specific simulation with user isolation.
    
    🔒 SECURITY: Users can only access their own simulations.
    """
    try:
        simulation = await enterprise_simulation_service.get_user_simulation(
            user_id=current_user.id,
            simulation_id=simulation_id,
            db=db
        )
        
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found or access denied")
        
        return simulation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE] Failed to get simulation {simulation_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve simulation")

@router.get("/simulations", response_model=List[SimulationResponse])
async def list_enterprise_simulations(
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None
):
    """
    🏢 List user's simulations with pagination and filtering.
    
    🔒 SECURITY: Only returns simulations owned by the authenticated user.
    """
    try:
        simulations = await enterprise_simulation_service.get_user_simulations(
            user_id=current_user.id,
            db=db,
            limit=limit,
            offset=offset,
            status_filter=status
        )
        
        logger.info(f"✅ [ENTERPRISE] Retrieved {len(simulations)} simulations for user {current_user.id}")
        return simulations
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE] Failed to list simulations for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve simulations")

@router.delete("/simulations/{simulation_id}")
async def delete_enterprise_simulation(
    simulation_id: str,
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    🏢 Delete a simulation with user verification.
    
    🔒 SECURITY: Users can only delete their own simulations.
    """
    try:
        success = await enterprise_simulation_service.delete_user_simulation(
            user_id=current_user.id,
            simulation_id=simulation_id,
            db=db
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Simulation not found or access denied")
        
        return {"message": "Simulation deleted successfully", "simulation_id": simulation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE] Failed to delete simulation {simulation_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete simulation")

@router.get("/simulations/{simulation_id}/status", response_model=dict)
async def get_enterprise_simulation_status(
    simulation_id: str,
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    🏢 Get simulation status with real-time progress.
    
    🔒 SECURITY: Users can only check status of their own simulations.
    """
    try:
        simulation = await enterprise_simulation_service.get_user_simulation(
            user_id=current_user.id,
            simulation_id=simulation_id,
            db=db
        )
        
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found or access denied")
        
        # Get real-time progress from progress store
        from shared.progress_store import get_progress
        progress_data = get_progress(simulation_id)
        
        return {
            "simulation_id": simulation_id,
            "status": simulation.status,
            "message": simulation.message,
            "progress": progress_data,
            "created_at": simulation.created_at,
            "updated_at": simulation.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE] Failed to get status for simulation {simulation_id}, user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve simulation status")

async def _execute_enterprise_simulation(user_id: int, simulation_id: str, request: SimulationRequest):
    """
    🏢 Execute simulation in background with enterprise monitoring.
    
    This function bridges to the existing simulation engines while maintaining user isolation.
    """
    try:
        logger.info(f"🏢 [ENTERPRISE] Starting simulation execution: {simulation_id} for user {user_id}")
        
        # Import the existing simulation execution logic
        from simulation.service import run_simulation_with_engine
        from database import SessionLocal
        
        # Update status to running
        db = SessionLocal()
        try:
            await enterprise_simulation_service.update_simulation_status(
                user_id=user_id,
                simulation_id=simulation_id,
                status="running",
                message="Simulation is executing",
                db=db
            )
            
            # Execute the simulation using existing engine
            # The engine doesn't need to know about users - we handle isolation at the API level
            result = await run_simulation_with_engine(
                sim_id=simulation_id,
                file_id=request.file_id,
                mc_inputs=request.variables,
                constants=request.constants,
                target_cell=request.target_cells[0] if request.target_cells else None,  # For single target
                iterations=request.iterations,
                engine_type=request.engine_type
            )
            
            # Update status to completed with results
            await enterprise_simulation_service.update_simulation_status(
                user_id=user_id,
                simulation_id=simulation_id,
                status="completed",
                message="Simulation completed successfully",
                results=result,
                db=db
            )
            
            logger.info(f"✅ [ENTERPRISE] Simulation {simulation_id} completed successfully for user {user_id}")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE] Simulation {simulation_id} failed for user {user_id}: {e}")
        
        # Update status to failed
        db = SessionLocal()
        try:
            await enterprise_simulation_service.update_simulation_status(
                user_id=user_id,
                simulation_id=simulation_id,
                status="failed",
                message=f"Simulation failed: {str(e)}",
                db=db
            )
        finally:
            db.close()

# Health check endpoint
@router.get("/health")
async def enterprise_health_check():
    """Enterprise health check endpoint."""
    return {
        "status": "healthy",
        "service": "enterprise-simulation-service",
        "version": "1.0.0",
        "capabilities": [
            "user-isolation",
            "audit-logging", 
            "secure-multi-tenancy"
        ]
    }
