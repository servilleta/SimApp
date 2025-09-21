from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from database import get_db
from auth.auth0_dependencies import get_current_active_auth0_user
from models import User
from saved_simulations.schemas import (
    SaveSimulationRequest, 
    SavedSimulationResponse, 
    SavedSimulationListResponse,
    LoadSimulationResponse
)
from saved_simulations.service import (
    save_simulation,
    load_simulation,
    get_user_simulations,
    delete_simulation
)

router = APIRouter(tags=["saved_simulations"])

@router.post("/save", response_model=SavedSimulationResponse)
async def save_simulation_endpoint(
    request: SaveSimulationRequest,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Save current simulation with its Excel file and configuration"""
    return await save_simulation(db, current_user.id, request)

@router.get("", response_model=SavedSimulationListResponse)
async def get_saved_simulations(
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Get all saved simulations for the current user"""
    simulations = await get_user_simulations(db, current_user.id)
    return SavedSimulationListResponse(simulations=simulations)

@router.get("/{simulation_id}/load", response_model=LoadSimulationResponse)
async def load_simulation_endpoint(
    simulation_id: int,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Load a saved simulation and restore its Excel file"""
    return await load_simulation(db, current_user.id, simulation_id)

@router.delete("/{simulation_id}")
async def delete_simulation_endpoint(
    simulation_id: int,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Delete a saved simulation"""
    await delete_simulation(db, current_user.id, simulation_id)
    return {"message": "Simulation deleted successfully"} 