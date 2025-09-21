from fastapi import APIRouter, HTTPException, Query, Path, Depends
from uuid import UUID
from typing import List # Keep List for response model if needed, e.g. List[ResultSummary]

from results.schemas import ResultSummary, ResultsListResponse # Using new schemas
from simulation.schemas import SimulationResponse # Changed to absolute import
from results.service import get_all_simulation_results_summary, get_result_details_by_id # Corrected imports
from auth.auth0_dependencies import get_current_active_auth0_user # Changed to get_current_active_user
from auth.schemas import User # Changed to absolute

router = APIRouter(
    prefix="/results", # Added a prefix for clarity
    tags=["results"],
    responses={404: {"description": "Not found"}}
)

@router.get("/", response_model=ResultsListResponse)
async def list_simulation_results(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    current_user: User = Depends(get_current_active_auth0_user) # Added dependency
):
    """Retrieve a list of all simulation result summaries."""
    try:
        results_list = await get_all_simulation_results_summary(skip=skip, limit=limit)
        return results_list
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Failed to retrieve result summaries: {str(e)}")

@router.get("/{simulation_id}", response_model=SimulationResponse) # Reusing SimulationResponse from simulation module for full detail
async def get_specific_simulation_result(
    simulation_id: UUID = Path(..., title="The ID of the simulation result to retrieve"),
    current_user: User = Depends(get_current_active_auth0_user) # Added dependency
):
    """Get the detailed results of a specific simulation by its ID."""
    # This endpoint is similar to the one in simulation/router.py but under the /results path.
    # It uses the get_result_details_by_id from results.service, which currently points to the shared store.
    result = await get_result_details_by_id(simulation_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Results for simulation ID '{simulation_id}' not found.")
    return result 