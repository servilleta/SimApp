from typing import List, Optional
from uuid import UUID

from results.schemas import ResultSummary, ResultsListResponse
from simulation.service import SIMULATION_RESULTS_STORE # Changed to absolute
from simulation.schemas import SimulationResponse # Changed to absolute

async def get_all_simulation_results_summary(skip: int = 0, limit: int = 100) -> ResultsListResponse:
    """Retrieve a paginated list of all simulation result summaries."""
    # This is a basic implementation using the in-memory store.
    # In a real application, this would query a database.
    
    all_responses: List[SimulationResponse] = list(SIMULATION_RESULTS_STORE.values())
    
    summaries: List[ResultSummary] = []
    for resp in all_responses:
        summary = ResultSummary(
            simulation_id=resp.simulation_id,
            status=resp.status,
            message=resp.message
            # Potentially add more fields here if resp.results exists
        )
        summaries.append(summary)
        
    paginated_summaries = summaries[skip : skip + limit]
    
    return ResultsListResponse(results=paginated_summaries, total=len(summaries))

async def get_result_details_by_id(simulation_id: UUID) -> Optional[SimulationResponse]:
    """Retrieve full details for a specific simulation result by its ID."""
    # This reuses the function from simulation.service for consistency
    # as it already fetches the complete SimulationResponse.
    return SIMULATION_RESULTS_STORE.get(simulation_id) 