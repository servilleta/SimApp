from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import UUID

# Re-using or extending simulation schemas might be appropriate here.
# For now, defining a simple placeholder if specific result views are needed.

class ResultSummary(BaseModel):
    simulation_id: UUID
    status: str
    message: Optional[str] = None
    # Add other specific summary fields if needed, e.g., mean, median if completed

class ResultsListResponse(BaseModel):
    results: List[ResultSummary]
    total: int 