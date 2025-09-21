from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime

class SaveSimulationRequest(BaseModel):
    name: str
    description: Optional[str] = None
    file_id: str
    simulation_config: Dict[str, Any]  # Contains input variables, target cells, iterations, etc.
    simulation_results: Optional[Dict[str, Any]] = None  # Contains saved simulation results

class SavedSimulationResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    original_filename: str
    simulation_config: Dict[str, Any]  # Include simulation config for KPI display
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class SavedSimulationListResponse(BaseModel):
    simulations: List[SavedSimulationResponse]

class LoadSimulationResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    original_filename: str
    file_id: str  # New file_id for the restored Excel file
    simulation_config: Dict[str, Any]
    simulation_results: Optional[Dict[str, Any]] = None  # Include saved simulation results
    created_at: datetime
    file_info: Optional[Dict[str, Any]] = None  # Include Excel data directly
    
    class Config:
        from_attributes = True 