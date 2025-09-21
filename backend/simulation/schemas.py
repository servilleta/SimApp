from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
import pandas as pd

class TargetStatistics(BaseModel):
    """Statistics for a single target in multi-target simulation"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    percentiles: Dict[str, float]  # e.g., {"5": 1.23, "10": 2.34, ...}
    histogram: Dict[str, Any]  # Histogram data with bins and values

class SimulationResultStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VariableConfig(BaseModel):
    name: str = Field(..., example="B5", description="Cell reference (e.g., 'B5', 'C10')")
    sheet_name: str = Field(..., example="Sheet1", description="Name of the Excel sheet")
    min_value: float = Field(..., example=0.05, description="Minimum value for triangular distribution")
    most_likely: float = Field(..., example=0.15, description="Most likely value for triangular distribution")
    max_value: float = Field(..., example=0.35, description="Maximum value for triangular distribution")

class ConstantConfig(BaseModel):
    name: str = Field(..., example="C10", description="Cell reference for constant value")
    sheet_name: str = Field(..., example="Sheet1", description="Name of the Excel sheet")
    value: float = Field(..., example=1000000.0, description="Fixed value to set in the cell")

class SimulationRequest(BaseModel):
    simulation_id: Optional[str] = Field(None, description="Optional simulation ID for tracking")
    file_id: str = Field(..., example="file_12345", description="ID of the uploaded Excel file")
    result_cell_coordinate: str = Field(..., example="J25", description="Cell coordinate for the main target result")
    result_cell_sheet_name: str = Field(..., example="Sheet1", description="Sheet name containing the target result cell")
    variables: List[VariableConfig] = Field(..., description="List of variable configurations for Monte Carlo simulation")
    constants: Optional[List[ConstantConfig]] = Field(default=[], description="Optional list of constant values to set during simulation")
    iterations: int = Field(default=1000, example=10000, description="Number of Monte Carlo iterations to run")
    engine_type: str = Field(default="ultra", example="ultra", description="Simulation engine type (ultra, standard, etc.)")
    original_filename: Optional[str] = Field(None, description="Original filename of the uploaded Excel file")
    batch_id: Optional[str] = Field(None, description="Batch ID for grouping multiple simulations")
    target_cells: Optional[List[str]] = Field(default=None, example=["J25", "K25"], description="List of target cells for multi-target simulations")
    target_cells_info: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        example=[
            {"cell": "J25", "sheet_name": "Sheet1", "description": "Portfolio Value"},
            {"cell": "K25", "sheet_name": "Sheet1", "description": "Total Return %"}
        ],
        description="Additional metadata for target cells including sheet names"
    )
    temp_id: Optional[str] = Field(None, description="Temporary ID from frontend for WebSocket mapping")

class SimulationResult(BaseModel):
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentiles: Dict[str, float]
    histogram: Optional[Dict[str, Any]] = None
    iterations_run: int
    errors: Optional[List[str]] = None
    sensitivity_analysis: Optional[List[Dict[str, Any]]] = None
    target_display_name: Optional[str] = None  # Display name for the target cell

class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    message: Optional[str] = None
    results: Optional[SimulationResult] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    original_filename: Optional[str] = None
    engine_type: Optional[str] = None
    user: Optional[str] = None
    target_name: Optional[str] = None
    result_cell_coordinate: Optional[str] = None  # Cell coordinate (e.g., "E8", "F8", "G8")
    batch_id: Optional[str] = None  # ID of parent batch simulation
    batch_simulation_ids: Optional[List[str]] = None  # IDs of child simulations in batch
    multi_target_result: Optional['MultiTargetSimulationResult'] = None  # Full multi-target results
    file_id: Optional[str] = None  # Excel file ID needed for restoring simulation view
    
    # Simulation configuration fields for restoration
    variables_config: Optional[List[Dict[str, Any]]] = None  # Monte Carlo variables configuration
    target_cell: Optional[str] = None  # Target cell coordinate
    iterations_requested: Optional[int] = None  # Number of iterations requested
    
    # Frontend compatibility: Top-level result fields for easier access
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None  # Standard deviation (frontend expects 'std')
    min: Optional[float] = None  
    max: Optional[float] = None
    iterations_run: Optional[int] = None
    histogram: Optional[Dict[str, Any]] = None

class EngineSelectionRequest(BaseModel):
    simulation_id: str
    file_path: Optional[str] = None
    mc_inputs: List[VariableConfig] = []

class EngineInfo(BaseModel):
    id: str
    name: str
    description: str
    best_for: str
    max_iterations: int

class EngineRecommendation(BaseModel):
    recommended_engine: str
    reason: str
    complexity_analysis: Dict[str, Any]
    available_engines: List[EngineInfo]

class SimulationExecutionRequest(BaseModel):
    simulation_id: str
    engine_type: str = "ultra"
    confirm_engine: bool = True

class BatchSimulationRequest(BaseModel):
    file_id: str
    target_cells: List[str]
    variables: List[VariableConfig]
    constants: Optional[List[ConstantConfig]] = []
    iterations: int = 1000
    engine_type: str = "ultra"
    original_filename: Optional[str] = None

class BatchSimulationResponse(BaseModel):
    batch_id: str
    simulation_ids: List[str]
    status: str
    message: Optional[str] = None

class MultiTargetSimulationResult(BaseModel):
    """
    Result schema for multi-target Monte Carlo simulations.
    This ensures all targets are calculated with the SAME random values per iteration.
    """
    target_results: Dict[str, List[float]]  # Results for each target
    correlations: Dict[str, Dict[str, float]]  # Target-to-target correlations
    iteration_data: List[Dict[str, float]]  # All values per iteration for correlation analysis
    total_iterations: int
    targets: List[str]
    statistics: Dict[str, TargetStatistics]  # Pre-calculated statistics for each target
    sensitivity_data: Optional[Dict[str, List[Dict[str, Any]]]] = None  # Input variable impact for each target
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_target_statistics(self, target: str) -> TargetStatistics:
        """Get statistics for a specific target"""
        if target in self.statistics:
            return self.statistics[target]
            
        values = self.target_results[target]
        
        # Generate histogram (handle NaN values)
        import math
        valid_values = [v for v in values if not math.isnan(v)]
        
        if len(valid_values) > 0:
            counts, bin_edges = np.histogram(valid_values, bins=50)
            histogram = {
                "bins": bin_edges.tolist(),
                "values": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
                "counts": counts.tolist()
            }
        else:
            # All values are NaN - create empty histogram
            histogram = {
                "bins": [0.0, 1.0],  # Default range
                "values": [0],
                "bin_edges": [0.0, 1.0],
                "counts": [0]
            }
        
        stats = TargetStatistics(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            median=float(np.median(values)),
            percentiles={
                "5": float(np.percentile(values, 5)),
                "10": float(np.percentile(values, 10)),
                "25": float(np.percentile(values, 25)),
                "75": float(np.percentile(values, 75)),
                "90": float(np.percentile(values, 90)),
                "95": float(np.percentile(values, 95))
            },
            histogram=histogram
        )
        return stats
    
    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Get correlation matrix for all targets"""
        return self.correlations
    
    def convert_to_legacy_result(self, primary_target: str) -> SimulationResult:
        """Convert multi-target result to legacy single-target format for backward compatibility"""
        if primary_target not in self.target_results:
            raise ValueError(f"Primary target {primary_target} not found in results")
        
        target_stats = self.get_target_statistics(primary_target)
        values = self.target_results[primary_target]
        
        # Generate histogram (handle NaN values)
        import math
        valid_values = [v for v in values if not math.isnan(v)]
        
        if len(valid_values) > 0:
            counts, bin_edges = np.histogram(valid_values, bins=50)
            histogram = {
                "bins": bin_edges.tolist(),
                "values": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
                "counts": counts.tolist()
            }
        else:
            # All values are NaN - create empty histogram
            histogram = {
                "bins": [0.0, 1.0],  # Default range
                "values": [0],
                "bin_edges": [0.0, 1.0],
                "counts": [0]
            }
        
        # Convert correlations to sensitivity analysis format
        sensitivity_analysis = []
        if primary_target in self.correlations:
            for target, corr_value in self.correlations[primary_target].items():
                sensitivity_analysis.append({
                    "variable": target,
                    "correlation": corr_value,
                    "impact": abs(corr_value)
                })
        
        return SimulationResult(
            mean=target_stats.mean,
            median=target_stats.median,
            std_dev=target_stats.std,
            min_value=target_stats.min,
            max_value=target_stats.max,
            percentiles=target_stats.percentiles,
            histogram=histogram,
            iterations_run=self.total_iterations,
            errors=[],
            sensitivity_analysis=sensitivity_analysis,
            target_display_name=primary_target
        ) 