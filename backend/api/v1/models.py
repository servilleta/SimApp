"""
B2B API v1 - Pydantic models for request/response validation
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DistributionType(str, Enum):
    """Supported distribution types for Monte Carlo variables."""
    TRIANGULAR = "triangular"
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"


class Distribution(BaseModel):
    """Distribution configuration for a Monte Carlo variable."""
    type: DistributionType
    # Triangular distribution parameters
    min: Optional[float] = Field(None, description="Minimum value (triangular/uniform)")
    mode: Optional[float] = Field(None, description="Most likely value (triangular)")
    max: Optional[float] = Field(None, description="Maximum value (triangular/uniform)")
    # Normal distribution parameters
    mean: Optional[float] = Field(None, description="Mean value (normal/lognormal)")
    std: Optional[float] = Field(None, description="Standard deviation (normal/lognormal)")


class MonteCarloVariable(BaseModel):
    """Monte Carlo variable definition."""
    cell: str = Field(..., description="Cell reference (e.g., 'B5' or 'Sheet2!C10' for multi-sheet)")
    name: Optional[str] = Field(None, description="Human-readable variable name")
    distribution: Distribution


class ModelUploadResponse(BaseModel):
    """Response from model upload endpoint."""
    model_id: str = Field(..., description="Unique model identifier")
    status: str = Field(..., description="Upload status")
    processing_time_estimate: str = Field(..., description="Estimated processing time")
    formulas_count: int = Field(..., description="Number of formulas detected")
    variables_detected: List[Dict[str, Any]] = Field(..., description="Auto-detected variables")
    created_at: datetime = Field(..., description="Upload timestamp")


class SimulationConfig(BaseModel):
    """Monte Carlo simulation configuration."""
    iterations: int = Field(..., ge=1000, le=10000000, description="Number of Monte Carlo iterations")
    variables: List[MonteCarloVariable] = Field(..., description="Variable definitions")
    output_cells: List[str] = Field(..., description="Cells to analyze (e.g., ['J25', 'Sheet2!K25'] for multi-sheet)")
    confidence_levels: List[float] = Field(default=[0.95, 0.99], description="Confidence levels for analysis")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


class SimulationRequest(BaseModel):
    """Request to run Monte Carlo simulation."""
    model_id: str = Field(..., description="Model ID from upload")
    simulation_config: SimulationConfig


class SimulationResponse(BaseModel):
    """Response from simulation start endpoint."""
    simulation_id: str = Field(..., description="Unique simulation identifier")
    status: str = Field(..., description="Simulation status")
    estimated_completion: datetime = Field(..., description="Estimated completion time")
    progress_url: str = Field(..., description="URL to check progress")
    credits_consumed: float = Field(..., description="API credits consumed")


class StatisticalResults(BaseModel):
    """Statistical analysis results for a cell."""
    mean: float
    std: float
    min: float
    max: float
    percentiles: Dict[str, float] = Field(..., description="Percentile values (5, 25, 50, 75, 95)")
    var_95: float = Field(..., description="Value at Risk (95% confidence)")
    var_99: float = Field(..., description="Value at Risk (99% confidence)")


class CellResults(BaseModel):
    """Results for a specific output cell."""
    cell_name: str = Field(..., description="Human-readable cell name")
    statistics: StatisticalResults
    distribution_data: Dict[str, Any] = Field(..., description="Histogram and distribution data")


class SimulationResults(BaseModel):
    """Complete simulation results."""
    simulation_id: str
    status: str
    execution_time: str
    iterations_completed: int
    results: Dict[str, CellResults] = Field(..., description="Results per output cell")
    download_links: Dict[str, str] = Field(..., description="Links to detailed data")
    created_at: datetime


class ProgressUpdate(BaseModel):
    """Real-time progress information."""
    simulation_id: str
    status: str
    progress: Dict[str, Any] = Field(..., description="Progress details")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance information")


class APIError(BaseModel):
    """Standardized API error response."""
    code: int
    message: str
    details: Optional[str] = None
    documentation_url: Optional[str] = None


class HealthCheck(BaseModel):
    """API health check response."""
    status: str
    timestamp: datetime
    version: str
    gpu_available: bool
    system_metrics: Dict[str, Any]
