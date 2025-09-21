"""
🚀 BULLETPROOF PROGRESS SCHEMA
Unified progress data transfer objects to fix frontend/backend schema mismatch
"""

from pydantic import BaseModel
from typing import Dict, Optional, Any
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)

class PhaseProgress(BaseModel):
    """Individual phase progress tracking"""
    stage: str
    progress: float = 0.0
    completed: bool = False

class VariableProgress(BaseModel):
    """Individual variable progress tracking"""
    name: str
    progress: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    iterations: int = 0
    totalIterations: int = 0

class EngineInfo(BaseModel):
    """Engine detection and performance info"""
    engine: Optional[str] = None
    engine_type: Optional[str] = None
    gpu_acceleration: Optional[bool] = None
    detected: bool = False

class FormulaMetrics(BaseModel):
    """Formula analysis metrics"""
    total_formulas: int = 0
    relevant_formulas: int = 0
    analysis_method: Optional[str] = None
    cache_hits: int = 0
    chunks_processed: int = 0

class ProgressDTO(BaseModel):
    """
    🎯 UNIFIED PROGRESS DTO
    Standardized progress response that matches frontend UnifiedProgressTracker expectations
    """
    # Core progress fields (legacy compatibility)
    simulation_id: str
    progress_percentage: float = 0.0
    current_iteration: int = 0
    total_iterations: int = 0
    status: str = "pending"
    
    # Legacy fields for backward compatibility
    message: Optional[str] = None
    stage_description: Optional[str] = None
    
    # Frontend-expected schema
    overallProgress: float = 0.0
    currentStage: str = "Initializing..."
    phases: Dict[str, PhaseProgress] = {}
    variables: Dict[str, VariableProgress] = {}
    
    # Enhanced metadata
    streaming_mode: bool = False
    memory_efficient: bool = False
    stage: Optional[str] = None
    stage_description: Optional[str] = None
    
    # Engine and performance info
    engineInfo: Optional[EngineInfo] = None
    formulaMetrics: Optional[FormulaMetrics] = None
    
    # Timing info
    # Provide both snake_case and camelCase for backward-compatibility
    start_time: Optional[str] = None  # snake_case – expected by frontend
    startTime: Optional[float] = None
    estimatedTimeRemaining: Optional[float] = None
    timestamp: float = time.time()
    
    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"

def create_progress_dto(
    simulation_id: str,
    raw_progress: Dict[str, Any],
    target_variables: Optional[list] = None
) -> ProgressDTO:
    """
    🔧 PROGRESS ADAPTER
    Converts raw backend progress to frontend-expected DTO schema
    """
    
    # Extract raw fields
    progress_percentage = raw_progress.get('progress_percentage', 0.0)
    current_iteration = raw_progress.get('current_iteration', 0)
    total_iterations = raw_progress.get('total_iterations', 0)
    status = raw_progress.get('status', 'pending')
    streaming_mode = raw_progress.get('streaming_mode', False)
    memory_efficient = raw_progress.get('memory_efficient', False)
    stage = raw_progress.get('stage', 'simulation')
    
    # Calculate overall progress and stage description
    overall_progress = progress_percentage
    current_stage = determine_current_stage(progress_percentage, status, stage, streaming_mode)
    
    # Create phases based on progress
    phases = create_phases_from_progress(progress_percentage, status, streaming_mode)
    
    # Create variables tracking
    variables = create_variables_from_progress(
        simulation_id, 
        progress_percentage, 
        current_iteration, 
        total_iterations, 
        status,
        target_variables
    )
    
    # Create engine info
    engine_info = create_engine_info(raw_progress)
    
    # Create formula metrics
    formula_metrics = create_formula_metrics(raw_progress)
    
    # Fix timestamp handling - convert string timestamps to float
    start_time_str = raw_progress.get('start_time') or raw_progress.get('startTime')
    start_time_float = None
    
    if start_time_str:
        try:
            if isinstance(start_time_str, str):
                # Parse ISO timestamp string to float
                from datetime import datetime
                dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                start_time_float = dt.timestamp()
            elif isinstance(start_time_str, (int, float)):
                start_time_float = float(start_time_str)
        except Exception as e:
            logger.warning(f"Failed to parse start_time '{start_time_str}': {e}")
            start_time_float = None
    
    # CRITICAL FIX: Preserve user and filename data for persistence
    dto_data = {
        'simulation_id': simulation_id,
        'progress_percentage': progress_percentage,
        'current_iteration': current_iteration,
        'total_iterations': total_iterations,
        'status': status,
        
        # Legacy fields for backward compatibility
        'message': raw_progress.get('message') or current_stage,
        'stage_description': raw_progress.get('stage_description') or current_stage,
        
        'overallProgress': overall_progress,
        'currentStage': current_stage,
        'phases': phases,
        'variables': variables,
        'streaming_mode': streaming_mode,
        'memory_efficient': memory_efficient,
        'stage': stage,
        'engineInfo': engine_info,
        'formulaMetrics': formula_metrics,
        'start_time': start_time_str,  # Keep string for backward compatibility
        'startTime': start_time_float,  # Use float for proper DTO validation
        'estimatedTimeRemaining': raw_progress.get('estimatedTimeRemaining'),
        
        # CRITICAL FIX: Preserve user and filename data from raw progress
        'user': raw_progress.get('user'),
        'original_filename': raw_progress.get('original_filename'),
        'file_id': raw_progress.get('file_id'),
        'target_variables': raw_progress.get('target_variables'),
    }
    
    # Remove None values to keep clean data
    dto_data = {k: v for k, v in dto_data.items() if v is not None}
    
    return ProgressDTO(**dto_data)

def determine_current_stage(progress: float, status: str, stage: str, streaming: bool) -> str:
    """Determine human-readable current stage"""
    if status == "completed":
        return "Completed"
    elif status == "failed":
        return "Failed"
    elif status == "cancelled":
        return "Cancelled"
    elif progress == 0:
        return "Initializing..."
    elif progress < 5:
        return "Preparing simulation..."
    elif streaming and progress < 100:
        return f"Running Monte Carlo Simulation (Streaming Mode) - {progress:.1f}%"
    elif progress < 100:
        return f"Running Monte Carlo Simulation - {progress:.1f}%"
    else:
        return "Finalizing results..."

def create_phases_from_progress(progress: float, status: str, streaming: bool) -> Dict[str, PhaseProgress]:
    """Create phase tracking from raw progress"""
    phases = {
        "initialization": PhaseProgress(stage="File Upload & Validation"),
        "parsing": PhaseProgress(stage="Parsing Excel File"),
        "smart_analysis": PhaseProgress(stage="Smart Dependency Analysis"),
        "analysis": PhaseProgress(stage="Formula Analysis"),
        "simulation": PhaseProgress(stage="Running Monte Carlo Simulation"),
        "results": PhaseProgress(stage="Generating Results")
    }
    
    if progress >= 100 or status == "completed":
        # All phases complete
        for phase in phases.values():
            phase.progress = 100.0
            phase.completed = True
    elif progress > 0:
        # Simulation phase active
        phases["initialization"].progress = 100.0
        phases["initialization"].completed = True
        phases["parsing"].progress = 100.0
        phases["parsing"].completed = True
        phases["smart_analysis"].progress = 100.0
        phases["smart_analysis"].completed = True
        phases["analysis"].progress = 100.0
        phases["analysis"].completed = True
        phases["simulation"].progress = progress
        phases["simulation"].completed = progress >= 100
    else:
        # Initialization phase
        phases["initialization"].progress = 50.0
    
    return phases

def create_variables_from_progress(
    simulation_id: str, 
    progress: float, 
    current_iteration: int, 
    total_iterations: int, 
    status: str,
    target_variables: Optional[list] = None
) -> Dict[str, VariableProgress]:
    """Create variable tracking from simulation progress - Enhanced for multi-target"""
    
    variables = {}
    
    # Handle multi-target simulations
    if target_variables and len(target_variables) > 1:
        # Create entries for each target variable
        for idx, var_name in enumerate(target_variables):
            var_key = f"{simulation_id}_target_{idx}"
            variables[var_key] = VariableProgress(
                name=var_name,
                progress=progress,
                status=status,
                iterations=current_iteration,
                totalIterations=total_iterations
            )
    elif target_variables and len(target_variables) == 1:
        # Single target
        variables[simulation_id] = VariableProgress(
            name=target_variables[0],
            progress=progress,
            status=status,
            iterations=current_iteration,
            totalIterations=total_iterations
        )
    else:
        # Default fallback
        variables[simulation_id] = VariableProgress(
            name="Target Variable",
            progress=progress,
            status=status,
            iterations=current_iteration,
            totalIterations=total_iterations
        )
    
    return variables

def create_engine_info(raw_progress: Dict[str, Any]) -> EngineInfo:
    """Create engine info from raw progress"""
    # 🔧 CRITICAL FIX: Respect explicit engineInfo from backend - NO AUTO-DETECTION OVERRIDE
    
    # Extract engine info from nested engineInfo if present (HIGHEST PRIORITY)
    nested_engine_info = raw_progress.get('engineInfo', {})
    
    # If we have ANY engineInfo from backend, use it directly (even partial)
    if nested_engine_info:
        return EngineInfo(
            engine=nested_engine_info.get('engine', 'WorldClassMonteCarloEngine'),
            engine_type=nested_engine_info.get('engine_type', 'Enhanced'),
            gpu_acceleration=nested_engine_info.get('gpu_acceleration', False),
            detected=nested_engine_info.get('detected', True)
        )
    
    # Only use explicit top-level fields if no nested engineInfo exists
    engine_name = raw_progress.get('engine', 'WorldClassMonteCarloEngine')
    engine_type = raw_progress.get('engine_type')
    
    # If streaming_mode flag is present and no explicit engine_type set, classify as Streaming
    if raw_progress.get('streaming_mode') and not engine_type:
        return EngineInfo(
            engine=engine_name,
            engine_type='Streaming',
            gpu_acceleration=False,  # Streaming engine uses vectorized CPU/Arrow paths
            detected=True
        )
    
    # If we have explicit engine_type at top level, respect it completely
    if engine_type:
        gpu_acceleration = raw_progress.get('gpu_acceleration')
        if gpu_acceleration is None:
            # Default based on engine type
            if engine_type == 'Enhanced':
                gpu_acceleration = True   # Enhanced uses GPU acceleration
            elif engine_type == 'Ultra':
                gpu_acceleration = True   # Ultra uses GPU acceleration
            else:
                gpu_acceleration = False  # Standard uses CPU only
        
        return EngineInfo(
            engine=engine_name,
            engine_type=engine_type,
            gpu_acceleration=gpu_acceleration,
            detected=True
        )
    
    # LAST RESORT: Only auto-detect if NO explicit engine info exists anywhere
    # This should rarely happen with our fixes
    if 'Ultra' in engine_name or 'UltraMonteCarloEngine' in engine_name:
        engine_type = 'Ultra'
        gpu_acceleration = True
    elif 'WorldClass' in engine_name or 'Enhanced' in engine_name:
        engine_type = 'Enhanced'
        gpu_acceleration = True
    else:
        engine_type = 'Standard'
        gpu_acceleration = False
    
    return EngineInfo(
        engine=engine_name,
        engine_type=engine_type,
        gpu_acceleration=gpu_acceleration,
        detected=True
    )

def create_formula_metrics(raw_progress: Dict[str, Any]) -> FormulaMetrics:
    """Create formula metrics from raw progress"""
    return FormulaMetrics(
        total_formulas=raw_progress.get('total_formulas', 0),
        relevant_formulas=raw_progress.get('relevant_formulas', 0),
        analysis_method=raw_progress.get('analysis_method', 'Streaming' if raw_progress.get('streaming_mode') else 'Standard'),
        cache_hits=raw_progress.get('cache_hits', 0),
        chunks_processed=raw_progress.get('chunks_processed', 0)
    ) 