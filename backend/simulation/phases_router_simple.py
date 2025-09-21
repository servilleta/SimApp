"""
Simplified phase-based simulation API endpoints.
This version works without complex modular dependencies.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
import asyncio
import logging
import json
import uuid
import os
from datetime import datetime

# Simple imports that should work
from simulation.schemas import SimulationRequest, SimulationResult
from excel_parser.service import get_formulas_for_file, get_constants_for_file
from simulation.formula_utils import get_evaluation_order
from shared.progress_store import get_progress, set_progress

logger = logging.getLogger(__name__)
router = APIRouter()

# Store phase results in memory (in production, use Redis)
phase_results = {}

from pydantic import BaseModel

class Phase1Request(BaseModel):
    file_id: str

class Phase2Request(BaseModel):
    file_id: str

class Phase3Request(BaseModel):
    file_id: str
    target_cells: List[str]
    target_sheet: str

class Phase4Request(BaseModel):
    file_id: str
    target_cells: List[str]
    target_sheet: str

@router.post("/phase1/validate")
async def phase1_file_validation(request: Phase1Request) -> Dict[str, Any]:
    """Phase 1: File Upload & Validation"""
    try:
        file_id = request.file_id
        logger.info(f"🔍 Phase 1: Validating file {file_id}")
        
        # Check if file exists
        file_path = os.path.join("uploads", f"{file_id}.xlsx")
        if not os.path.exists(file_path):
            # Try with the original filename pattern
            import glob
            pattern = os.path.join("uploads", f"{file_id}_*.xlsx")
            files = glob.glob(pattern)
            if files:
                file_path = files[0]
            else:
                raise HTTPException(status_code=404, detail="File not found")
        
        # Basic validation
        file_stats = os.stat(file_path)
        
        result = {
            "status": "success",
            "file_id": file_id,
            "file_size": file_stats.st_size,
            "file_path": file_path,
            "validated_at": datetime.utcnow().isoformat()
        }
        
        # Store result
        phase_key = f"{file_id}_phase1"
        phase_results[phase_key] = result
        
        return result
        
    except Exception as e:
        logger.error(f"Phase 1 error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/phase2/parse")
async def phase2_parse_excel(request: Phase2Request) -> Dict[str, Any]:
    """Phase 2: Parse Excel File"""
    try:
        file_id = request.file_id
        logger.info(f"📄 Phase 2: Parsing Excel file {file_id}")
        
        # Get formulas
        formulas = await get_formulas_for_file(file_id)
        
        # Get constants
        constants = await get_constants_for_file(file_id)
        
        # Count formulas and constants
        formula_count = sum(len(sheet_formulas) for sheet_formulas in formulas.values())
        constant_count = len(constants)
        
        result = {
            "status": "success",
            "file_id": file_id,
            "formula_count": formula_count,
            "constant_count": constant_count,
            "sheets": list(formulas.keys()),
            "parsed_at": datetime.utcnow().isoformat()
        }
        
        # Store detailed data
        phase_key = f"{file_id}_phase2"
        phase_results[phase_key] = {
            **result,
            "formulas": formulas,
            "constants": constants
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Phase 2 error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/phase3/dependency")
async def phase3_dependency_analysis(request: Phase3Request) -> Dict[str, Any]:
    """Phase 3: Smart Dependency Analysis"""
    try:
        file_id = request.file_id
        target_cells = request.target_cells
        target_sheet = request.target_sheet
        logger.info(f"🔗 Phase 3: Dependency analysis for {target_cells} in {file_id}")
        
        # Get phase 2 results
        phase2_key = f"{file_id}_phase2"
        if phase2_key not in phase_results:
            raise HTTPException(status_code=400, detail="Phase 2 not completed")
        
        formulas = phase_results[phase2_key]["formulas"]
        
        # Analyze dependencies for each target
        dependency_info = {}
        total_dependencies = 0
        
        for target_cell in target_cells:
            try:
                # Get evaluation order
                evaluation_order = get_evaluation_order(
                    target_sheet,    # Note: parameter order is different
                    target_cell,
                    formulas,
                    set(),           # Empty set for mc_input_cells
                    engine_type="power"  # Use enhanced limits
                )
                
                # For now, we'll use evaluation_order length for both metrics
                dependency_info[target_cell] = {
                    "formulas_in_chain": len(evaluation_order),
                    "direct_dependencies": len(evaluation_order)  # Simplified
                }
                total_dependencies += len(evaluation_order)
            except Exception as e:
                logger.warning(f"Could not analyze dependencies for {target_cell}: {e}")
                dependency_info[target_cell] = {
                    "formulas_in_chain": 0,
                    "direct_dependencies": 0,
                    "error": str(e)
                }
        
        result = {
            "status": "success",
            "file_id": file_id,
            "target_cells": target_cells,
            "total_dependencies": total_dependencies,
            "dependency_info": dependency_info,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        # Store detailed data
        phase_key = f"{file_id}_phase3"
        phase_results[phase_key] = result
        
        return result
        
    except Exception as e:
        logger.error(f"Phase 3 error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/phase4/formula-analysis")
async def phase4_formula_analysis(request: Phase4Request) -> Dict[str, Any]:
    """Phase 4: Formula Analysis & Optimization"""
    try:
        file_id = request.file_id
        target_cells = request.target_cells
        target_sheet = request.target_sheet
        logger.info(f"📊 Phase 4: Formula analysis for {target_cells}")
        
        # Get previous phase results
        phase2_key = f"{file_id}_phase2"
        if phase2_key not in phase_results:
            raise HTTPException(status_code=400, detail="Phase 2 not completed")
        
        phase2_data = phase_results[phase2_key]
        formulas = phase2_data["formulas"]
        
        # Analyze formula complexity
        formula_stats = {
            "simple": 0,
            "complex": 0,
            "gpu_eligible": 0,
            "functions_used": set()
        }
        
        for sheet_formulas in formulas.values():
            for formula in sheet_formulas.values():
                # Simple analysis
                if any(func in formula.upper() for func in ["SUM", "AVERAGE", "COUNT", "MIN", "MAX"]):
                    formula_stats["simple"] += 1
                    formula_stats["gpu_eligible"] += 1
                elif any(func in formula.upper() for func in ["VLOOKUP", "HLOOKUP", "INDEX", "MATCH"]):
                    formula_stats["complex"] += 1
                else:
                    formula_stats["simple"] += 1
                
                # Extract functions
                import re
                functions = re.findall(r'([A-Z]+)\(', formula.upper())
                formula_stats["functions_used"].update(functions)
        
        result = {
            "status": "success",
            "file_id": file_id,
            "formula_stats": {
                **formula_stats,
                "functions_used": list(formula_stats["functions_used"])
            },
            "optimization_potential": formula_stats["gpu_eligible"] / max(1, formula_stats["simple"] + formula_stats["complex"]),
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        # Store result
        phase_key = f"{file_id}_phase4"
        phase_results[phase_key] = result
        
        return result
        
    except Exception as e:
        logger.error(f"Phase 4 error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/phase5/simulate")
async def phase5_run_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Phase 5: Run Monte Carlo Simulation"""
    try:
        logger.info(f"🎲 Phase 5: Running simulation for {request.result_cell_coordinate}")
        
        # Check previous phases
        phase2_key = f"{request.file_id}_phase2"
        if phase2_key not in phase_results:
            raise HTTPException(
                status_code=400, 
                detail="Previous phases not completed. Please run phases 1-4 first."
            )
        
        # Create simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Initialize progress
        set_progress(simulation_id, {
            "status": "running",
            "stage": "simulation",
            "progress_percentage": 0,
            "current_iteration": 0,
            "total_iterations": request.iterations,
            "message": "Starting simulation..."
        })
        
        # Run simulation in background
        background_tasks.add_task(
            run_power_simulation_task,
            simulation_id,
            request,
            phase_results[phase2_key]
        )
        
        return {
            "status": "started",
            "simulation_id": simulation_id,
            "target_cell": request.result_cell_coordinate,
            "iterations": request.iterations,
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Phase 5 error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/phase6/results/{simulation_id}")
async def phase6_generate_results(simulation_id: str) -> Dict[str, Any]:
    """Phase 6: Generate Final Results"""
    try:
        logger.info(f"📈 Phase 6: Generating results for {simulation_id}")
        
        # Get simulation progress
        progress = get_progress(simulation_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        if progress.get("status") != "completed":
            return {
                "status": "pending",
                "message": "Simulation still running",
                "progress": progress.get("progress_percentage", 0)
            }
        
        # Get raw results
        raw_results_key = f"{simulation_id}_raw_results"
        if raw_results_key not in phase_results:
            raise HTTPException(status_code=404, detail="Raw results not found")
        
        raw_results = phase_results[raw_results_key]
        
        # Generate enhanced results
        import numpy as np
        values = raw_results["values"]
        
        # Calculate statistics
        result = {
            "status": "success",
            "simulation_id": simulation_id,
            "statistics": {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "percentiles": {
                    "p5": float(np.percentile(values, 5)),
                    "p25": float(np.percentile(values, 25)),
                    "p50": float(np.percentile(values, 50)),
                    "p75": float(np.percentile(values, 75)),
                    "p95": float(np.percentile(values, 95))
                }
            },
            "histogram": {
                "counts": np.histogram(values, bins=50)[0].tolist(),
                "bin_edges": np.histogram(values, bins=50)[1].tolist()
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Phase 6 error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_power_simulation_task(
    simulation_id: str,
    request: SimulationRequest,
    phase2_data: Dict
):
    """Background task to run Power Engine simulation"""
    try:
        from simulation.power_engine import PowerMonteCarloEngine
        
        # Initialize Power Engine
        engine = PowerMonteCarloEngine()
        
        # Get data from phase 2
        formulas = phase2_data["formulas"]
        constants = phase2_data["constants"]
        
        # Run simulation
        result = await engine.run_simulation(
            simulation_id=simulation_id,
            file_id=request.file_id,
            variables=request.variables,
            iterations=request.iterations,
            target_cell_coordinate=request.result_cell_coordinate,
            target_sheet_name=request.result_cell_sheet_name,
            formulas=formulas,
            constants=constants
        )
        
        # Store raw results
        raw_results_key = f"{simulation_id}_raw_results"
        phase_results[raw_results_key] = {
            "values": result.raw_values if hasattr(result, 'raw_values') else [],
            "sensitivity": result.sensitivity_analysis if hasattr(result, 'sensitivity_analysis') else None
        }
        
        # Update progress
        set_progress(simulation_id, {
            "status": "completed",
            "progress_percentage": 100,
            "message": "Simulation completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Simulation task error: {str(e)}")
        set_progress(simulation_id, {
            "status": "failed",
            "error": str(e),
            "message": f"Simulation failed: {str(e)}"
        })

@router.get("/debug/phase-results/{file_id}")
async def get_phase_results(file_id: str) -> Dict[str, Any]:
    """Debug endpoint to see all phase results for a file"""
    results = {}
    for key, value in phase_results.items():
        if key.startswith(file_id):
            # Don't include raw formula/constant data in debug output
            if isinstance(value, dict):
                filtered_value = {k: v for k, v in value.items() 
                                if k not in ['formulas', 'constants', 'raw_values']}
                results[key] = filtered_value
            else:
                results[key] = value
    return results 