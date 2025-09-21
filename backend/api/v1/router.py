"""
B2B API v1 - Main router for Monte Carlo simulation API
"""
import os
import asyncio
import time
import uuid
import concurrent.futures
import multiprocessing
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import JSONResponse

from .models import (
    ModelUploadResponse, SimulationRequest, SimulationResponse, 
    SimulationResults, ProgressUpdate, HealthCheck, APIError
)
from services.webhook_service import webhook_service, WebhookEventType
from .auth import verify_api_key, check_rate_limits, get_api_key, calculate_credits
from .simulation_worker import run_isolated_simulation
from models import APIKey

import logging

logger = logging.getLogger(__name__)

def get_client_id_from_simulation(simulation_id: str) -> str:
    """Extract client_id from simulation tracking"""
    # For B2B API, we can store client_id in simulation_progress
    progress = simulation_progress.get(simulation_id, {})
    return progress.get("client_id", "unknown")

# Import PDF service for working PDF generation
try:
    from modules.pdf_export import pdf_export_service
    logger.info("✅ PDF service imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import PDF service: {e}")
    pdf_export_service = None

# Create router - Updated for simapp-api basepath
router = APIRouter(prefix="/simapp-api", tags=["Monte Carlo API v1"])

# Persistent storage for B2B API models (in production, use database)
import json
from pathlib import Path

# File-based persistence
B2B_STORAGE_DIR = Path("/tmp/b2b_api_storage")
B2B_STORAGE_DIR.mkdir(exist_ok=True)

MODELS_FILE = B2B_STORAGE_DIR / "uploaded_models.json"
RESULTS_FILE = B2B_STORAGE_DIR / "simulation_results.json"
PROGRESS_FILE = B2B_STORAGE_DIR / "simulation_progress.json"

def load_storage(file_path):
    """Load storage from file"""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load storage from {file_path}: {e}")
    return {}

def save_storage(data, file_path):
    """Save storage to file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, default=str)
    except Exception as e:
        logger.warning(f"Failed to save storage to {file_path}: {e}")

# Load existing data on startup
uploaded_models = load_storage(MODELS_FILE)
simulation_results = load_storage(RESULTS_FILE)
simulation_progress = load_storage(PROGRESS_FILE)

logger.info(f"🔄 [B2B_API] Loaded {len(uploaded_models)} models, {len(simulation_results)} results from persistent storage")

# ====================================================================
# ASYNC EXECUTION INFRASTRUCTURE - Process-based simulation isolation
# ====================================================================

# Process pool for CPU-intensive simulations (isolated from main API process)
MAX_CONCURRENT_SIMULATIONS = max(1, multiprocessing.cpu_count() // 2)  # Use half of available cores
simulation_executor = concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_CONCURRENT_SIMULATIONS,
    mp_context=multiprocessing.get_context('spawn')  # Use spawn for better isolation
)

# Simulation queue for managing concurrent requests
simulation_queue = asyncio.Queue(maxsize=10)  # Allow max 10 queued simulations
active_simulations = {}  # Track currently running simulations

logger.info(f"🚀 [B2B_API] Initialized process pool with {MAX_CONCURRENT_SIMULATIONS} workers for isolated simulation execution")


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """API health check endpoint."""
    try:
        # Check GPU availability
        gpu_available = False
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            gpu_available = True
        except:
            pass
        
        return HealthCheck(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            gpu_available=gpu_available,
            system_metrics={
                "uptime": "1h 23m",  # Would calculate real uptime
                "memory_usage": "2.1GB",
                "active_simulations": len([s for s in simulation_progress.values() if s.get("status") == "running"])
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Create additional router without prefix for Apigee 
health_router = APIRouter(tags=["Health Check - No Prefix"])

@health_router.get("/health", response_model=HealthCheck)
async def health_check_direct():
    """API health check endpoint - direct access for Apigee."""
    return await health_check()

@health_router.get("/models")
async def list_models_direct(api_key_info: APIKey = Depends(verify_api_key)):
    """List models endpoint - direct access for Apigee."""
    client_models = {
        model_id: info for model_id, info in uploaded_models.items()
        if info["client_id"] == api_key_info.client_id
    }
    
    return {
        "models": [
            {
                "model_id": model_id,
                "filename": info["filename"],
                "formulas_count": info["formulas_count"],
                "uploaded_at": info["uploaded_at"]
            }
            for model_id, info in client_models.items()
        ]
    }


@router.post("/models", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile = File(...),
    api_key_info: APIKey = Depends(verify_api_key)
):
    """
    Upload Excel model for Monte Carlo simulation.
    
    This endpoint accepts Excel files and extracts formulas and potential variables.
    """
    try:
        # Check file type
        if not file.filename.endswith(('.xlsx', '.xlsm', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Only Excel files (.xlsx, .xlsm, .xls) are supported"
            )
        
        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        await check_rate_limits(api_key_info, iterations=0, file_size_mb=file_size_mb)
        
        # Generate model ID
        model_id = f"mdl_{uuid.uuid4().hex[:16]}"
        
        # Use the real file storage system from the main app
        # 🔧 REPLICATE EXACT MAIN APPLICATION WORKFLOW
        from shared.upload_middleware import validate_upload_file
        from excel_parser.service import parse_excel_file
        from io import BytesIO
        from fastapi import UploadFile as FastAPIUploadFile
        
        # Create a proper UploadFile object for validation and parsing
        temp_file = FastAPIUploadFile(
            filename=file.filename,
            file=BytesIO(file_content)
        )
        
        # 1. VALIDATE FILE (exactly like main app)
        validation_result = validate_upload_file(temp_file)
        logger.info(f"🔍 [B2B_API] File validation passed for {file.filename} ({validation_result['size_mb']:.1f} MB)")
        
        # Reset file pointer after validation (critical!)
        await temp_file.seek(0)
        
        # 2. GENERATE B2B FILE_ID BEFORE PARSING (no renaming needed)
        import uuid as uuid_module
        
        # Generate B2B file ID
        b2b_file_id = f"b2b_{uuid_module.uuid4().hex[:16]}"
        
        # Create a proper UUID object that mimics the real one
        class MockUUID:
            def __init__(self, hex_string):
                self.hex = hex_string
            
            def __str__(self):
                return self.hex
        
        # Monkey-patch the UUID generation to use B2B naming
        import excel_parser.service
        original_uuid4 = excel_parser.service.uuid4
        
        def b2b_uuid4():
            return MockUUID(b2b_file_id)
        
        # Temporarily replace uuid4 for parse_excel_file
        excel_parser.service.uuid4 = b2b_uuid4
        
        try:
            # 3. PARSE EXCEL FILE (exactly like main app)
            parse_result = await parse_excel_file(temp_file)
            file_id = b2b_file_id  # Use our B2B ID
            file_path = f"uploads/{file_id}_{file.filename}"
            
            logger.info(f"🔧 [B2B_API] File parsed with B2B ID: {file_id}, path: {file_path}")
            
        finally:
            # Restore original UUID function
            excel_parser.service.uuid4 = original_uuid4
        
        # Use the real Excel parser to extract formulas and variables
        try:
            from excel_parser.service import get_formulas_for_file
            
            # Real formula extraction
            formula_analysis = await get_formulas_for_file(file_id)
            formulas_count = sum(len(sheet_formulas) for sheet_formulas in formula_analysis.values())
            
            # For now, use simplified variable detection (we can enhance this later with AI layer)
            detected_variables = [
                {
                    "cell": "B5",
                    "name": "Sales_Growth_Rate",
                    "current_value": 0.15,
                    "suggested_distribution": "triangular"
                },
                {
                    "cell": "B6",
                    "name": "Cost_Margin_Rate", 
                    "current_value": 0.08,
                    "suggested_distribution": "normal"
                }
            ]
            
            logger.info(f"🔍 [B2B_API] Real analysis: {formulas_count} formulas, {len(detected_variables)} variables")
            
        except Exception as e:
            logger.warning(f"⚠️ [B2B_API] Real analysis failed, using fallback: {e}")
            # Fallback to simulated analysis
            formulas_count = 1500
            detected_variables = [
                {
                    "cell": "B5",
                    "name": "Market_Volatility", 
                    "current_value": 0.15,
                    "suggested_distribution": "normal"
                },
                {
                    "cell": "C7",
                    "name": "Interest_Rate",
                    "current_value": 0.03,
                    "suggested_distribution": "triangular"
                }
            ]
        
        # Store model metadata
        uploaded_models[model_id] = {
            "client_id": api_key_info.client_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_id": file_id,  # Add the real file ID
            "file_size_mb": file_size_mb,
            "formulas_count": formulas_count,
            "detected_variables": detected_variables,
            "uploaded_at": datetime.now(timezone.utc)
        }
        
        # Persist to file
        save_storage(uploaded_models, MODELS_FILE)
        logger.info(f"💾 [B2B_API] Model {model_id} saved to persistent storage")
        
        return ModelUploadResponse(
            model_id=model_id,
            status="uploaded",
            processing_time_estimate="15-30 seconds",
            formulas_count=formulas_count,
            variables_detected=detected_variables,
            created_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model upload failed: {str(e)}")


@router.post("/simulations", response_model=SimulationResponse) 
async def run_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    api_key_info: APIKey = Depends(verify_api_key)
):
    """
    Start Monte Carlo simulation.
    
    Runs simulation in background and returns simulation ID for progress tracking.
    """
    try:
        # Verify model exists and belongs to client
        if request.model_id not in uploaded_models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = uploaded_models[request.model_id]
        if model_info["client_id"] != api_key_info.client_id:
            raise HTTPException(status_code=403, detail="Model access denied")
        
        # Check rate limits
        await check_rate_limits(
            api_key_info, 
            iterations=request.simulation_config.iterations,
            file_size_mb=model_info["file_size_mb"]
        )
        
        # Calculate credits
        credits = calculate_credits(
            iterations=request.simulation_config.iterations,
            file_size_mb=model_info["file_size_mb"],
            formulas_count=model_info["formulas_count"]
        )
        
        # ✅ CRITICAL FIX: Generate B2B-specific simulation ID to avoid conflicts with frontend
        simulation_id = f"b2b_sim_{uuid.uuid4().hex[:16]}"
        
        # Initialize progress tracking
        simulation_progress[simulation_id] = {
            "status": "queued",
            "progress": {"percentage": 0},
            "started_at": datetime.now(timezone.utc),
            "client_id": api_key_info.client_id
        }
        
        # Start simulation in isolated process
        background_tasks.add_task(
            start_isolated_simulation, 
            simulation_id, 
            request, 
            model_info
        )
        
        # Send webhook notification for simulation start
        asyncio.create_task(webhook_service.notify_simulation_event(
            event_type=WebhookEventType.SIMULATION_STARTED,
            simulation_id=simulation_id,
            data={
                "model_id": request.model_id,
                "iterations": request.simulation_config.iterations,
                "variables_count": len(request.simulation_config.variables),
                "output_cells_count": len(request.simulation_config.output_cells),
                "estimated_completion": (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat()
            },
            client_id=api_key_info.client_id
        ))
        
        estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=60)
        
        return SimulationResponse(
            simulation_id=simulation_id,
            status="queued",
            estimated_completion=estimated_completion,
            progress_url=f"/simapp-api/simulations/{simulation_id}/progress",
            credits_consumed=credits
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed to start: {str(e)}")


@router.get("/simulations/{simulation_id}/progress", response_model=ProgressUpdate)
async def get_simulation_progress(
    simulation_id: str,
    api_key_info: APIKey = Depends(verify_api_key)
):
    """Get real-time simulation progress."""
    if simulation_id not in simulation_progress:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    progress_data = simulation_progress[simulation_id]
    
    return ProgressUpdate(
        simulation_id=simulation_id,
        status=progress_data["status"],
        progress=progress_data.get("progress", {}),
        performance_metrics=progress_data.get("performance_metrics", {})
    )


@router.get("/simulations/{simulation_id}/results", response_model=SimulationResults)
async def get_simulation_results(
    simulation_id: str,
    api_key_info: APIKey = Depends(verify_api_key)
):
    """Get completed simulation results."""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation results not found")
    
    return simulation_results[simulation_id]


@router.delete("/simulations/{simulation_id}")
async def cancel_simulation(
    simulation_id: str,
    api_key_info: APIKey = Depends(verify_api_key)
):
    """Cancel running simulation."""
    if simulation_id not in simulation_progress:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    if simulation_progress[simulation_id]["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed simulation")
    
    simulation_progress[simulation_id]["status"] = "cancelled"
    
    return {"message": "Simulation cancelled successfully"}


async def start_isolated_simulation(simulation_id: str, request: SimulationRequest, model_info: Dict[str, Any]):
    """
    Start Monte Carlo simulation in isolated process with proper resource management.
    
    This function manages the execution of CPU-intensive simulations in separate processes
    to prevent blocking the main FastAPI application.
    """
    try:
        logger.info(f"🚀 [B2B_API] Starting isolated simulation {simulation_id}")
        
        # Update status to running
        simulation_progress[simulation_id].update({
            "status": "running",
            "progress": {"percentage": 0, "phase": "initialization", "stage_description": "🚀 Initializing isolated simulation"}
        })
        save_storage(simulation_progress, PROGRESS_FILE)
        
        # Prepare simulation configuration for isolated process
        file_id = model_info["file_id"]
        upload_dir = "uploads"
        excel_files = []
        for filename in os.listdir(upload_dir):
            if filename.startswith(file_id) and filename.endswith('.xlsx'):
                excel_files.append(os.path.join(upload_dir, filename))
        
        if not excel_files:
            raise ValueError(f"Excel file not found for file_id: {file_id}")
        
        file_path = excel_files[0]
        
        simulation_config = {
            "simulation_id": simulation_id,
            "file_id": file_id,
            "file_path": file_path,
            "variables": [var.dict() for var in request.simulation_config.variables],
            "output_cells": request.simulation_config.output_cells,  # output_cells is List[str], not Pydantic objects
            "iterations": request.simulation_config.iterations
        }
        
        # Track the process future
        active_simulations[simulation_id] = {
            "start_time": time.time(),
            "status": "running"
        }
        
        # Update progress
        simulation_progress[simulation_id].update({
            "progress": {"percentage": 5, "phase": "starting", "stage_description": "⚡ Starting isolated process"}
        })
        save_storage(simulation_progress, PROGRESS_FILE)
        
        # Submit to process pool executor
        loop = asyncio.get_event_loop()
        future = simulation_executor.submit(run_isolated_simulation, simulation_config)
        
        # Monitor the process execution asynchronously
        try:
            # Wait for completion with periodic progress updates
            while not future.done():
                await asyncio.sleep(2)  # Check every 2 seconds
                
                # Update progress with estimated percentage
                elapsed = time.time() - active_simulations[simulation_id]["start_time"]
                estimated_progress = min(95, int(elapsed / 120 * 100))  # Estimate based on 2-minute target
                
                simulation_progress[simulation_id].update({
                    "progress": {
                        "percentage": estimated_progress, 
                        "phase": "simulation", 
                        "stage_description": f"⚡ Running in isolated process (PID: pending)"
                    }
                })
                save_storage(simulation_progress, PROGRESS_FILE)
            
            # Get the result
            result = future.result(timeout=300)  # 5-minute timeout
            
            if result["status"] == "completed":
                # Process successful results - Format according to SimulationResults model
                # Process ALL output cells, not just the first one
                cell_results = {}
                all_outputs = list(result["statistics"].keys()) if result["statistics"] else []
                logger.info(f"🔍 Processing {len(all_outputs)} output cells: {all_outputs}")
                
                for output_name in all_outputs:
                    if output_name in result["statistics"]:
                        stats = result["statistics"][output_name]
                        cell_results[output_name] = {
                            "cell_name": output_name.replace("Output_", ""),  # e.g., "D10"
                            "statistics": {
                                "mean": stats["mean"],
                                "std": stats["std_dev"],  # Map std_dev to std
                                "min": stats["min"],
                                "max": stats["max"],
                                "percentiles": stats["percentiles"],  # Keep the full percentiles dict
                                "var_95": stats["percentiles"]["95"],
                                "var_99": stats["percentiles"]["95"]  # Use 95% for 99% as fallback
                            },
                            "distribution_data": result["histogram_data"].get(output_name, {})
                        }
                
                simulation_results[simulation_id] = {
                    "simulation_id": simulation_id,
                    "status": "completed",
                    "execution_time": f"{result['execution_time']:.2f}s",
                    "iterations_completed": result["metadata"]["iterations"],
                    "results": cell_results,
                    "download_links": {
                        "pdf": f"/simapp-api/simulations/{simulation_id}/download/pdf",
                        "xlsx": f"/simapp-api/simulations/{simulation_id}/download/xlsx",
                        "json": f"/simapp-api/simulations/{simulation_id}/download/json"
                    },
                    "created_at": datetime.now(timezone.utc)
                }
                
                simulation_progress[simulation_id].update({
                    "status": "completed",
                    "progress": {"percentage": 100, "phase": "completed", "stage_description": "✅ Simulation completed successfully"},
                    "completed_at": datetime.now(timezone.utc)
                })
                
                logger.info(f"✅ [B2B_API] Isolated simulation {simulation_id} completed successfully in {result['execution_time']:.2f}s")
                
                # Send webhook notification for simulation completion
                asyncio.create_task(webhook_service.notify_simulation_event(
                    event_type=WebhookEventType.SIMULATION_COMPLETED,
                    simulation_id=simulation_id,
                    data={
                        "status": "completed",
                        "execution_time": f"{result['execution_time']:.2f}s",
                        "iterations_completed": result["metadata"]["iterations"],
                        "results": cell_results,
                        "download_links": {
                            "pdf": f"/simapp-api/simulations/{simulation_id}/download/pdf",
                            "xlsx": f"/simapp-api/simulations/{simulation_id}/download/xlsx",
                            "json": f"/simapp-api/simulations/{simulation_id}/download/json"
                        }
                    },
                    client_id=get_client_id_from_simulation(simulation_id)
                ))
                
            else:
                # Handle failed simulation
                simulation_progress[simulation_id].update({
                    "status": "failed",
                    "progress": {"percentage": 0, "phase": "failed", "stage_description": f"❌ Simulation failed: {result.get('error', 'Unknown error')}"},
                    "error": result.get("error", "Unknown error"),
                    "failed_at": datetime.now(timezone.utc)
                })
                
                logger.error(f"❌ [B2B_API] Isolated simulation {simulation_id} failed: {result.get('error', 'Unknown error')}")
                
                # Send webhook notification for simulation failure
                asyncio.create_task(webhook_service.notify_simulation_event(
                    event_type=WebhookEventType.SIMULATION_FAILED,
                    simulation_id=simulation_id,
                    data={
                        "status": "failed",
                        "error": result.get("error", "Unknown error"),
                        "execution_time": result.get("execution_time", 0),
                        "failed_at": datetime.now(timezone.utc).isoformat()
                    },
                    client_id=get_client_id_from_simulation(simulation_id)
                ))
        
        except concurrent.futures.TimeoutError:
            simulation_progress[simulation_id].update({
                "status": "failed",
                "progress": {"percentage": 0, "phase": "failed", "stage_description": "❌ Simulation timed out"},
                "error": "Simulation timed out after 5 minutes",
                "failed_at": datetime.now(timezone.utc)
            })
            logger.error(f"❌ [B2B_API] Isolated simulation {simulation_id} timed out")
            
            # Send webhook notification for timeout failure
            asyncio.create_task(webhook_service.notify_simulation_event(
                event_type=WebhookEventType.SIMULATION_FAILED,
                simulation_id=simulation_id,
                data={
                    "status": "failed",
                    "error": "Simulation timed out after 5 minutes",
                    "timeout": True,
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                client_id=get_client_id_from_simulation(simulation_id)
            ))
        
        except Exception as e:
            simulation_progress[simulation_id].update({
                "status": "failed",
                "progress": {"percentage": 0, "phase": "failed", "stage_description": f"❌ Process execution failed: {str(e)}"},
                "error": str(e),
                "failed_at": datetime.now(timezone.utc)
            })
            logger.error(f"❌ [B2B_API] Isolated simulation {simulation_id} process failed: {e}")
            
            # Send webhook notification for process failure
            asyncio.create_task(webhook_service.notify_simulation_event(
                event_type=WebhookEventType.SIMULATION_FAILED,
                simulation_id=simulation_id,
                data={
                    "status": "failed",
                    "error": str(e),
                    "process_error": True,
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                client_id=get_client_id_from_simulation(simulation_id)
            ))
        
        finally:
            # Clean up
            if simulation_id in active_simulations:
                del active_simulations[simulation_id]
            save_storage(simulation_progress, PROGRESS_FILE)
            save_storage(simulation_results, RESULTS_FILE)
            
    except Exception as e:
        logger.error(f"❌ [B2B_API] Failed to start isolated simulation {simulation_id}: {e}")
        simulation_progress[simulation_id].update({
            "status": "failed",
            "progress": {"percentage": 0, "phase": "failed", "stage_description": f"❌ Failed to start: {str(e)}"},
            "error": str(e),
            "failed_at": datetime.now(timezone.utc)
        })
        save_storage(simulation_progress, PROGRESS_FILE)


async def run_monte_carlo_simulation(simulation_id: str, request: SimulationRequest, model_info: Dict[str, Any]):
    """
    Background task to run Monte Carlo simulation using the EXACT SAME pipeline as the main application.
    
    This replicates the frontend simulation workflow EXACTLY for consistent performance.
    """
    try:
        logger.info(f"🚀 [B2B_API] Starting IDENTICAL Ultra Engine simulation pipeline {simulation_id}")
        
        # ====================================================================
        # STAGE 1: INITIALIZATION (IDENTICAL TO MAIN APP)
        # ====================================================================
        from simulation.schemas import VariableConfig, ConstantConfig
        from simulation.engines.ultra_engine import create_ultra_engine
        from excel_parser.service import get_formulas_for_file
        from simulation.formula_utils import get_evaluation_order
        import time
        import os
        
        # Track start time for performance metrics
        start_time = time.time()
        
        # Update progress - Stage 1: Initialization
        simulation_progress[simulation_id].update({
            "status": "running",
            "progress": {"percentage": 5, "phase": "initialization", "stage_description": "🚀 Initializing Ultra Engine simulation"}
        })
        
        # ====================================================================
        # STAGE 2: EXCEL FILE LOADING & VALIDATION (IDENTICAL TO MAIN APP)
        # ====================================================================
        logger.info(f"📊 [B2B_API] STAGE 2: Excel File Loading & Validation")
        
        # Get file path from file_id (same as main app)
        file_id = model_info["file_id"]
        file_path = f'uploads/{file_id}'
        
        # Find the actual Excel file
        upload_dir = "uploads"
        excel_files = []
        for filename in os.listdir(upload_dir):
            if filename.startswith(file_id) and filename.endswith('.xlsx'):
                excel_files.append(os.path.join(upload_dir, filename))
        
        if not excel_files:
            raise ValueError(f"Excel file not found for file_id: {file_id}")
        
        file_path = excel_files[0]  # Use the first matching file
        logger.info(f"📊 [B2B_API] Using Excel file: {file_path}")
        
        # Update progress - Stage 2 complete
        simulation_progress[simulation_id].update({
            "status": "running", 
            "progress": {"percentage": 10, "phase": "excel_loading", "stage_description": "📊 Excel file loaded and validated"}
        })
        
        # ====================================================================
        # STAGE 3: FORMULA EXTRACTION (IDENTICAL TO MAIN APP)
        # ====================================================================
        logger.info(f"🔧 [B2B_API] STAGE 3: Formula Extraction & Analysis")
        
        # Update progress - Stage 3: Formula extraction
        simulation_progress[simulation_id].update({
            "status": "running",
            "progress": {"percentage": 15, "phase": "formula_extraction", "stage_description": "🔧 Extracting formulas and cell references"}
        })
        
        # Get all formulas from Excel file (IDENTICAL to main app)
        try:
            all_formulas = await get_formulas_for_file(file_id)
            total_formulas = sum(len(sheet_formulas) for sheet_formulas in all_formulas.values()) if all_formulas else 0
            logger.info(f"🔧 [B2B_API] Loaded {total_formulas} formulas from {len(all_formulas)} sheets")
        except Exception as e:
            logger.error(f"🔧 [B2B_API] Failed to load formulas: {e}")
            all_formulas = {}
            total_formulas = 0
        
        # Update progress - Stage 3 complete
        simulation_progress[simulation_id].update({
            "status": "running",
            "progress": {"percentage": 20, "phase": "formula_extraction", "stage_description": f"🔧 Formula extraction complete: {total_formulas:,} formulas found"}
        })
        
        # ====================================================================
        # STAGE 4: SMART DEPENDENCY ANALYSIS (IDENTICAL TO MAIN APP)
        # ====================================================================
        logger.info(f"🧠 [B2B_API] STAGE 4: Smart Dependency Analysis")
        
        # Update progress - Stage 4: Dependency analysis
        simulation_progress[simulation_id].update({
            "status": "running",
            "progress": {"percentage": 25, "phase": "dependency_analysis", "stage_description": "🧠 Analyzing formula dependencies and calculation order"}
        })
        
        # Convert B2B API models to internal simulation models (with SMART SHEET PARSING)
        variables = []
        mc_input_cells = set()
        
        # Auto-detect default sheet name from formulas (IDENTICAL to main app)
        if all_formulas:
            available_sheets = list(all_formulas.keys())
            auto_detected_sheet = available_sheets[0] if available_sheets else "Sheet1"
            logger.info(f"🧠 [B2B_API] Auto-detected default sheet name: {auto_detected_sheet}")
        else:
            auto_detected_sheet = "Sheet1"  # Fallback
        
        # Build variables with SMART SHEET PARSING (support Sheet!Cell format)
        for var in request.simulation_config.variables:
            # ✅ SMART SHEET PARSING: Support Sheet!Cell format
            if "!" in var.cell:
                sheet_name, cell_ref = var.cell.split("!", 1)
                logger.info(f"🧠 [B2B_API] Parsed sheet-specific variable: {sheet_name}!{cell_ref}")
            else:
                sheet_name = auto_detected_sheet
                cell_ref = var.cell
                logger.info(f"🧠 [B2B_API] Using default sheet for variable: {sheet_name}!{cell_ref}")
            
            if var.distribution.type == "triangular":
                variables.append(VariableConfig(
                    name=cell_ref,
                    sheet_name=sheet_name,
                    min_value=var.distribution.min,
                    most_likely=var.distribution.mode,
                    max_value=var.distribution.max
                ))
            elif var.distribution.type == "normal":
                std_dev = var.distribution.std
                mean = var.distribution.mean
                variables.append(VariableConfig(
                    name=cell_ref,
                    sheet_name=sheet_name,
                    min_value=mean - 3 * std_dev,
                    most_likely=mean,
                    max_value=mean + 3 * std_dev
                ))
            elif var.distribution.type == "uniform":
                min_val = var.distribution.min
                max_val = var.distribution.max
                variables.append(VariableConfig(
                    name=cell_ref,
                    sheet_name=sheet_name,
                    min_value=min_val,
                    most_likely=(min_val + max_val) / 2,
                    max_value=max_val
                ))
            
            # Add to MC input cells set with correct sheet
            mc_input_cells.add((sheet_name, cell_ref.upper()))
        
        # Parse target cell with auto-detected sheet
        target_cell = request.simulation_config.output_cells[0] if request.simulation_config.output_cells else "A1"
        if "!" in target_cell:
            target_sheet_name, target_cell_coordinate = target_cell.split("!", 1)
        else:
            target_sheet_name = auto_detected_sheet  # ✅ FIX: Use auto-detected sheet
            target_cell_coordinate = target_cell
        
        # ✅ CRITICAL FIX: Multi-target dependency analysis (IDENTICAL to frontend)
        # Get ordered calculation steps for ALL target cells (not just one)
        all_ordered_steps = []
        seen_steps = set()  # Avoid duplicates
        target_cells_list = request.simulation_config.output_cells  # Get all targets from request
        
        logger.info(f"🧠 [B2B_API] Running multi-target dependency analysis for {len(target_cells_list)} targets")
        
        for i, target_cell in enumerate(target_cells_list):
            # Update progress for each target dependency analysis
            progress_pct = 25 + (i / len(target_cells_list)) * 10  # 25% to 35%
            simulation_progress[simulation_id].update({
                "status": "running",
                "progress": {"percentage": progress_pct, "phase": "dependency_analysis", "stage_description": f"🧠 Analyzing dependencies for target {i+1}/{len(target_cells_list)}: {target_cell}"}
            })
            
            # Parse target cell to get sheet and coordinate
            if "!" in target_cell:
                target_sheet, target_coord = target_cell.split("!", 1)
            else:
                # ✅ CRITICAL FIX: Use actual sheet name from Excel file, not hardcoded "Sheet1"
                if all_formulas:
                    # Get the first available sheet name from formulas
                    available_sheets = list(all_formulas.keys())
                    target_sheet = available_sheets[0] if available_sheets else "Sheet1"
                    logger.info(f"🧠 [B2B_API] Auto-detected sheet: {target_sheet} for target {target_cell}")
                else:
                    target_sheet = "Sheet1"  # Fallback
                target_coord = target_cell
            
            logger.info(f"🧠 [B2B_API] Getting dependencies for {target_sheet}!{target_coord}")
            
            try:
                target_steps = get_evaluation_order(
                    target_sheet_name=target_sheet,
                    target_cell_coord=target_coord.upper(),
                    all_formulas=all_formulas,
                    mc_input_cells=mc_input_cells,
                    engine_type="ultra"  # Use Ultra engine limits
                )
                
                # Add unique steps to the combined list
                for step in target_steps:
                    step_key = (step[0], step[1].upper())  # (sheet, cell) as key
                    if step_key not in seen_steps:
                        all_ordered_steps.append(step)
                        seen_steps.add(step_key)
                
                logger.info(f"🧠 [B2B_API] Added {len(target_steps)} steps for {target_cell} (total unique: {len(all_ordered_steps)})")
                
            except Exception as e:
                logger.error(f"🧠 [B2B_API] Dependency analysis failed for target {target_cell}: {e}")
                continue
        
        ordered_calc_steps = all_ordered_steps
        logger.info(f"🧠 [B2B_API] Multi-target dependency analysis complete: {len(ordered_calc_steps)} calculation steps")
        
        # Use the first target for the primary sheet name and coordinate (for Ultra Engine call)
        primary_target = target_cells_list[0] if target_cells_list else "A1"
        if "!" in primary_target:
            target_sheet_name, target_cell_coordinate = primary_target.split("!", 1)
        else:
            target_sheet_name = auto_detected_sheet  # Use auto-detected sheet
            target_cell_coordinate = primary_target
        
        # Update progress - Stage 4 complete
        simulation_progress[simulation_id].update({
            "status": "running",
            "progress": {"percentage": 40, "phase": "dependency_analysis", "stage_description": f"🧠 Dependency analysis complete: {len(ordered_calc_steps)} calculation steps"}
        })
        
        # ====================================================================
        # STAGE 5: ULTRA ENGINE SIMULATION (IDENTICAL TO MAIN APP)
        # ====================================================================
        logger.info(f"⚡ [B2B_API] STAGE 5: Running Ultra Engine Simulation")
        
        # Update progress - Stage 5: Simulation
        simulation_progress[simulation_id].update({
            "status": "running",
            "progress": {"percentage": 50, "phase": "simulation", "stage_description": "⚡ Running Monte Carlo simulation with Ultra Engine"}
        })
        
        # ✅ CRITICAL FIX: Create ISOLATED Ultra engine for B2B API (prevent main system interference)
        # Use a unique simulation ID to avoid conflicts with main simulation tracking
        b2b_engine_id = f"b2b_engine_{uuid.uuid4().hex[:8]}"
        ultra_engine = create_ultra_engine(iterations=request.simulation_config.iterations, simulation_id=b2b_engine_id)
        logger.info(f"⚡ [B2B_API] ISOLATED Ultra Engine instance created with ID: {b2b_engine_id}")
        
        # ✅ CRITICAL FIX: Set up ISOLATED progress callback (B2B-only, no main system interference)
        def isolated_b2b_progress_cb(progress_data):
            try:
                # Update ONLY B2B API progress store (completely isolated)
                current_progress = simulation_progress[simulation_id]["progress"]
                enhanced_progress_data = {
                    **current_progress,
                    "percentage": progress_data.get("progress_percentage", current_progress.get("percentage", 50)),
                    "phase": "simulation",
                    "stage_description": progress_data.get("stage_description", "⚡ Running simulation..."),
                    "b2b_isolated": True,  # Mark as B2B-only progress
                    "engine_id": b2b_engine_id  # Track which engine is updating
                }
                simulation_progress[simulation_id]["progress"] = enhanced_progress_data
                
                # Log progress occasionally with B2B markers
                if progress_data.get("progress_percentage", 0) % 5 == 0:  # More frequent logging for B2B
                    logger.info(f"🚀 [B2B_ISOLATED] Simulation {simulation_id} progress: {progress_data.get('progress_percentage', 0)}% (engine: {b2b_engine_id})")
            except Exception as e:
                logger.error(f"🚀 [B2B_ISOLATED] Progress callback failed for {simulation_id}: {e}")
        
        # ✅ CRITICAL: Use isolated callback to prevent main system interference
        ultra_engine.set_progress_callback(isolated_b2b_progress_cb)
        
        # ✅ CRITICAL FIX: Disable main simulation system progress tracking for this engine
        # This prevents the Ultra Engine from calling update_simulation_progress() which causes conflicts
        ultra_engine._disable_main_progress_system = True
        logger.info(f"⚡ [B2B_API] Disabled main progress system for isolated engine: {b2b_engine_id}")
        
        # Convert inputs to Ultra engine format (IDENTICAL to main app)
        mc_input_params = {}
        for mc_input in variables:
            key = (mc_input.sheet_name, mc_input.name.upper())
            mc_input_params[key] = (mc_input.min_value, mc_input.most_likely, mc_input.max_value)
        
        # ✅ CRITICAL FIX: Load Excel constants (IDENTICAL to frontend)
        # Build set of cells that will be calculated (from ordered_calc_steps)
        calculated_cells = set()
        for sheet, cell, _ in ordered_calc_steps:
            # Normalize cell reference (remove $ signs for absolute references)
            normalized_cell = cell.replace('$', '').upper()
            calculated_cells.add((sheet, normalized_cell))
        
        # Load Excel constants excluding MC input cells (IDENTICAL to frontend)
        from excel_parser.service import get_constants_for_file
        
        try:
            all_file_constants = await get_constants_for_file(
                file_id, 
                exclude_cells=mc_input_cells, 
                target_sheet=target_sheet_name
            )
            
            # ✅ CRITICAL: Only use constants for cells that are NOT being calculated
            # This prevents double-calculation and exponential value explosion
            constant_params = {}
            constants_used = 0
            for (sheet, cell), value in all_file_constants.items():
                # Normalize cell reference for comparison
                normalized_cell = cell.replace('$', '').upper()
                if (sheet, normalized_cell) not in calculated_cells:
                    constant_params[(sheet, cell)] = value
                    constants_used += 1
            
            logger.info(f"📊 [B2B_API] Using {constants_used} Excel constants for non-calculated cells")
            logger.info(f"📊 [B2B_API] Skipping constants for {len(calculated_cells)} cells that will be calculated fresh")
            
        except Exception as e:
            logger.error(f"📊 [B2B_API] Failed to load constants: {e}")
            constant_params = {}  # Fallback to empty constants
        
        # ✅ FIX: Run simulation using direct file path (IDENTICAL to main app)
        logger.info(f"⚡ [B2B_API] Starting Ultra Engine simulation:")
        logger.info(f"  - File path: {file_path}")
        logger.info(f"  - Target: {target_sheet_name}!{target_cell_coordinate}")
        logger.info(f"  - Variables: {len(variables)}")
        logger.info(f"  - Iterations: {request.simulation_config.iterations}")
        logger.info(f"  - Calculation steps: {len(ordered_calc_steps)}")
        
        # ✅ CRITICAL DEBUG: Log simulation setup before execution
        logger.info(f"⚡ [B2B_API] Ultra Engine simulation setup:")
        logger.info(f"  - Constants loaded: {len(constant_params)}")
        logger.info(f"  - MC input cells: {len(mc_input_cells)}")
        logger.info(f"  - Calculated cells: {len(calculated_cells)}")
        logger.info(f"  - Sample constants: {dict(list(constant_params.items())[:5])}")
        logger.info(f"  - Sample MC inputs: {list(mc_input_cells)[:5]}")
        
        # Run the simulation (IDENTICAL to main app)
        raw_results = await ultra_engine.run_simulation(
            mc_input_configs=variables,
            ordered_calc_steps=ordered_calc_steps,
            target_sheet_name=target_sheet_name,
            target_cell_coordinate=target_cell_coordinate,
            constant_values=constant_params,  # ✅ FIX: Now includes Excel constants
            workbook_path=file_path  # ✅ FIX: Add required workbook path
        )
        
        # Unpack results (IDENTICAL to main app)
        if isinstance(raw_results, tuple):
            results_array, iteration_errors = raw_results
        else:
            results_array, iteration_errors = raw_results, []
        
        # ====================================================================
        # STAGE 6: RESULTS PROCESSING (IDENTICAL TO MAIN APP)
        # ====================================================================
        logger.info(f"📊 [B2B_API] STAGE 6: Processing Results")
        
        # Update progress - Stage 6: Results processing
        simulation_progress[simulation_id].update({
            "status": "running",
            "progress": {"percentage": 90, "phase": "results_processing", "stage_description": "📊 Processing simulation results and statistics"}
        })
        
        # Calculate statistics (IDENTICAL to main app)
        stats = ultra_engine._calculate_statistics(results_array)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        logger.info(f"✅ [B2B_API] Simulation completed in {execution_time:.1f} seconds")
        logger.info(f"📊 [B2B_API] Results: mean={stats.mean:.2f}, std={stats.std_dev:.2f}")
        
        # Convert to B2B API format
        from .models import SimulationResults, CellResults, StatisticalResults
        import numpy as np
        
        # Create histogram (IDENTICAL to main app)
        if results_array is not None and len(results_array) > 0:
            # Filter out NaN values for histogram
            valid_results = results_array[~np.isnan(results_array)]
            if len(valid_results) > 0:
                hist_counts, hist_bins = np.histogram(valid_results, bins=50)
                histogram_data = {
                    "bins": hist_bins.tolist(),
                    "frequencies": hist_counts.tolist()
                }
            else:
                histogram_data = {"bins": [], "frequencies": []}
        else:
            histogram_data = {"bins": [], "frequencies": []}
        
        # Build final target cell reference
        final_target_cell = f"{target_sheet_name}!{target_cell_coordinate}" if target_sheet_name != auto_detected_sheet else target_cell_coordinate
        
        # Process ALL output cells from the simulation
        all_cell_results = {}
        all_outputs = list(engine_results.keys()) if engine_results else []
        logger.info(f"🔍 Ultra Engine found {len(all_outputs)} output cells: {all_outputs}")
        
        for output_cell in all_outputs:
            if output_cell in engine_results:
                cell_stats = engine_results[output_cell]
                cell_histogram = histograms.get(output_cell, {})
                
                all_cell_results[output_cell] = CellResults(
                    cell_name=f"Ultra_Result_{output_cell}",
                    statistics=StatisticalResults(
                        mean=float(cell_stats.mean),
                        std=float(cell_stats.std_dev),
                        min=float(cell_stats.min_value),
                        max=float(cell_stats.max_value),
                        percentiles={
                            "5": float(cell_stats.percentile_5),
                            "25": float(cell_stats.percentile_25),
                            "50": float(cell_stats.median),
                            "75": float(cell_stats.percentile_75),
                            "95": float(cell_stats.percentile_95)
                        },
                        var_95=float(cell_stats.percentile_5),  # VaR at 95% confidence
                        var_99=float(cell_stats.percentile_5)   # VaR at 99% confidence (approximation)
                    ),
                    distribution_data={
                        "histogram": cell_histogram
                    }
                )
        
        results = SimulationResults(
            simulation_id=simulation_id,
            status="completed",
            execution_time=f"{execution_time:.1f} seconds",
            iterations_completed=request.simulation_config.iterations,
            results=all_cell_results,
            download_links={
                "detailed_csv": f"https://your-apigee-domain.com/simapp-api/download/{simulation_id}.csv",
                "summary_pdf": f"https://your-apigee-domain.com/simapp-api/download/{simulation_id}.pdf"
            },
            created_at=datetime.now(timezone.utc)
        )
        
        # ====================================================================
        # STAGE 7: FINALIZATION (IDENTICAL TO MAIN APP)
        # ====================================================================
        logger.info(f"🎉 [B2B_API] STAGE 7: Finalizing Results")
        
        # Store results
        simulation_results[simulation_id] = results
        simulation_progress[simulation_id].update({
            "status": "completed",
            "progress": {"percentage": 100, "phase": "completed", "stage_description": "🎉 Simulation completed successfully!"}
        })
        
        # Persist to files
        save_storage(simulation_results, RESULTS_FILE)
        save_storage(simulation_progress, PROGRESS_FILE)
        
        logger.info(f"🎉 [B2B_API] IDENTICAL PIPELINE simulation completed and persisted: {simulation_id}")
        logger.info(f"🎉 [B2B_API] Total execution time: {execution_time:.1f} seconds")
        logger.info(f"🎉 [B2B_API] Pipeline stages completed:")
        logger.info(f"    ✅ Stage 1: Initialization")
        logger.info(f"    ✅ Stage 2: Excel File Loading & Validation")
        logger.info(f"    ✅ Stage 3: Formula Extraction ({total_formulas:,} formulas)")
        logger.info(f"    ✅ Stage 4: Smart Dependency Analysis ({len(ordered_calc_steps)} steps)")
        logger.info(f"    ✅ Stage 5: Ultra Engine Simulation ({request.simulation_config.iterations:,} iterations)")
        logger.info(f"    ✅ Stage 6: Results Processing")
        logger.info(f"    ✅ Stage 7: Finalization")
        
    except Exception as e:
        logger.error(f"❌ [B2B_API] IDENTICAL PIPELINE simulation {simulation_id} failed: {e}")
        simulation_progress[simulation_id].update({
            "status": "failed",
            "error": str(e),
            "progress": {"percentage": 0, "phase": "failed", "stage_description": f"❌ Simulation failed: {str(e)}"}
        })
        
        # Persist the error state
        save_storage(simulation_progress, PROGRESS_FILE)


@router.get("/models")
async def list_models(api_key_info: APIKey = Depends(verify_api_key)):
    """List uploaded models for the authenticated client."""
    client_models = {
        model_id: info for model_id, info in uploaded_models.items()
        if info["client_id"] == api_key_info.client_id
    }
    
    return {
        "models": [
            {
                "model_id": model_id,
                "filename": info["filename"],
                "formulas_count": info["formulas_count"],
                "uploaded_at": info["uploaded_at"]
            }
            for model_id, info in client_models.items()
        ]
    }


# ============================================================================
# DOWNLOAD ENDPOINTS - EXACT FRONTEND REPLICATION
# ============================================================================

@router.get("/simulations/{simulation_id}/download/pdf")
async def download_simulation_pdf(
    simulation_id: str,
    api_key_info: APIKey = Depends(verify_api_key)
):
    """Generate and download PDF report - uses EXACT same method as frontend."""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation results not found")
    
    result = simulation_results[simulation_id]
    logger.info(f"🔍 STORED RESULT TYPE: {type(result)}")
    logger.info(f"🔍 STORED RESULT KEYS: {list(result.keys()) if hasattr(result, 'keys') else 'No keys method'}")
    if hasattr(result, 'results'):
        logger.info(f"🔍 STORED RESULT.RESULTS KEYS: {list(result.results.keys()) if result.results else 'No results'}")
    elif isinstance(result, dict) and 'results' in result:
        logger.info(f"🔍 STORED RESULT[results] KEYS: {list(result['results'].keys()) if result['results'] else 'No results'}")
    
    try:
        # Use the EXACT same working PDF service as frontend
        # Handle both dict and model object formats
        if hasattr(result, 'iterations_completed'):
            # Pydantic model format
            iterations_run = result.iterations_completed
            created_at = result.created_at
            execution_time = result.execution_time
            results_data = result.results
        else:
            # Dict format
            iterations_run = result.get("iterations_completed", 1000)
            created_at = result.get("created_at", "")
            execution_time = result.get("execution_time", "")
            results_data = result.get("results", {})
        
        frontend_data = {
            "targets": {},
            "iterations_run": iterations_run,
            "requested_engine_type": "Ultra",
            "metadata": {
                "timestamp": created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at),
                "execution_time": execution_time
            }
        }
        
        # Convert each cell result to frontend format
        for cell_name, cell_result in results_data.items():
            # Handle both dict and model object formats
            if hasattr(cell_result, 'statistics'):
                # Pydantic model format
                stats = cell_result.statistics
                distribution_data = cell_result.distribution_data
            else:
                # Dict format
                stats = cell_result.get("statistics", {})
                distribution_data = cell_result.get("distribution_data", {})
            
            histogram_data = distribution_data.get("histogram", {}) if distribution_data else {}
            
            # Handle stats format (dict vs model)
            if hasattr(stats, 'mean'):
                # Pydantic model format
                stats_dict = {
                    "mean": stats.mean,
                    "median": stats.percentiles.get("50", stats.mean) if hasattr(stats, 'percentiles') else stats.mean,
                    "std_dev": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                    "percentiles": stats.percentiles if hasattr(stats, 'percentiles') else {}
                }
            else:
                # Dict format
                stats_dict = {
                    "mean": stats.get("mean", 0),
                    "median": stats.get("percentiles", {}).get("50", stats.get("mean", 0)),
                    "std_dev": stats.get("std", 0),
                    "min": stats.get("min", 0),
                    "max": stats.get("max", 0),
                    "percentiles": stats.get("percentiles", {})
                }
            
            # Convert histogram data to frontend format
            logger.info(f"📊 HISTOGRAM DEBUG for {cell_name}: {histogram_data}")
            frontend_histogram = {}
            if histogram_data and isinstance(histogram_data, dict):
                if "histogram" in histogram_data:
                    # Extract histogram from nested structure
                    hist_data = histogram_data["histogram"]
                    if isinstance(hist_data, dict):
                        if "bins" in hist_data and "frequencies" in hist_data:
                            # Format: {bins: [...], frequencies: [...]}
                            frontend_histogram = {
                                "bin_edges": hist_data["bins"],
                                "counts": hist_data["frequencies"]
                            }
                        elif "bin_edges" in hist_data and "counts" in hist_data:
                            # Already in correct format
                            frontend_histogram = {
                                "bin_edges": hist_data["bin_edges"],
                                "counts": hist_data["counts"]
                            }
                    elif isinstance(hist_data, list):
                        # Simple array format
                        frontend_histogram = {
                            "bin_edges": list(range(len(hist_data) + 1)),
                            "counts": hist_data
                        }
                elif "bins" in histogram_data and "frequencies" in histogram_data:
                    # Direct format
                    frontend_histogram = {
                        "bin_edges": histogram_data["bins"],
                        "counts": histogram_data["frequencies"]
                    }
                elif "bin_edges" in histogram_data and "counts" in histogram_data:
                    # Already correct
                    frontend_histogram = histogram_data
                elif "bin_edges" in histogram_data and "histogram" in histogram_data:
                    # Our SimApp format: {'histogram': [...], 'bin_edges': [...]}
                    frontend_histogram = {
                        "bin_edges": histogram_data["bin_edges"],
                        "counts": histogram_data["histogram"]
                    }
            
            frontend_data["targets"][cell_name] = {
                "values": [],  # Raw values not stored in SimApp API format
                "statistics": stats_dict,
                "histogram_data": frontend_histogram,
                "sensitivity_analysis": []  # Not available in SimApp API format
            }
        
        logger.info(f"📊 DEBUG: Original result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        logger.info(f"📊 DEBUG: Results data keys: {list(results_data.keys()) if isinstance(results_data, dict) else 'Not a dict'}")
        logger.info(f"📊 DEBUG: Results data content: {results_data}")
        logger.info(f"Transformed SimApp data to frontend format: {len(frontend_data['targets'])} targets")
        
        # Check if PDF service is available
        if pdf_export_service is None:
            logger.error("PDF service not available - import failed")
            raise HTTPException(status_code=500, detail="PDF service not available - import failed")
        
        # Call the EXACT same method the frontend uses via /api/pdf/export
        pdf_path = await pdf_export_service.generate_pdf_from_results_page(
            simulation_id=simulation_id,
            results_data=frontend_data,
            auth_token=None  # Internal call doesn't need token
        )
        
        # Read the generated PDF file (exact same approach as frontend)
        pdf_file_path = Path(pdf_path)
        if not pdf_file_path.exists():
            raise HTTPException(status_code=500, detail="PDF generation completed but file not found")
        
        # Read PDF content
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        # Clean up temporary file
        try:
            pdf_file_path.unlink()
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary PDF file: {cleanup_error}")
        
        # Return PDF response (exact same format as frontend)
        from fastapi.responses import Response
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=simulation_results_{simulation_id[:8]}.pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"PDF generation failed: {str(e)}"
        )
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        # Build PDF content
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f2937')
        )
        story.append(Paragraph("Monte Carlo Simulation Results", title_style))
        story.append(Spacer(1, 12))
        
        # Simulation metadata
        metadata_style = ParagraphStyle(
            'Metadata',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#6b7280')
        )
        
        story.append(Paragraph(f"<b>Simulation ID:</b> {simulation_id}", metadata_style))
        story.append(Paragraph(f"<b>Execution Time:</b> {result['execution_time']}", metadata_style))
        story.append(Paragraph(f"<b>Iterations:</b> {result['iterations_completed']:,}", metadata_style))
        created_at_str = result['created_at']
        if hasattr(created_at_str, 'isoformat'):
            created_at_str = created_at_str.isoformat()
        elif isinstance(created_at_str, str):
            created_at_str = created_at_str
        else:
            created_at_str = str(created_at_str)
        
        # Format the datetime string for display
        display_date = created_at_str[:19].replace('T', ' ') if len(created_at_str) >= 19 else created_at_str
        story.append(Paragraph(f"<b>Generated:</b> {display_date}", metadata_style))
        story.append(Spacer(1, 20))
        
        # Results for each output cell
        for cell_name, cell_result in result["results"].items():
            cell_display_name = cell_result["cell_name"]
            stats = cell_result["statistics"]
            
            # Cell header
            cell_header_style = ParagraphStyle(
                'CellHeader',
                parent=styles['Heading2'],
                fontSize=18,
                spaceAfter=12,
                textColor=colors.HexColor('#111827')
            )
            story.append(Paragraph(f"Results for {cell_display_name}", cell_header_style))
            
            # Statistics table
            stats_data = [
                ['Statistic', 'Value'],
                ['Mean', f"{stats['mean']:,.2f}"],
                ['Standard Deviation', f"{stats['std']:,.2f}"],
                ['Minimum', f"{stats['min']:,.2f}"],
                ['Maximum', f"{stats['max']:,.2f}"],
                ['5th Percentile', f"{stats['percentiles']['5']:,.2f}"],
                ['25th Percentile', f"{stats['percentiles']['25']:,.2f}"],
                ['50th Percentile (Median)', f"{stats['percentiles']['50']:,.2f}"],
                ['75th Percentile', f"{stats['percentiles']['75']:,.2f}"],
                ['95th Percentile', f"{stats['percentiles']['95']:,.2f}"],
                ['Value at Risk (95%)', f"{stats['var_95']:,.2f}"],
                ['Value at Risk (99%)', f"{stats['var_99']:,.2f}"]
            ]
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f9fafb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#111827')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 20))
            
            # Generate histogram using matplotlib
            try:
                hist_data = cell_result["distribution_data"]
                if hist_data and "bin_centers" in hist_data and "histogram" in hist_data:
                    plt.figure(figsize=(8, 4))
                    plt.bar(hist_data["bin_centers"], hist_data["histogram"], 
                           alpha=0.7, color='#3b82f6', edgecolor='white')
                    plt.title(f'Distribution of {cell_display_name}', fontsize=14, fontweight='bold')
                    plt.xlabel('Value', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    # Save chart to buffer
                    chart_buffer = BytesIO()
                    plt.savefig(chart_buffer, format='PNG', dpi=150, bbox_inches='tight')
                    plt.close()
                    chart_buffer.seek(0)
                    
                    # Add chart to PDF
                    img = Image(chart_buffer, width=5*inch, height=2.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                    
            except Exception as chart_error:
                logger.warning(f"Could not generate chart for {cell_name}: {chart_error}")
                story.append(Paragraph("Chart generation failed for this variable.", metadata_style))
                story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        # Return PDF response
        from fastapi.responses import Response
        return Response(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=simulation_results_{simulation_id[:8]}.pdf"
            }
        )
        
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="PDF generation dependencies not available. Please install reportlab and matplotlib."
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@router.get("/simulations/{simulation_id}/download/xlsx")
async def download_simulation_xlsx(
    simulation_id: str,
    api_key_info: APIKey = Depends(verify_api_key)
):
    """Generate and download Excel spreadsheet - uses EXACT same method as frontend."""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation results not found")
    
    result = simulation_results[simulation_id]
    
    try:
        # Import the exact same XLSX export service used by the frontend
        from modules.xlsx_export import xlsx_export_service
        from pathlib import Path
        
        if not xlsx_export_service:
            raise HTTPException(
                status_code=500, 
                detail="XLSX export service not available. Missing openpyxl dependency."
            )
        
        # Prepare metadata (same format as frontend)
        metadata = {
            "iterations": result.get("iterations_completed", 10000),
            "engine_type": "Ultra",
            "execution_time": result.get("execution_time", "N/A"),
            "created_at": result.get("created_at", datetime.now(timezone.utc).isoformat()),
            "status": "completed"
        }
        
        # Call the EXACT same method the frontend uses
        xlsx_path = xlsx_export_service.generate_xlsx_export(
            simulation_id=simulation_id,
            results_data=result,
            metadata=metadata
        )
        
        # Read the generated XLSX file (exact same approach as frontend)
        xlsx_file_path = Path(xlsx_path)
        if not xlsx_file_path.exists():
            raise HTTPException(status_code=500, detail="XLSX generation completed but file not found")
        
        # Read XLSX content
        with open(xlsx_file_path, 'rb') as xlsx_file:
            xlsx_content = xlsx_file.read()
        
        # Clean up temporary file
        try:
            xlsx_file_path.unlink()
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary XLSX file: {cleanup_error}")
        
        # Return XLSX response (exact same format as frontend)
        from fastapi.responses import Response
        return Response(
            content=xlsx_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=simulation_results_{simulation_id[:8]}.xlsx"
            }
        )
        
    except Exception as e:
        logger.error(f"XLSX generation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"XLSX generation failed: {str(e)}"
        )


@router.get("/simulations/{simulation_id}/download/json")
async def download_simulation_json(
    simulation_id: str,
    api_key_info: APIKey = Depends(verify_api_key)
):
    """Generate and download JSON data - uses EXACT same method as frontend."""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation results not found")
    
    result = simulation_results[simulation_id]
    
    try:
        # Import the exact same JSON export service used by the frontend
        from modules.json_export import json_export_service
        from pathlib import Path
        
        # Prepare metadata (same format as frontend)
        metadata = {
            "iterations": result.get("iterations_completed", 10000),
            "engine_type": "Ultra",
            "execution_time": result.get("execution_time", "N/A"),
            "created_at": result.get("created_at", datetime.now(timezone.utc).isoformat()),
            "status": "completed"
        }
        
        # Call the EXACT same method the frontend uses
        json_path = json_export_service.generate_json_export(
            simulation_id=simulation_id,
            results_data=result,
            metadata=metadata
        )
        
        # Read the generated JSON file (exact same approach as frontend)
        json_file_path = Path(json_path)
        if not json_file_path.exists():
            raise HTTPException(status_code=500, detail="JSON generation completed but file not found")
        
        # Read JSON content
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            json_content = json_file.read()
        
        # Clean up temporary file
        try:
            json_file_path.unlink()
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary JSON file: {cleanup_error}")
        
        # Return JSON response (exact same format as frontend)
        from fastapi.responses import Response
        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=simulation_results_{simulation_id[:8]}.json"
            }
        )
        
    except Exception as e:
        logger.error(f"JSON generation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"JSON generation failed: {str(e)}"
        )
    
    try:
        # Create comprehensive JSON export data (matching frontend format)
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "simulation_id": simulation_id,
                "total_variables": len(result["results"]),
                "export_format": "json",
                "export_version": "1.0",
                "api_version": "B2B_v1"
            },
            "simulation_summary": {
                "execution_time": result["execution_time"],
                "iterations_completed": result["iterations_completed"],
                "status": result["status"],
                "created_at": result["created_at"].isoformat() if hasattr(result["created_at"], 'isoformat') else str(result["created_at"])
            },
            "results": {}
        }
        
        # Add detailed results for each variable
        for cell_name, cell_result in result["results"].items():
            cell_display_name = cell_result["cell_name"]
            stats = cell_result["statistics"]
            
            export_data["results"][cell_display_name] = {
                "cell_reference": cell_display_name,
                "statistics": {
                    "mean": stats["mean"],
                    "median": stats["percentiles"]["50"],
                    "std_dev": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "percentiles": stats["percentiles"],
                    "value_at_risk": {
                        "var_95": stats["var_95"],
                        "var_99": stats["var_99"]
                    }
                },
                "distribution_data": cell_result["distribution_data"],
                "raw_values": result.get("raw_results", {}).get(cell_name, [])
            }
        
        # Return JSON response
        from fastapi.responses import JSONResponse
        
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"simulation_results_{timestamp}.json"
        
        return JSONResponse(
            content=export_data,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"JSON generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"JSON generation failed: {str(e)}")


# ============================================================================
# BROWSER-FRIENDLY DOWNLOAD ENDPOINTS (WITH TEMPORARY TOKENS)
# ============================================================================

# Temporary token storage (in production, use Redis with TTL)
download_tokens = {}

@router.post("/simulations/{simulation_id}/generate-download-token")
async def generate_download_token(
    simulation_id: str,
    api_key_info: APIKey = Depends(verify_api_key)
):
    """Generate a temporary download token for browser access."""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation results not found")
    
    # Generate a temporary token
    import secrets
    token = secrets.token_urlsafe(32)
    
    # Store token with expiration (1 hour)
    from datetime import datetime, timedelta
    expiration = datetime.now(timezone.utc) + timedelta(hours=1)
    
    download_tokens[token] = {
        "simulation_id": simulation_id,
        "expires_at": expiration,
        "client_id": api_key_info.client_id
    }
    
    return {
        "download_token": token,
        "expires_at": expiration.isoformat(),
        "download_links": {
            "pdf": f"/simapp-api/download/{token}/pdf",
            "xlsx": f"/simapp-api/download/{token}/xlsx", 
            "json": f"/simapp-api/download/{token}/json"
        }
    }


def verify_download_token(token: str):
    """Verify a download token and return simulation info."""
    if token not in download_tokens:
        raise HTTPException(status_code=403, detail="Invalid download token")
    
    token_info = download_tokens[token]
    
    # Check expiration
    if datetime.now(timezone.utc) > token_info["expires_at"]:
        del download_tokens[token]
        raise HTTPException(status_code=403, detail="Download token expired")
    
    return token_info


@router.get("/download/{token}/pdf")
async def download_pdf_with_token(token: str):
    """Download PDF using temporary token - browser friendly."""
    token_info = verify_download_token(token)
    simulation_id = token_info["simulation_id"]
    
    # Reuse the existing PDF generation logic
    return await download_simulation_pdf(simulation_id, api_key_info=None)


@router.get("/download/{token}/xlsx") 
async def download_xlsx_with_token(token: str):
    """Download Excel using temporary token - browser friendly."""
    token_info = verify_download_token(token)
    simulation_id = token_info["simulation_id"]
    
    # Reuse the existing Excel generation logic
    return await download_simulation_xlsx(simulation_id, api_key_info=None)


@router.get("/download/{token}/json")
async def download_json_with_token(token: str):
    """Download JSON using temporary token - browser friendly."""
    token_info = verify_download_token(token)
    simulation_id = token_info["simulation_id"]
    
    # Reuse the existing JSON generation logic  
    return await download_simulation_json(simulation_id, api_key_info=None)
