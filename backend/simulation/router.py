from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Response, WebSocket, WebSocketDisconnect
from uuid import uuid4
import logging
import math
import asyncio
from typing import List, Dict, Any, Optional
import time

# Import settings for configuration
from config import settings

from simulation.schemas import SimulationRequest, SimulationResponse
from simulation.service import (
    initiate_simulation,
    get_simulation_status_or_results,
    cancel_simulation_task,
    get_simulation_cache_stats,
    _clear_specific_simulation_cache,
    get_cache_stats,
    clear_all_cache,
    clear_user_cache,
    _remove_from_cache
)
from auth.auth0_dependencies import get_current_active_auth0_user
from models import User
from shared.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

# Temporary ID mapping store (in-memory for now)
_temp_id_mapping = {}

def _handle_temp_id(simulation_id: str) -> str:
    """
    Handle temporary simulation IDs by mapping them to real IDs if available.
    
    This solves the race condition where frontend starts polling with temp IDs
    before the backend has processed the simulation request.
    """
    # Check if this is a temp ID
    if simulation_id.startswith('temp_'):
        # Check if we have a mapping to a real ID
        if simulation_id in _temp_id_mapping:
            real_id = _temp_id_mapping[simulation_id]
            logger.debug(f"TEMP_ID_MAPPING: {simulation_id} -> {real_id}")
            return real_id
        else:
            # No mapping yet - return temp ID (backend will handle gracefully)
            logger.debug(f"TEMP_ID_MAPPING: No mapping found for {simulation_id}")
            return simulation_id
    
    return simulation_id

def register_temp_id_mapping(temp_id: str, real_id: str):
    """Register a mapping from temporary ID to real simulation ID"""
    if temp_id.startswith('temp_'):
        _temp_id_mapping[temp_id] = real_id
        logger.info(f"TEMP_ID_MAPPING: Registered {temp_id} -> {real_id}")

def _normalize_simulation_id(simulation_id: str) -> str:
    """
    Map child ids back to parent id for compatibility.
    
    Handles multiple suffix patterns:
    - '_target_N' (single suffix)
    - '_target_N_target_N' (double suffix)
    - '_target_N_target_N_target_N' (triple suffix)
    - Any other child patterns that may emerge
    - Temporary IDs (temp_*)
    """
    # First handle temp IDs
    simulation_id = _handle_temp_id(simulation_id)
    try:
        import re
        # Generalized regex to handle any number of _target_ suffixes and other child patterns
        patterns = [
            r"^(?P<parent>.+?)(?:_target_\d+)+$",  # Multiple _target_ suffixes
            r"^(?P<parent>.+?)(?:_child_\d+)+$",   # Potential _child_ suffixes
            r"^(?P<parent>.+?)(?:_sub_\d+)+$",     # Potential _sub_ suffixes
        ]
        
        for pattern in patterns:
            m = re.match(pattern, simulation_id)
            if m:
                normalized_id = m.group("parent")
                logger.info(f"ID normalization: {simulation_id} -> {normalized_id}")
                return normalized_id
        
        return simulation_id
    except Exception as e:
        logger.error(f"Error normalizing simulation ID {simulation_id}: {e}")
        return simulation_id

# Import shared sanitization utilities
from shared.data_sanitizer import sanitize_data_structure, sanitize_simulation_response

router = APIRouter(
    tags=["Simulations"],
    responses={404: {"description": "Not found"}}
)

def _extract_targets_from_message(message):
    """Extract target variables from batch simulation message"""
    if not message:
        return "Target Variable"
    
    # Pattern: "Batch simulation completed: 3 targets processed - D8, F8, G8"
    import re
    match = re.search(r'targets processed - (.+)$', message)
    if match:
        return match.group(1).strip()
    
    return "Target Variable"

def _extract_target_count_from_message(message, target_name):
    """Extract target count from message or target_name"""
    if target_name and ',' in target_name:
        return len(target_name.split(', '))
    
    if not message:
        return 1
    
    # Pattern: "Batch simulation completed: 3 targets processed"
    import re
    match = re.search(r'(\d+) targets processed', message)
    if match:
        return int(match.group(1))
    
    # Fallback: count from extracted targets
    targets = _extract_targets_from_message(message)
    if targets != "Target Variable" and ',' in targets:
        return len(targets.split(', '))
    
    return 1

@router.get("/history")
async def get_simulation_history_admin(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get simulation history for admin users"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        from database import get_db
        from models import SimulationResult
        from sqlalchemy.orm import Session
        
        db = next(get_db())
        
        # Get recent simulations for admin view - ONLY PARENT SIMULATIONS
        # Parent simulations have messages starting with "Batch simulation" or have target_name filled
        simulations = db.query(SimulationResult).filter(
            (SimulationResult.message.like('Batch simulation%')) |
            (SimulationResult.target_name.isnot(None) & (SimulationResult.target_name != ''))
        ).order_by(
            SimulationResult.created_at.desc()
        ).limit(50).all()
        
        history = []
        for sim in simulations:
            history.append({
                "simulation_id": sim.simulation_id,
                "status": sim.status,
                "user_id": sim.user_id,
                "user": sim.user.username if sim.user else "Unknown",
                "email": sim.user.email if sim.user else "Unknown",
                "engine_type": sim.engine_type,
                "file_name": sim.original_filename if sim.original_filename and sim.original_filename != "Unknown" else "Unknown",
                "target_variables": sim.target_name if sim.target_name else _extract_targets_from_message(sim.message),
                "simulation_count": _extract_target_count_from_message(sim.message, sim.target_name),
                "iterations_requested": sim.iterations_requested,
                "iterations_run": sim.iterations_run,
                "created_at": sim.created_at.isoformat() if sim.created_at else None,
                "started_at": sim.started_at.isoformat() if sim.started_at else None,
                "completed_at": sim.completed_at.isoformat() if sim.completed_at else None,
                "message": sim.message
            })
        
        # Return array directly instead of wrapped in object
        return history
        
    except Exception as e:
        logger.error(f"Error fetching simulation history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch simulation history: {str(e)}")

@router.post("/create-id")
async def create_simulation_id(current_user: User = Depends(get_current_active_auth0_user)):
    """
    🚀 NEW: Create a real simulation ID before starting simulation
    This eliminates the temporary ID mapping system
    """
    import uuid
    simulation_id = str(uuid.uuid4())
    logger.info(f"🚀 [API] Created new simulation ID: {simulation_id} for user: {current_user.email}")
    return {"simulation_id": simulation_id}

@router.post("/run", response_model=SimulationResponse, status_code=202)
async def create_simulation_run(
    request: SimulationRequest, 
    background_tasks: BackgroundTasks, 
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Initiate a Monte Carlo simulation. The simulation runs as a background task."""
    
    # 🚨 CRITICAL DEBUGGING: Comprehensive simulation creation logging
    logger.info(f"🚨 [SIM_CREATE] ===== SIMULATION CREATION REQUEST RECEIVED =====")
    logger.info(f"🚨 [SIM_CREATE] User: {current_user.email if hasattr(current_user, 'email') else current_user.username}")
    logger.info(f"🚨 [SIM_CREATE] Request simulation_id: {request.simulation_id}")
    logger.info(f"🚨 [SIM_CREATE] Request file_id: {request.file_id}")
    logger.info(f"🚨 [SIM_CREATE] Request target_cells: {request.target_cells}")
    logger.info(f"🚨 [SIM_CREATE] Request target_cells_count: {len(request.target_cells) if request.target_cells else 0}")
    logger.info(f"🚨 [SIM_CREATE] Request engine_type: {request.engine_type}")
    logger.info(f"🚨 [SIM_CREATE] Request iterations: {request.iterations}")
    logger.info(f"🚨 [SIM_CREATE] Request variables count: {len(request.variables) if request.variables else 0}")
    logger.info(f"🚨 [SIM_CREATE] About to call initiate_simulation()")
    
    try:
        response = await initiate_simulation(request, background_tasks, current_user)
        
        logger.info(f"🚨 [SIM_CREATE] ✅ SUCCESS: initiate_simulation returned simulation_id: {response.simulation_id}")
        logger.info(f"🚨 [SIM_CREATE] ✅ SUCCESS: Response status: {response.status}")
        
        # CRITICAL FIX: Register temp ID mapping to solve race condition
        if request.simulation_id and request.simulation_id.startswith('temp_') and response.simulation_id:
            register_temp_id_mapping(request.simulation_id, response.simulation_id)
            logger.info(f"🚨 [SIM_CREATE] ✅ TEMP_ID_MAPPING: Registered {request.simulation_id} -> {response.simulation_id}")
        
        logger.info(f"🚨 [SIM_CREATE] ===== SIMULATION CREATION COMPLETED =====")
        
        return response
    except Exception as e:
        logger.error(f"🚨 [SIM_CREATE] ❌ ERROR: Failed to initiate simulation: {str(e)}", exc_info=True)
        logger.error(f"🚨 [SIM_CREATE] ===== SIMULATION CREATION FAILED =====")
        raise HTTPException(status_code=500, detail=f"Failed to initiate simulation processing: {str(e)}")

@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_single_simulation_results(
    simulation_id: str, 
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Retrieve the status or results of a specific simulation."""
    logger.info(f"🔍 [DEBUG] GET /simulations/{simulation_id} called by user: {current_user.username if hasattr(current_user, 'username') else 'unknown'}")
    normalized_id = _normalize_simulation_id(simulation_id)
    if normalized_id != simulation_id:
        logger.info(f"Compatibility: mapping child id {simulation_id} -> parent {normalized_id}")
    
    response = await get_simulation_status_or_results(normalized_id)
    if not response:
        logger.error(f"🔍 [DEBUG] No response from get_simulation_status_or_results for {normalized_id}")
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    logger.info(f"🔍 [DEBUG] Original response file_id: {getattr(response, 'file_id', 'NOT_FOUND')}")
    logger.info(f"🔍 [DEBUG] Original response variables_config: {getattr(response, 'variables_config', 'NOT_FOUND') is not None}")
    
    # CRITICAL FIX: Sanitize all float values to prevent NaN JSON serialization errors
    try:
        # Convert response to dict, sanitize, then convert back
        response_dict = response.dict() if hasattr(response, 'dict') else response
        logger.info(f"🔍 [DEBUG] Dict file_id: {response_dict.get('file_id')}")
        logger.info(f"🔍 [DEBUG] Dict variables_config: {response_dict.get('variables_config') is not None}")
        
        sanitized_dict = sanitize_simulation_response(response_dict)
        logger.info(f"🔍 [DEBUG] Sanitized file_id: {sanitized_dict.get('file_id')}")
        logger.info(f"🔍 [DEBUG] Sanitized variables_config: {sanitized_dict.get('variables_config') is not None}")
        
        # Test JSON serialization to ensure it works
        import json
        json.dumps(sanitized_dict)
        
        # Convert back to response model if needed
        if hasattr(response, 'dict'):
            response = SimulationResponse(**sanitized_dict)
            logger.info(f"🔍 [DEBUG] Final response file_id: {response.file_id}")
            logger.info(f"🔍 [DEBUG] Final response variables_config: {response.variables_config is not None}")
        else:
            response = sanitized_dict
            
    except Exception as e:
        logger.error(f"Error sanitizing response for {simulation_id}: {e}")
        # Fallback to basic sanitization
        response = sanitize_simulation_response(response)
    
    logger.info(f"🔍 [DEBUG] Returning response for {simulation_id}")
    return response

@router.get("/{simulation_id}/multi-target", response_model=dict)
async def get_multi_target_results(
    simulation_id: str, 
    current_user: User = Depends(get_current_active_auth0_user)
):
    """
    🎯 MULTI-TARGET: Get full multi-target simulation results including correlations
    
    This endpoint provides access to the complete multi-target simulation results
    including correlation matrices, individual target statistics, and iteration data.
    """
    from simulation.service import SIMULATION_RESULTS_STORE
    from shared.progress_store import get_progress
    
    logger.info(f"🎯 [MULTI_TARGET_API] Fetching multi-target results for {simulation_id}")
    
    # Get simulation response
    sim_response = SIMULATION_RESULTS_STORE.get(simulation_id)
    if not sim_response:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Check if this is a multi-target simulation
    progress_data = get_progress(simulation_id)
    is_multi_target = progress_data and progress_data.get("multi_target", False)
    
    if not is_multi_target:
        raise HTTPException(
            status_code=400, 
            detail="This simulation is not a multi-target simulation. Use the regular results endpoint."
        )
    
    # Get the multi-target result
    multi_target_result = getattr(sim_response, 'multi_target_result', None)
    if not multi_target_result:
        raise HTTPException(
            status_code=404, 
            detail="Multi-target results not available. Simulation may still be running or failed."
        )
    
    # Convert to serializable format
    try:
        result_data = {
            "simulation_id": simulation_id,
            "simulation_type": "multi_target_monte_carlo",
            "targets": multi_target_result.targets,
            "total_iterations": multi_target_result.total_iterations,
            
            # Individual target results
            "target_results": multi_target_result.target_results,
            
            # Target statistics
            "target_statistics": {
                target: multi_target_result.get_target_statistics(target).dict()
                for target in multi_target_result.targets
            },
            
            # Correlation matrix
            "correlations": multi_target_result.correlations,
            "correlation_matrix": multi_target_result.get_correlation_matrix(),
            
            # Analysis summary
            "analysis_summary": {
                "targets_count": len(multi_target_result.targets),
                "iterations_completed": multi_target_result.total_iterations,
                "correlation_pairs": len(multi_target_result.targets) * (len(multi_target_result.targets) - 1) // 2,
                "valid_results": {
                    target: len([v for v in values if not math.isnan(v)]) 
                    for target, values in multi_target_result.target_results.items()
                }
            },
            
            # Metadata
            "metadata": {
                "calculated_at": sim_response.updated_at,
                "engine_type": progress_data.get("engine_type", "ultra"),
                "original_filename": progress_data.get("original_filename"),
                "user": progress_data.get("user")
            }
        }
        
        # Sanitize for JSON serialization
        from simulation.utils import sanitize_data_structure
        result_data = sanitize_data_structure(result_data)
        
        logger.info(f"🎯 [MULTI_TARGET_API] Successfully prepared multi-target results for {simulation_id}")
        return result_data
        
    except Exception as e:
        logger.error(f"❌ [MULTI_TARGET_API] Error preparing results for {simulation_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error preparing multi-target results: {str(e)}"
        )

@router.get("/{simulation_id}/status", response_model=SimulationResponse)
async def get_simulation_status(
    simulation_id: str, 
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Retrieve the status of a specific simulation (alias for compatibility)."""
    logger.info(f"Fetching status for simulation_id: {simulation_id}")
    normalized_id = _normalize_simulation_id(simulation_id)
    if normalized_id != simulation_id:
        logger.info(f"Compatibility: mapping child id {simulation_id} -> parent {normalized_id}")
    response = await get_simulation_status_or_results(normalized_id)
    if not response:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # CRITICAL FIX: Sanitize all float values to prevent NaN JSON serialization errors
    try:
        # Convert response to dict, sanitize, then convert back
        response_dict = response.dict() if hasattr(response, 'dict') else response
        sanitized_dict = sanitize_simulation_response(response_dict)
        
        # Test JSON serialization to ensure it works
        import json
        json.dumps(sanitized_dict)
        
        # Convert back to response model if needed
        if hasattr(response, 'dict'):
            response = SimulationResponse(**sanitized_dict)
        else:
            response = sanitized_dict
            
    except Exception as e:
        logger.error(f"Error sanitizing response for {simulation_id}: {e}")
        # Fallback to basic sanitization
        response = sanitize_simulation_response(response)
    
    return response

@router.post("/{simulation_id}/cancel", status_code=200)
async def cancel_simulation(
    simulation_id: str,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Request to cancel a running simulation."""
    logger.info(f"Received cancel request for sim_id: {simulation_id}")
    try:
        result = await cancel_simulation_task(simulation_id)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error cancelling simulation {simulation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cancel simulation.")

@router.get("/{simulation_id}/cancel", status_code=200)
async def cancel_simulation(
    simulation_id: str, 
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Cancel a running simulation"""
    logger.info(f"Cancelling simulation: {simulation_id}")
    try:
        result = await cancel_simulation_task(simulation_id)
        return result
    except Exception as e:
        logger.error(f"Error cancelling simulation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel simulation: {str(e)}") 

@router.get("/{simulation_id}/progress")
async def get_simulation_progress_no_auth(simulation_id: str, response: Response):
    """Get real-time simulation progress status with optimized performance and fallback"""
    start_time = time.time()
    server_timestamp = time.time()
    logger.debug(f"PROGRESS_ENDPOINT: Request received for {simulation_id}")
    
    # Set cache control headers for 1-2 second caching
    response.headers["Cache-Control"] = "public, max-age=2, must-revalidate"
    response.headers["Pragma"] = "cache"
    response.headers["Expires"] = str(int(time.time() + 2))
    
    # CORS headers are handled by app-level middleware
    
    try:
        # Import shared progress store with async support
        from shared.progress_store import _progress_store
        
        # Normalize simulation ID to handle child simulation IDs
        normalized_id = _normalize_simulation_id(simulation_id)
        if normalized_id != simulation_id:
            logger.debug(f"PROGRESS_ENDPOINT: Normalized {simulation_id} -> {normalized_id}")
        
        # Quick Redis health check if needed
        if _progress_store._should_check_redis_health():
            asyncio.create_task(_progress_store._redis_health_check())
        
        # 🚀 OPTIMIZATION: Try ultra-fast async progress retrieval with intelligent caching
        try:
            # Check for cached response first (LRU cache)
            cached_response = _progress_store._get_cached_progress(normalized_id)
            if cached_response:
                elapsed_time = time.time() - start_time
                logger.debug(f"⚡ CACHE HIT: Progress for {simulation_id} in {elapsed_time*1000:.1f}ms")
                return sanitize_simulation_response({
                    **cached_response,
                    "simulation_id": simulation_id,  # Return original ID
                    "server_timestamp": server_timestamp,
                    "_performance": {
                        "retrieval_time_ms": round(elapsed_time * 1000, 2),
                        "source": "cache"
                    }
                })
            
            # 🚀 ULTRA-FAST: Get from store with very short timeout for instant response
            progress_data = await asyncio.wait_for(
                _progress_store.get_progress_async(normalized_id),
                timeout=0.5  # 🚀 ULTRA-FAST: 500ms max for instant response during simulation
            )
        except asyncio.TimeoutError:
            logger.warning(f"PROGRESS_ENDPOINT: Timeout getting progress for {simulation_id}, using memory fallback")
            # 🚀 OPTIMIZATION: Ultra-fast memory fallback to prevent blocking
            progress_data = None
            # Try memory store directly without Redis
            _progress_store._ensure_fallback_store()
            with _progress_store._fallback_lock:
                progress_data = _progress_store._fallback_store.get(normalized_id)
        
        if progress_data:
            elapsed_time = time.time() - start_time
            logger.debug(f"PROGRESS_ENDPOINT: Found progress for {simulation_id} in {elapsed_time:.3f}s: {progress_data.get('progress_percentage', 0)}%")
            
            # Update server timestamp for freshness
            
            # Sanitize float values to prevent NaN JSON serialization errors
            sanitized_response = sanitize_simulation_response({
                "simulation_id": simulation_id,  # Return original ID for frontend consistency
                "status": progress_data.get("status", "unknown"),
                "progress_percentage": progress_data.get("progress_percentage", 0.0),
                "current_iteration": progress_data.get("current_iteration", 0),
                "total_iterations": progress_data.get("total_iterations", 0),
                "stage": progress_data.get("stage", "unknown"),
                "stage_description": progress_data.get("stage_description", "Processing..."),
                "message": progress_data.get("stage_description", "Processing..."),
                "timestamp": progress_data.get("timestamp", server_timestamp),
                "server_timestamp": server_timestamp,
                "start_time": progress_data.get("start_time"),  # Include start_time if present
                "target_count": progress_data.get("target_count") or len(progress_data.get("target_variables", [])) or 1,
                "engine_info": {
                    "engine": progress_data.get("engine", "Unknown"),
                    "engine_type": progress_data.get("engine_type", "unknown"),
                    "gpu_acceleration": progress_data.get("gpu_acceleration", False)
                },
                "_performance": {
                    "retrieval_time_ms": round((time.time() - start_time) * 1000, 2),
                    "redis_circuit_open": _progress_store._is_redis_circuit_open(),
                    "source": "redis" if not _progress_store._is_redis_circuit_open() else "memory"
                }
            })
            
            return sanitized_response
        else:
            elapsed_time = time.time() - start_time
            logger.warning(f"PROGRESS_ENDPOINT: No progress found for {simulation_id} after {elapsed_time:.3f}s")
            
            # CRITICAL FIX: Handle temp IDs gracefully to prevent race condition
            if simulation_id.startswith('temp_'):
                logger.debug(f"PROGRESS_ENDPOINT: Temp ID {simulation_id} - simulation may be starting")
                # Return pending status for temp IDs instead of 404
                return sanitize_simulation_response({
                    "simulation_id": simulation_id,
                    "status": "initializing", 
                    "progress_percentage": 0.0,
                    "message": "Simulation request received, processing...",
                    "timestamp": time.time(),
                    "server_timestamp": server_timestamp
                })
            
            # Check if simulation exists in database before returning 404
            try:
                from simulation.service import get_simulation_status_or_results
                sim_status = await asyncio.wait_for(
                    get_simulation_status_or_results(normalized_id),
                    timeout=settings.PROGRESS_ENDPOINT_TIMEOUT
                )
                if sim_status and sim_status.status in ["completed", "failed", "cancelled"]:
                    # Simulation exists but is finished - return graceful response
                    return sanitize_simulation_response({
                        "simulation_id": simulation_id,
                        "status": sim_status.status,
                        "progress_percentage": 100.0 if sim_status.status == "completed" else 0.0,
                        "message": "Simulation completed",
                        "timestamp": server_timestamp,
                        "server_timestamp": server_timestamp
                    })
            except:
                pass  # Fall through to 404
            
            # Return 404 for not found
            raise HTTPException(status_code=404, detail='Simulation not found or completed')
            
    except HTTPException:
        # Re-raise HTTPExceptions (like 404)
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"PROGRESS_ENDPOINT: Error getting progress for {simulation_id} after {elapsed_time:.3f}s: {e}", exc_info=True)
        
        # Return graceful degradation response with 202 status
        response.status_code = 202
        return sanitize_simulation_response({
            "simulation_id": simulation_id,
            "status": "processing", 
            "progress_percentage": 0.0,
            "message": "Still processing, please retry",
            "timestamp": time.time(),
            "server_timestamp": time.time(),
            "_error": "Progress retrieval temporarily unavailable"
        }) 


# 🛡️ ANTI-STALE DATA ENDPOINTS
@router.get("/cache/stats")
async def get_cache_statistics(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """
    📊 Get current simulation cache statistics for monitoring and debugging
    """
    try:
        stats = get_simulation_cache_stats()
        logger.info(f"📊 [CACHE_STATS] Requested by user: {getattr(current_user, 'email', 'unknown')}")
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"📊 [CACHE_STATS] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.delete("/cache/clear/{simulation_id}")
async def clear_simulation_cache(
    simulation_id: str,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """
    🧹 Clear cache for a specific simulation ID - useful for debugging stale data issues
    """
    try:
        logger.info(f"🧹 [MANUAL_CLEANUP] User {getattr(current_user, 'email', 'unknown')} clearing cache for: {simulation_id}")
        await _clear_specific_simulation_cache(simulation_id)
        
        return {
            "status": "success",
            "message": f"Cache cleared for simulation {simulation_id}",
            "simulation_id": simulation_id,
            "cleared_by": getattr(current_user, 'email', 'unknown')
        }
    except Exception as e:
        logger.error(f"🧹 [MANUAL_CLEANUP] Error clearing cache for {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.post("/cache/clear-all")
async def clear_all_simulation_cache_endpoint(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """
    🧹 Clear ALL simulation cache - use with caution! 
    This is useful when experiencing system-wide stale data issues.
    """
    try:
        logger.warning(f"🧹 [FULL_CLEANUP] User {getattr(current_user, 'email', 'unknown')} requesting FULL cache clear")
        
        # Import and call the comprehensive cleanup function
        from simulation.engines.service import clear_all_simulation_cache
        result = clear_all_simulation_cache()
        
        logger.info(f"🧹 [FULL_CLEANUP] Full cache clear completed by {getattr(current_user, 'email', 'unknown')}")
        
        return {
            "status": "success",
            "message": "All simulation cache cleared successfully",
            "result": result,
            "cleared_by": getattr(current_user, 'email', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"🧹 [FULL_CLEANUP] Error during full cache clear: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear all cache: {str(e)}")

# ===== 🧹 ENHANCED CACHE MANAGEMENT ENDPOINTS =====

@router.get("/cache/stats/comprehensive")
async def get_comprehensive_cache_stats(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get comprehensive cache statistics and health information"""
    try:
        logger.info("Comprehensive cache stats request")
        stats = get_cache_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Failed to get comprehensive cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@router.delete("/cache/user/clear")
async def clear_current_user_cache(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Clear all cached simulations for the current user"""
    try:
        user_id = str(getattr(current_user, 'id', getattr(current_user, 'sub', 'unknown')))
        logger.info(f"Clear user cache request by user: {getattr(current_user, 'email', 'unknown')} (ID: {user_id})")
        result = clear_user_cache(user_id)
        return result
    except Exception as e:
        logger.error(f"Failed to clear user cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear user cache: {str(e)}")

@router.delete("/cache/remove/{simulation_id}")
async def remove_specific_simulation_from_cache(
    simulation_id: str,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Remove a specific simulation from cache with validation"""
    try:
        logger.info(f"Remove simulation cache request: {simulation_id}")
        removed = _remove_from_cache(simulation_id, reason="manual_removal")
        
        if removed:
            return {
                "status": "success",
                "message": f"Simulation {simulation_id} removed from cache",
                "simulation_id": simulation_id
            }
        else:
            return {
                "status": "not_found",
                "message": f"Simulation {simulation_id} not found in cache",
                "simulation_id": simulation_id
            }
    except Exception as e:
        logger.error(f"Failed to remove simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove simulation: {str(e)}")

@router.post("/cache/cleanup/expired")
async def trigger_expired_cache_cleanup(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Manually trigger cleanup of expired cache items"""
    try:
        logger.info("Manual expired cache cleanup triggered")
        
        # Import the cleanup function
        from simulation.service import _cleanup_cache_if_needed
        _cleanup_cache_if_needed()
        
        # Get updated stats
        stats = get_cache_stats()
        
        return {
            "status": "success",
            "message": "Expired cache cleanup completed",
            "updated_stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to trigger expired cache cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Expired cache cleanup failed: {str(e)}")

# WebSocket endpoint moved to main.py for better routing