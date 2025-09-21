from typing import Dict, Optional, List, Tuple, Any, Set
from uuid import UUID, uuid4
from fastapi import BackgroundTasks, HTTPException
import asyncio
from datetime import datetime, timezone
import numpy as np
import math
import logging
import signal
import os
import tempfile
import traceback
import json
import time
import re

from config import settings
from simulation.schemas import (
    SimulationRequest, 
    SimulationResponse, 
    SimulationResult, 
    VariableConfig,
    ConstantConfig
)
from shared.progress_store import set_progress, get_progress
from excel_parser.service import get_formulas_for_file
from simulation.engine import MonteCarloSimulation

SIMULATION_RESULTS_STORE: Dict[str, SimulationResponse] = {}
logger = logging.getLogger(__name__)
SIMULATION_START_TIMES = {}
SIMULATION_CANCELLATION_STORE: Dict[str, bool] = {}
# 🚀 CRITICAL: Global mapping for temp_id to real_id for dual progress storage
SIMULATION_TEMP_ID_MAPPING: Dict[str, str] = {}

# ===== 🧹 CACHE MANAGEMENT SYSTEM =====
CACHE_METADATA: Dict[str, Dict[str, Any]] = {}
MAX_CACHE_AGE_HOURS = 24  # Maximum age for cached results
MAX_CACHE_SIZE = 100  # Maximum number of cached simulations

def _add_to_cache(simulation_id: str, result: SimulationResponse, user_id: str) -> None:
    """Add simulation result to cache with metadata tracking"""
    try:
        # Add result to cache
        # Use new cache management system (this is in _add_to_cache function, keep as is)
        SIMULATION_RESULTS_STORE[simulation_id] = result
        
        # Add metadata
        CACHE_METADATA[simulation_id] = {
            'created_at': datetime.now(timezone.utc),
            'user_id': user_id,
            'last_accessed': datetime.now(timezone.utc),
            'access_count': 1
        }
        
        # Trigger cache cleanup if needed
        _cleanup_cache_if_needed()
        
        logger.info(f"🧹 [CACHE] Added simulation {simulation_id} to cache (total: {len(SIMULATION_RESULTS_STORE)})")
        
    except Exception as e:
        logger.error(f"🚨 [CACHE] Failed to add {simulation_id} to cache: {e}")

def _update_cache_access(simulation_id: str) -> None:
    """Update cache access metadata"""
    try:
        if simulation_id in CACHE_METADATA:
            CACHE_METADATA[simulation_id]['last_accessed'] = datetime.now(timezone.utc)
            CACHE_METADATA[simulation_id]['access_count'] = CACHE_METADATA[simulation_id].get('access_count', 0) + 1
    except Exception as e:
        logger.error(f"🚨 [CACHE] Failed to update access for {simulation_id}: {e}")

def _is_cache_expired(simulation_id: str) -> bool:
    """Check if cached result has expired"""
    try:
        if simulation_id not in CACHE_METADATA:
            return True
            
        created_at = CACHE_METADATA[simulation_id]['created_at']
        age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
        
        is_expired = age_hours > MAX_CACHE_AGE_HOURS
        if is_expired:
            logger.info(f"🧹 [CACHE] Simulation {simulation_id} expired (age: {age_hours:.1f}h)")
        
        return is_expired
        
    except Exception as e:
        logger.error(f"🚨 [CACHE] Failed to check expiration for {simulation_id}: {e}")
        return True

def _cleanup_cache_if_needed() -> None:
    """Clean up cache if it exceeds size limits or contains expired items"""
    try:
        # Remove expired items
        expired_ids = []
        for sim_id in list(CACHE_METADATA.keys()):
            if _is_cache_expired(sim_id):
                expired_ids.append(sim_id)
        
        for sim_id in expired_ids:
            _remove_from_cache(sim_id, reason="expired")
        
        # Remove oldest items if cache is too large
        if len(SIMULATION_RESULTS_STORE) > MAX_CACHE_SIZE:
            # Sort by last_accessed (oldest first)
            sorted_items = sorted(
                CACHE_METADATA.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            items_to_remove = len(SIMULATION_RESULTS_STORE) - MAX_CACHE_SIZE
            for i in range(items_to_remove):
                sim_id = sorted_items[i][0]
                _remove_from_cache(sim_id, reason="cache_full")
        
        if expired_ids or items_to_remove > 0:
            logger.info(f"🧹 [CACHE] Cleanup complete: {len(expired_ids)} expired, {items_to_remove if 'items_to_remove' in locals() else 0} removed for space")
            
    except Exception as e:
        logger.error(f"🚨 [CACHE] Cache cleanup failed: {e}")

def _remove_from_cache(simulation_id: str, reason: str = "manual") -> bool:
    """Remove simulation from cache"""
    try:
        removed = False
        
        if simulation_id in SIMULATION_RESULTS_STORE:
            del SIMULATION_RESULTS_STORE[simulation_id]
            removed = True
        
        if simulation_id in CACHE_METADATA:
            del CACHE_METADATA[simulation_id]
        
        # Also clean up related stores
        if simulation_id in SIMULATION_START_TIMES:
            del SIMULATION_START_TIMES[simulation_id]
            
        if simulation_id in SIMULATION_CANCELLATION_STORE:
            del SIMULATION_CANCELLATION_STORE[simulation_id]
            
        # 🚀 CRITICAL: Clean up temp_id mapping
        if simulation_id in SIMULATION_TEMP_ID_MAPPING:
            del SIMULATION_TEMP_ID_MAPPING[simulation_id]
        
        if removed:
            logger.info(f"🧹 [CACHE] Removed simulation {simulation_id} from cache (reason: {reason})")
        
        return removed
        
    except Exception as e:
        logger.error(f"🚨 [CACHE] Failed to remove {simulation_id} from cache: {e}")
        return False

def clear_all_cache() -> Dict[str, Any]:
    """Clear all cached simulations"""
    try:
        cache_size = len(SIMULATION_RESULTS_STORE)
        
        SIMULATION_RESULTS_STORE.clear()
        CACHE_METADATA.clear()
        SIMULATION_START_TIMES.clear()
        SIMULATION_CANCELLATION_STORE.clear()
        SIMULATION_TEMP_ID_MAPPING.clear()
        
        logger.info(f"🧹 [CACHE] Cleared all cache ({cache_size} items)")
        
        return {
            "cleared_count": cache_size,
            "status": "success",
            "message": f"Cleared {cache_size} cached simulations"
        }
        
    except Exception as e:
        logger.error(f"🚨 [CACHE] Failed to clear cache: {e}")
        return {
            "cleared_count": 0,
            "status": "error",
            "message": f"Failed to clear cache: {e}"
        }

def clear_user_cache(user_id: str) -> Dict[str, Any]:
    """Clear all cached simulations for a specific user"""
    try:
        user_simulations = []
        
        for sim_id, metadata in CACHE_METADATA.items():
            if metadata.get('user_id') == user_id:
                user_simulations.append(sim_id)
        
        cleared_count = 0
        for sim_id in user_simulations:
            if _remove_from_cache(sim_id, reason="user_clear"):
                cleared_count += 1
        
        logger.info(f"🧹 [CACHE] Cleared {cleared_count} cached simulations for user {user_id}")
        
        return {
            "cleared_count": cleared_count,
            "status": "success",
            "message": f"Cleared {cleared_count} cached simulations for user"
        }
        
    except Exception as e:
        logger.error(f"🚨 [CACHE] Failed to clear user cache for {user_id}: {e}")
        return {
            "cleared_count": 0,
            "status": "error",
            "message": f"Failed to clear user cache: {e}"
        }

async def _enhanced_cache_cleanup(current_user: dict) -> None:
    """Enhanced cache cleanup for pre-simulation environment preparation"""
    try:
        logger.info("🛡️ [ANTI_STALE] Starting enhanced cache cleanup...")
        
        # 1. Automatic cleanup of expired items
        _cleanup_cache_if_needed()
        
        # 2. Check cache utilization
        stats = get_cache_stats()
        utilization = stats.get("cache_utilization_percent", 0)
        
        if utilization > 80:
            logger.warning(f"🧹 [CACHE] High cache utilization: {utilization}%")
            
            # Clear older entries to make room
            now = datetime.now(timezone.utc)
            old_simulations = []
            
            for sim_id, metadata in CACHE_METADATA.items():
                age_hours = (now - metadata['created_at']).total_seconds() / 3600
                if age_hours > 2:  # Remove items older than 2 hours when cache is full
                    old_simulations.append(sim_id)
            
            for sim_id in old_simulations[:10]:  # Remove up to 10 old simulations
                _remove_from_cache(sim_id, reason="high_utilization_cleanup")
        
        # 3. Log cleanup results
        final_stats = get_cache_stats()
        logger.info(f"🛡️ [ANTI_STALE] Cache cleanup complete. Current utilization: {final_stats.get('cache_utilization_percent', 0)}%")
        
    except Exception as e:
        logger.error(f"🚨 [CACHE] Enhanced cache cleanup failed: {e}")

def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    try:
        total_simulations = len(SIMULATION_RESULTS_STORE)
        total_metadata = len(CACHE_METADATA)
        
        # Calculate age statistics
        now = datetime.now(timezone.utc)
        ages = []
        user_counts = {}
        
        for sim_id, metadata in CACHE_METADATA.items():
            age_hours = (now - metadata['created_at']).total_seconds() / 3600
            ages.append(age_hours)
            
            user_id = metadata.get('user_id', 'unknown')
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        stats = {
            "total_cached_simulations": total_simulations,
            "metadata_entries": total_metadata,
            "max_cache_size": MAX_CACHE_SIZE,
            "max_cache_age_hours": MAX_CACHE_AGE_HOURS,
            "cache_utilization_percent": round((total_simulations / MAX_CACHE_SIZE) * 100, 1) if MAX_CACHE_SIZE > 0 else 0,
            "users_with_cached_data": len(user_counts),
            "simulations_per_user": user_counts
        }
        
        if ages:
            stats.update({
                "average_age_hours": round(sum(ages) / len(ages), 1),
                "oldest_simulation_hours": round(max(ages), 1),
                "newest_simulation_hours": round(min(ages), 1)
            })
        
        return stats
        
    except Exception as e:
        logger.error(f"🚨 [CACHE] Failed to get cache stats: {e}")
        return {"error": str(e)}

async def initiate_simulation(request: SimulationRequest, background_tasks: BackgroundTasks, current_user: dict) -> SimulationResponse:
    # 🔧 PHASE 28: Debug batch detection logic
    logger.info(f"🔧 [PHASE28] BATCH_DETECTION: target_cells={request.target_cells}")
    logger.info(f"🔧 [PHASE28] BATCH_DETECTION: target_cells_length={len(request.target_cells) if request.target_cells else 'None'}")
    logger.info(f"🔧 [PHASE28] BATCH_DETECTION: condition_result={request.target_cells and len(request.target_cells) > 1}")
    
    # Check if this is a batch simulation request (multiple target cells)
    if request.target_cells and len(request.target_cells) > 1:
        logger.info(f"🔧 [PHASE28] CALLING initiate_batch_simulation with {len(request.target_cells)} targets")
        return await initiate_batch_simulation(request, background_tasks, current_user)
    else:
        logger.info(f"🔧 [PHASE28] CALLING single simulation (not batch)")
    
    # 🛡️ PRE-SIMULATION VALIDATION & CLEANUP
    await _ensure_clean_simulation_environment(request, current_user)
    
    # Single target simulation (existing logic)
    sim_id = request.simulation_id
    if sim_id is None:
        from uuid import uuid4
        sim_id = str(uuid4())
        logger.info(f"Generated new simulation_id: {sim_id}")
        # Update the request object with the generated simulation_id
        request.simulation_id = sim_id
    
    # Ensure single target is set for legacy compatibility
    if not request.result_cell_coordinate and request.target_cells:
        request.result_cell_coordinate = request.target_cells[0]
    
    # Get display name for the target cell
    target_display_name = request.result_cell_coordinate  # Default to cell coordinate
    if request.target_cells_info:
        for cell_info in request.target_cells_info:
            if cell_info["name"] == request.result_cell_coordinate:
                target_display_name = cell_info["display_name"]
                logger.info(f"🔧 [TARGET_DISPLAY_NAME] {cell_info['name']} -> {cell_info['display_name']}")
                break
    
    user_name = getattr(current_user, 'email', getattr(current_user, 'username', 'unknown'))
    
    # 🚨 CRITICAL FIX: Create database record IMMEDIATELY  
    logger.info(f"🚨 [DB_CREATE] Creating database record for simulation: {sim_id}")
    try:
        from simulation.database_service import SimulationDatabaseService
        db_service = SimulationDatabaseService()
        
        request_data = {
            'simulation_id': sim_id,
            'original_filename': request.original_filename,
            'engine_type': request.engine_type,
            'result_cell_coordinate': request.result_cell_coordinate,
            'file_id': request.file_id,
            'iterations': request.iterations,
            'mc_inputs': request.variables,
            'constants': request.constants
        }
        
        db_simulation = db_service.create_simulation(
            simulation_id=sim_id,
            user_id=current_user.id,
            request_data=request_data
        )
        
        logger.info(f"🚨 [DB_CREATE] ✅ SUCCESS: Database record created for simulation: {sim_id}")
        
    except Exception as e:
        logger.error(f"🚨 [DB_CREATE] ❌ ERROR: Failed to create database record for {sim_id}: {e}", exc_info=True)
        # Don't fail the simulation if database creation fails, but log it critically
        logger.error(f"🚨 [DB_CREATE] ❌ CONTINUING WITHOUT DATABASE RECORD - THIS EXPLAINS THE ISSUE!")
    
    response = SimulationResponse(
        simulation_id=sim_id,
        status="pending",
        message="Simulation has been queued.",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        original_filename=request.original_filename,
        engine_type=request.engine_type,
        target_name=target_display_name,  # ✅ Use display name instead of cell coordinate
        user=user_name,
        # 🚨 CRITICAL FIX: Include missing fields so sidebar click works
        file_id=request.file_id,
        variables_config=[var.dict() for var in request.variables],
        target_cell=request.result_cell_coordinate,
        iterations_requested=request.iterations
    )
    # 🚨 CRITICAL DEBUG: Log when adding to memory store
    logger.info(f"🚨 [MEMORY_STORE] Adding simulation to SIMULATION_RESULTS_STORE: {sim_id}")
    logger.info(f"🚨 [MEMORY_STORE] Initial status: {response.status}")
    logger.info(f"🚨 [MEMORY_STORE] Initial progress: {getattr(response, 'progress_percentage', 'none')}%")
    
    # Store using cache management system
    user_id = getattr(current_user, 'id', getattr(current_user, 'sub', 'unknown'))
    _add_to_cache(sim_id, response, str(user_id))
    
    # 🚀 OPTIMIZATION: Set immediate progress so frontend can track from the start
    immediate_progress = {
        "status": "pending",
        "progress_percentage": 0,
        "stage": "queued",
        "stage_description": "Single simulation queued",
        "simulation_id": sim_id,
        "user": user_name,
        "original_filename": request.original_filename,
        "file_id": request.file_id,
        "engine_type": request.engine_type,
        "target_cell": request.result_cell_coordinate,
        "target_display_name": target_display_name
    }
    
    # 🚀 CRITICAL: Get temp_id for dual progress storage (Single Simulation)
    temp_id = getattr(request, 'temp_id', None) or getattr(request, 'tempId', None)
    
    # 🚀 CRITICAL: Store temp_id mapping for ongoing progress updates
    if temp_id and temp_id.startswith('temp_'):
        SIMULATION_TEMP_ID_MAPPING[sim_id] = temp_id
        logger.info(f"🔗 [TEMP_ID_MAPPING] Stored mapping: {sim_id} -> {temp_id}")
    
    # 🚀 CRITICAL FIX: Use async set_progress_async to avoid blocking the fallback lock
    from shared.progress_store import set_progress_async
    import asyncio
    asyncio.create_task(set_progress_async(sim_id, immediate_progress, temp_id, bypass_merge=True))
    
    # 🚀 CRITICAL: Broadcast WebSocket ID mapping for instant progress tracking (Single Simulation)
    try:
        logger.info(f"🔧 [WebSocket] DEBUG: request.temp_id = {temp_id}")
        logger.info(f"🔧 [WebSocket] DEBUG: request.__dict__ = {request.__dict__}")
        if temp_id and temp_id.startswith('temp_'):
            # Import and call WebSocket manager
            try:
                from websocket_manager import websocket_manager
                import asyncio
                
                # Schedule the WebSocket broadcast (non-blocking)
                asyncio.create_task(websocket_manager.send_simulation_id_mapping(temp_id, sim_id))
                logger.info(f"🚀 [WebSocket] Scheduled ID mapping broadcast: {temp_id} -> {sim_id}")
                
            except Exception as ws_error:
                logger.warning(f"⚠️ [WebSocket] Failed to schedule ID mapping broadcast: {ws_error}")
                # Continue without WebSocket - this is not critical for simulation success
        else:
            logger.warning(f"🔧 [WebSocket] No valid temp_id found in request for ID mapping. temp_id = {temp_id}")
        
    except Exception as mapping_error:
        logger.warning(f"⚠️ [WebSocket] Failed to process ID mapping: {mapping_error}")
        # Continue without WebSocket - this is not critical for simulation success
    
    request._user_context = current_user
    background_tasks.add_task(run_monte_carlo_simulation_task, request)
    
    logger.info(f"🚀 [SINGLE_SIM] ⚡ INSTANT RESPONSE: Queued simulation {sim_id} for immediate tracking")
    
    return response

async def get_simulation_status_or_results(simulation_id: str) -> Optional[SimulationResponse]:
    """Get simulation status or results from memory with database fallback"""
    
    # 🚨 CRITICAL DEBUG: Log what's in memory store vs database
    logger.info(f"🚨 [MEMORY_STORE] get_simulation_status_or_results called for: {simulation_id}")
    logger.info(f"🚨 [MEMORY_STORE] Found in SIMULATION_RESULTS_STORE: {simulation_id in SIMULATION_RESULTS_STORE}")
    
    if simulation_id in SIMULATION_RESULTS_STORE:
        # Check if cache entry is expired
        if _is_cache_expired(simulation_id):
            logger.info(f"🧹 [CACHE] Removing expired simulation {simulation_id}")
            _remove_from_cache(simulation_id, reason="expired_on_access")
            # Fall through to database if expired
        else:
            # Update access tracking
            _update_cache_access(simulation_id)
            
            in_memory_item = SIMULATION_RESULTS_STORE[simulation_id]
            
            # ✅ CRITICAL FIX: Check if the item in memory is already a SimulationResponse
            # If it is, return it directly. If not, fall through to database to construct proper response.
            from .schemas import SimulationResponse
            if isinstance(in_memory_item, SimulationResponse):
                logger.info(f"🚨 [MEMORY_STORE] Returning SimulationResponse from memory for {simulation_id}.")
                return in_memory_item
            
            # If it's a raw result (e.g., MultiTargetSimulationResult or SimulationResult from engine),
            # or any other non-SimulationResponse object, we need to fetch the full data from the DB
            # to construct a proper SimulationResponse.
            logger.info(f"🚨 [MEMORY_STORE] In-memory item for {simulation_id} is not a SimulationResponse ({type(in_memory_item)}). Falling through to DB.")
            # Fall through to database to construct a proper SimulationResponse
    
    # 🔥 NEW: Database fallback for historical simulations
    logger.info(f"📦 [DATABASE_FALLBACK] Not found in memory, checking database for: {simulation_id}")
    try:
        from database import get_db
        from models import SimulationResult as SimulationResultModel
        from .schemas import SimulationResponse, SimulationResult
        
        # Get database session
        db_gen = get_db()
        db = next(db_gen)
        
        try:
            # Query the database for this simulation
            db_simulation = db.query(SimulationResultModel).filter(
                SimulationResultModel.simulation_id == simulation_id
            ).first()
            
            if db_simulation:
                logger.info(f"📦 [DATABASE_FALLBACK] Found historical simulation in database: {simulation_id}")
                
                # ✅ CRITICAL FIX: Use the database model's to_simulation_response method
                # This method properly handles both single-target and multi-target results
                response = db_simulation.to_simulation_response()
                logger.info(f"📦 [DATABASE_FALLBACK] Database model response - Status: {response.status}")
                logger.info(f"📦 [DATABASE_FALLBACK] Database model response - Has results: {response.results is not None}")
                logger.info(f"📦 [DATABASE_FALLBACK] Database model response - Has multi_target_result: {response.multi_target_result is not None}")
                logger.info(f"📦 [DATABASE_FALLBACK] Database model response - File ID: {response.file_id}")
                
                # ✅ CACHE OPTIMIZATION: Store the full SimulationResponse back in memory for future requests
                SIMULATION_RESULTS_STORE[simulation_id] = response
                _update_cache_access(simulation_id)  # Track access
                logger.info(f"📦 [DATABASE_FALLBACK] Cached SimulationResponse in memory for future requests: {simulation_id}")
                
                logger.info(f"📦 [DATABASE_FALLBACK] Successfully created response for historical simulation: {simulation_id}")
                return response
            else:
                logger.info(f"📦 [DATABASE_FALLBACK] Simulation not found in database: {simulation_id}")
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"📦 [DATABASE_FALLBACK] Error accessing database for {simulation_id}: {e}")
    
    return None

async def cancel_simulation_task(simulation_id: str) -> Dict[str, any]:
    if simulation_id not in SIMULATION_RESULTS_STORE:
        raise HTTPException(status_code=404, detail="Simulation not found")
    SIMULATION_CANCELLATION_STORE[simulation_id] = True
    if simulation_id in SIMULATION_RESULTS_STORE:
        SIMULATION_RESULTS_STORE[simulation_id].status = "cancelled"
        SIMULATION_RESULTS_STORE[simulation_id].message = "Simulation cancellation requested."
        SIMULATION_RESULTS_STORE[simulation_id].updated_at = datetime.now(timezone.utc).isoformat()
    return {"message": "Simulation cancellation requested.", "simulation_id": simulation_id}

def update_simulation_progress(simulation_id: str, progress_data: Dict[str, Any]):
    try:

        
        # Check if this is a completion update (100% or completed status) - these should never be smoothed/skipped
        is_completion = (
            progress_data.get('progress_percentage') == 100 or 
            progress_data.get('progress_percentage') >= 99.9 or
            progress_data.get('status') == 'completed' or
            progress_data.get('status') == 'failed' or
            progress_data.get('status') == 'cancelled'
        )
        
        if is_completion:
            # Force completion updates to bypass smoothing
            from simulation.progress_smoother import force_simulation_progress
            if 'progress_percentage' in progress_data:
                force_simulation_progress(simulation_id, progress_data['progress_percentage'])

        else:
            # 🚀 TEMPORARILY DISABLED: Progress smoothing to allow all updates through
            # Apply progress smoothing to avoid jarring jumps for regular updates
            # from simulation.progress_smoother import smooth_simulation_progress
            # smoothed_data = smooth_simulation_progress(simulation_id, progress_data)
            
            # If smoother says to skip this update, return early
            # if smoothed_data is None:
            #     logger.info(f"🔧 [DEBUG] Progress smoother skipped update for {simulation_id}")
            #     return
            
            logger.info(f"🔧 [DEBUG] Progress smoother DISABLED - allowing direct update for {simulation_id}: {progress_data.get('progress_percentage', 'N/A')}%")
            
            # Use original data without smoothing
            # progress_data = smoothed_data
        
        # CRITICAL FIX: Preserve existing metadata to prevent persistence failure
        from shared.progress_store import get_progress
        existing_progress = get_progress(simulation_id) or {}
        
        # 🔧 DEBUG: Log what we're merging to identify backwards progress source
        incoming_progress = progress_data.get('progress_percentage', 'N/A')
        existing_progress_pct = existing_progress.get('progress_percentage', 'N/A')
        logger.info(f"🔧 [MERGE_DEBUG] {simulation_id}: incoming={incoming_progress}%, existing={existing_progress_pct}%")
        
        # Preserve critical fields that should never be overwritten
        preserve_fields = [
            "user", "original_filename", "file_id", "engine_type", "target_cell",
            "simulation_id", "start_time", "batch_id", "child_simulations", "total_targets"
        ]
        
        for field in preserve_fields:
            if field not in progress_data and field in existing_progress:
                progress_data[field] = existing_progress[field]
                logger.debug(f"🔧 [MERGE_DEBUG] Preserved field '{field}': {existing_progress[field]}")
        
        # 🔧 CRITICAL: Verify progress_percentage was NOT overwritten 
        final_progress = progress_data.get('progress_percentage', 'N/A')
        logger.info(f"🔧 [MERGE_DEBUG] {simulation_id}: final={final_progress}% (should match incoming={incoming_progress}%)")
        
        # Add start time if available
        if simulation_id in SIMULATION_START_TIMES:
            progress_data["start_time"] = SIMULATION_START_TIMES[simulation_id]
        
        # 🚀 CRITICAL: Get temp_id for dual storage during ongoing progress updates
        temp_id = SIMULATION_TEMP_ID_MAPPING.get(simulation_id)
        
        if temp_id:
            logger.info(f"🚀 [DUAL_STORAGE] Using temp_id for progress update: {simulation_id} -> {temp_id} ({progress_data.get('progress_percentage', 'N/A')}%)")
        
        # 🚀 CRITICAL FIX: Use async to avoid blocking fallback lock
        # 🚀 ULTIMATE FIX: Direct synchronous bridge update (faster than async tasks)
        from shared.progress_store import _progress_store
        
        try:
            logger.info(f"🔧 [DIRECT_UPDATE] About to update bridge for {simulation_id}")
            # Direct bridge storage (thread-safe, no async overhead)
            _progress_store._set_progress_bridge(simulation_id, progress_data)
            logger.info(f"✅ [DIRECT_UPDATE] Bridge updated successfully for {simulation_id}: {progress_data.get('progress_percentage', 'N/A')}%")
        except Exception as e:
            logger.error(f"❌ [DIRECT_UPDATE] Bridge update FAILED for {simulation_id}: {e}")
        
        # 🔧 PHASE23 FIX: WebSocket notifications handled by progress_store.set_progress()
        # Removed duplicate WebSocket notification - progress_store already sends it
            
    except Exception as e:
        logger.warning(f"Failed to update progress for simulation {simulation_id}: {e}")

async def run_monte_carlo_simulation_task(request: SimulationRequest):
    sim_id = request.simulation_id
    logger.info(f"🚀 Starting Monte Carlo simulation task: {sim_id}")
    start_time = datetime.now(timezone.utc)
    SIMULATION_START_TIMES[sim_id] = start_time.isoformat()
    
    try:
        # CRITICAL FIX: Set up initial progress with user and filename metadata (like initiate_simulation does)
        user_name = 'unknown'
        if hasattr(request, '_user_context') and request._user_context:
            user_context = request._user_context
            if hasattr(user_context, 'email') and user_context.email:
                user_name = user_context.email
            elif hasattr(user_context, 'username') and user_context.username:
                user_name = user_context.username
        
        # Get target display name from target_cells_info
        target_display_name = request.result_cell_coordinate  # Default to cell coordinate
        if request.target_cells_info:
            for cell_info in request.target_cells_info:
                if cell_info["name"] == request.result_cell_coordinate:
                    target_display_name = cell_info["display_name"]
                    logger.info(f"🔧 [TASK] Found target display name: {request.result_cell_coordinate} -> {target_display_name}")
                    break
        
        # Reset progress tracking for clean start
        from simulation.progress_smoother import reset_simulation_progress
        reset_simulation_progress(sim_id)
        
        # Set up initial progress with complete metadata
        initial_progress = {
            "status": "running",
            "progress_percentage": 1,
            "message": "Initializing engine...",
            "simulation_id": sim_id,
            "start_time": start_time.isoformat(),
            "user": user_name,
            "original_filename": request.original_filename,
            "file_id": request.file_id,
            "engine_type": request.engine_type,
            "target_cell": request.result_cell_coordinate,
            "target_display_name": target_display_name  # Store display name in progress
        }
        
        # Use set_progress directly to establish initial metadata
        from shared.progress_store import set_progress
        set_progress(sim_id, initial_progress)
        
        logger.info(f"🔍 [TASK] Initial progress set for {sim_id} with user: {user_name}, filename: {request.original_filename}, target: {target_display_name}")
        
        logger.info(f"🔧 [TASK] About to call run_simulation_with_engine for {sim_id}")
        sim_result = await run_simulation_with_engine(
            sim_id=sim_id,
            file_id=request.file_id,
            mc_inputs=request.variables,
            constants=request.constants,
            target_cell=request.result_cell_coordinate,
            iterations=request.iterations,
            engine_type=request.engine_type
        )
        logger.info(f"🎯 [TASK] {request.engine_type.upper()} Engine returned results for {sim_id}: {type(sim_result)} with {len(sim_result.histogram['values']) if sim_result and sim_result.histogram and sim_result.histogram.get('values') else 0} results")
        
        if not SIMULATION_CANCELLATION_STORE.get(sim_id, False):
            logger.info(f"✅ [TASK] Processing successful completion for {sim_id}")
            response = SIMULATION_RESULTS_STORE.get(sim_id)
            if response:
                logger.info(f"📊 [TASK] Updating response object for {sim_id}")
                response.results = sim_result
                response.status = "completed"
                response.message = "Simulation completed successfully."
                response.target_name = target_display_name  # ✅ FIXED: Set target display name in results
                response.updated_at = datetime.now(timezone.utc).isoformat()
                
                # Add target display name to results metadata
                if sim_result:
                    sim_result.target_display_name = target_display_name
                    logger.info(f"🔧 [TASK] Added target display name to sim_result: {target_display_name}")
                
                # UPDATE PROGRESS STORE TO MARK AS COMPLETED
                logger.info(f"🏁 [TASK] Updating progress store to completed for {sim_id}")
                # Force progress to 100% without smoothing for completion
                from simulation.progress_smoother import force_simulation_progress
                force_simulation_progress(sim_id, 100.0)
                
                update_simulation_progress(sim_id, {
                    "status": "completed",
                    "progress_percentage": 100,
                    "current_iteration": request.iterations,
                    "total_iterations": request.iterations,
                    "stage": "completed",
                    "stage_description": "Simulation completed successfully",
                    "message": "Simulation completed successfully.",
                    "completed_at": datetime.now(timezone.utc).isoformat()
                })
                logger.info(f"🎉 [TASK] Task fully completed for {sim_id}")
        else:
            await _mark_simulation_cancelled(sim_id)
            
    except Exception as e:
        error_message = f"An unexpected error occurred in simulation task: {str(e)}"
        logger.error(f"❌ Simulation {sim_id} failed: {error_message}", exc_info=True)
        await _mark_simulation_failed(sim_id, error_message)

async def _mark_simulation_failed(sim_id: str, error_message: str):
    response = SIMULATION_RESULTS_STORE.get(sim_id)
    if response and response.status != "failed":
        response.status = "failed"
        response.message = error_message
        response.updated_at = datetime.now(timezone.utc).isoformat()
        
        # CRITICAL FIX: Mark failed simulations with 0% progress, not 100%
        # This prevents batch monitor from counting failures as completions
        logger.error(f"❌ [SIMULATION_FAILED] Marking {sim_id} as FAILED (not completed): {error_message}")
        update_simulation_progress(sim_id, {
            "status": "failed", 
            "progress_percentage": 0,  # Failed = 0%, not 100%
            "message": error_message,
            "current_iteration": 0,    # No iterations completed
            "total_iterations": 0,     # No iterations attempted
            "stage": "failed",
            "stage_description": f"Simulation failed: {error_message}"
        })

async def _mark_simulation_cancelled(sim_id: str):
    response = SIMULATION_RESULTS_STORE.get(sim_id)
    if response and response.status != "cancelled":
        response.status = "cancelled"
        response.message = "Simulation cancelled by user."
        response.updated_at = datetime.now(timezone.utc).isoformat()
        update_simulation_progress(sim_id, {"status": "cancelled", "progress_percentage": 100, "message": "Simulation cancelled."})

async def run_simulation_with_engine(sim_id: str, file_id: str, mc_inputs: List[VariableConfig], constants: List[ConstantConfig], target_cell: str, iterations: int, engine_type: str = "cpu") -> SimulationResult:
    logger.info(f"⚙️ Running simulation {sim_id} with {engine_type.upper()} Engine")
    logger.info(f"⚙️ Target cell format: '{target_cell}'")
    
    try:
        # Handle target cell format - it might be "Sheet!Cell" or just "Cell"
        if '!' in target_cell:
            target_sheet_name, target_cell_name = target_cell.split('!', 1)
        else:
            # If no sheet specified, get the actual sheet name from Excel file
            try:
                # Get formulas to determine the actual sheet name
                all_formulas = await get_formulas_for_file(file_id)
                if all_formulas:
                    # Use the first sheet name found in formulas
                    target_sheet_name = list(all_formulas.keys())[0]
                    logger.info(f"⚙️ No sheet specified, detected actual sheet: {target_sheet_name}")
                else:
                    target_sheet_name = "Sheet1"  # Fallback
                    logger.info(f"⚙️ No formulas found, using fallback: {target_sheet_name}")
            except Exception as e:
                logger.warning(f"⚙️ Could not detect sheet name: {e}, using fallback")
                target_sheet_name = "Sheet1"  # Fallback
            
            target_cell_name = target_cell
            logger.info(f"⚙️ Final target: {target_sheet_name}!{target_cell_name}")
        
        logger.info(f"⚙️ Parsed target: sheet='{target_sheet_name}', cell='{target_cell_name}'")
        
        # Create progress callback function
        def progress_callback(progress_data):
            """Progress callback that updates the progress store"""
            try:
                # If progress_data is a dict, use it directly
                if isinstance(progress_data, dict):
                    progress_data["simulation_id"] = sim_id
                    progress_data["engine"] = f"{engine_type.title()}MonteCarloEngine"
                    progress_data["engine_type"] = engine_type  # CRITICAL: Ensure engine_type is always set
                    progress_data["target_cell"] = f"{target_sheet_name}!{target_cell_name}"  # CRITICAL: Add target_cell for persistence
                    progress_data["gpu_acceleration"] = engine_type in ["enhanced", "ultra"]
                    
                    # Add start time if available
                    if sim_id in SIMULATION_START_TIMES:
                        progress_data["start_time"] = SIMULATION_START_TIMES[sim_id]
                    
                    update_simulation_progress(sim_id, progress_data)
                else:
                    logger.warning(f"Progress callback received non-dict data: {progress_data}")
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
        
        # Import the necessary modules
        from excel_parser.service import get_constants_for_file, resolve_file_path
        from excel_parser.dependency_tracker import get_monte_carlo_dependent_cells
        from simulation.formula_utils import get_evaluation_order
        
        # 🚀 PROGRESS UPDATE: Excel file parsing (5%)
        update_simulation_progress(sim_id, {
            "status": "running",
            "progress_percentage": 5,
            "stage": "initialization",
            "stage_description": "Parsing Excel file structure",
            "message": f"Loading formulas from Excel file..."
        })
        
        # Get all formulas from the Excel file
        all_formulas = await get_formulas_for_file(file_id)
        
        # Get MC input cells
        mc_input_cells = set()
        mc_variables_list = []  # For dependency tracking
        for var_config in mc_inputs:
            mc_cell = (var_config.sheet_name, var_config.name.upper())
            mc_input_cells.add(mc_cell)
            mc_variables_list.append(mc_cell)
        
        # 🚀 PROGRESS UPDATE: Dependency analysis (10%)
        update_simulation_progress(sim_id, {
            "status": "running",
            "progress_percentage": 10,
            "stage": "initialization", 
            "stage_description": "Analyzing cell dependencies",
            "message": f"Processing {len(mc_input_cells)} Monte Carlo variables..."
        })
        
        # Get evaluation order for formulas FIRST (before determining constants)
        ordered_calc_steps = get_evaluation_order(target_sheet_name, target_cell_name, all_formulas, mc_input_cells, engine_type)
        
        # 🚀 PROGRESS UPDATE: Formula evaluation order calculated (15%)
        update_simulation_progress(sim_id, {
            "status": "running", 
            "progress_percentage": 15,
            "stage": "initialization",
            "stage_description": "Building formula execution order",
            "message": f"Ordered {len(ordered_calc_steps)} calculation steps"
        })
        
        # Build set of cells that will be calculated
        calculated_cells = set()
        for sheet, cell, _ in ordered_calc_steps:
            # Normalize cell reference (remove $ signs for absolute references)
            normalized_cell = cell.replace('$', '').upper()
            calculated_cells.add((sheet, normalized_cell))
        
        # Get constants from the target sheet only (prevents multi-sheet contamination)
        all_file_constants = await get_constants_for_file(file_id, exclude_cells=mc_input_cells, target_sheet=target_sheet_name)
        
        # CRITICAL: Only use constants for cells that are NOT being calculated
        # This prevents double-calculation and exponential value explosion
        constant_values = {}
        constants_used = 0
        for (sheet, cell), value in all_file_constants.items():
            # Normalize cell reference for comparison
            normalized_cell = cell.replace('$', '').upper()
            if (sheet, normalized_cell) not in calculated_cells:
                constant_values[(sheet, cell)] = value
                constants_used += 1
        
        logger.info(f"📊 [CONSTANTS] Using {constants_used} Excel constants for non-calculated cells")
        logger.info(f"📊 [CONSTANTS] Skipping constants for {len(calculated_cells)} cells that will be calculated fresh")
        
        # Add user-provided constants
        for constant in constants:
            # Assume constants apply to the target sheet if no sheet specified
            sheet_for_constant = target_sheet_name
            constant_values[(sheet_for_constant, constant.name.upper())] = constant.value
        
        # Track direct MC dependents for logging
        direct_dependent_cells = set()
        if mc_variables_list:
            try:
                # Get file path for dependency analysis using robust resolution
                file_path = resolve_file_path(file_id)
                if file_path:
                    from excel_parser.dependency_tracker import get_monte_carlo_direct_dependents
                    direct_dependent_cells = get_monte_carlo_direct_dependents(file_path, mc_variables_list)
                    logger.info(f"🎯 [DIRECT_DEPENDENTS] Found {len(direct_dependent_cells)} cells that directly reference MC variables")
            except Exception as e:
                logger.warning(f"⚠️  [DEPENDENCY_TRACKER] Direct dependency tracking failed: {e}")
        
        # 🔥 ROBUST MONTE CARLO: Process ALL formulas in topological order
        # The key is using the right mix of constants and MC-varied values
        logger.info(f"🎯 [FULL_EVALUATION] Processing complete Excel model with {len(ordered_calc_steps)} formulas")
        logger.info(f"💡 [FULL_EVALUATION] Direct MC dependents: {len(direct_dependent_cells)} cells")
        logger.info(f"📊 [FULL_EVALUATION] Using Excel constants for {len(constant_values)} pre-calculated values")
        
        # Log diagnostic information
        if ordered_calc_steps:
            logger.info(f"🔍 [FULL_EVALUATION] Formula evaluation order:")
            logger.info(f"   First 5 formulas: ")
            for i, (s, c, f) in enumerate(ordered_calc_steps[:5]):
                logger.info(f"   {i+1}. {s}!{c} = {f[:100]}...")
            logger.info(f"   ...")
            logger.info(f"   Last 5 formulas: ")
            for i, (s, c, f) in enumerate(ordered_calc_steps[-5:]):
                logger.info(f"   {len(ordered_calc_steps)-4+i}. {s}!{c} = {f[:100]}...")
            logger.info(f"   Target cell: {target_sheet_name}!{target_cell_name}")
        
        # NO FILTERING - We need ALL formulas for proper Monte Carlo simulation
        # The topological order ensures each cell is calculated exactly once per iteration
        # Using constants prevents re-calculation of non-MC-dependent cells
        
        # 🚀 PROGRESS UPDATE: Engine initialization (25%)
        update_simulation_progress(sim_id, {
            "status": "running",
            "progress_percentage": 25,
            "stage": "initialization",
            "stage_description": "Starting simulation engine",
            "message": f"Initializing {engine_type} engine..."
        })
        
        # Select the appropriate engine based on engine_type
        if engine_type == "ultra":
            logger.info(f"🔧 [PHASE22] Attempting Ultra Monte Carlo engine (UltraMonteCarloEngine) for engine type: {engine_type}")
            
            try:
                from simulation.engines.ultra_engine import UltraMonteCarloEngine
                
                # Create Ultra engine with correct parameters
                engine = UltraMonteCarloEngine(
                    iterations=iterations,
                    simulation_id=sim_id
                )
                
                # Set progress callback
                engine.set_progress_callback(progress_callback)
                
                # Run the Ultra engine - it returns (results_array, iteration_errors)
                logger.info(f"🔧 [PHASE22] Ultra engine initialized, starting simulation...")
                results_array, iteration_errors = await engine.run_simulation(
                    mc_input_configs=mc_inputs,
                    ordered_calc_steps=ordered_calc_steps,
                    target_sheet_name=target_sheet_name,
                    target_cell_coordinate=target_cell_name,
                    constant_values=constant_values,
                    workbook_path=f'uploads/{file_id}'
                )
                logger.info(f"🔧 [PHASE22] Ultra engine completed successfully")
                
            except RuntimeError as e:
                if "GPU validation failed" in str(e):
                    logger.warning(f"🔧 [PHASE22] Ultra Engine GPU validation failed: {e}")
                    logger.info(f"🔧 [PHASE22] Falling back to Enhanced Engine (CPU-based)")
                    
                    # PHASE 22: ENGINE FALLBACK - Switch to Enhanced engine
                    from simulation.engines.enhanced_engine import EnhancedMonteCarloEngine
                    
                    engine = EnhancedMonteCarloEngine(
                        iterations=iterations,
                        simulation_id=sim_id
                    )
                    engine.set_progress_callback(progress_callback)
                    
                    # Run Enhanced engine instead
                    results_array, iteration_errors = await engine.run_simulation(
                        mc_input_configs=mc_inputs,
                        ordered_calc_steps=ordered_calc_steps,
                        target_sheet_name=target_sheet_name,
                        target_cell_coordinate=target_cell_name,
                        constant_values=constant_values,
                        workbook_path=f'uploads/{file_id}'
                    )
                    logger.info(f"🔧 [PHASE22] Enhanced engine fallback completed successfully")
                else:
                    # Other RuntimeError, re-raise
                    raise
            except Exception as e:
                logger.error(f"🔧 [PHASE22] Ultra Engine failed with unexpected error: {e}")
                logger.info(f"🔧 [PHASE22] Falling back to Enhanced Engine due to unexpected error")
                
                # Fallback for any other unexpected error
                from simulation.engines.enhanced_engine import EnhancedMonteCarloEngine
                
                engine = EnhancedMonteCarloEngine(
                    iterations=iterations,
                    simulation_id=sim_id
                )
                engine.set_progress_callback(progress_callback)
                
                results_array, iteration_errors = await engine.run_simulation(
                    mc_input_configs=mc_inputs,
                    ordered_calc_steps=ordered_calc_steps,
                    target_sheet_name=target_sheet_name,
                    target_cell_coordinate=target_cell_name,
                    constant_values=constant_values,
                    workbook_path=f'uploads/{file_id}'
                )
                logger.info(f"🔧 [PHASE22] Enhanced engine fallback completed successfully")
            
        else:
            logger.info(f"Using CPU Monte Carlo engine (MonteCarloSimulation) for engine type: {engine_type}")
            
            # Create and configure the CPU engine
            engine = MonteCarloSimulation(iterations=iterations)
            engine.set_progress_callback(progress_callback)
            
            # Run the simulation using the CPU engine - it returns (results_array, iteration_errors)
            results_array, iteration_errors = await engine.run_simulation(
                mc_input_configs=mc_inputs,
                ordered_calc_steps=ordered_calc_steps,
                target_sheet_name=target_sheet_name,
                target_cell_coordinate=target_cell_name,
                constant_values=constant_values
            )
        
        # Convert results to SimulationResult format
        from simulation.schemas import SimulationResult
        from shared.histogram_service import generate_histogram_statistics
        
        if results_array is not None and len(results_array) > 0:
            # Generate histogram and statistics
            histogram_data = generate_histogram_statistics(results_array)
            
            # Calculate basic statistics
            mean_value = float(np.mean(results_array))
            median_value = float(np.median(results_array))
            std_dev = float(np.std(results_array))
            min_value = float(np.min(results_array))
            max_value = float(np.max(results_array))
            
            # Calculate percentiles
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                percentiles[str(p)] = float(np.percentile(results_array, p))
            
            # Get sensitivity analysis from Ultra engine if available
            sensitivity_analysis = None
            if engine_type == "ultra" and hasattr(engine, 'get_sensitivity_analysis'):
                try:
                    ultra_sensitivity = engine.get_sensitivity_analysis()
                    if ultra_sensitivity and 'tornado_chart' in ultra_sensitivity:
                        # Convert Ultra engine format to schema-compatible format
                        sensitivity_analysis = ultra_sensitivity['tornado_chart']
                        logger.info(f"🔧 [ULTRA] Retrieved sensitivity analysis: {len(sensitivity_analysis)} variables")
                    else:
                        logger.warning("🔧 [ULTRA] No tornado chart data in sensitivity analysis")
                        sensitivity_analysis = None
                except Exception as e:
                    logger.warning(f"🔧 [ULTRA] Failed to get sensitivity analysis: {e}")
                    sensitivity_analysis = None
            
            # Create simulation result with all required fields
            simulation_result = SimulationResult(
                histogram=histogram_data,
                mean=mean_value,
                median=median_value,
                std_dev=std_dev,
                min_value=min_value,
                max_value=max_value,
                percentiles=percentiles,
                iterations_run=len(results_array),
                errors=iteration_errors if iteration_errors else [],
                sensitivity_analysis=sensitivity_analysis
            )
            
            return simulation_result
        else:
            # Return empty result if no valid results
            return SimulationResult(
                histogram={"values": [], "counts": []},
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                percentiles={"5": 0.0, "10": 0.0, "25": 0.0, "50": 0.0, "75": 0.0, "90": 0.0, "95": 0.0},
                iterations_run=0,
                errors=[]
            )
    except Exception as e:
        logger.error(f"❌ {engine_type.upper()} Engine execution failed for {sim_id}: {e}", exc_info=True)
        raise 

async def initiate_batch_simulation(request: SimulationRequest, background_tasks: BackgroundTasks, current_user: dict) -> SimulationResponse:
    """
    🎯 CRITICAL FIX: Handle TRUE multi-target Monte Carlo simulation
    
    This replaces the broken parent/child pattern with a single simulation
    that calculates ALL targets using the SAME random values per iteration.
    """
    # 🛡️ PRE-SIMULATION VALIDATION & CLEANUP (for batch simulations too)
    await _ensure_clean_simulation_environment(request, current_user)
    
    batch_id = request.batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(request.target_cells)}targets"
    
    logger.info(f"🎯 [MULTI_TARGET] Starting TRUE multi-target simulation {batch_id}")
    logger.info(f"🎯 [MULTI_TARGET] Targets: {request.target_cells}")
    logger.info(f"🎯 [MULTI_TARGET] Engine: {request.engine_type}")
    
    # Create mapping from cell coordinates to display names
    target_cell_display_names = {}
    if request.target_cells_info:
        for cell_info in request.target_cells_info:
            target_cell_display_names[cell_info["name"]] = cell_info["display_name"]
            logger.info(f"🔧 [TARGET_DISPLAY_NAME] {cell_info['name']} -> {cell_info['display_name']}")
    
    # ✅ CRITICAL FIX: Create ONE multi-target simulation instead of multiple separate simulations
    # 🚀 FIXED: Use provided simulation_id instead of always generating new one
    sim_id = request.simulation_id
    if sim_id is None:
        from uuid import uuid4
        sim_id = str(uuid4())
        logger.info(f"Generated new batch simulation_id: {sim_id}")
        # Update the request object with the generated simulation_id
        request.simulation_id = sim_id
    else:
        logger.info(f"Using provided batch simulation_id: {sim_id}")
    user_name = getattr(current_user, 'email', getattr(current_user, 'username', 'unknown'))
    
    # 🚀 TEMPORARY FIX: Skip database record creation to prevent 94-second API blocking
    # The database operation was causing the API to block for 94 seconds instead of returning immediately
    # We'll defer database record creation to the background task
    logger.info(f"🚀 [TEMP_FIX] Skipping immediate database record creation to prevent API blocking for: {sim_id}")
    logger.info(f"🚀 [TEMP_FIX] Database record will be created in background task instead")
    
    # Create ONE multi-target simulation request
    multi_target_request = SimulationRequest(
        simulation_id=sim_id,
        file_id=request.file_id,
        result_cell_coordinate=request.target_cells[0],  # Primary target for compatibility
        result_cell_sheet_name=request.result_cell_sheet_name,
        target_cells=request.target_cells,  # ✅ ALL targets in single simulation
        target_cells_info=request.target_cells_info,
        variables=request.variables,
        constants=request.constants,
        iterations=request.iterations,
        engine_type=request.engine_type,
        original_filename=request.original_filename,
        batch_id=batch_id
    )
    multi_target_request._user_context = current_user
    
    # Create display names list
    target_display_names_list = [target_cell_display_names.get(cell, cell) for cell in request.target_cells]
    
    # Create response for multi-target simulation
    response = SimulationResponse(
        simulation_id=sim_id,
        status="pending",
        message=f"Multi-target simulation with {len(request.target_cells)} targets queued",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        original_filename=request.original_filename,
        engine_type=request.engine_type,
        target_name=", ".join(target_display_names_list),  # Combined target names
        user=user_name,
        batch_id=batch_id,
        batch_simulation_ids=[],  # No child simulations needed
        # 🚨 CRITICAL FIX: Include missing fields so sidebar click works for multi-target
        file_id=request.file_id,
        variables_config=[var.dict() for var in request.variables],
        target_cell=request.target_cells[0],  # Primary target for compatibility
        iterations_requested=request.iterations
    )
    
    # Store simulation response using cache management system
    user_id = getattr(current_user, 'id', getattr(current_user, 'sub', 'unknown'))
    _add_to_cache(sim_id, response, str(user_id))
    
    # Set up simulation progress
    start_time = datetime.now(timezone.utc)
    SIMULATION_START_TIMES[sim_id] = start_time.isoformat()
    
    multi_target_progress = {
        "status": "running",
        "progress_percentage": 0,
        "message": f"Multi-target simulation starting with {len(request.target_cells)} targets",
        "simulation_id": sim_id,
        "start_time": start_time.isoformat(),
        "user": user_name,
        "original_filename": request.original_filename,
        "file_id": request.file_id,
        "engine_type": request.engine_type,
        "target_cell": ", ".join(request.target_cells),
        "batch_id": batch_id,
        "total_targets": len(request.target_cells),
        "multi_target": True,  # Flag to indicate this is a true multi-target simulation
        # ✅ Enhanced multi-target metadata
        "target_cells": request.target_cells,  # Individual targets array
        "target_display_names": target_display_names_list,  # Display names array
        "target_variables": target_display_names_list,  # For progress schema compatibility
        "correlations_pending": True,  # Will be calculated
        "simulation_type": "multi_target_monte_carlo"
    }
    
    # 🚀 OPTIMIZATION: Set immediate progress so frontend can track from the start
    immediate_progress = {
        "status": "pending",
        "progress_percentage": 0,
        "stage": "queued",
        "stage_description": f"Multi-target simulation queued with {len(request.target_cells)} targets",
        "simulation_id": sim_id,
        "start_time": start_time.isoformat(),
        "user": user_name,
        "original_filename": request.original_filename,
        "file_id": request.file_id,
        "engine_type": request.engine_type,
        "target_cell": ", ".join(request.target_cells),
        "batch_id": batch_id,
        "total_targets": len(request.target_cells),
        "multi_target": True,
        "target_cells": request.target_cells,
        "target_display_names": target_display_names_list,
        "target_variables": target_display_names_list,
        "correlations_pending": True,
        "simulation_type": "multi_target_monte_carlo"
    }
    
    # 🚀 CRITICAL: Get temp_id for dual progress storage (Multi-target)
    temp_id = getattr(request, 'temp_id', None)
    
    # 🚀 CRITICAL: Store temp_id mapping for ongoing progress updates
    if temp_id and temp_id.startswith('temp_'):
        SIMULATION_TEMP_ID_MAPPING[sim_id] = temp_id
        logger.info(f"🔗 [TEMP_ID_MAPPING] Stored mapping: {sim_id} -> {temp_id}")
    
    # 🚀 CRITICAL FIX: Use async set_progress_async to avoid blocking the fallback lock
    from shared.progress_store import set_progress_async
    import asyncio
    asyncio.create_task(set_progress_async(sim_id, immediate_progress, temp_id, bypass_merge=True))
    
    # 🚀 CRITICAL: Broadcast WebSocket ID mapping for instant progress tracking
    try:
        logger.info(f"🔧 [WebSocket] DEBUG: request.temp_id = {temp_id}")
        logger.info(f"🔧 [WebSocket] DEBUG: request.__dict__ = {request.__dict__}")
        if temp_id and temp_id.startswith('temp_'):
            # Import and call WebSocket manager
            try:
                from websocket_manager import websocket_manager
                import asyncio
                
                # Schedule the WebSocket broadcast (non-blocking)
                asyncio.create_task(websocket_manager.send_simulation_id_mapping(temp_id, sim_id))
                logger.info(f"🚀 [WebSocket] Scheduled ID mapping broadcast: {temp_id} -> {sim_id}")
                
            except Exception as ws_error:
                logger.warning(f"⚠️ [WebSocket] Failed to schedule ID mapping broadcast: {ws_error}")
                # Continue without WebSocket - this is not critical for simulation success
        else:
            logger.warning(f"🔧 [WebSocket] No valid temp_id found in request for ID mapping. temp_id = {temp_id}")
        
    except Exception as mapping_error:
        logger.warning(f"⚠️ [WebSocket] Failed to process ID mapping: {mapping_error}")
        # Continue without WebSocket - this is not critical for simulation success
    
    # 🚀 FINAL FIX: Use threading to completely decouple from FastAPI event loop
    # This ensures the simulation runs independently without blocking HTTP response
    import threading
    import asyncio
    
    def run_simulation_in_thread():
        """Run simulation with Redis connection fix"""
        try:
            # Create a fresh event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 🔧 CRITICAL FIX: Initialize Redis connection in this thread's event loop
            async def init_and_run():
                logger.info(f"🔧 [THREAD_TASK] Starting Redis connection reset for thread: {sim_id}")
                
                # Reset Redis connection for this thread's async event loop
                from shared.progress_store import reset_redis_connection_async
                try:
                    await reset_redis_connection_async()
                    logger.info(f"✅ [THREAD_TASK] Redis connection reset successful: {sim_id}")
                except Exception as e:
                    logger.error(f"❌ [THREAD_TASK] Redis connection reset failed: {sim_id} - {e}")
                    # Try backup approach - force reconnection
                    from shared.progress_store import _progress_store
                    _progress_store.redis_client = None
                    _progress_store.async_redis_client = None
                    logger.warning(f"🔄 [THREAD_TASK] Forced Redis reset as fallback: {sim_id}")
                
                # Now run the simulation with proper Redis access
                return await run_multi_target_simulation_task(multi_target_request)
            
            # Run the simulation with Redis fix
            result = loop.run_until_complete(init_and_run())
            logger.info(f"✅ [THREAD_TASK] Simulation completed successfully: {sim_id}")
            
        except Exception as e:
            logger.error(f"🚨 [THREAD_TASK] Simulation failed: {sim_id} - {e}")
        finally:
            try:
                loop.close()
            except:
                pass
    
    # Start in separate thread - this is fire-and-forget
    thread = threading.Thread(
        target=run_simulation_in_thread,
        name=f"sim_{sim_id}",
        daemon=True
    )
    thread.start()
    logger.info(f"🧵 [THREAD_TASK] Started simulation {sim_id} in thread: {thread.name}")
    
    logger.info(f"🚀 [ASYNC_TASK] Started simulation {sim_id} as async task")
    
    logger.info(f"🎯 [MULTI_TARGET] ⚡ INSTANT RESPONSE: Queued TRUE multi-target simulation {sim_id}")
    logger.info(f"🎯 [MULTI_TARGET] Targets: {request.target_cells}")
    logger.info(f"🎯 [MULTI_TARGET] Frontend can now track progress immediately!")
    
    return response


async def run_multi_target_simulation_task(request: SimulationRequest):
    """
    🎯 CRITICAL FIX: Execute TRUE multi-target Monte Carlo simulation
    
    This function replaces the broken parent/child pattern with a single simulation
    that calculates ALL targets using the SAME random values per iteration.
    """
    sim_id = request.simulation_id
    logger.info(f"🎯 [MULTI_TARGET_TASK] Starting multi-target simulation {sim_id}")
    logger.info(f"🎯 [MULTI_TARGET_TASK] Targets: {request.target_cells}")
    logger.info(f"🎯 [MULTI_TARGET_TASK] Engine: {request.engine_type}")
    
    try:
        # For multi-target simulations, we only support Ultra engine for now
        # as it has the proper multi-target implementation
        if request.engine_type == "ultra":
            result = await _run_ultra_multi_target_simulation(
                sim_id=sim_id,
                file_id=request.file_id,
                mc_inputs=request.variables,
                constants=request.constants,
                target_cells=request.target_cells,
                iterations=request.iterations
            )
        else:
            # For other engines, fallback to sequential execution with warning
            logger.warning(f"🎯 [MULTI_TARGET_TASK] Engine {request.engine_type} doesn't support true multi-target. Using sequential fallback.")
            result = await _run_sequential_fallback_for_multi_target(
                sim_id=sim_id,
                file_id=request.file_id,
                mc_inputs=request.variables,
                constants=request.constants,
                target_cells=request.target_cells,
                iterations=request.iterations,
                engine_type=request.engine_type
            )
        
        logger.info(f"🎯 [MULTI_TARGET_TASK] Completed successfully: {sim_id}")
        
    except Exception as e:
        logger.error(f"❌ [MULTI_TARGET_TASK] Failed: {sim_id}: {e}", exc_info=True)
        
        # Update progress store with error
        from shared.progress_store import get_progress, set_progress
        progress_data = get_progress(sim_id) or {}
        progress_data.update({
            "status": "failed",
            "progress_percentage": 0,
            "message": f"Multi-target simulation failed: {str(e)}"
        })
        set_progress(sim_id, progress_data)
        
        # Update results store with error
        if sim_id in SIMULATION_RESULTS_STORE:
            SIMULATION_RESULTS_STORE[sim_id].status = "failed"
            SIMULATION_RESULTS_STORE[sim_id].message = f"Multi-target simulation failed: {str(e)}"


async def _run_ultra_multi_target_simulation(
    sim_id: str,
    file_id: str,
    mc_inputs: List[VariableConfig],
    constants: List[ConstantConfig],
    target_cells: List[str],
    iterations: int
) -> None:
    """Run multi-target simulation using Ultra engine"""
    logger.info(f"🎯 [ULTRA_MULTI_TARGET] Starting {sim_id} with {len(target_cells)} targets")
    
    try:
        # ✅ INITIALIZATION: Set initial progress with target count
        update_simulation_progress(sim_id, {
            "status": "running",
            "progress_percentage": 0,
            "stage": "initialization", 
            "stage_description": f"Starting multi-target simulation for {len(target_cells)} targets",
            "target_count": len(target_cells),  # ✅ CRITICAL: Set target count from beginning
            "engine_type": "ultra",
            "gpu_acceleration": True
        })
        
        # 📊 STEP 1: Excel File Loading
        update_simulation_progress(sim_id, {
            "progress_percentage": 2,
            "stage": "excel_loading",
            "stage_description": "📊 Loading Excel file and validating structure",
            "target_count": len(target_cells),
            "current_step": "file_validation"
        })
        
        # Import the Ultra engine
        from simulation.engines.ultra_engine import create_ultra_engine
        
        # Create Ultra engine instance
        ultra_engine = create_ultra_engine(iterations=iterations, simulation_id=sim_id)
        
        # Set up progress callback
        def progress_cb(progress_data):
            try:
                enhanced_progress_data = {**progress_data, "simulation_id": sim_id}
                if sim_id in SIMULATION_START_TIMES:
                    enhanced_progress_data["start_time"] = SIMULATION_START_TIMES[sim_id]
                update_simulation_progress(sim_id, enhanced_progress_data)
            except Exception as e:
                logger.debug(f"Progress relay failed for {sim_id}: {e}")
        
        ultra_engine.set_progress_callback(progress_cb)
        
        # Get file path
        file_path = f'uploads/{file_id}'
        
        # Convert inputs to the format expected by Ultra engine
        mc_input_params = {}
        for mc_input in mc_inputs:
            key = (mc_input.sheet_name, mc_input.name.upper())
            mc_input_params[key] = (mc_input.min_value, mc_input.most_likely, mc_input.max_value)
        
        # ✅ CRITICAL FIX: Restore sophisticated constants loading from working baseline
        # Build set of cells that will be calculated (from ordered_calc_steps)
        calculated_cells = set()
        
        # Get all formulas and MC input cells first (needed for constants loading)
        mc_input_cells = set()
        for var_config in mc_inputs:
            mc_input_cells.add((var_config.sheet_name, var_config.name.upper()))
        
        # 📊 STEP 2: Excel Parsing Progress
        update_simulation_progress(sim_id, {
            "progress_percentage": 5,
            "stage": "excel_parsing",
            "stage_description": "📊 Parsing Excel file structure and data",
            "target_count": len(target_cells),
            "current_step": "reading_sheets"
        })
        
        # Get all formulas from the Excel file
        try:
            from excel_parser.service import get_formulas_for_file
            
            # Update progress during formula extraction
            update_simulation_progress(sim_id, {
                "progress_percentage": 8,
                "stage": "excel_parsing",
                "stage_description": "📊 Extracting formulas and cell references",
                "target_count": len(target_cells),
                "current_step": "extracting_formulas"
            })
            
            all_formulas = await get_formulas_for_file(file_id)
            total_formulas = sum(len(sheet_formulas) for sheet_formulas in all_formulas.values()) if all_formulas else 0
            
            logger.info(f"🎯 [ULTRA_MULTI_TARGET] Loaded {total_formulas} formulas from {len(all_formulas)} sheets")
            
            # Excel parsing completed
            update_simulation_progress(sim_id, {
                "progress_percentage": 12,
                "stage": "excel_parsing",
                "stage_description": f"📊 Excel parsing complete: {total_formulas:,} formulas found",
                "target_count": len(target_cells),
                "current_step": "parsing_complete",
                "total_formulas": total_formulas
            })
            
        except Exception as e:
            logger.error(f"🎯 [ULTRA_MULTI_TARGET] Failed to load formulas: {e}")
            all_formulas = {}
            update_simulation_progress(sim_id, {
                "progress_percentage": 12,
                "stage": "excel_parsing",
                "stage_description": "⚠️ Excel parsing completed with limited formula detection",
                "target_count": len(target_cells),
                "current_step": "parsing_fallback"
            })
        
        # 🔧 STEP 3: Formula Dependency Analysis
        update_simulation_progress(sim_id, {
            "progress_percentage": 15,
            "stage": "dependency_analysis",
            "stage_description": "🔧 Analyzing formula dependencies and calculation order",
            "target_count": len(target_cells),
            "current_step": "dependency_mapping"
        })
        
        # Get ordered calculation steps for ALL target cells
        from simulation.formula_utils import get_evaluation_order
        
        # ✅ CRITICAL FIX: Get dependencies for ALL targets, not just the first one
        all_ordered_steps = []
        seen_steps = set()  # Avoid duplicates
        
        for i, target_cell in enumerate(target_cells):
            # Update progress for each target dependency analysis
            progress_pct = 15 + (i / len(target_cells)) * 10  # 15% to 25%
            update_simulation_progress(sim_id, {
                "progress_percentage": progress_pct,
                "stage": "dependency_analysis",
                "stage_description": f"🔧 Analyzing dependencies for target {i+1}/{len(target_cells)}: {target_cell}",
                "target_count": len(target_cells),
                "current_step": f"analyzing_target_{i+1}"
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
                    logger.info(f"🎯 [ULTRA_MULTI_TARGET] Auto-detected sheet: {target_sheet} for target {target_cell}")
                else:
                    target_sheet = "Sheet1"  # Fallback
                target_coord = target_cell
            
            logger.info(f"🎯 [ULTRA_MULTI_TARGET] Getting dependencies for {target_sheet}!{target_coord}")
            
            target_steps = get_evaluation_order(
                target_sheet_name=target_sheet,
                target_cell_coord=target_coord,
                all_formulas=all_formulas,
                mc_input_cells=mc_input_cells,
                engine_type='ultra'
            )
            
            # Add unique steps to the combined list
            for step in target_steps:
                step_key = (step[0], step[1].upper())  # (sheet, cell) as key
                if step_key not in seen_steps:
                    all_ordered_steps.append(step)
                    seen_steps.add(step_key)
            
            logger.info(f"🎯 [ULTRA_MULTI_TARGET] Added {len(target_steps)} steps for {target_cell} (total unique: {len(all_ordered_steps)})")
        
        ordered_calc_steps = all_ordered_steps
        logger.info(f"🎯 [ULTRA_MULTI_TARGET] Found {len(ordered_calc_steps)} calculation steps")
        
        # Dependency analysis completed
        update_simulation_progress(sim_id, {
            "progress_percentage": 25,
            "stage": "dependency_analysis",
            "stage_description": f"🔧 Formula dependency analysis complete: {len(ordered_calc_steps)} calculation steps",
            "target_count": len(target_cells),
            "current_step": "dependency_complete",
            "calculation_steps": len(ordered_calc_steps)
        })
        
        # 🔍 SHEET CONSISTENCY DEBUG: Log all sheets involved
        mc_input_sheets = set()
        target_sheets = set()
        calc_step_sheets = set()
        
        for mc_sheet, mc_cell in mc_input_cells:
            mc_input_sheets.add(mc_sheet)
            
        for target_cell in target_cells:
            if "!" in target_cell:
                sheet_name = target_cell.split("!", 1)[0]
                target_sheets.add(sheet_name)
            else:
                # Auto-detected sheet logic (from earlier in this function)
                if all_formulas:
                    available_sheets = list(all_formulas.keys())
                    target_sheets.add(available_sheets[0] if available_sheets else "Sheet1")
                    
        for sheet, cell, formula in ordered_calc_steps[:10]:  # Check first 10
            calc_step_sheets.add(sheet)
            
        logger.info(f"🔍 [SHEET_DEBUG] Monte Carlo input sheets: {mc_input_sheets}")
        logger.info(f"🔍 [SHEET_DEBUG] Target cell sheets: {target_sheets}")
        logger.info(f"🔍 [SHEET_DEBUG] Calculation step sheets (sample): {calc_step_sheets}")
        
        # 🚨 SHEET MISMATCH WARNING
        all_sheets = mc_input_sheets | target_sheets | calc_step_sheets
        if len(all_sheets) > 1:
            logger.warning(f"🚨 [SHEET_MISMATCH] Multiple sheets detected: {all_sheets}")
            logger.warning(f"🚨 [SHEET_MISMATCH] This could cause Monte Carlo variable lookup failures!")
        else:
            logger.info(f"✅ [SHEET_CONSISTENCY] All operations on single sheet: {all_sheets}")
        
        # 🔧 SHEET MISMATCH FIX: Inject Monte Carlo variables on ALL calculation sheets
        # This ensures that formulas on any sheet can find the Monte Carlo values
        if len(all_sheets) > 1:
            original_mc_inputs = list(mc_input_cells)
            expanded_mc_inputs = set()
            
            for orig_sheet, orig_cell in original_mc_inputs:
                for calculation_sheet in calc_step_sheets:
                    expanded_mc_inputs.add((calculation_sheet, orig_cell))
                    
            mc_input_cells = expanded_mc_inputs
            logger.info(f"🔧 [MC_MULTI_SHEET] Expanded MC inputs from {len(original_mc_inputs)} to {len(mc_input_cells)} across sheets: {calc_step_sheets}")
        
        # ✅ CRITICAL FIX: Build calculated_cells set (from working baseline)
        for sheet, cell, _ in ordered_calc_steps:
            # Normalize cell reference (remove $ signs for absolute references)
            normalized_cell = cell.replace('$', '').upper()
            calculated_cells.add((sheet, normalized_cell))
        
        # ✅ CRITICAL FIX: Load Excel constants properly (from working baseline)
        from excel_parser.service import get_constants_for_file
        
        # Get first target sheet for constants loading (primary sheet)
        primary_target_sheet = "Sheet1"  # Default fallback
        if target_cells:
            first_target = target_cells[0]
            if "!" in first_target:
                primary_target_sheet = first_target.split("!", 1)[0]
            elif all_formulas:
                available_sheets = list(all_formulas.keys())
                primary_target_sheet = available_sheets[0] if available_sheets else "Sheet1"
        
        # Load Excel constants excluding MC input cells
        all_file_constants = await get_constants_for_file(
            file_id, 
            exclude_cells=mc_input_cells, 
            target_sheet=primary_target_sheet
        )
        
        # ✅ CRITICAL: Only use constants for cells that are NOT being calculated
        # This prevents double-calculation and exponential value explosion
        constant_values = {}
        constants_used = 0
        for (sheet, cell), value in all_file_constants.items():
            # Normalize cell reference for comparison
            normalized_cell = cell.replace('$', '').upper()
            if (sheet, normalized_cell) not in calculated_cells:
                constant_values[(sheet, cell)] = value
                constants_used += 1
        
        logger.info(f"📊 [CONSTANTS] Using {constants_used} Excel constants for non-calculated cells")
        logger.info(f"📊 [CONSTANTS] Skipping constants for {len(calculated_cells)} cells that will be calculated fresh")
        
        # Add user-provided constants (from ConstantConfig list)
        for constant in constants:
            # Assume constants apply to the primary target sheet if no sheet specified
            sheet_for_constant = primary_target_sheet
            constant_values[(sheet_for_constant, constant.name.upper())] = constant.value
        
        # 🎯 STEP 4: Multi-target Setup
        update_simulation_progress(sim_id, {
            "progress_percentage": 30,
            "stage": "multi_target_setup",
            "stage_description": f"🎯 Setting up multi-target simulation for {len(target_cells)} targets",
            "target_count": len(target_cells),
            "current_step": "target_configuration",
            "targets": target_cells,
            "calculation_steps": len(ordered_calc_steps),
            "constants_loaded": constants_used
        })
        
        # 💾 STEP 5: GPU Memory Allocation
        update_simulation_progress(sim_id, {
            "progress_percentage": 35,
            "stage": "gpu_initialization",
            "stage_description": "💾 Allocating GPU memory and initializing Monte Carlo engine",
            "target_count": len(target_cells),
            "current_step": "gpu_memory_allocation"
        })
        
        # ⚡ STEP 6: Monte Carlo Initialization
        update_simulation_progress(sim_id, {
            "progress_percentage": 40,
            "stage": "monte_carlo_init",
            "stage_description": f"⚡ Initializing Monte Carlo engine with {iterations:,} iterations",
            "target_count": len(target_cells),
            "current_step": "monte_carlo_setup",
            "iterations": iterations,
            "gpu_acceleration": True
        })
        
        # ✅ CRITICAL: Run TRUE multi-target simulation with proper constants
        multi_target_result = await ultra_engine.run_multi_target_simulation(
            target_cells=target_cells,
            mc_input_configs=mc_inputs,
            ordered_calc_steps=ordered_calc_steps,
            constant_values=constant_values,  # ✅ Use loaded Excel constants + user constants
            workbook_path=file_path
        )
        
        logger.info(f"🎯 [ULTRA_MULTI_TARGET] Simulation completed successfully")
        logger.info(f"🎯 [ULTRA_MULTI_TARGET] Targets calculated: {len(multi_target_result.targets)}")
        logger.info(f"🎯 [ULTRA_MULTI_TARGET] Iterations completed: {multi_target_result.total_iterations}")
        logger.info(f"🎯 [ULTRA_MULTI_TARGET] Correlations calculated: {len(multi_target_result.correlations)}")
        
        # ✅ SIMPLIFIED: Store multi-target result directly without artificial child simulations
        # Ultra engine already calculated all targets correctly in each iteration
        
        from shared.result_store import set_result
        
        # Update main simulation response with complete multi-target information
        if sim_id in SIMULATION_RESULTS_STORE:
            target_display_names_list = target_cells  # Use cell names as display names
            
            logger.info(f"🔧 [RESULT_STORAGE] Updating SIMULATION_RESULTS_STORE for {sim_id}")
            
            # Set simulation as completed with multi-target result
            SIMULATION_RESULTS_STORE[sim_id].status = "completed"
            SIMULATION_RESULTS_STORE[sim_id].message = f"Multi-target simulation completed: {len(target_cells)} targets analyzed"
            SIMULATION_RESULTS_STORE[sim_id].target_name = ", ".join(target_display_names_list)
            SIMULATION_RESULTS_STORE[sim_id].multi_target_result = multi_target_result
            SIMULATION_RESULTS_STORE[sim_id].updated_at = datetime.now(timezone.utc).isoformat()
            
            # Remove batch complexity - frontend will read multi_target_result directly
            SIMULATION_RESULTS_STORE[sim_id].batch_simulation_ids = None
            
            logger.info(f"🔧 [RESULT_STORAGE] About to serialize and store result for {sim_id}")
            
            try:
                # Test serialization first
                result_dict = SIMULATION_RESULTS_STORE[sim_id].dict()
                logger.info(f"🔧 [RESULT_STORAGE] Serialization successful, size: {len(str(result_dict))} chars")
                
                # ✅ CRITICAL: Persist result to Redis  
                set_result(sim_id, result_dict)
                logger.info(f"🔧 [RESULT_STORAGE] set_result() call completed for {sim_id}")
                
                # Verify storage worked
                from shared.result_store import get_result
                stored_result = get_result(sim_id)
                if stored_result:
                    logger.info(f"✅ [RESULT_STORAGE] Result successfully stored and verified for {sim_id}")
                else:
                    logger.error(f"❌ [RESULT_STORAGE] Result was not stored properly for {sim_id}")
                    
            except Exception as storage_error:
                logger.error(f"❌ [RESULT_STORAGE] Failed to store result for {sim_id}: {storage_error}", exc_info=True)
            
            logger.info(f"🎯 [ULTRA_SIMPLIFIED] Completed multi-target simulation: {sim_id}")
            logger.info(f"🎯 [ULTRA_SIMPLIFIED] Targets: {target_cells}")
            logger.info(f"🎯 [ULTRA_SIMPLIFIED] Multi-target result stored directly")
        else:
            logger.error(f"❌ [RESULT_STORAGE] Simulation {sim_id} not found in SIMULATION_RESULTS_STORE!")
        
        # Update progress store with detailed completion info
        from shared.progress_store import set_progress
        correlation_pairs = len(target_cells) * (len(target_cells) - 1) // 2
        final_progress = {
            "status": "completed",
            "progress_percentage": 100,
            "message": f"Multi-target simulation completed: {len(target_cells)} targets analyzed with {correlation_pairs} correlations",
            "simulation_id": sim_id,
            "targets_calculated": len(target_cells),
            "correlations_available": True,
            "correlations_calculated": correlation_pairs,
            "multi_target_complete": True,
            "target_cells": target_cells,  # Preserve target cell list
            "completion_summary": {
                "targets": len(target_cells),
                "iterations": multi_target_result.total_iterations,
                "correlations": correlation_pairs,
                "has_valid_results": len([t for t in multi_target_result.target_results.values() if len(t) > 0]) > 0
            }
        }
        set_progress(sim_id, final_progress)
        
        logger.info(f"🎯 [ULTRA_MULTI_TARGET] Results stored for {sim_id}")
        
        # 🔧 CRITICAL FIX: Persist simplified multi-target simulation to database
        try:
            from persistence_logging.persistence import persist_simulation_run, build_simulation_summary
            
            # Get progress metadata for persistence
            from shared.progress_store import get_progress
            progress_data = get_progress(sim_id)
            
            # Build simulation summary for persistence
            summary = build_simulation_summary(
                simulation_id=sim_id,
                status="completed",
                message=f"Multi-target simulation completed: {len(target_cells)} targets analyzed",
                engine_type="ultra",
                iterations_requested=iterations,
                variables_config=[var.dict() for var in mc_inputs],  # ✅ CRITICAL: Include variables config
                target_cell=target_cells[0] if target_cells else None,  # Use first target as primary
                started_at=progress_data.get("start_time") if progress_data else None
            )
            
            # ✅ CRITICAL FIX: Add multi-target result to summary for database persistence
            if multi_target_result:
                import json
                # Convert multi-target result to JSON for database storage
                summary['multi_target_result'] = json.dumps(multi_target_result.dict())
                logger.info(f"📦 [MULTI_TARGET_PERSISTENCE] Added multi-target result to summary for {sim_id}")
            
            # Persist to database asynchronously
            persist_success = await persist_simulation_run(summary)
            if persist_success:
                logger.info(f"📦 [DURABLE_LOG] Successfully persisted simplified multi-target simulation {sim_id} to database")
            else:
                logger.warning(f"📦 [DURABLE_LOG] Failed to persist simplified multi-target simulation {sim_id} to database")
                
        except Exception as persist_error:
            logger.error(f"📦 [DURABLE_LOG] Error persisting simplified multi-target simulation {sim_id}: {persist_error}")
            # Don't fail the simulation if persistence fails
        
    except Exception as e:
        logger.error(f"❌ [ULTRA_MULTI_TARGET] Failed: {e}", exc_info=True)
        raise


async def _run_sequential_fallback_for_multi_target(
    sim_id: str,
    file_id: str,
    mc_inputs: List[VariableConfig],
    constants: List[ConstantConfig],
    target_cells: List[str],
    iterations: int,
    engine_type: str
) -> None:
    """
    Fallback to sequential execution for engines that don't support true multi-target.
    
    WARNING: This produces mathematically incorrect results for correlation analysis
    but maintains backward compatibility.
    """
    logger.warning(f"🎯 [SEQUENTIAL_FALLBACK] Using sequential execution for {engine_type} engine")
    logger.warning(f"🎯 [SEQUENTIAL_FALLBACK] This will NOT produce valid correlations between targets")
    
    # For now, just run the first target as a single simulation
    # TODO: Implement proper sequential fallback that at least warns about correlation issues
    primary_target = target_cells[0]
    
    logger.info(f"🎯 [SEQUENTIAL_FALLBACK] Running primary target only: {primary_target}")
    
    try:
        result = await run_simulation_with_engine(
            sim_id=sim_id,
            file_id=file_id,
            mc_inputs=mc_inputs,
            constants=constants,
            target_cell=primary_target,
            iterations=iterations,
            engine_type=engine_type
        )
        
        # Update simulation response
        if sim_id in SIMULATION_RESULTS_STORE:
            SIMULATION_RESULTS_STORE[sim_id].status = "completed"
            SIMULATION_RESULTS_STORE[sim_id].results = result
            SIMULATION_RESULTS_STORE[sim_id].message = f"Sequential fallback completed for primary target: {primary_target}"
        
        logger.warning(f"🎯 [SEQUENTIAL_FALLBACK] Completed with limitations - only primary target calculated")
        
    except Exception as e:
        logger.error(f"❌ [SEQUENTIAL_FALLBACK] Failed: {e}", exc_info=True)
        raise


async def monitor_batch_simulation(parent_sim_id: str, child_simulation_ids: List[str]):
    """Monitor child simulations and update parent simulation progress"""
    
    # 🔧 PHASE 28: Comprehensive function entry debugging
    logger.info(f"🔧 [PHASE28] ENTRY: monitor_batch_simulation called")
    logger.info(f"🔧 [PHASE28] ENTRY: parent_sim_id={parent_sim_id}")
    logger.info(f"🔧 [PHASE28] ENTRY: child_simulation_ids={child_simulation_ids}")
    logger.info(f"🔧 [PHASE28] ENTRY: Number of children={len(child_simulation_ids) if child_simulation_ids else 0}")
    
    try:
        logger.info(f"🔧 [PHASE28] Attempting imports...")
        import asyncio
        from shared.progress_store import get_progress, get_progress_store
        logger.info(f"🔧 [PHASE28] Imports successful")
        
        logger.info(f"🔧 [PHASE28] About to start batch monitor execution")
        
        # Legacy logging for compatibility
        logger.info(f"🔍 [BATCH_MONITOR] Starting batch monitor for parent {parent_sim_id} with {len(child_simulation_ids)} children")
        
        completed_children = 0
        total_children = len(child_simulation_ids)
        iteration = 0
        
        while completed_children < total_children:
            iteration += 1
            logger.info(f"🔧 [PHASE28] Starting monitoring iteration {iteration}")
            logger.info(f"🔍 [BATCH_MONITOR] Iteration {iteration} for parent {parent_sim_id}: checking {total_children} children")
            
            # 🚀 PROGRESS STALL FIX: Aggregate child progress into parent progress
            total_progress = 0.0
            current_completed = 0
            
            for child_id in child_simulation_ids:
                try:
                    # 🚀 PERFORMANCE FIX: Use direct Redis access to avoid DTO transformation overhead
                    progress_store = get_progress_store()
                    if progress_store.redis_client:
                        key = progress_store._get_key(child_id)
                        raw_value = progress_store.redis_client.get(key)
                        if raw_value:
                            child_progress_data = json.loads(raw_value)
                            child_percentage = child_progress_data.get('progress_percentage', 0)
                            child_status = child_progress_data.get('status', 'unknown')
                            
                            total_progress += child_percentage
                            if child_percentage >= 100 or child_status == 'completed':
                                current_completed += 1
                            
                            if iteration % 5 == 1:  # Log every 5th iteration to reduce noise
                                logger.info(f"🔍 [BATCH_MONITOR] Child {child_id[:8]}... status: {child_status}, progress: {child_percentage:.1f}%")
                    else:
                        # Fallback to standard method if Redis unavailable
                        child_progress_data = progress_store.get_progress(child_id)
                        if child_progress_data:
                            child_percentage = child_progress_data.get('progress_percentage', 0)
                            total_progress += child_percentage
                            if child_percentage >= 100:
                                current_completed += 1
                except Exception as e:
                    logger.warning(f"🔧 [PROGRESS_AGGREGATION] Failed to get progress for child {child_id}: {e}")
            
            # Calculate average progress across all children
            if total_children > 0:
                avg_progress = total_progress / total_children
                # Update parent progress with aggregated value
                parent_progress_data = {
                    'progress_percentage': avg_progress,
                    'status': 'completed' if current_completed == total_children else 'running',
                    'current_iteration': current_completed * 1000,  # Approximate
                    'total_iterations': total_children * 1000,
                    'child_completion_count': current_completed,
                    'total_children': total_children,
                    'message': f'Batch simulation: {current_completed}/{total_children} targets completed'
                }
                
                try:
                    get_progress_store().set_progress(parent_sim_id, parent_progress_data)
                    logger.info(f"📊 [PROGRESS_AGGREGATION] Updated parent {parent_sim_id} progress: {avg_progress:.1f}% ({current_completed}/{total_children} children completed)")
                except Exception as e:
                    logger.error(f"❌ [PROGRESS_AGGREGATION] Failed to update parent progress: {e}")
            
            completed_children = current_completed
            
            logger.info(f"🔧 [PHASE28] About to sleep for 2 seconds...")
            await asyncio.sleep(2)  # Check every 2 seconds
            logger.info(f"🔧 [PHASE28] Sleep completed, continuing with iteration {iteration}")
            
            # Check status of all child simulations with CONTINUOUS progress aggregation
            completed_children = 0
            failed_children = 0
            child_results = []
            total_progress = 0.0  # CRITICAL FIX: Aggregate actual progress from all children
            
            for child_id in child_simulation_ids:
                try:
                    child_progress = get_progress(child_id)
                    if child_progress:
                        status = child_progress.get("status", "unknown")
                        child_progress_pct = child_progress.get("progress_percentage", 0.0)
                        
                        logger.info(f"🔍 [BATCH_MONITOR] Child {child_id[:8]}... status: {status}, progress: {child_progress_pct:.1f}%")
                        
                        if status == "completed":
                            completed_children += 1
                            total_progress += 100.0  # Completed = 100%
                            # Collect results from completed children
                            child_response = SIMULATION_RESULTS_STORE.get(child_id)
                            if child_response and child_response.results:
                                child_results.append({
                                    "simulation_id": child_id,
                                    "target_cell": child_response.target_name,
                                    "results": child_response.results
                                })
                        elif status == "failed":
                            failed_children += 1
                            total_progress += 0.0  # Failed = 0%
                            logger.warning(f"❌ [BATCH_MONITOR] Child {child_id[:8]}... FAILED - not counting towards parent completion")
                        elif status == "running":
                            total_progress += child_progress_pct  # CRITICAL: Use actual child progress
                        else:
                            total_progress += 0.0  # Pending/unknown = 0%
                except Exception as e:
                    logger.warning(f"🔍 [BATCH_MONITOR] Error checking child {child_id}: {e}")
                    total_progress += 0.0  # Error = 0%
            
            # CRITICAL FIX: Parent progress = average of all children's actual progress
            parent_progress = total_progress / total_children if total_children > 0 else 0.0
            logger.info(f"🔍 [BATCH_MONITOR] Parent {parent_sim_id}: {completed_children}/{total_children} completed, avg progress: {parent_progress:.1f}%")
        
        try:
            # Get existing parent progress to preserve critical metadata
            from shared.progress_store import get_progress
            existing_parent_progress = get_progress(parent_sim_id) or {}
            
            # CRITICAL FIX: Determine parent status based on successful vs failed children
            successful_children = completed_children  # Only successful completions count
            all_children_finished = (completed_children + failed_children) >= total_children
            
            if all_children_finished:
                if failed_children > 0 and completed_children == 0:
                    parent_status = "failed"  # All children failed
                    parent_message = f"Batch simulation failed: {failed_children}/{total_children} targets failed"
                elif failed_children > 0:
                    parent_status = "partial"  # Some succeeded, some failed
                    parent_message = f"Batch simulation partial: {completed_children}/{total_children} succeeded, {failed_children} failed"
                else:
                    parent_status = "completed"  # All succeeded
                    parent_message = f"Batch simulation completed: {completed_children}/{total_children} targets successful"
            else:
                parent_status = "running"
                parent_message = f"Batch simulation: {completed_children}/{total_children} completed, {failed_children} failed"
            
            # CRITICAL FIX: Calculate aggregated iteration data from children
            total_current_iterations = 0
            total_max_iterations = 0
            
            for child_id in child_simulation_ids:
                try:
                    child_progress = get_progress(child_id)
                    if child_progress:
                        total_current_iterations += child_progress.get("current_iteration", 0)
                        total_max_iterations += child_progress.get("total_iterations", 0)
                except Exception as e:
                    logger.warning(f"🔍 [BATCH_MONITOR] Error getting iteration data for child {child_id}: {e}")
            
            parent_progress_data = {
                "status": parent_status,
                "progress_percentage": parent_progress,
                "message": parent_message,
                "completed_children": completed_children,
                "total_children": total_children,
                "failed_children": failed_children,
                # CRITICAL FIX: Include aggregated iteration data
                "current_iteration": total_current_iterations,
                "total_iterations": total_max_iterations,
                # Add engine info from children
                "engine": "ultra",
                "engine_type": "ultra",
                "gpu_acceleration": True,
                # CRITICAL: Preserve metadata for persistence
                "target_cell": existing_parent_progress.get("target_cell"),
                "user": existing_parent_progress.get("user"),
                "original_filename": existing_parent_progress.get("original_filename"),
                "file_id": existing_parent_progress.get("file_id")
            }
            
            # CRITICAL FIX: Send parent progress update immediately via WebSocket
            logger.info(f"🔧 [BATCH_MONITOR] Sending parent progress update: {parent_progress:.1f}%")
            update_simulation_progress(parent_sim_id, parent_progress_data)
            
            # If all children completed, mark parent as complete and aggregate results
            if completed_children == total_children:
                logger.info(f"🎯 [BATCH_MONITOR] All children completed for parent {parent_sim_id}")
                
                # Update parent response with individual child results (not aggregated)
                parent_response = SIMULATION_RESULTS_STORE.get(parent_sim_id)
                if parent_response and child_results:
                    # 🎯 CRITICAL FIX: Store child simulation IDs in parent response for persistence
                    # (not exposed to frontend, but needed for admin logs and database)
                    parent_response.batch_simulation_ids = child_simulation_ids
                    
                    # 🎯 NEW APPROACH: Ensure child results are available in SIMULATION_RESULTS_STORE
                    # so frontend can access them individually
                    individual_results = []
                    
                    for child_result in child_results:
                        if child_result["results"]:
                            target_cell = child_result["target_cell"]
                            child_id = child_result["simulation_id"]
                            
                            # Make sure child result is in SIMULATION_RESULTS_STORE
                            child_response = SIMULATION_RESULTS_STORE.get(child_id)
                            if child_response:
                                # Ensure child response has correct status
                                child_response.status = "completed"
                                child_response.updated_at = datetime.now(timezone.utc).isoformat()
                                SIMULATION_RESULTS_STORE[child_id] = child_response
                                individual_results.append(child_response)
                                
                                logger.info(f"🎯 [BATCH_MONITOR] Child result {child_id[:8]}... ({target_cell}) available in store")
                    
                    # Update parent with summary information
                    parent_response.status = "completed"
                    parent_response.message = f"Batch simulation completed: {len(child_results)} targets processed - {', '.join([cr['target_cell'] for cr in child_results])}"
                    parent_response.updated_at = datetime.now(timezone.utc).isoformat()
                    
                    # Store the parent response
                    SIMULATION_RESULTS_STORE[parent_sim_id] = parent_response
                    
                    logger.info(f"🎯 [BATCH_MONITOR] Parent simulation {parent_sim_id} completed with {len(individual_results)} individual child results for: {', '.join([cr['target_cell'] for cr in child_results])}")
                    
                    # 🔧 CRITICAL FIX: Persist parent simulation to database
                    try:
                        from persistence_logging.persistence import persist_simulation_run, build_simulation_summary
                        
                        # Build parent simulation summary for persistence
                        parent_summary = build_simulation_summary(
                            simulation_id=parent_sim_id,
                            status="completed",
                            message=f"Batch simulation completed: {len(child_results)} targets processed - {', '.join([cr['target_cell'] for cr in child_results])}",
                            engine_type=existing_parent_progress.get("engine_type"),
                            iterations_requested=None,  # Parent doesn't have iterations
                            target_cell=existing_parent_progress.get("target_cell"),
                            started_at=existing_parent_progress.get("start_time")
                        )
                        
                        # Persist parent simulation to database
                        persist_success = await persist_simulation_run(parent_summary)
                        if persist_success:
                            logger.info(f"📦 [DURABLE_LOG] Successfully persisted parent simulation {parent_sim_id} to database")
                        else:
                            logger.warning(f"📦 [DURABLE_LOG] Failed to persist parent simulation {parent_sim_id} to database")
                        
                    except Exception as persist_error:
                        logger.error(f"📦 [DURABLE_LOG] Error persisting parent simulation {parent_sim_id}: {persist_error}")
                        # Don't fail the batch if persistence fails
                
                parent_progress_data["status"] = "completed"
                parent_progress_data["message"] = f"Batch simulation completed: {len(child_results)} targets processed"
                parent_progress_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            
            update_simulation_progress(parent_sim_id, parent_progress_data)
            
        except Exception as e:
            logger.error(f"🔍 [BATCH_MONITOR] Error updating parent progress: {e}")
    
        logger.info(f"🔧 [PHASE28] Batch monitor main loop completed successfully")
        logger.info(f"🎯 [BATCH_MONITOR] Batch monitor completed for parent {parent_sim_id}")
        return parent_sim_id
    
    except Exception as e:
        logger.error(f"🔧 [PHASE28] CRITICAL ERROR in batch monitor function: {e}")
        logger.error(f"🔧 [PHASE28] Exception type: {type(e).__name__}")
        logger.error(f"🔧 [PHASE28] Exception args: {e.args}")
        import traceback
        logger.error(f"🔧 [PHASE28] Full traceback: {traceback.format_exc()}")
        logger.error(f"🔍 [BATCH_MONITOR] CRITICAL ERROR in batch monitor for parent {parent_sim_id}: {e}", exc_info=True)
        return None 


# 🛡️ ANTI-STALE DATA PROTECTION SYSTEM
async def _ensure_clean_simulation_environment(request: SimulationRequest, current_user: dict):
    """
    🛡️ COMPREHENSIVE PRE-SIMULATION CLEANUP & VALIDATION
    
    This function ensures we have a clean environment before running any simulation
    to prevent stale data issues and zero results problems.
    """
    logger.info("🛡️ [ANTI_STALE] Starting pre-simulation environment validation...")
    
    # 1. Validate and potentially clear existing simulation data
    if request.simulation_id:
        await _validate_existing_simulation_id(request.simulation_id)
    
    # 2. Clear old/stale simulation cache using new cache management system
    await _enhanced_cache_cleanup(current_user)
    
    # 3. Validate file accessibility using robust file resolution and clear stale cache
    file_path = await _validate_simulation_file_access(request.file_id)
    if file_path:
        # Extract actual file_id from resolved path
        actual_file_id = os.path.basename(file_path).split('_')[0]
        if actual_file_id != request.file_id:
            logger.warning(f"🔄 [FILE_MAPPING] File ID changed: {request.file_id} -> {actual_file_id}")
            logger.warning(f"🧹 [CACHE_INVALIDATION] Clearing cache for potential stale data")
            await _clear_file_related_cache(actual_file_id, request.file_id)
    
    # 4. Log user context for tracking
    user_email = getattr(current_user, 'email', getattr(current_user, 'username', 'unknown'))
    logger.info(f"🛡️ [ANTI_STALE] Simulation environment validated for user: {user_email}")


async def _validate_existing_simulation_id(simulation_id: str):
    """
    🔍 Check if simulation ID already exists and handle appropriately
    """
    logger.info(f"🔍 [SIMULATION_ID_CHECK] Validating simulation ID: {simulation_id}")
    
    # Check in-memory store
    if simulation_id in SIMULATION_RESULTS_STORE:
        existing_sim = SIMULATION_RESULTS_STORE[simulation_id]
        existing_status = getattr(existing_sim, 'status', 'unknown')
        
        logger.warning(f"🚨 [SIMULATION_ID_CHECK] Found existing simulation {simulation_id} with status: {existing_status}")
        
        # If simulation is not completed or failed, clear it to prevent confusion
        if existing_status in ['pending', 'running']:
            logger.warning(f"🧹 [SIMULATION_ID_CHECK] Clearing non-completed simulation {simulation_id}")
            await _clear_specific_simulation_cache(simulation_id)
        elif existing_status == 'completed':
            # For completed simulations, log but don't auto-clear (user might want to re-access)
            logger.info(f"📋 [SIMULATION_ID_CHECK] Existing completed simulation found - will be overwritten")
            await _clear_specific_simulation_cache(simulation_id)
    
    # Check progress store
    try:
        from shared.progress_store import get_progress
        existing_progress = get_progress(simulation_id)
        if existing_progress:
            logger.info(f"🔍 [SIMULATION_ID_CHECK] Found existing progress data for {simulation_id}")
            # Clear it to ensure fresh start
            await _clear_specific_simulation_cache(simulation_id)
    except Exception as e:
        logger.debug(f"🔍 [SIMULATION_ID_CHECK] Could not check progress store: {e}")


async def _conditional_cache_cleanup():
    """
    🧹 Conditionally clear cache based on memory usage and age
    """
    try:
        # Check if we have too many simulations in memory
        current_sim_count = len(SIMULATION_RESULTS_STORE)
        MAX_SIMULATIONS = 100  # Maximum simulations to keep in memory
        
        if current_sim_count > MAX_SIMULATIONS:
            logger.info(f"🧹 [CONDITIONAL_CLEANUP] Memory store has {current_sim_count} simulations (max: {MAX_SIMULATIONS})")
            
            # Use existing cleanup service
            from shared.file_cleanup import file_cleanup_service
            cleanup_result = file_cleanup_service.cleanup_simulation_results(max_results=MAX_SIMULATIONS)
            logger.info(f"🧹 [CONDITIONAL_CLEANUP] Cleanup completed: {cleanup_result}")
        else:
            logger.debug(f"🧹 [CONDITIONAL_CLEANUP] Memory store size OK: {current_sim_count}/{MAX_SIMULATIONS}")
            
    except Exception as e:
        logger.warning(f"🧹 [CONDITIONAL_CLEANUP] Cleanup failed: {e}")


async def _validate_simulation_file_access(file_identifier: str):
    """
    📁 Validate that the simulation file is accessible using robust file resolution
    """
    logger.info(f"📁 [FILE_ACCESS_CHECK] Validating file access for: {file_identifier}")
    
    try:
        # Use robust file resolution to find the correct file
        from excel_parser.service import resolve_file_path
        
        file_path = resolve_file_path(file_identifier)
        if file_path:
            logger.info(f"✅ [FILE_ACCESS_CHECK] File resolved successfully: {file_path}")
            
            # Extract the actual file_id for logging
            import os
            actual_file_id = os.path.basename(file_path).split('_')[0]
            logger.info(f"📁 [FILE_MAPPING] '{file_identifier}' -> file_id: {actual_file_id}")
            
            # Verify file can be read
            with open(file_path, 'rb') as f:
                # Just check if we can read the first few bytes
                header = f.read(1024)
                if len(header) > 0:
                    logger.info(f"✅ [FILE_ACCESS_CHECK] File is readable: {len(header)} bytes header")
                    return file_path
        
        logger.error(f"❌ [FILE_ACCESS_CHECK] File not found or not accessible: {file_identifier}")
        return None
        
    except Exception as e:
        logger.warning(f"📁 [FILE_ACCESS_CHECK] Could not validate file access: {e}")
        return None


async def _clear_specific_simulation_cache(simulation_id: str):
    """
    🧹 Clear cache for a specific simulation ID from all stores
    """
    logger.info(f"🧹 [SPECIFIC_CLEANUP] Clearing cache for simulation: {simulation_id}")
    
    try:
        # Clear from in-memory store
        if simulation_id in SIMULATION_RESULTS_STORE:
            del SIMULATION_RESULTS_STORE[simulation_id]
            logger.debug(f"🧹 [SPECIFIC_CLEANUP] Cleared from SIMULATION_RESULTS_STORE")
        
        # Clear from progress store
        try:
            from shared.progress_store import clear_progress
            clear_progress(simulation_id)
            logger.debug(f"🧹 [SPECIFIC_CLEANUP] Cleared from progress store")
        except Exception as e:
            logger.debug(f"🧹 [SPECIFIC_CLEANUP] Could not clear progress store: {e}")
        
        # Clear from result store
        try:
            from shared.result_store import _result_store
            if hasattr(_result_store, 'delete'):
                _result_store.delete(simulation_id)
                logger.debug(f"🧹 [SPECIFIC_CLEANUP] Cleared from result store")
        except Exception as e:
            logger.debug(f"🧹 [SPECIFIC_CLEANUP] Could not clear result store: {e}")
        
        # Also clear related target simulations for multi-target
        for i in range(10):  # Clear up to 10 targets (reasonable limit)
            target_id = f"{simulation_id}_target_{i}"
            if target_id in SIMULATION_RESULTS_STORE:
                del SIMULATION_RESULTS_STORE[target_id]
                logger.debug(f"🧹 [SPECIFIC_CLEANUP] Cleared target simulation: {target_id}")
        
        logger.info(f"🧹 [SPECIFIC_CLEANUP] Successfully cleared cache for {simulation_id}")
        
    except Exception as e:
        logger.error(f"🧹 [SPECIFIC_CLEANUP] Error clearing cache for {simulation_id}: {e}")


def get_simulation_cache_stats() -> dict:
    """
    📊 Get current simulation cache statistics for monitoring
    """
    try:
        stats = {
            "in_memory_simulations": len(SIMULATION_RESULTS_STORE),
            "simulation_start_times": len(SIMULATION_START_TIMES),
            "cancellation_store": len(SIMULATION_CANCELLATION_STORE),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            stats["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        return stats
    except Exception as e:
        logger.error(f"📊 [CACHE_STATS] Error getting cache stats: {e}")


async def _clear_file_related_cache(new_file_id: str, old_file_id: str):
    """
    🧹 Clear all cached data related to file changes to prevent stale data issues
    """
    try:
        logger.info(f"🧹 [FILE_CACHE_CLEAR] Clearing cache for file change: {old_file_id} -> {new_file_id}")
        
        # Clear simulation results cache entries that might be using old file data
        cache_keys_to_remove = []
        for key, metadata in CACHE_METADATA.items():
            if 'file_id' in metadata and metadata['file_id'] in [old_file_id, new_file_id]:
                cache_keys_to_remove.append(key)
        
        for key in cache_keys_to_remove:
            if key in SIMULATION_RESULTS_STORE:
                del SIMULATION_RESULTS_STORE[key]
                logger.info(f"🧹 [FILE_CACHE_CLEAR] Removed result cache: {key}")
            if key in CACHE_METADATA:
                del CACHE_METADATA[key]
                logger.info(f"🧹 [FILE_CACHE_CLEAR] Removed metadata cache: {key}")
        
        # Clear any file-specific Arrow cache
        import os
        from pathlib import Path
        arrow_cache_files = [
            Path(f"/app/cache/{old_file_id}.feather"),
            Path(f"/app/cache/{new_file_id}.feather")
        ]
        
        for cache_file in arrow_cache_files:
            if cache_file.exists():
                try:
                    os.remove(cache_file)
                    logger.info(f"🧹 [FILE_CACHE_CLEAR] Removed Arrow cache: {cache_file}")
                except Exception as e:
                    logger.warning(f"⚠️ [FILE_CACHE_CLEAR] Could not remove Arrow cache {cache_file}: {e}")
        
        logger.info(f"✅ [FILE_CACHE_CLEAR] File-related cache cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ [FILE_CACHE_CLEAR] Cache cleanup failed: {e}")
        return {"error": str(e)}