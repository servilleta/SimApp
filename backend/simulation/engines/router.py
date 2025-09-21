from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Path
# from uuid import UUID # No longer needed for path if using str
from datetime import datetime
import logging # Added
import dateutil.parser

from ..schemas import SimulationRequest, SimulationResponse, EngineRecommendation, EngineSelectionRequest, SimulationExecutionRequest
# Removed SimulationResult as it's encapsulated in SimulationResponse from service
from .service import initiate_simulation, get_simulation_status_or_results, get_engine_recommendation, sanitize_float
import math
from auth.dependencies import get_current_active_user # Added
from auth.schemas import User # Added

logger = logging.getLogger(__name__) # Added

def sanitize_data_structure(data):
    """
    Recursively sanitize any data structure to remove NaN/inf values.
    This is a comprehensive catch-all to prevent JSON serialization errors.
    """
    if data is None:
        return data
    elif isinstance(data, (int, str, bool)):
        return data
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0.0
        return data
    elif isinstance(data, dict):
        return {key: sanitize_data_structure(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_data_structure(item) for item in data]
    else:
        # For any other types, try to convert to string as fallback
        try:
            return str(data)
        except:
            return None

router = APIRouter(
    # prefix="/simulations", # Prefix is typically defined in main.py app.include_router
    tags=["Simulations"], # Capitalized Tag
    responses={404: {"description": "Not found"}}
)

@router.post("/run", response_model=SimulationResponse, status_code=202) # 202 Accepted for background tasks
async def create_simulation_run(request: SimulationRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_active_user)): # Added dependency
    """Initiate a Monte Carlo simulation with the provided parameters. 
       The simulation runs as a background task."""
    try:
        # ✅ MODIFIED: Pass the current_user to the service layer
        response = await initiate_simulation(request, background_tasks, current_user)
        # Attach user for logging/reporting
        try:
            from simulation.service import SIMULATION_RESULTS_STORE
            if response.simulation_id in SIMULATION_RESULTS_STORE:
                setattr(SIMULATION_RESULTS_STORE[response.simulation_id], "username", current_user.username)
        except Exception as e:
            logger.debug(f"Failed to inject username into SIMULATION_RESULTS_STORE: {e}")
        return response
    except Exception as e:
        # Log the exception e
        # This part might not be reached if initiate_simulation is robust
        # but good for unforeseen issues before task queuing.
        logger.error(f"Error initiating simulation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initiate simulation processing: {str(e)}")

@router.post("/recommend-engine", response_model=EngineRecommendation)
async def recommend_engine(request: EngineSelectionRequest, current_user: User = Depends(get_current_active_user)):
    """Get engine recommendation based on file complexity"""
    try:
        # The service layer now handles all recommendation logic.
        # This endpoint can be simplified or used for more complex future analysis.
        # For now, we get a file_path and pass it to the service.
        
        file_path = request.file_path
        if not file_path:
             raise HTTPException(status_code=400, detail="file_path is required for engine recommendation.")

        # The service function `get_engine_recommendation` now contains the simplified logic
        # that will always recommend the 'enhanced' engine.
        recommendation = await get_engine_recommendation(file_path, request.mc_inputs)
        
        logger.info(f"Engine recommendation requested. Recommended: '{recommendation.recommended_engine}'")
        return recommendation
        
    except Exception as e:
        logger.error(f"Engine recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Engine recommendation failed: {str(e)}")

# -------------------------------------------------------------------------
#  STATIC admin listing routes (must come BEFORE dynamic /{simulation_id})
# -------------------------------------------------------------------------

@router.get("/active", response_model=list)
async def list_active_simulations_admin2(current_user: User = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")
    from simulation.service import SIMULATION_RESULTS_STORE
    return [
        {
            "simulation_id": sim_id,
            "status": sim.status,
            "user": getattr(sim, "username", "n/a"),
            "created_at": sim.created_at,
            "message": sim.message,
        }
        for sim_id, sim in SIMULATION_RESULTS_STORE.items() if sim.status in ("running", "pending")
    ]

@router.get("/debug-redis-data", include_in_schema=True)
async def debug_redis_data(current_user: User = Depends(get_current_active_user)):
    """Debug endpoint to see what data is stored in Redis"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")
    
    try:
        import redis
        import json
        r = redis.Redis(host='redis', port=6379, db=0)
        
        # Get all result keys
        result_keys = r.keys("simulation:results:*")
        debug_data = []
        
        for key in result_keys[:5]:  # Limit to first 5 entries
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            sim_id = key_str.split(':')[-1]
            
            result_data = r.get(key)
            if result_data:
                try:
                    result_json = json.loads(result_data)
                    debug_data.append({
                        "simulation_id": sim_id,
                        "data_keys": list(result_json.keys()),
                        "user": result_json.get('user'),
                        "original_filename": result_json.get('original_filename'),
                        "filename": result_json.get('filename'),
                        "file_name": result_json.get('file_name'),
                        "created_at": result_json.get('created_at'),
                        "target_name": result_json.get('target_name'),
                        "engine_type": result_json.get('engine_type')
                    })
                except json.JSONDecodeError:
                    debug_data.append({
                        "simulation_id": sim_id,
                        "error": "Could not parse JSON"
                    })
        
        return {
            "total_keys": len(result_keys),
            "sample_data": debug_data
        }
    
    except Exception as e:
        return {"error": str(e)}

@router.get("/history", response_model=list)
async def list_simulation_history_admin2(current_user: User = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")
    
    # 🚀 ENHANCED: Prioritize database data for completed simulations, then Redis
    from shared.progress_store import get_all_progress_keys, get_progress
    from .service import SIMULATION_RESULTS_STORE
    import redis
    import json
    from collections import defaultdict
    
    history = []
    simulation_jobs = defaultdict(list)  # Group by job
    
    # 1. First, get completed simulations from DATABASE for better data accuracy
    try:
        from database import get_db
        from models import SimulationResult, User as UserModel
        from sqlalchemy.orm import Session
        
        db = next(get_db())
        
        # Get recent simulations from database with user info
        db_simulations = db.query(SimulationResult).join(
            UserModel, SimulationResult.user_id == UserModel.id
        ).order_by(SimulationResult.created_at.desc()).limit(100).all()
        
        for sim in db_simulations:
            # Create better filename display
            display_filename = sim.original_filename
            if not display_filename or display_filename == "Unknown":
                # Fallback to simulation ID prefix when filename is unknown
                display_filename = f"Simulation {sim.simulation_id[:8]}..."
            
            # Extract user info
            username = sim.user.username if sim.user else 'unknown'
            
            # Use creation time for job grouping
            created_time = sim.created_at.isoformat() if sim.created_at else ''
            
            # Simple job key for database simulations (they're individual jobs)
            job_key = f"{sim.simulation_id}_db"
            
            simulation_jobs[job_key].append({
                "simulation_id": sim.simulation_id,
                "status": sim.status,
                "user": username,
                "file_name": display_filename,
                "created_at": created_time,
                "updated_at": sim.updated_at.isoformat() if sim.updated_at else created_time,
                "target_name": sim.target_name or "Target Variable",
                "engine_type": sim.engine_type or "unknown",
                "source": "database"  # Mark as database source
            })
            
        logger.info(f"HISTORY: Loaded {len(db_simulations)} simulations from database")
            
    except Exception as e:
        logger.warning(f"Could not fetch simulations from database: {e}")
    
    # 2. Get active simulations from progress store (Redis)
    try:
        all_progress_keys = get_all_progress_keys()
        for key in all_progress_keys:
            sim_id = key.split(':')[-1]
            
            # Skip if already loaded from database
            if any(sim_id in [s["simulation_id"] for s in job_sims] for job_sims in simulation_jobs.values()):
                continue
                
            progress_data = get_progress(sim_id)
            if progress_data:
                # Enhanced filename extraction with better fallbacks
                filename = (progress_data.get('original_filename') or 
                          progress_data.get('filename') or 
                          f"Simulation {sim_id[:8]}...")  # Fallback to sim ID prefix
                
                start_time = progress_data.get('start_time', '')
                user = progress_data.get('user', progress_data.get('username', 'n/a'))
                file_id = progress_data.get('file_id', '')
                
                # Improved job grouping using file_id and time windows
                import datetime
                try:
                    if 'T' in str(start_time):
                        dt = datetime.datetime.fromisoformat(str(start_time).replace('Z', '+00:00'))
                        time_window = dt.strftime('%Y-%m-%d_%H:%M')[:-1] + '0'  # Round to 10-minute windows
                        if file_id:
                            job_key = f"{file_id}_{time_window}"
                        else:
                            job_key = f"{filename}_{time_window}"
                    else:
                        if file_id:
                            job_key = f"{file_id}_unknown"
                        else:
                            job_key = f"{filename}_unknown"
                except:
                    if file_id:
                        job_key = f"{file_id}_unknown"
                    else:
                        job_key = f"{filename}_unknown"
                
                simulation_jobs[job_key].append({
                    "simulation_id": sim_id,
                    "status": progress_data.get('status', 'unknown'),
                    "user": user,
                    "file_name": filename,
                    "created_at": start_time,
                    "updated_at": progress_data.get('timestamp', start_time),
                    "target_name": (
                        progress_data.get('variables', {}).get(sim_id, {}).get('name') or
                        progress_data.get('target_name') or
                        progress_data.get('result_cell_coordinate') or
                        'Target Variable'
                    ),
                    "engine_type": progress_data.get('engineInfo', {}).get('engine_type', 'Unknown'),
                    "source": "redis_progress"  # Mark as Redis progress source
                })
    except Exception as e:
        logger.warning(f"Could not fetch active simulations from progress store: {e}")
    
    # 3. Get any remaining completed simulations from Redis results store
    try:
        r = redis.Redis(host='redis', port=6379, db=0)
        result_keys = r.keys("simulation:results:*")
        
        for key in result_keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            sim_id = key_str.split(':')[-1]
            
            # Skip if already in progress store or database
            if any(sim_id in [s["simulation_id"] for s in job_sims] for job_sims in simulation_jobs.values()):
                continue
            
            result_data = r.get(key)
            if result_data:
                try:
                    result_json = json.loads(result_data)
                    
                    # Enhanced filename extraction with better fallbacks
                    filename = (result_json.get('original_filename') or 
                              result_json.get('filename') or 
                              result_json.get('file_name') or 
                              f"Simulation {sim_id[:8]}...")  # Fallback to sim ID prefix
                    
                    # Extract user from multiple possible fields  
                    user = (result_json.get('user') or 
                           result_json.get('username') or 
                           'n/a')
                    
                    file_id = result_json.get('file_id', '')
                    created_time = result_json.get('created_at', '')
                    
                    # Job grouping for Redis results
                    import datetime
                    try:
                        if 'T' in str(created_time):
                            dt = datetime.datetime.fromisoformat(str(created_time).replace('Z', '+00:00'))
                            time_window = dt.strftime('%Y-%m-%d_%H:%M')[:-1] + '0'
                            if file_id:
                                job_key = f"{file_id}_{time_window}"
                            else:
                                job_key = f"{filename}_{time_window}"
                        else:
                            if file_id:
                                job_key = f"{file_id}_unknown"
                            else:
                                job_key = f"{filename}_unknown"
                    except:
                        if file_id:
                            job_key = f"{file_id}_unknown"
                        else:
                            job_key = f"{filename}_unknown"
                    
                    simulation_jobs[job_key].append({
                        "simulation_id": sim_id,
                        "status": result_json.get('status', 'completed'),
                        "user": user,
                        "file_name": filename,
                        "created_at": created_time,
                        "updated_at": result_json.get('updated_at', created_time),
                        "target_name": result_json.get('target_name', 'Target Variable'),
                        "engine_type": result_json.get('engine_type', 'Unknown'),
                        "source": "redis_results"  # Mark as Redis results source
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse results data for {sim_id}")
                    
    except Exception as e:
        logger.warning(f"Could not fetch completed simulations from results store: {e}")
    
    # 4. Convert grouped simulations to history entries (one per job)
    for job_key, job_simulations in simulation_jobs.items():
        if not job_simulations:
            continue
            
        # Sort simulations in this job by creation time
        job_simulations.sort(key=lambda x: x.get("created_at") or "0")
        
        # Use the first simulation as the representative
        primary_sim = job_simulations[0]
        
        # Determine overall job status
        statuses = [sim["status"] for sim in job_simulations]
        if "running" in statuses or "streaming" in statuses or "pending" in statuses:
            overall_status = "running"
        elif all(status == "completed" for status in statuses):
            overall_status = "completed"
        elif "failed" in statuses or "error" in statuses:
            overall_status = "failed"
        else:
            overall_status = statuses[0]
        
        # Create combined target names
        target_names = [sim["target_name"] for sim in job_simulations]
        combined_targets = ", ".join(target_names) if len(target_names) <= 3 else f"{', '.join(target_names[:3])} (+{len(target_names)-3} more)"
        
        history.append({
            "simulation_id": primary_sim["simulation_id"],  # Use primary sim ID for actions
            "job_id": job_key,  # Add job identifier
            "simulation_count": len(job_simulations),  # Number of target variables
            "target_variables": combined_targets,  # Combined target names
            "status": overall_status,
            "user": primary_sim["user"],
            "file_name": primary_sim["file_name"],  # Now has better fallback handling
            "engine_type": primary_sim["engine_type"],
            "cached": overall_status == "completed",
            "created_at": primary_sim["created_at"],
            "updated_at": max(sim.get("updated_at", "") for sim in job_simulations),
            "related_simulations": [sim["simulation_id"] for sim in job_simulations]  # For bulk operations
        })
    
    # Sort by creation time (newest first)
    history.sort(key=lambda x: x.get("created_at") or "0", reverse=True)
    return history

@router.get("/{simulation_id}/status")
async def get_simulation_progress(simulation_id: str = Path(..., title="The ID of the simulation to get progress for")):
    """Get real-time simulation progress status - NO AUTHENTICATION REQUIRED"""
    logger.info(f"GET_STATUS: Received request for sim_id: {simulation_id}")
    try:
        # Import shared progress store
        from shared.progress_store import get_progress_store, get_progress
        progress_store = get_progress_store()
        
        logger.info(f"GET_STATUS: Using Redis-backed progress store for sim_id: {simulation_id}")

        progress_data = get_progress(simulation_id)
        
        if progress_data:
            logger.info(f"GET_STATUS: Found progress for {simulation_id}: {progress_data}")
            # Ensure progress_data has a 'progress' key for the frontend
            if 'progress' not in progress_data:
                # Import sanitize_float for safe JSON serialization
                from .service import sanitize_float
                
                # Sanitize all float values to prevent NaN serialization errors
                progress_details = {
                    "percentage": sanitize_float(progress_data.get("progress_percentage", 0.0)),
                    "stage": progress_data.get("stage", "calculating"),
                    "current_iteration": int(progress_data.get("current_iteration", 0)),
                    "total_iterations": int(progress_data.get("total_iterations", 0)),
                    "current_batch": int(progress_data.get("current_batch", 0)),
                    "total_batches": int(progress_data.get("total_batches", 0)),
                    "estimated_time_remaining": sanitize_float(progress_data.get("estimated_time_remaining", 0.0)) if progress_data.get("estimated_time_remaining") is not None else None,
                    "current_target": progress_data.get("current_target", None),
                    "processing_speed": sanitize_float(progress_data.get("processing_speed", 0.0)) if progress_data.get("processing_speed") is not None else None
                }
                progress_data['progress'] = progress_details
            
            # Sanitize all float values in the main response to prevent JSON errors
            sanitized_response = {
                "simulation_id": simulation_id,
            }
            
            # Safely add all progress_data fields with sanitization
            for key, value in progress_data.items():
                if key in ["progress_percentage", "estimated_time_remaining", "processing_speed"] and value is not None:
                    sanitized_response[key] = sanitize_float(value)
                elif key in ["current_iteration", "total_iterations", "current_batch", "total_batches"] and value is not None:
                    sanitized_response[key] = int(value)
                else:
                    sanitized_response[key] = value
            
            # Apply final comprehensive sanitization to catch any remaining NaN values
            return sanitize_data_structure(sanitized_response)
        else:
            logger.warning(f"GET_STATUS: sim_id {simulation_id} not found in progress_store.")
            # Simulation not found in progress store
            return sanitize_data_structure({
                "simulation_id": simulation_id,
                "status": "not_found",
                "progress": {},
                "message": "Simulation not found or completed"
            })
    except Exception as e:
        logger.error(f"Error getting simulation progress for {simulation_id}: {e}", exc_info=True)
        return sanitize_data_structure({
            "simulation_id": simulation_id,
            "status": "error",
            "progress": {},
            "message": str(e)
        })

@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_single_simulation_results(simulation_id: str = Path(..., title="The ID of the simulation to retrieve"), current_user: User = Depends(get_current_active_user)): # Added dependency
    """Get the status or results of a previously initiated simulation."""
    # Convert simulation_id to string if it comes as UUID, though Path should give str based on type hint
    # simulation_id_str = str(simulation_id)
    
    results_response = await get_simulation_status_or_results(simulation_id) # Pass str ID
    
    if not results_response:
        raise HTTPException(status_code=404, detail=f"Simulation with ID '{simulation_id}' not found.")
    
    # Final safety check: ensure response is JSON serializable
    try:
        # Apply additional sanitization as a safety net
        from .service import sanitize_simulation_response
        results_response = sanitize_simulation_response(results_response)
        
        # Apply comprehensive data structure sanitization
        results_response_dict = sanitize_data_structure(results_response.dict())
        
        # Test JSON serialization
        import json
        json.dumps(results_response_dict)
        
        # Convert back to response model
        from ..schemas import SimulationResponse
        results_response = SimulationResponse(**results_response_dict)
        
    except Exception as e:
        logger.warning(f"Error applying final sanitization to simulation {simulation_id}: {e}")
    
    return results_response

@router.post("/{simulation_id}/cancel", status_code=200)
async def cancel_simulation(
    simulation_id: str = Path(..., title="The ID of the simulation to cancel"), 
    current_user: User = Depends(get_current_active_user)
):
    """Cancel a running simulation."""
    logger.info(f"CANCEL: Received cancel request for sim_id: {simulation_id}")
    try:
        from .service import cancel_simulation_task
        result = await cancel_simulation_task(simulation_id)
        
        if result["success"]:
            logger.info(f"CANCEL: Successfully cancelled simulation {simulation_id}")
            return {
                "simulation_id": simulation_id,
                "status": "cancelled",
                "message": "Simulation cancelled successfully"
            }
        else:
            logger.warning(f"CANCEL: Failed to cancel simulation {simulation_id}: {result['message']}")
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        logger.error(f"Error cancelling simulation {simulation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel simulation: {str(e)}")

@router.delete("/{simulation_id}", status_code=200)
async def delete_simulation(
    simulation_id: str = Path(..., title="The ID of the simulation to delete"),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a simulation job and all related simulations (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")
    
    logger.info(f"DELETE: Received delete request for sim_id: {simulation_id}")
    try:
        from .service import SIMULATION_RESULTS_STORE
        from shared.progress_store import get_progress, clear_progress
        import redis
        import os
        import glob
        import json
        
        # 🚀 ENHANCED: Find all related simulations in the same job
        related_simulations = [simulation_id]  # Start with the requested one
        
        # Try to find the job this simulation belongs to by checking the history
        try:
            # Get the history data to find related simulations
            history = await list_simulation_history_admin2(current_user)
            
            # Find the job entry that contains this simulation_id
            target_job = None
            for job_entry in history:
                if (job_entry.get("simulation_id") == simulation_id or 
                    simulation_id in job_entry.get("related_simulations", [])):
                    target_job = job_entry
                    break
            
            if target_job and target_job.get("related_simulations"):
                related_simulations = target_job["related_simulations"]
                logger.info(f"DELETE: Found job with {len(related_simulations)} related simulations: {related_simulations}")
            else:
                logger.warning(f"DELETE: Could not find job for simulation {simulation_id}, using fallback method")
                # Fallback to the original logic if history lookup fails
                # Check if it's an active simulation
                progress_data = get_progress(simulation_id)
                if progress_data:
                    filename = progress_data.get('original_filename', 'Unknown')
                    start_time = progress_data.get('start_time', '')
                    job_key = f"{filename}_{start_time}"
                    logger.info(f"DELETE: Found active job key: {job_key}")
                    
                    # Find all simulations with the same job key
                    from shared.progress_store import get_all_progress_keys
                    all_keys = get_all_progress_keys()
                    for key in all_keys:
                        other_sim_id = key.split(':')[-1]
                        other_progress = get_progress(other_sim_id)
                        if other_progress:
                            other_job_key = f"{other_progress.get('original_filename', 'Unknown')}_{other_progress.get('start_time', '')}"
                            if other_job_key == job_key and other_sim_id not in related_simulations:
                                related_simulations.append(other_sim_id)
                else:
                    # Check if it's a completed simulation
                    r = redis.Redis(host='redis', port=6379, db=0)
                    result_data = r.get(f"simulation:results:{simulation_id}")
                    if result_data:
                        result_json = json.loads(result_data)
                        filename = result_json.get('original_filename', 'Unknown')
                        created_time = result_json.get('created_at', '')
                        
                        # Find all simulations with similar job characteristics
                        result_keys = r.keys("simulation:results:*")
                        for key in result_keys:
                            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                            other_sim_id = key_str.split(':')[-1]
                            if other_sim_id == simulation_id:
                                continue
                                
                            other_result_data = r.get(key)
                            if other_result_data:
                                try:
                                    other_result_json = json.loads(other_result_data)
                                    other_filename = other_result_json.get('original_filename', 'Unknown')
                                    other_created_time = other_result_json.get('created_at', '')
                                    
                                    # Group by filename and time (within 1 minute)
                                    if (other_filename == filename and 
                                        abs(float(created_time or 0) - float(other_created_time or 0)) < 60):
                                        related_simulations.append(other_sim_id)
                                except (json.JSONDecodeError, ValueError):
                                    continue
        except Exception as e:
            logger.warning(f"Could not find related simulations: {e}")
        
        logger.info(f"DELETE: Found {len(related_simulations)} related simulations: {related_simulations}")
        
        # Delete all related simulations
        deleted_from = []
        total_deleted = 0
        
        for sim_id in related_simulations:
            sim_deleted_from = []
            
            # 1. Remove from results store (if exists)
            if sim_id in SIMULATION_RESULTS_STORE:
                del SIMULATION_RESULTS_STORE[sim_id]
                sim_deleted_from.append("results_store")
                logger.info(f"Deleted from SIMULATION_RESULTS_STORE: {sim_id}")
            
            # 2. Remove from Redis progress store
            try:
                clear_progress(sim_id)
                sim_deleted_from.append("redis_progress")
                logger.info(f"Deleted from Redis progress store: {sim_id}")
            except Exception as e:
                logger.warning(f"Could not delete {sim_id} from Redis progress store: {e}")
            
            # 3. Remove from Redis results store
            try:
                r = redis.Redis(host='redis', port=6379, db=0)
                if r.delete(f"simulation:results:{sim_id}"):
                    sim_deleted_from.append("redis_results")
                    logger.info(f"Deleted from Redis results store: {sim_id}")
            except Exception as e:
                logger.warning(f"Could not delete {sim_id} from Redis results store: {e}")
            
            # 4. Clean up result files
            result_files = glob.glob(f"results/*{sim_id}*")
            for file_path in result_files:
                try:
                    os.remove(file_path)
                    sim_deleted_from.append(f"file:{file_path}")
                    logger.info(f"Deleted result file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete result file {file_path}: {e}")
            
            # 5. Clean up cache files
            cache_files = glob.glob(f"cache/*{sim_id}*")
            for file_path in cache_files:
                try:
                    os.remove(file_path)
                    sim_deleted_from.append(f"cache:{file_path}")
                    logger.info(f"Deleted cache file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete cache file {file_path}: {e}")
            
            if sim_deleted_from:
                deleted_from.extend(sim_deleted_from)
                total_deleted += 1
        
        if total_deleted == 0:
            logger.warning(f"DELETE: No simulations were found to delete for job containing {simulation_id}")
        
        logger.info(f"DELETE: Successfully deleted {total_deleted} simulations from job containing {simulation_id}")
        return {
            "simulation_id": simulation_id,
            "status": "deleted",
            "message": f"Deleted {total_deleted} simulation(s) from job: {', '.join(set(deleted_from)) if deleted_from else 'no stores (already deleted)'}",
            "deleted_simulations": related_simulations,
            "deleted_count": total_deleted,
            "deleted_from": list(set(deleted_from))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting simulation {simulation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete simulation: {str(e)}")

@router.post("/{simulation_id}/clean-cache", status_code=200)
async def clean_simulation_cache(
    simulation_id: str = Path(..., title="The ID of the simulation to clean cache for"),
    current_user: User = Depends(get_current_active_user)
):
    """Clean cache files for a specific simulation (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")
    
    logger.info(f"CLEAN_CACHE: Received clean cache request for sim_id: {simulation_id}")
    try:
        import os
        import glob
        
        cleaned_files = []
        
        # Clean up cache files
        cache_files = glob.glob(f"cache/*{simulation_id}*")
        for file_path in cache_files:
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                cleaned_files.append({"file": file_path, "size": file_size})
                logger.info(f"Cleaned cache file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean cache file {file_path}: {e}")
        
        # Also clean temporary files
        temp_files = glob.glob(f"uploads/*{simulation_id}*")
        for file_path in temp_files:
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                cleaned_files.append({"file": file_path, "size": file_size})
                logger.info(f"Cleaned temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean temp file {file_path}: {e}")
        
        total_size = sum(f["size"] for f in cleaned_files)
        
        logger.info(f"CLEAN_CACHE: Successfully cleaned {len(cleaned_files)} files ({total_size} bytes) for simulation {simulation_id}")
        return {
            "simulation_id": simulation_id,
            "status": "cache_cleaned",
            "message": f"Cache cleaned successfully - {len(cleaned_files)} files removed",
            "files_cleaned": len(cleaned_files),
            "bytes_freed": total_size
        }
        
    except Exception as e:
        logger.error(f"Error cleaning cache for simulation {simulation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clean cache: {str(e)}")

@router.post("/clear-all-cache", status_code=200)
async def clear_all_simulation_cache_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Clear ALL simulation cache to ensure fresh start (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")
    
    logger.info("CLEAR_ALL_CACHE: Received request to clear all simulation cache")
    try:
        from .service import clear_all_simulation_cache
        result = clear_all_simulation_cache()
        
        logger.info("CLEAR_ALL_CACHE: Successfully cleared all simulation cache")
        return result
        
    except Exception as e:
        logger.error(f"Error clearing all simulation cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear all cache: {str(e)}")

@router.post("/ensure-fresh-start", status_code=200)
async def ensure_fresh_simulation_start(
    current_user: User = Depends(get_current_active_user)
):
    """Ensure completely fresh start for new simulations by clearing all cache."""
    logger.info(f"FRESH_START: User {current_user.username} requesting fresh simulation start")
    try:
        from .service import clear_all_simulation_cache
        
        # Clear all cached results
        result = clear_all_simulation_cache()
        
        logger.info(f"FRESH_START: Fresh start prepared for user {current_user.username}")
        return {
            "status": "success",
            "message": "Fresh simulation start prepared - all cache cleared",
            "user": current_user.username,
            "cache_cleared": result,
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Error preparing fresh start: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to prepare fresh start: {str(e)}")

@router.post("/clear-redis-progress", status_code=200)
async def clear_redis_progress_data(
    current_user: User = Depends(get_current_active_user)
):
    """Clear old simulation progress data from Redis to prevent multiple simulation confusion."""
    logger.info(f"REDIS_CLEANUP: User {current_user.username} requesting Redis progress cleanup")
    try:
        import redis
        import json
        
        # Connect to Redis
        r = redis.Redis(host='redis', port=6379, db=0)
        
        # Get all simulation keys
        keys = r.keys('simulation:*')
        cleaned_simulations = []
        kept_simulations = []
        
        for key in keys:
            key_str = key.decode('utf-8')
            sim_id = key_str.replace('simulation:', '')
            
            # Get simulation data
            data = r.get(key)
            if data:
                try:
                    sim_data = json.loads(data.decode('utf-8'))
                    engine_info = sim_data.get('engineInfo', {})
                    engine_type = engine_info.get('engine_type', 'Unknown')
                    status = sim_data.get('status', 'Unknown')
                    progress = sim_data.get('progress_percentage', 0)
                    
                    # Keep only active simulations with recent activity
                    if status in ['running', 'streaming'] and progress > 0:
                        kept_simulations.append({
                            'simulation_id': sim_id,
                            'engine_type': engine_type,
                            'status': status,
                            'progress': progress
                        })
                        logger.info(f"REDIS_CLEANUP: Keeping active simulation {sim_id} ({engine_type}, {status}, {progress}%)")
                    else:
                        # Delete old/stuck simulations
                        r.delete(key)
                        cleaned_simulations.append({
                            'simulation_id': sim_id,
                            'engine_type': engine_type,
                            'status': status,
                            'progress': progress
                        })
                        logger.info(f"REDIS_CLEANUP: Removed old simulation {sim_id} ({engine_type}, {status}, {progress}%)")
                        
                except json.JSONDecodeError:
                    # Delete invalid data
                    r.delete(key)
                    cleaned_simulations.append({
                        'simulation_id': sim_id,
                        'engine_type': 'Invalid',
                        'status': 'corrupted',
                        'progress': 0
                    })
                    logger.info(f"REDIS_CLEANUP: Removed corrupted simulation data {sim_id}")
        
        logger.info(f"REDIS_CLEANUP: Cleaned {len(cleaned_simulations)} simulations, kept {len(kept_simulations)} active ones")
        return {
            "status": "success",
            "message": f"Redis cleanup completed - removed {len(cleaned_simulations)} old simulations",
            "user": current_user.username,
            "cleaned_count": len(cleaned_simulations),
            "kept_count": len(kept_simulations),
            "cleaned_simulations": cleaned_simulations,
            "kept_simulations": kept_simulations,
            "timestamp": logger.info.__globals__.get('datetime', __import__('datetime')).datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning Redis progress data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clean Redis progress data: {str(e)}")

@router.get("/active-legacy", include_in_schema=False)
async def get_active_simulations(current_user: User = Depends(get_current_active_user)):
    """Return all simulations that are running or pending (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")

    from simulation.service import SIMULATION_RESULTS_STORE
    active = []
    for sim_id, sim in SIMULATION_RESULTS_STORE.items():
        if sim.status in ("running", "pending"):
            active.append({
                "simulation_id": sim_id,
                "status": sim.status,
                "user": getattr(sim, "username", "n/a"),
                "created_at": sim.created_at,
                "message": sim.message,
            })
    return active

@router.get("/history-legacy", include_in_schema=False)
async def get_simulation_history(current_user: User = Depends(get_current_active_user)):
    """Return history of all simulations (any status). Admin only."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")

    from simulation.service import SIMULATION_RESULTS_STORE
    history = []
    for sim_id, sim in SIMULATION_RESULTS_STORE.items():
        history.append({
            "simulation_id": sim_id,
            "status": sim.status,
            "user": getattr(sim, "username", "n/a"),
            "file_name": getattr(sim, "file_name", None),
            "cached": sim.status == "completed",
            "created_at": sim.created_at,
            "updated_at": sim.updated_at,
        })
    history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return history

@router.post("/run-with-engine/{sim_id}")
async def run_simulation_with_specific_engine(
    sim_id: str,
    request: SimulationExecutionRequest,
    background_tasks: BackgroundTasks
):
    """Run simulation with specific engine"""
    try:
        logger.info(f"Running simulation {sim_id} with {request.engine_type} engine")
        
        # Get simulation configuration from storage
        # This would be implemented based on your simulation storage
        # For now, we'll expect the parameters to be passed
        
        # Start background task for simulation
        background_tasks.add_task(
            _run_simulation_background,
            sim_id,
            request.engine_type
        )
        
        return {
            "message": f"Simulation {sim_id} started with {request.engine_type} engine",
            "simulation_id": sim_id,
            "engine": request.engine_type,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start simulation with specific engine: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation start failed: {str(e)}")

async def _run_simulation_background(sim_id: str, engine_type: str):
    """Background task to run simulation with specific engine"""
    try:
        # This would get the actual simulation parameters from storage
        # and call run_simulation_with_engine
        logger.info(f"Background simulation {sim_id} with {engine_type} starting...")
        
        # Placeholder - in real implementation, would fetch params and call:
        # result = await run_simulation_with_engine(sim_id, file_path, mc_inputs, constants, target_cell, iterations, engine_type)
        
    except Exception as e:
        logger.error(f"Background simulation {sim_id} failed: {e}")

@router.get("/active-old", include_in_schema=False)
async def _legacy_active_route(*args, **kwargs):
    """Placeholder to avoid path collision after reordering."""
    return []

@router.get("/history-old", include_in_schema=False)
async def _legacy_history_route(*args, **kwargs):
    """Placeholder to avoid path collision after reordering."""
    return []

# -------------------------------------------------------------------------
#  Admin listing routes placed BEFORE dynamic /{simulation_id} to ensure
#  they are matched first.
# -------------------------------------------------------------------------

# ACTIVE simulations (running / pending)
@router.get("/active-dupe", include_in_schema=False)
async def _unused_active_route(current_user: User = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")

    from simulation.service import SIMULATION_RESULTS_STORE
    active = []
    for sim_id, sim in SIMULATION_RESULTS_STORE.items():
        if sim.status in ("running", "pending"):
            active.append({
                "simulation_id": sim_id,
                "status": sim.status,
                "user": getattr(sim, "username", "n/a"),
                "created_at": sim.created_at,
                "message": sim.message,
            })
    return active

# HISTORY of all simulations
@router.get("/history-dupe", include_in_schema=False)
async def _unused_history_route(current_user: User = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")

    from simulation.service import SIMULATION_RESULTS_STORE
    history = []
    for sim_id, sim in SIMULATION_RESULTS_STORE.items():
        history.append({
            "simulation_id": sim_id,
            "status": sim.status,
            "user": getattr(sim, "username", "n/a"),
            "file_name": getattr(sim, "file_name", None),
            "cached": sim.status == "completed",
            "created_at": sim.created_at,
            "updated_at": sim.updated_at,
        })

    history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return history

@router.delete("/job/{job_id}", status_code=200)
async def delete_simulation_job(
    job_id: str = Path(..., title="The job ID of the simulation job to delete"),
    current_user: User = Depends(get_current_active_user)
):
    """Delete an entire simulation job by job_id (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")
    
    logger.info(f"DELETE_JOB: Received delete request for job_id: {job_id}")
    try:
        # Get the history data to find related simulations for this job
        history = await list_simulation_history_admin2(current_user)
        
        # Find the job entry
        target_job = None
        for job_entry in history:
            if job_entry.get("job_id") == job_id:
                target_job = job_entry
                break
        
        if not target_job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        related_simulations = target_job.get("related_simulations", [])
        if not related_simulations:
            raise HTTPException(status_code=404, detail=f"No simulations found for job {job_id}")
        
        logger.info(f"DELETE_JOB: Found {len(related_simulations)} simulations in job {job_id}: {related_simulations}")
        
        # Use the existing delete logic for the first simulation, which will delete all related ones
        first_simulation_id = related_simulations[0]
        delete_result = await delete_simulation(first_simulation_id, current_user)
        
        # Update the response to indicate this was a job deletion
        delete_result["job_id"] = job_id
        delete_result["message"] = f"Deleted job '{job_id}' containing {len(related_simulations)} simulation(s)"
        
        return delete_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}") 