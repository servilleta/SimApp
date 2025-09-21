"""
Durable Logging Persistence Module

This module handles persisting completed simulation runs to the database
for reliable admin reporting and historical analysis.

Implements the durable logging enhancement from logbug.txt:
- Keeps real-time Redis/WebSocket progress flow untouched
- Persists single summary row per completed simulation
- Enables reliable admin reporting without hot-path overhead
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database import get_db
from models import SimulationResult as SimulationResultModel, User
from shared.persistent_excel_storage import move_simulation_file_to_persistent_storage
from shared.progress_store import get_progress

logger = logging.getLogger(__name__)

# Environment flag to enable/disable persistence (rollback strategy)
ENABLE_RUN_PERSISTENCE = os.getenv('ENABLE_RUN_PERSISTENCE', 'true').lower() == 'true'


async def persist_simulation_run(summary: Dict[str, Any]) -> bool:
    """
    Persist a completed simulation run to the database.
    
    Args:
        summary: Dictionary containing simulation summary data
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not ENABLE_RUN_PERSISTENCE:
        logger.debug("Run persistence disabled via ENABLE_RUN_PERSISTENCE=false")
        return False
        
    try:
        # Get database session
        db_gen = get_db()
        db: Session = next(db_gen)
        
        try:
            # CRITICAL FIX: Get user by identifier (username or email)
            user_identifier = summary.get('user_identifier')
            user = None
            
            if user_identifier and user_identifier != 'unknown':
                # Try username first (most common in our progress data)
                user = db.query(User).filter(User.username == user_identifier).first()
                if not user:
                    # Fallback to email lookup
                    user = db.query(User).filter(User.email == user_identifier).first()
            
            if not user:
                # Smart fallback: try to find admin user or any user
                logger.warning(f"User not found for simulation {summary.get('simulation_id')}: {user_identifier}")
                
                # Try to find admin user as fallback
                user = db.query(User).filter(User.is_admin == True).first()
                if not user:
                    # Fallback to any user in the system
                    user = db.query(User).first()
                
                if user:
                    logger.info(f"Using fallback user {user.username} for simulation {summary.get('simulation_id')}")
                else:
                    logger.error(f"No users found in database for simulation {summary.get('simulation_id')}")
                    return False
            
            # Check if simulation already exists
            existing = db.query(SimulationResultModel).filter(
                SimulationResultModel.simulation_id == summary['simulation_id']
            ).first()
            
            if existing:
                # Update existing record
                for key, value in summary.items():
                    if key in ['simulation_id']:
                        continue  # Skip primary key
                    if key == 'user_identifier':
                        continue  # Skip, we use user_id
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                
                existing.user_id = user.id
                existing.updated_at = datetime.now(timezone.utc)
                logger.info(f"Updated existing simulation record: {summary['simulation_id']}")
                
            else:
                # Create new record
                sim_result = SimulationResultModel(
                    simulation_id=summary['simulation_id'],
                    user_id=user.id,
                    status=summary.get('status', 'completed'),
                    message=summary.get('message'),
                    original_filename=summary.get('original_filename'),
                    engine_type=summary.get('engine_type'),
                    target_name=summary.get('target_name'),
                    file_id=summary.get('file_id'),
                    iterations_requested=summary.get('iterations_requested'),
                    variables_config=summary.get('variables_config'),
                    constants_config=summary.get('constants_config'),
                    target_cell=summary.get('target_cell'),
                    mean=summary.get('mean'),
                    median=summary.get('median'),
                    std_dev=summary.get('std_dev'),
                    min_value=summary.get('min_value'),
                    max_value=summary.get('max_value'),
                    percentiles=summary.get('percentiles'),
                    histogram=summary.get('histogram'),
                    iterations_run=summary.get('iterations_run'),
                    sensitivity_analysis=summary.get('sensitivity_analysis'),
                    errors=summary.get('errors'),
                    multi_target_result=summary.get('multi_target_result'),  # ✅ CRITICAL: Save multi-target results
                    started_at=_parse_datetime(summary.get('started_at')),
                    completed_at=_parse_datetime(summary.get('completed_at')) or datetime.now(timezone.utc)
                )
                
                db.add(sim_result)
                logger.info(f"Created new simulation record: {summary['simulation_id']}")
            
            # Commit the transaction
            db.commit()
            logger.info(f"Successfully persisted simulation run: {summary['simulation_id']}")
            
            # 📁 EXCEL PERSISTENCE: Move Excel file to long-term storage for completed simulations
            file_id = summary.get('file_id')
            original_filename = summary.get('original_filename')
            
            if file_id and original_filename and summary.get('status') == 'completed':
                try:
                    persistent_path = move_simulation_file_to_persistent_storage(file_id, original_filename)
                    if persistent_path:
                        logger.info(f"📁 Excel file moved to persistent storage: {file_id} -> {persistent_path}")
                    else:
                        logger.warning(f"📁 Failed to move Excel file to persistent storage: {file_id}")
                except Exception as e:
                    logger.error(f"📁 Error moving Excel file to persistent storage for {file_id}: {e}")
            
            return True
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Database integrity error persisting simulation {summary.get('simulation_id')}: {e}")
            return False
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error persisting simulation {summary.get('simulation_id')}: {e}")
            return False
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to get database session for simulation {summary.get('simulation_id')}: {e}")
        return False


def build_simulation_summary(
    simulation_id: str,
    results: Optional[Any] = None,
    status: str = "completed",
    message: Optional[str] = None,
    engine_type: Optional[str] = None,
    iterations_requested: Optional[int] = None,
    variables_config: Optional[list] = None,
    constants_config: Optional[list] = None,
    target_cell: Optional[str] = None,
    started_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a simulation summary dictionary from simulation data.
    
    Args:
        simulation_id: Unique simulation identifier
        results: SimulationResult object with statistics
        status: Final status (completed/failed/cancelled)
        message: Status message
        engine_type: Engine used for simulation
        iterations_requested: Number of iterations requested
        variables_config: Monte Carlo variables configuration
        constants_config: Constants configuration
        target_cell: Target cell coordinate
        started_at: ISO timestamp when simulation started
        
    Returns:
        Dictionary containing simulation summary
    """
    try:
        # Get additional data from progress store
        progress_data = get_progress(simulation_id) or {}
        
        # Build base summary
        summary = {
            'simulation_id': simulation_id,
            'status': status,
            'message': message,
            'engine_type': engine_type or progress_data.get('engine_type') or (progress_data.get('engineInfo', {}).get('engine_type')),
            'iterations_requested': iterations_requested,
            'variables_config': variables_config,
            'constants_config': constants_config,
            'target_cell': target_cell or progress_data.get('target_cell'),
            'started_at': started_at or progress_data.get('start_time'),
            'completed_at': datetime.now(timezone.utc).isoformat(),
            
            # Extract user and filename from progress data
            # CRITICAL FIX: Handle both username and email properly
            'user_identifier': progress_data.get('user', 'unknown'),
            'original_filename': progress_data.get('original_filename', 'Unknown'),
            'file_id': progress_data.get('file_id'),
            'target_name': progress_data.get('target_name') or target_cell,
        }
        
        # Add results data if available
        if results and hasattr(results, 'mean'):
            summary.update({
                'mean': results.mean,
                'median': results.median,
                'std_dev': results.std_dev,
                'min_value': results.min_value,
                'max_value': results.max_value,
                'percentiles': results.percentiles,
                'histogram': results.histogram,
                'iterations_run': results.iterations_run,
                'sensitivity_analysis': results.sensitivity_analysis,
                'errors': results.errors
            })
        
        # Calculate duration if we have start time
        if summary.get('started_at') and summary.get('completed_at'):
            try:
                start_dt = _parse_datetime(summary['started_at'])
                end_dt = _parse_datetime(summary['completed_at'])
                if start_dt and end_dt:
                    duration = (end_dt - start_dt).total_seconds()
                    summary['duration_sec'] = int(duration)
            except Exception as e:
                logger.debug(f"Could not calculate duration for {simulation_id}: {e}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error building simulation summary for {simulation_id}: {e}")
        return {
            'simulation_id': simulation_id,
            'status': status,
            'message': message or f"Error building summary: {str(e)}",
            'user_identifier': 'unknown',
            'original_filename': 'Unknown',
            'completed_at': datetime.now(timezone.utc).isoformat()
        }


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string to datetime object."""
    if not dt_str:
        return None
    
    try:
        # Handle both with and without timezone info
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1] + '+00:00'
        
        if '+' in dt_str or dt_str.endswith('Z'):
            return datetime.fromisoformat(dt_str)
        else:
            # Assume UTC if no timezone info
            return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
            
    except Exception as e:
        logger.debug(f"Could not parse datetime '{dt_str}': {e}")
        return None


async def reconcile_missing_simulations():
    """
    Daily reconciliation job to find completed simulations in Redis
    that were not persisted to the database.
    
    This is the safety net mentioned in the logbug.txt plan.
    """
    if not ENABLE_RUN_PERSISTENCE:
        logger.debug("Reconciliation skipped - persistence disabled")
        return
    
    try:
        from shared.progress_store import get_all_progress_keys
        
        # Get all Redis simulation keys
        redis_keys = get_all_progress_keys()
        simulation_ids = []
        
        for key in redis_keys:
            if key.startswith('simulation:progress:'):
                sim_id = key.replace('simulation:progress:', '')
                simulation_ids.append(sim_id)
        
        if not simulation_ids:
            logger.info("No simulations found in Redis for reconciliation")
            return
        
        # Get database session
        db_gen = get_db()
        db: Session = next(db_gen)
        
        try:
            # Find simulations that exist in Redis but not in DB
            existing_ids = db.query(SimulationResultModel.simulation_id).filter(
                SimulationResultModel.simulation_id.in_(simulation_ids)
            ).all()
            existing_ids = [row[0] for row in existing_ids]
            
            missing_ids = [sim_id for sim_id in simulation_ids if sim_id not in existing_ids]
            
            if not missing_ids:
                logger.info("No missing simulations found during reconciliation")
                return
            
            logger.info(f"Found {len(missing_ids)} missing simulations, attempting to reconcile")
            
            # Process each missing simulation
            reconciled = 0
            for sim_id in missing_ids:
                try:
                    progress_data = get_progress(sim_id)
                    if not progress_data:
                        continue
                    
                    # Only reconcile completed simulations
                    if progress_data.get('status') not in ['completed', 'failed', 'cancelled']:
                        continue
                    
                    # Build summary from progress data
                    summary = build_simulation_summary(
                        simulation_id=sim_id,
                        status=progress_data.get('status', 'completed'),
                        message=progress_data.get('message'),
                        engine_type=progress_data.get('engine_type') or (progress_data.get('engineInfo', {}).get('engine_type')),
                        started_at=progress_data.get('start_time')
                    )
                    
                    # Persist the simulation
                    if await persist_simulation_run(summary):
                        reconciled += 1
                        logger.info(f"Reconciled simulation: {sim_id}")
                    
                except Exception as e:
                    logger.error(f"Error reconciling simulation {sim_id}: {e}")
                    continue
            
            logger.info(f"Reconciliation complete: {reconciled}/{len(missing_ids)} simulations reconciled")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during reconciliation: {e}") 