"""
Database service for simulation results

This service replaces the in-memory SIMULATION_RESULTS_STORE with proper
database persistence using PostgreSQL/SQLite.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from fastapi import HTTPException
import logging

from database import get_db
from models import SimulationResult as SimulationResultModel, UserUsageMetrics, UserSubscription
from simulation.schemas import SimulationResponse, SimulationResult, VariableConfig, ConstantConfig

logger = logging.getLogger(__name__)


class SimulationDatabaseService:
    """
    Database service for simulation results management
    Replaces in-memory SIMULATION_RESULTS_STORE with persistent storage
    """

    def __init__(self):
        self.logger = logger

    def create_simulation(
        self, 
        simulation_id: str, 
        user_id: int,
        request_data: Dict[str, Any]
    ) -> SimulationResultModel:
        """Create a new simulation record in the database"""
        db = next(get_db())
        try:
            # Extract data from request
            variables_config = None
            constants_config = None
            
            if 'mc_inputs' in request_data:
                variables_config = [var.dict() if hasattr(var, 'dict') else var for var in request_data['mc_inputs']]
            if 'constants' in request_data:
                constants_config = [const.dict() if hasattr(const, 'dict') else const for const in request_data['constants']]

            simulation = SimulationResultModel(
                simulation_id=simulation_id,
                user_id=user_id,
                status="pending",
                message="Simulation has been queued.",
                original_filename=request_data.get('original_filename'),
                engine_type=request_data.get('engine_type'),
                target_name=request_data.get('result_cell_coordinate'),
                file_id=request_data.get('file_id'),
                iterations_requested=request_data.get('iterations'),
                variables_config=variables_config,
                constants_config=constants_config,
                target_cell=request_data.get('result_cell_coordinate'),
                created_at=datetime.now(timezone.utc)
            )
            
            db.add(simulation)
            db.commit()
            db.refresh(simulation)
            
            self.logger.info(f"Created simulation record: {simulation_id}")
            return simulation
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error creating simulation {simulation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create simulation: {str(e)}")
        finally:
            db.close()

    def get_simulation(self, simulation_id: str) -> Optional[SimulationResultModel]:
        """Get a simulation by ID"""
        db = next(get_db())
        try:
            simulation = db.query(SimulationResultModel).filter(
                SimulationResultModel.simulation_id == simulation_id
            ).first()
            return simulation
        finally:
            db.close()

    def update_simulation_status(
        self, 
        simulation_id: str, 
        status: str, 
        message: Optional[str] = None,
        progress_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update simulation status and progress"""
        db = next(get_db())
        try:
            simulation = db.query(SimulationResultModel).filter(
                SimulationResultModel.simulation_id == simulation_id
            ).first()
            
            if not simulation:
                self.logger.warning(f"Simulation not found for status update: {simulation_id}")
                return False
            
            # Update basic fields
            simulation.status = status
            if message:
                simulation.message = message
            simulation.updated_at = datetime.now(timezone.utc)
            
            # Update timestamps based on status
            if status == "running" and not simulation.started_at:
                simulation.started_at = datetime.now(timezone.utc)
            elif status in ["completed", "failed", "cancelled"] and not simulation.completed_at:
                simulation.completed_at = datetime.now(timezone.utc)
            
            db.commit()
            self.logger.debug(f"Updated simulation {simulation_id} status to {status}")
            return True
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error updating simulation {simulation_id}: {e}")
            return False
        finally:
            db.close()

    def save_simulation_results(
        self, 
        simulation_id: str, 
        results: SimulationResult
    ) -> bool:
        """Save simulation results to database"""
        db = next(get_db())
        try:
            simulation = db.query(SimulationResultModel).filter(
                SimulationResultModel.simulation_id == simulation_id
            ).first()
            
            if not simulation:
                self.logger.error(f"Simulation not found for results save: {simulation_id}")
                return False
            
            # Update results fields
            simulation.status = "completed"
            simulation.message = "Simulation completed successfully."
            simulation.mean = results.mean
            simulation.median = results.median
            simulation.std_dev = results.std_dev
            simulation.min_value = results.min_value
            simulation.max_value = results.max_value
            simulation.percentiles = results.percentiles
            simulation.histogram = results.histogram
            simulation.iterations_run = results.iterations_run
            simulation.sensitivity_analysis = results.sensitivity_analysis
            simulation.errors = results.errors
            simulation.completed_at = datetime.now(timezone.utc)
            simulation.updated_at = datetime.now(timezone.utc)
            
            db.commit()
            self.logger.info(f"Saved results for simulation: {simulation_id}")
            return True
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error saving results for simulation {simulation_id}: {e}")
            return False
        finally:
            db.close()

    def get_simulation_response(self, simulation_id: str) -> Optional[SimulationResponse]:
        """Get simulation as SimulationResponse object"""
        simulation = self.get_simulation(simulation_id)
        if simulation:
            return simulation.to_simulation_response()
        return None

    def get_user_simulations(
        self, 
        user_id: int, 
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SimulationResultModel]:
        """Get simulations for a specific user"""
        db = next(get_db())
        try:
            query = db.query(SimulationResultModel).filter(
                SimulationResultModel.user_id == user_id
            )
            
            if status:
                query = query.filter(SimulationResultModel.status == status)
            
            simulations = query.order_by(
                desc(SimulationResultModel.created_at)
            ).offset(offset).limit(limit).all()
            
            return simulations
        finally:
            db.close()

    def get_running_simulations(self, user_id: Optional[int] = None) -> List[SimulationResultModel]:
        """Get all running simulations, optionally filtered by user"""
        db = next(get_db())
        try:
            query = db.query(SimulationResultModel).filter(
                SimulationResultModel.status.in_(["pending", "running"])
            )
            
            if user_id:
                query = query.filter(SimulationResultModel.user_id == user_id)
            
            simulations = query.order_by(
                desc(SimulationResultModel.created_at)
            ).all()
            
            return simulations
        finally:
            db.close()

    def delete_simulation(self, simulation_id: str, user_id: Optional[int] = None) -> bool:
        """Delete a simulation (with optional user ownership check)"""
        db = next(get_db())
        try:
            query = db.query(SimulationResultModel).filter(
                SimulationResultModel.simulation_id == simulation_id
            )
            
            if user_id:
                query = query.filter(SimulationResultModel.user_id == user_id)
            
            simulation = query.first()
            if not simulation:
                return False
            
            db.delete(simulation)
            db.commit()
            self.logger.info(f"Deleted simulation: {simulation_id}")
            return True
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error deleting simulation {simulation_id}: {e}")
            return False
        finally:
            db.close()

    def cleanup_old_simulations(self, retention_days: int = 30) -> Dict[str, int]:
        """Clean up old simulation results based on retention policy"""
        db = next(get_db())
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timezone.timedelta(days=retention_days)
            
            # Count simulations to be deleted
            count_query = db.query(SimulationResultModel).filter(
                SimulationResultModel.created_at < cutoff_date
            )
            total_to_delete = count_query.count()
            
            # Delete old simulations
            deleted_count = count_query.delete()
            db.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} old simulations (older than {retention_days} days)")
            
            return {
                "deleted_count": deleted_count,
                "retention_days": retention_days,
                "cutoff_date": cutoff_date.isoformat()
            }
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error cleaning up old simulations: {e}")
            return {"deleted_count": 0, "error": str(e)}
        finally:
            db.close()

    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get overall simulation statistics"""
        db = next(get_db())
        try:
            # Get counts by status
            status_counts = db.query(
                SimulationResultModel.status,
                func.count(SimulationResultModel.id).label('count')
            ).group_by(SimulationResultModel.status).all()
            
            # Get total count
            total_count = db.query(SimulationResultModel).count()
            
            # Get recent activity (last 24 hours)
            recent_cutoff = datetime.now(timezone.utc) - timezone.timedelta(hours=24)
            recent_count = db.query(SimulationResultModel).filter(
                SimulationResultModel.created_at >= recent_cutoff
            ).count()
            
            return {
                "total_simulations": total_count,
                "recent_simulations_24h": recent_count,
                "status_breakdown": {status: count for status, count in status_counts},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        finally:
            db.close()

    def mark_simulation_failed(self, simulation_id: str, error_message: str) -> bool:
        """Mark a simulation as failed with error message"""
        return self.update_simulation_status(
            simulation_id=simulation_id,
            status="failed",
            message=error_message
        )

    def mark_simulation_cancelled(self, simulation_id: str) -> bool:
        """Mark a simulation as cancelled"""
        return self.update_simulation_status(
            simulation_id=simulation_id,
            status="cancelled",
            message="Simulation was cancelled by user request."
        )

    def get_user_current_usage(self, user_id: int) -> Dict[str, int]:
        """Get current usage statistics for a user (current month)"""
        db = next(get_db())
        try:
            # Get current month start
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Count simulations this month
            simulations_count = db.query(SimulationResultModel).filter(
                and_(
                    SimulationResultModel.user_id == user_id,
                    SimulationResultModel.created_at >= month_start
                )
            ).count()
            
            # Count running simulations
            running_count = db.query(SimulationResultModel).filter(
                and_(
                    SimulationResultModel.user_id == user_id,
                    SimulationResultModel.status.in_(["pending", "running"])
                )
            ).count()
            
            # Sum total iterations
            iterations_result = db.query(
                func.sum(SimulationResultModel.iterations_requested)
            ).filter(
                and_(
                    SimulationResultModel.user_id == user_id,
                    SimulationResultModel.created_at >= month_start,
                    SimulationResultModel.iterations_requested.isnot(None)
                )
            ).scalar()
            
            total_iterations = iterations_result or 0
            
            return {
                "simulations_this_month": simulations_count,
                "running_simulations": running_count,
                "total_iterations_this_month": total_iterations,
                "period_start": month_start.isoformat()
            }
        finally:
            db.close()


# Global instance
simulation_db_service = SimulationDatabaseService() 