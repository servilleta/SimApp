"""
🏢 ENTERPRISE SIMULATION SERVICE
Replaces the global SIMULATION_RESULTS_STORE with user-isolated database service.

This is the CRITICAL security fix for multi-user deployment.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from fastapi import HTTPException, Depends

from database import get_db
from models import SimulationResult, User as UserModel
from simulation.schemas import SimulationRequest, SimulationResponse
from auth.auth0_dependencies import get_current_active_auth0_user

logger = logging.getLogger(__name__)

class EnterpriseSimulationService:
    """
    🏢 Enterprise-grade simulation service with complete user data isolation.
    
    This service ensures that:
    - Users can only access their own simulations
    - All operations are tenant-aware
    - Complete audit trail for compliance
    - Zero cross-user data contamination
    """
    
    def __init__(self):
        self.audit_logger = EnterpriseAuditLogger()
    
    async def get_user_simulation(
        self, 
        user_id: int, 
        simulation_id: str, 
        db: Session
    ) -> Optional[SimulationResponse]:
        """
        Get a specific simulation for a user with complete isolation.
        
        🔒 SECURITY: Only returns simulations owned by the requesting user.
        """
        try:
            # Query with user isolation - CRITICAL for security
            simulation = db.query(SimulationResult).filter(
                and_(
                    SimulationResult.user_id == user_id,
                    SimulationResult.simulation_id == simulation_id
                )
            ).first()
            
            if not simulation:
                await self.audit_logger.log_access_attempt(
                    user_id=user_id,
                    simulation_id=simulation_id,
                    action="simulation_access_denied",
                    reason="simulation_not_found_or_access_denied"
                )
                return None
            
            # Log successful access
            await self.audit_logger.log_access_attempt(
                user_id=user_id,
                simulation_id=simulation_id,
                action="simulation_accessed",
                reason="authorized_access"
            )
            
            # Convert database model to response format
            return self._convert_to_simulation_response(simulation)
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE] Failed to get simulation {simulation_id} for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=simulation_id,
                error=str(e),
                action="get_user_simulation"
            )
            raise HTTPException(status_code=500, detail="Failed to retrieve simulation")
    
    async def create_user_simulation(
        self, 
        user_id: int, 
        request: SimulationRequest, 
        db: Session
    ) -> SimulationResponse:
        """
        Create a new simulation with automatic user association.
        
        🔒 SECURITY: Automatically associates simulation with authenticated user.
        """
        try:
            # Generate unique simulation ID
            simulation_id = request.simulation_id or str(uuid.uuid4())
            
            # Create database record with user isolation
            simulation_record = SimulationResult(
                simulation_id=simulation_id,  # Use simulation_id field name
                user_id=user_id,  # 🔒 CRITICAL: User association
                file_id=request.file_id,
                original_filename=request.original_filename,
                engine_type=request.engine_type,
                target_cell=", ".join(request.target_cells) if request.target_cells else None,
                variables_config=request.variables,
                constants_config=request.constants,
                iterations_requested=request.iterations,
                status="pending",
                message="Simulation has been queued for processing."
            )
            
            db.add(simulation_record)
            db.commit()
            db.refresh(simulation_record)
            
            # Log simulation creation
            await self.audit_logger.log_simulation_created(
                user_id=user_id,
                simulation_id=simulation_id,
                request_details={
                    "engine_type": request.engine_type,
                    "iterations": request.iterations,
                    "target_cells": request.target_cells
                }
            )
            
            logger.info(f"✅ [ENTERPRISE] Created simulation {simulation_id} for user {user_id}")
            
            return self._convert_to_simulation_response(simulation_record)
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE] Failed to create simulation for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=simulation_id,
                error=str(e),
                action="create_user_simulation"
            )
            raise HTTPException(status_code=500, detail="Failed to create simulation")
    
    async def update_simulation_status(
        self, 
        user_id: int, 
        simulation_id: str, 
        status: str, 
        message: str = None,
        results: Dict[str, Any] = None,
        db: Session = None
    ) -> bool:
        """
        Update simulation status with user verification.
        
        🔒 SECURITY: Verifies user ownership before allowing updates.
        """
        try:
            # Verify user ownership before update
            simulation = db.query(SimulationResult).filter(
                and_(
                    SimulationResult.user_id == user_id,
                    SimulationResult.simulation_id == simulation_id
                )
            ).first()
            
            if not simulation:
                await self.audit_logger.log_access_attempt(
                    user_id=user_id,
                    simulation_id=simulation_id,
                    action="simulation_update_denied",
                    reason="simulation_not_found_or_access_denied"
                )
                return False
            
            # Update simulation
            simulation.status = status
            simulation.updated_at = datetime.now(timezone.utc)
            
            if message:
                simulation.message = message
            
            if results:
                simulation.results = results
            
            db.commit()
            
            # Log successful update
            await self.audit_logger.log_simulation_updated(
                user_id=user_id,
                simulation_id=simulation_id,
                old_status=simulation.status,
                new_status=status
            )
            
            logger.info(f"✅ [ENTERPRISE] Updated simulation {simulation_id} status to {status} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE] Failed to update simulation {simulation_id} for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=simulation_id,
                error=str(e),
                action="update_simulation_status"
            )
            return False
    
    async def get_user_simulations(
        self, 
        user_id: int, 
        db: Session,
        limit: int = 50,
        offset: int = 0,
        status_filter: str = None
    ) -> List[SimulationResponse]:
        """
        Get all simulations for a user with pagination and filtering.
        
        🔒 SECURITY: Only returns simulations owned by the requesting user.
        """
        try:
            query = db.query(SimulationResult).filter(
                SimulationResult.user_id == user_id
            )
            
            # Apply status filter if provided
            if status_filter:
                query = query.filter(SimulationResult.status == status_filter)
            
            # Apply pagination
            simulations = query.order_by(
                SimulationResult.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            # Convert to response format
            simulation_responses = [
                self._convert_to_simulation_response(sim) 
                for sim in simulations
            ]
            
            # Log access
            await self.audit_logger.log_bulk_access(
                user_id=user_id,
                action="get_user_simulations",
                count=len(simulation_responses)
            )
            
            logger.info(f"✅ [ENTERPRISE] Retrieved {len(simulation_responses)} simulations for user {user_id}")
            return simulation_responses
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE] Failed to get simulations for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=None,
                error=str(e),
                action="get_user_simulations"
            )
            raise HTTPException(status_code=500, detail="Failed to retrieve simulations")
    
    async def delete_user_simulation(
        self, 
        user_id: int, 
        simulation_id: str, 
        db: Session
    ) -> bool:
        """
        Delete a simulation with user verification.
        
        🔒 SECURITY: Verifies user ownership before allowing deletion.
        """
        try:
            # Verify user ownership before deletion
            simulation = db.query(SimulationResult).filter(
                and_(
                    SimulationResult.user_id == user_id,
                    SimulationResult.simulation_id == simulation_id
                )
            ).first()
            
            if not simulation:
                await self.audit_logger.log_access_attempt(
                    user_id=user_id,
                    simulation_id=simulation_id,
                    action="simulation_deletion_denied",
                    reason="simulation_not_found_or_access_denied"
                )
                return False
            
            # Store simulation info for audit before deletion
            simulation_info = {
                "id": simulation.id,
                "status": simulation.status,
                "created_at": simulation.created_at.isoformat(),
                "engine_type": simulation.engine_type
            }
            
            # Delete simulation
            db.delete(simulation)
            db.commit()
            
            # Log successful deletion
            await self.audit_logger.log_simulation_deleted(
                user_id=user_id,
                simulation_id=simulation_id,
                simulation_info=simulation_info
            )
            
            logger.info(f"✅ [ENTERPRISE] Deleted simulation {simulation_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [ENTERPRISE] Failed to delete simulation {simulation_id} for user {user_id}: {e}")
            await self.audit_logger.log_error(
                user_id=user_id,
                simulation_id=simulation_id,
                error=str(e),
                action="delete_user_simulation"
            )
            return False
    
    def _convert_to_simulation_response(self, simulation: SimulationResult) -> SimulationResponse:
        """Convert database model to API response format."""
        # Build results dictionary if simulation is completed
        results = None
        if simulation.status == "completed" and simulation.mean is not None:
            results = {
                "mean": simulation.mean,
                "median": simulation.median,
                "std": simulation.std_dev,
                "min": simulation.min_value,
                "max": simulation.max_value,
                "percentiles": simulation.percentiles,
                "histogram": simulation.histogram,
                "iterations": simulation.iterations_run,
                "sensitivity_analysis": simulation.sensitivity_analysis,
                "errors": simulation.errors
            }
        
        return SimulationResponse(
            simulation_id=simulation.simulation_id,
            status=simulation.status,
            message=simulation.message,
            created_at=simulation.created_at.isoformat() if simulation.created_at else None,
            updated_at=simulation.updated_at.isoformat() if simulation.updated_at else None,
            original_filename=simulation.original_filename,
            engine_type=simulation.engine_type,
            target_name=simulation.target_cell,
            user=f"user_{simulation.user_id}",  # Don't expose actual user info
            results=results,
            multi_target_result=simulation.multi_target_result,
            progress_percentage=0  # Will need to be calculated from progress store
        )

class EnterpriseAuditLogger:
    """
    🔍 Enterprise audit logging for compliance (SOC 2, GDPR, etc.)
    """
    
    def __init__(self):
        self.audit_log = logging.getLogger("enterprise.audit")
    
    async def log_access_attempt(self, user_id: int, simulation_id: str, action: str, reason: str):
        """Log all access attempts for security auditing."""
        self.audit_log.info({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "simulation_id": simulation_id,
            "action": action,
            "reason": reason,
            "event_type": "access_attempt"
        })
    
    async def log_simulation_created(self, user_id: int, simulation_id: str, request_details: Dict):
        """Log simulation creation for audit trail."""
        self.audit_log.info({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "simulation_id": simulation_id,
            "action": "simulation_created",
            "request_details": request_details,
            "event_type": "data_creation"
        })
    
    async def log_simulation_updated(self, user_id: int, simulation_id: str, old_status: str, new_status: str):
        """Log simulation status updates."""
        self.audit_log.info({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "simulation_id": simulation_id,
            "action": "simulation_updated",
            "old_status": old_status,
            "new_status": new_status,
            "event_type": "data_modification"
        })
    
    async def log_simulation_deleted(self, user_id: int, simulation_id: str, simulation_info: Dict):
        """Log simulation deletion for compliance."""
        self.audit_log.info({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "simulation_id": simulation_id,
            "action": "simulation_deleted",
            "simulation_info": simulation_info,
            "event_type": "data_deletion"
        })
    
    async def log_bulk_access(self, user_id: int, action: str, count: int):
        """Log bulk data access operations."""
        self.audit_log.info({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "action": action,
            "record_count": count,
            "event_type": "bulk_access"
        })
    
    async def log_error(self, user_id: int, simulation_id: str, error: str, action: str):
        """Log errors for debugging and security monitoring."""
        self.audit_log.error({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "simulation_id": simulation_id,
            "action": action,
            "error": error,
            "event_type": "error"
        })

# Global enterprise service instance
enterprise_simulation_service = EnterpriseSimulationService()

# 🔄 MIGRATION COMPATIBILITY LAYER
# This allows gradual migration from the old global store

class LegacyCompatibilityLayer:
    """
    Temporary compatibility layer to support gradual migration from SIMULATION_RESULTS_STORE.
    This will be removed after complete migration.
    """
    
    def __init__(self):
        self.enterprise_service = enterprise_simulation_service
        logger.warning("🚨 [MIGRATION] Using legacy compatibility layer - should be removed after migration")
    
    async def get_simulation_with_fallback(
        self, 
        simulation_id: str, 
        user_id: int = None, 
        db: Session = None
    ) -> Optional[SimulationResponse]:
        """
        Get simulation with fallback to legacy store during migration.
        """
        # Try enterprise service first (new way)
        if user_id and db:
            result = await self.enterprise_service.get_user_simulation(user_id, simulation_id, db)
            if result:
                return result
        
        # Fallback to legacy store (old way) - REMOVE AFTER MIGRATION
        from simulation.service import SIMULATION_RESULTS_STORE
        if simulation_id in SIMULATION_RESULTS_STORE:
            logger.warning(f"🚨 [MIGRATION] Using legacy store for simulation {simulation_id}")
            return SIMULATION_RESULTS_STORE[simulation_id]
        
        return None

# Compatibility instance for gradual migration
legacy_compatibility = LegacyCompatibilityLayer()
