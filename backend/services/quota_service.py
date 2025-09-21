"""
Quota Enforcement Service for Monte Carlo Platform

Manages and enforces subscription-based quotas and limits for users
based on their subscription tier and current usage.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from models import User as UserModel, UserSubscription, UserUsageMetrics
from database import get_db

logger = logging.getLogger(__name__)

class QuotaService:
    """
    Service for enforcing subscription quotas and limits
    """
    
    @staticmethod
    def get_user_subscription_with_limits(db: Session, user_id: int) -> Tuple[UserSubscription, Dict[str, Any]]:
        """
        Get user subscription and their current limits
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Tuple of (UserSubscription, limits_dict)
        """
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == user_id
        ).first()
        
        if not user_subscription:
            # Create default free subscription
            user_subscription = UserSubscription(
                user_id=user_id,
                tier="free",
                status="active"
            )
            db.add(user_subscription)
            db.commit()
            db.refresh(user_subscription)
        
        limits = user_subscription.get_limits()
        return user_subscription, limits
    
    @staticmethod
    def get_current_usage(db: Session, user_id: int) -> UserUsageMetrics:
        """
        Get current month's usage metrics for a user
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            UserUsageMetrics instance (may be new)
        """
        # Get current month boundaries
        now = datetime.now(timezone.utc)
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month_start = current_month_start + relativedelta(months=1)
        
        usage_metrics = db.query(UserUsageMetrics).filter(
            UserUsageMetrics.user_id == user_id,
            UserUsageMetrics.period_start >= current_month_start,
            UserUsageMetrics.period_type == "monthly"
        ).first()
        
        if not usage_metrics:
            # Create new usage metrics for current month
            usage_metrics = UserUsageMetrics(
                user_id=user_id,
                period_start=current_month_start,
                period_end=next_month_start,
                period_type="monthly"
            )
            db.add(usage_metrics)
            db.commit()
            db.refresh(usage_metrics)
        
        return usage_metrics
    
    @staticmethod
    def check_iteration_limit(db: Session, user_id: int, requested_iterations: int) -> Tuple[bool, str]:
        """
        Check if user can run a simulation with the requested number of iterations
        
        Args:
            db: Database session
            user_id: User ID
            requested_iterations: Number of iterations requested
            
        Returns:
            Tuple of (allowed: bool, message: str)
        """
        try:
            subscription, limits = QuotaService.get_user_subscription_with_limits(db, user_id)
            max_iterations = limits.get("max_iterations", 5000)
            
            # Check if unlimited (-1)
            if max_iterations == -1:
                return True, "Unlimited iterations allowed"
            
            # Check if request exceeds per-simulation limit
            if requested_iterations > max_iterations:
                return False, f"Requested {requested_iterations:,} iterations exceeds your plan limit of {max_iterations:,} per simulation. Upgrade your plan for higher limits."
            
            return True, "Within iteration limits"
            
        except Exception as e:
            logger.error(f"Error checking iteration limit for user {user_id}: {str(e)}")
            return False, "Error checking iteration limits"
    
    @staticmethod
    def check_file_size_limit(db: Session, user_id: int, file_size_mb: float) -> Tuple[bool, str]:
        """
        Check if user can upload a file of the given size
        
        Args:
            db: Database session
            user_id: User ID
            file_size_mb: File size in MB
            
        Returns:
            Tuple of (allowed: bool, message: str)
        """
        try:
            subscription, limits = QuotaService.get_user_subscription_with_limits(db, user_id)
            max_file_size_mb = limits.get("file_size_mb", 10)
            
            # Check if unlimited (-1)
            if max_file_size_mb == -1:
                return True, "Unlimited file size allowed"
            
            if file_size_mb > max_file_size_mb:
                return False, f"File size {file_size_mb:.1f}MB exceeds your plan limit of {max_file_size_mb}MB. Upgrade your plan for larger file limits."
            
            return True, "Within file size limits"
            
        except Exception as e:
            logger.error(f"Error checking file size limit for user {user_id}: {str(e)}")
            return False, "Error checking file size limits"
    
    @staticmethod
    def check_concurrent_simulation_limit(db: Session, user_id: int) -> Tuple[bool, str]:
        """
        Check if user can start another concurrent simulation
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Tuple of (allowed: bool, message: str)
        """
        try:
            subscription, limits = QuotaService.get_user_subscription_with_limits(db, user_id)
            max_concurrent = limits.get("concurrent_simulations", 1)
            
            # Check if unlimited (-1)
            if max_concurrent == -1:
                return True, "Unlimited concurrent simulations allowed"
            
            # Count currently running simulations
            from models import SimulationResult
            running_count = db.query(SimulationResult).filter(
                SimulationResult.user_id == user_id,
                SimulationResult.status.in_(["pending", "running"])
            ).count()
            
            if running_count >= max_concurrent:
                return False, f"You have {running_count} simulations running. Your plan allows {max_concurrent} concurrent simulations. Wait for current simulations to complete or upgrade your plan."
            
            return True, f"Can run {max_concurrent - running_count} more concurrent simulation(s)"
            
        except Exception as e:
            logger.error(f"Error checking concurrent simulation limit for user {user_id}: {str(e)}")
            return False, "Error checking concurrent simulation limits"
    
    @staticmethod
    def check_formula_limit(db: Session, user_id: int, formula_count: int) -> Tuple[bool, str]:
        """
        Check if user can process a file with the given number of formulas
        
        Args:
            db: Database session
            user_id: User ID
            formula_count: Number of formulas in the file
            
        Returns:
            Tuple of (allowed: bool, message: str)
        """
        try:
            subscription, limits = QuotaService.get_user_subscription_with_limits(db, user_id)
            max_formulas = limits.get("max_formulas", 1000)
            
            # Check if unlimited (very high number)
            if max_formulas >= 1000000:
                return True, "High formula count allowed"
            
            if formula_count > max_formulas:
                return False, f"File contains {formula_count:,} formulas, which exceeds your plan limit of {max_formulas:,}. Upgrade your plan for higher formula limits."
            
            return True, "Within formula limits"
            
        except Exception as e:
            logger.error(f"Error checking formula limit for user {user_id}: {str(e)}")
            return False, "Error checking formula limits"
    
    @staticmethod
    def check_project_storage_limit(db: Session, user_id: int) -> Tuple[bool, str]:
        """
        Check if user can store another project
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Tuple of (allowed: bool, message: str)
        """
        try:
            subscription, limits = QuotaService.get_user_subscription_with_limits(db, user_id)
            max_projects = limits.get("projects_stored", 3)
            
            # Check if unlimited (-1)
            if max_projects == -1:
                return True, "Unlimited project storage allowed"
            
            # Count stored projects (completed simulations)
            from models import SimulationResult
            stored_projects = db.query(SimulationResult).filter(
                SimulationResult.user_id == user_id,
                SimulationResult.status == "completed"
            ).count()
            
            if stored_projects >= max_projects:
                return False, f"You have {stored_projects} stored projects. Your plan allows {max_projects} stored projects. Delete old projects or upgrade your plan."
            
            return True, f"Can store {max_projects - stored_projects} more project(s)"
            
        except Exception as e:
            logger.error(f"Error checking project storage limit for user {user_id}: {str(e)}")
            return False, "Error checking project storage limits"
    
    @staticmethod
    def check_api_call_limit(db: Session, user_id: int) -> Tuple[bool, str]:
        """
        Check if user can make another API call this month
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Tuple of (allowed: bool, message: str)
        """
        try:
            subscription, limits = QuotaService.get_user_subscription_with_limits(db, user_id)
            max_api_calls = limits.get("api_calls_per_month", 0)
            
            # Free tier has no API access
            if max_api_calls == 0:
                return False, "API access not included in your plan. Upgrade to Professional or higher for API access."
            
            # Check if unlimited (-1)
            if max_api_calls == -1:
                return True, "Unlimited API calls allowed"
            
            # Get current usage
            usage = QuotaService.get_current_usage(db, user_id)
            
            if usage.api_calls >= max_api_calls:
                return False, f"You have used {usage.api_calls} of {max_api_calls} API calls this month. Your monthly limit has been reached."
            
            return True, f"Can make {max_api_calls - usage.api_calls} more API call(s) this month"
            
        except Exception as e:
            logger.error(f"Error checking API call limit for user {user_id}: {str(e)}")
            return False, "Error checking API call limits"
    
    @staticmethod
    def get_gpu_priority(db: Session, user_id: int) -> str:
        """
        Get GPU priority level for user
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            GPU priority level (low, standard, high, premium, dedicated)
        """
        try:
            subscription, limits = QuotaService.get_user_subscription_with_limits(db, user_id)
            return limits.get("gpu_priority", "low")
        except Exception as e:
            logger.error(f"Error getting GPU priority for user {user_id}: {str(e)}")
            return "low"
    
    @staticmethod
    def record_simulation_usage(
        db: Session, 
        user_id: int, 
        iterations_run: int,
        file_size_mb: float = 0,
        engine_used: str = "power"
    ):
        """
        Record usage when a simulation is completed
        
        Args:
            db: Database session
            user_id: User ID
            iterations_run: Number of iterations that were run
            file_size_mb: Size of the file processed
            engine_used: Engine that was used
        """
        try:
            usage = QuotaService.get_current_usage(db, user_id)
            
            # Update counters
            usage.simulations_run += 1
            usage.total_iterations += iterations_run
            usage.total_file_size_mb += file_size_mb
            
            # Update engines used (JSON array)
            if usage.engines_used is None:
                usage.engines_used = []
            
            if engine_used not in usage.engines_used:
                usage.engines_used.append(engine_used)
            
            db.commit()
            logger.info(f"Recorded simulation usage for user {user_id}: {iterations_run} iterations")
            
        except Exception as e:
            logger.error(f"Error recording simulation usage for user {user_id}: {str(e)}")
    
    @staticmethod
    def record_api_call(db: Session, user_id: int):
        """
        Record an API call for usage tracking
        
        Args:
            db: Database session
            user_id: User ID
        """
        try:
            usage = QuotaService.get_current_usage(db, user_id)
            usage.api_calls += 1
            db.commit()
            
        except Exception as e:
            logger.error(f"Error recording API call for user {user_id}: {str(e)}")
    
    @staticmethod
    def record_file_upload(db: Session, user_id: int, file_size_mb: float):
        """
        Record a file upload for usage tracking
        
        Args:
            db: Database session
            user_id: User ID
            file_size_mb: Size of uploaded file in MB
        """
        try:
            usage = QuotaService.get_current_usage(db, user_id)
            usage.files_uploaded += 1
            usage.total_file_size_mb += file_size_mb
            db.commit()
            
        except Exception as e:
            logger.error(f"Error recording file upload for user {user_id}: {str(e)}")
