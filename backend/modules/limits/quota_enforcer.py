"""
Quota Enforcement Service

Provides real-time quota checking and enforcement for user limits.
Integrates with simulation service to prevent quota violations.
"""

import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from database import get_db
from models import UserSubscription, UserUsageMetrics, SimulationResult as SimulationResultModel
from modules.limits.service import TIER_LIMITS
from services.trial_service import trial_service

logger = logging.getLogger(__name__)


class QuotaEnforcer:
    """Real-time quota enforcement for user limits"""
    
    def __init__(self):
        self.logger = logger
    
    async def check_simulation_quota(self, user_id: int) -> Tuple[bool, str, Dict]:
        """
        Check if user can start a new simulation
        
        Returns:
            - bool: Whether simulation is allowed
            - str: Reason if not allowed
            - Dict: Current usage and limit info
        """
        db = next(get_db())
        try:
            # Get user subscription and limits
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if not subscription:
                # Check if user is eligible for trial or create free subscription
                trial_check = trial_service.check_and_enforce_trial_limits(user_id, db)
                allowed, reason, trial_info = trial_check
                
                if not allowed and "expired" in reason:
                    return False, reason, {"trial_expired": True}
                
                # Refresh subscription after potential trial creation
                subscription = db.query(UserSubscription).filter(
                    UserSubscription.user_id == user_id
                ).first()
                
                if not subscription:
                    # Create default free subscription as fallback
                    subscription = UserSubscription(
                        user_id=user_id,
                        tier="free",
                        status="active"
                    )
                    db.add(subscription)
                    db.commit()
                    db.refresh(subscription)
            
            # Check if trial has expired
            if subscription.is_trial and subscription.is_trial_expired():
                subscription.expire_trial()
                db.commit()
                return False, "Your 7-day trial has expired. Please upgrade to continue.", {
                    "trial_expired": True,
                    "tier": "free"
                }
            
            limits = TIER_LIMITS.get(subscription.tier, TIER_LIMITS["free"])
            usage_info = self._get_current_usage(user_id, db)
            
            # Check monthly simulation limit
            monthly_limit = limits["simulations_per_month"]
            if monthly_limit > 0:  # Not unlimited
                monthly_usage = usage_info["simulations_this_month"]
                if monthly_usage >= monthly_limit:
                    return False, f"Monthly simulation limit reached ({monthly_limit}). Upgrade to continue.", {
                        "usage": usage_info,
                        "limits": limits,
                        "quota_exceeded": "monthly_simulations"
                    }
            
            # Check concurrent simulation limit
            concurrent_limit = limits["concurrent_simulations"]
            if concurrent_limit > 0:  # Not unlimited
                concurrent_usage = usage_info["running_simulations"]
                if concurrent_usage >= concurrent_limit:
                    return False, f"Concurrent simulation limit reached ({concurrent_limit}). Wait for simulations to complete.", {
                        "usage": usage_info,
                        "limits": limits,
                        "quota_exceeded": "concurrent_simulations"
                    }
            
            return True, "Simulation allowed", {
                "usage": usage_info,
                "limits": limits,
                "quota_status": "within_limits"
            }
            
        except Exception as e:
            self.logger.error(f"Error checking simulation quota for user {user_id}: {e}")
            return False, "Error checking quota limits", {}
        finally:
            db.close()
    
    async def check_iteration_quota(self, user_id: int, requested_iterations: int) -> Tuple[bool, str]:
        """
        Check if user can run simulation with requested iterations
        
        Args:
            user_id: User ID
            requested_iterations: Number of iterations requested
            
        Returns:
            - bool: Whether iterations are allowed
            - str: Reason if not allowed
        """
        db = next(get_db())
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if not subscription:
                subscription = UserSubscription(
                    user_id=user_id,
                    tier="free",
                    status="active"
                )
                db.add(subscription)
                db.commit()
                db.refresh(subscription)
            
            limits = TIER_LIMITS.get(subscription.tier, TIER_LIMITS["free"])
            max_iterations = limits["iterations_per_simulation"]
            
            if max_iterations > 0 and requested_iterations > max_iterations:
                return False, f"Iteration limit exceeded. Maximum {max_iterations} iterations allowed for {subscription.tier} tier."
            
            return True, "Iterations allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking iteration quota for user {user_id}: {e}")
            return False, "Error checking iteration limits"
        finally:
            db.close()
    
    async def check_file_size_quota(self, user_id: int, file_size_mb: float) -> Tuple[bool, str]:
        """
        Check if user can upload file of given size
        
        Args:
            user_id: User ID
            file_size_mb: File size in megabytes
            
        Returns:
            - bool: Whether file size is allowed
            - str: Reason if not allowed
        """
        db = next(get_db())
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if not subscription:
                subscription = UserSubscription(
                    user_id=user_id,
                    tier="free",
                    status="active"
                )
                db.add(subscription)
                db.commit()
                db.refresh(subscription)
            
            limits = TIER_LIMITS.get(subscription.tier, TIER_LIMITS["free"])
            max_file_size = limits["file_size_mb"]
            
            if max_file_size > 0 and file_size_mb > max_file_size:
                return False, f"File size limit exceeded. Maximum {max_file_size}MB allowed for {subscription.tier} tier."
            
            return True, "File size allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking file size quota for user {user_id}: {e}")
            return False, "Error checking file size limits"
        finally:
            db.close()
    
    async def check_engine_access(self, user_id: int, engine_type: str) -> Tuple[bool, str]:
        """
        Check if user has access to requested engine
        
        Args:
            user_id: User ID
            engine_type: Engine type (power, arrow, enhanced, gpu, super)
            
        Returns:
            - bool: Whether engine access is allowed
            - str: Reason if not allowed
        """
        db = next(get_db())
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if not subscription:
                subscription = UserSubscription(
                    user_id=user_id,
                    tier="free",
                    status="active"
                )
                db.add(subscription)
                db.commit()
                db.refresh(subscription)
            
            limits = TIER_LIMITS.get(subscription.tier, TIER_LIMITS["free"])
            allowed_engines = limits["engines"]
            
            if engine_type not in allowed_engines:
                return False, f"Engine '{engine_type}' not available for {subscription.tier} tier. Upgrade to access advanced engines."
            
            # Special check for GPU access
            if engine_type in ["gpu", "super"] and not limits["gpu_access"]:
                return False, f"GPU acceleration not available for {subscription.tier} tier. Upgrade to Pro or Enterprise."
            
            return True, "Engine access allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking engine access for user {user_id}: {e}")
            return False, "Error checking engine access"
        finally:
            db.close()
    
    async def record_simulation_start(self, user_id: int, simulation_id: str, iterations: int, engine_type: str) -> None:
        """Record that a simulation has started for usage tracking"""
        db = next(get_db())
        try:
            # Update usage metrics
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            month_end = (month_start.replace(month=month_start.month + 1) if month_start.month < 12 
                        else month_start.replace(year=month_start.year + 1, month=1)) - timezone.utc.localize(datetime(1970, 1, 1)).replace(tzinfo=None)
            
            metrics = db.query(UserUsageMetrics).filter(
                and_(
                    UserUsageMetrics.user_id == user_id,
                    UserUsageMetrics.period_start == month_start,
                    UserUsageMetrics.period_type == "monthly"
                )
            ).first()
            
            if not metrics:
                metrics = UserUsageMetrics(
                    user_id=user_id,
                    period_start=month_start,
                    period_end=month_end,
                    period_type="monthly"
                )
                db.add(metrics)
            
            # Increment counters
            metrics.simulations_run += 1
            metrics.total_iterations += iterations
            
            if engine_type in ["gpu", "super"]:
                metrics.gpu_simulations += 1
            
            # Update engines used
            if metrics.engines_used is None:
                metrics.engines_used = []
            
            if engine_type not in metrics.engines_used:
                metrics.engines_used.append(engine_type)
            
            metrics.updated_at = now
            db.commit()
            
            self.logger.info(f"Recorded simulation start for user {user_id}: {simulation_id}")
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error recording simulation start for user {user_id}: {e}")
        finally:
            db.close()
    
    async def get_quota_status(self, user_id: int) -> Dict:
        """Get comprehensive quota status for user"""
        db = next(get_db())
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if not subscription:
                subscription = UserSubscription(
                    user_id=user_id,
                    tier="free",
                    status="active"
                )
                db.add(subscription)
                db.commit()
                db.refresh(subscription)
            
            limits = TIER_LIMITS.get(subscription.tier, TIER_LIMITS["free"])
            usage_info = self._get_current_usage(user_id, db)
            
            # Calculate quota percentages
            quota_status = {
                "subscription_tier": subscription.tier,
                "limits": limits,
                "usage": usage_info,
                "quotas": {}
            }
            
            # Monthly simulations quota
            if limits["simulations_per_month"] > 0:
                quota_status["quotas"]["monthly_simulations"] = {
                    "used": usage_info["simulations_this_month"],
                    "limit": limits["simulations_per_month"],
                    "percentage": (usage_info["simulations_this_month"] / limits["simulations_per_month"]) * 100,
                    "remaining": limits["simulations_per_month"] - usage_info["simulations_this_month"]
                }
            else:
                quota_status["quotas"]["monthly_simulations"] = {
                    "used": usage_info["simulations_this_month"],
                    "limit": -1,  # Unlimited
                    "percentage": 0,
                    "remaining": -1
                }
            
            # Concurrent simulations quota
            if limits["concurrent_simulations"] > 0:
                quota_status["quotas"]["concurrent_simulations"] = {
                    "used": usage_info["running_simulations"],
                    "limit": limits["concurrent_simulations"],
                    "percentage": (usage_info["running_simulations"] / limits["concurrent_simulations"]) * 100,
                    "remaining": limits["concurrent_simulations"] - usage_info["running_simulations"]
                }
            else:
                quota_status["quotas"]["concurrent_simulations"] = {
                    "used": usage_info["running_simulations"],
                    "limit": -1,  # Unlimited
                    "percentage": 0,
                    "remaining": -1
                }
            
            return quota_status
            
        except Exception as e:
            self.logger.error(f"Error getting quota status for user {user_id}: {e}")
            return {"error": str(e)}
        finally:
            db.close()
    
    def _get_current_usage(self, user_id: int, db: Session) -> Dict:
        """Get current usage statistics for a user"""
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
        
        # Sum total iterations this month
        iterations_result = db.query(
            func.sum(SimulationResultModel.iterations_requested)
        ).filter(
            and_(
                SimulationResultModel.user_id == user_id,
                SimulationResultModel.created_at >= month_start,
                SimulationResultModel.iterations_requested.isnot(None)
            )
        ).scalar()
        
        total_iterations = int(iterations_result or 0)
        
        return {
            "simulations_this_month": simulations_count,
            "running_simulations": running_count,
            "total_iterations_this_month": total_iterations,
            "period_start": month_start.isoformat(),
            "current_month": now.strftime("%Y-%m")
        }


# Global quota enforcer instance
quota_enforcer = QuotaEnforcer() 