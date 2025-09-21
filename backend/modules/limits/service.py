"""
Enhanced Limits service module - implements LimitsServiceProtocol

Handles user limits, quotas, and usage tracking with database persistence.
This service can be easily extracted to a microservice later.
"""

import logging
import json
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timezone, timedelta
from calendar import monthrange
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..base import BaseService, LimitsServiceProtocol
from database import get_db
from models import UserSubscription, UserUsageMetrics, SimulationResult as SimulationResultModel

logger = logging.getLogger(__name__)


# Default tier limits (can be overridden by database)
TIER_LIMITS = {
    "trial": {
        "simulations_per_month": 100,  # Professional level
        "iterations_per_simulation": 1000000,  # Professional level (1M)
        "concurrent_simulations": 10,  # Professional level
        "result_retention_days": 365,  # Professional level
        "file_size_mb": 10,  # Professional level (10MB)
        "gpu_access": True,  # Full access during trial
        "gpu_priority": "high",  # Professional level (10x faster)
        "support_response_hours": 24,  # Professional level
        "engines": ["power", "arrow", "enhanced", "ultra"],  # All engines
        "custom_integrations": False  # No custom integrations in trial
    },
    "free": {
        "simulations_per_month": 100,
        "iterations_per_simulation": 1000,
        "concurrent_simulations": 3,
        "result_retention_days": 30,
        "file_size_mb": 10,
        "gpu_access": False,
        "engines": ["power", "arrow", "enhanced"]
    },
    "starter": {
        "simulations_per_month": 50,
        "iterations_per_simulation": 100000,  # 100K
        "concurrent_simulations": 3,
        "result_retention_days": 90,
        "file_size_mb": 1,  # 1MB
        "gpu_access": True,
        "gpu_priority": "standard",
        "support_response_hours": 48,
        "engines": ["power", "arrow", "enhanced"],
        "custom_integrations": False
    },
    "professional": {
        "simulations_per_month": 100,
        "iterations_per_simulation": 1000000,  # 1M
        "concurrent_simulations": 10,
        "result_retention_days": 365,
        "file_size_mb": 10,  # 10MB
        "gpu_access": True,
        "gpu_priority": "high",  # 10x faster
        "support_response_hours": 24,
        "engines": ["power", "arrow", "enhanced", "ultra"],
        "custom_integrations": False
    },
    "enterprise": {
        "simulations_per_month": -1,  # Unlimited
        "iterations_per_simulation": -1,  # Unlimited
        "concurrent_simulations": -1,  # Unlimited
        "result_retention_days": -1,  # Unlimited
        "file_size_mb": -1,  # Unlimited
        "gpu_access": True,
        "gpu_priority": "dedicated",
        "support_response_hours": 4,
        "engines": ["power", "arrow", "enhanced", "ultra"],
        "custom_integrations": True
    }
}


class LimitsService(BaseService, LimitsServiceProtocol):
    """Enhanced Limits service with database persistence"""
    
    def __init__(self):
        super().__init__("limits")
        self._redis = None
        
    async def initialize(self) -> None:
        """Initialize the limits service"""
        await super().initialize()
        logger.info("Enhanced Limits service initialized with database persistence")

    def _get_user_subscription(self, user_id: int, db: Session) -> UserSubscription:
        """Get or create user subscription record"""
        subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == user_id
        ).first()
        
        if not subscription:
            # Create trial subscription for new users
            from services.trial_service import trial_service
            trial_result = trial_service.start_trial_for_user(user_id, db)
            
            if trial_result.get("success"):
                subscription = db.query(UserSubscription).filter(
                    UserSubscription.user_id == user_id
                ).first()
                logger.info(f"Started 7-day trial for new user {user_id}")
            else:
                # Fallback to free subscription
                subscription = UserSubscription(
                    user_id=user_id,
                    tier="free",
                    status="active"
                )
                db.add(subscription)
                db.commit()
                db.refresh(subscription)
                logger.info(f"Created free subscription for user {user_id}")
        
        return subscription

    def _get_current_usage_from_db(self, user_id: int, db: Session) -> Dict[str, int]:
        """Get current month usage from simulation records"""
        # Get current month boundaries
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
            "concurrent_simulations": running_count,  # For backward compatibility
            "running_simulations": running_count,
            "total_iterations_this_month": total_iterations,
            "period_start": month_start.isoformat(),
            "current_month": now.strftime("%Y-%m")
        }
    
    async def check_simulation_allowed(self, user_id: int) -> Tuple[bool, str]:
        """Check if user can run another simulation"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            limits = subscription.get_limits()
            usage = self._get_current_usage_from_db(user_id, db)
            
            # Check monthly simulation limit
            monthly_limit = limits.get("simulations_per_month", 100)
            if monthly_limit > 0:  # -1 means unlimited
                monthly_usage = usage.get("simulations_this_month", 0)
                if monthly_usage >= monthly_limit:
                    return False, f"Monthly simulation limit reached ({monthly_limit}). Upgrade to continue."
            
            # Check concurrent simulations
            concurrent_limit = limits.get("concurrent_simulations", 3)
            if concurrent_limit > 0:  # -1 means unlimited
                concurrent_usage = usage.get("running_simulations", 0)
                if concurrent_usage >= concurrent_limit:
                    return False, f"Concurrent simulation limit reached ({concurrent_limit}). Wait for simulations to complete."
            
            return True, "Simulation allowed"
            
        except Exception as e:
            logger.error(f"Error checking simulation limits for user {user_id}: {e}")
            return False, "Error checking limits"
        finally:
            db.close()

    async def check_iteration_limit(self, user_id: int, requested_iterations: int) -> Tuple[bool, str]:
        """Check if requested iterations are within user's limits"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            limits = subscription.get_limits()
            
            max_iterations = limits.get("iterations_per_simulation", 1000)
            if max_iterations > 0 and requested_iterations > max_iterations:
                return False, f"Iteration limit exceeded. Maximum {max_iterations} iterations per simulation for {subscription.tier} tier."
            
            return True, "Iterations allowed"
            
        except Exception as e:
            logger.error(f"Error checking iteration limits for user {user_id}: {e}")
            return False, "Error checking iteration limits"
        finally:
            db.close()

    async def check_engine_allowed(self, user_id: int, engine: str) -> Tuple[bool, str]:
        """Check if user can use the specified engine"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            limits = subscription.get_limits()
            
            # Check engine access
            allowed_engines = limits.get("engines", ["power"])
            if engine not in allowed_engines:
                return False, f"Engine '{engine}' not available for {subscription.tier} tier. Upgrade to access more engines."
            
            # Check GPU access for GPU-based engines
            if engine in ["enhanced", "super", "gpu"] and not limits.get("gpu_access", False):
                return False, f"GPU engines require paid subscription. Upgrade to access enhanced simulation capabilities."
            
            return True, "Engine allowed"
            
        except Exception as e:
            logger.error(f"Error checking engine access for user {user_id}: {e}")
            return False, "Error checking engine access"
        finally:
            db.close()
    
    async def increment_usage(self, user_id: int, metric: str, amount: int = 1) -> None:
        """Increment usage counter with database persistence"""
        db = next(get_db())
        try:
            # Get or create current month metrics
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
            
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
            
            # Update specific metrics
            if metric == "simulations":
                metrics.simulations_run += amount
            elif metric == "iterations":
                metrics.total_iterations += amount
            elif metric == "files":
                metrics.files_uploaded += amount
            elif metric == "api_calls":
                metrics.api_calls += amount
            elif metric == "gpu_simulations":
                metrics.gpu_simulations += amount
            
            metrics.updated_at = now
            db.commit()
            
            logger.debug(f"Usage incremented for user {user_id}: {metric} +{amount}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error incrementing usage for user {user_id}: {e}")
        finally:
            db.close()
    
    async def get_usage(self, user_id: int) -> Dict:
        """Get user's current usage from database"""
        db = next(get_db())
        try:
            usage = self._get_current_usage_from_db(user_id, db)
            
            # Add additional computed fields for backward compatibility
            usage.update({
                "user_id": user_id,
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "storage_used_mb": 0  # TODO: Implement if needed
            })
            
            return usage
            
        except Exception as e:
            logger.error(f"Error getting usage for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "current_month": datetime.now(timezone.utc).strftime("%Y-%m"),
                "simulations_this_month": 0,
                "concurrent_simulations": 0,
                "running_simulations": 0,
                "total_iterations_this_month": 0
            }
        finally:
            db.close()
    
    async def get_limits(self, user_id: int) -> Dict:
        """Get user's current limits from subscription"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            return subscription.get_limits()
            
        except Exception as e:
            logger.error(f"Error getting limits for user {user_id}: {e}")
            return TIER_LIMITS["free"]
        finally:
            db.close()

    async def get_comprehensive_limits_info(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive limits and usage information"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            limits = subscription.get_limits()
            usage = self._get_current_usage_from_db(user_id, db)
            
            # Calculate remaining quotas
            remaining = {}
            if limits["simulations_per_month"] > 0:
                remaining["simulations"] = max(0, limits["simulations_per_month"] - usage["simulations_this_month"])
            else:
                remaining["simulations"] = -1  # Unlimited
            
            remaining["concurrent_slots"] = max(0, limits["concurrent_simulations"] - usage["running_simulations"])
            
            # Check if approaching limits
            warnings = []
            upgrade_recommended = False
            
            if limits["simulations_per_month"] > 0:
                usage_percentage = (usage["simulations_this_month"] / limits["simulations_per_month"]) * 100
                if usage_percentage >= 90:
                    warnings.append("simulation_quota_90_percent")
                    upgrade_recommended = True
                elif usage_percentage >= 75:
                    warnings.append("simulation_quota_75_percent")
                    upgrade_recommended = subscription.tier == "free"
            
            if usage["running_simulations"] >= limits["concurrent_simulations"] * 0.8:
                warnings.append("concurrent_limit_approaching")
                upgrade_recommended = True
            
            return {
                "subscription": {
                    "tier": subscription.tier,
                    "status": subscription.status,
                    "stripe_customer_id": subscription.stripe_customer_id,
                    "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None
                },
                "limits": limits,
                "usage": usage,
                "remaining": remaining,
                "warnings": warnings,
                "upgrade_recommended": upgrade_recommended,
                "next_tier": self._get_next_tier(subscription.tier)
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive limits for user {user_id}: {e}")
            return {"error": str(e)}
        finally:
            db.close()

    def _get_next_tier(self, current_tier: str) -> Optional[str]:
        """Get the next upgrade tier"""
        tier_order = ["free", "basic", "pro", "enterprise"]
        try:
            current_index = tier_order.index(current_tier)
            if current_index < len(tier_order) - 1:
                return tier_order[current_index + 1]
        except ValueError:
            pass
        return None
    
    async def reset_monthly_usage(self, user_id: int) -> None:
        """Reset monthly usage counters (handled automatically by date-based queries)"""
        # With database persistence, monthly usage is automatically handled
        # by date-based queries. This method is kept for backward compatibility.
        logger.info(f"Monthly usage reset requested for user {user_id} (handled automatically)")
    
    async def check_file_size_allowed(self, user_id: int, file_size_mb: float) -> Tuple[bool, str]:
        """Check if file size is within user's limits"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            limits = subscription.get_limits()
            max_size = limits.get("file_size_mb", 10)
            
            if max_size > 0 and file_size_mb > max_size:
                return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size}MB) for {subscription.tier} tier. Upgrade to upload larger files."
            
            return True, "File size allowed"
            
        except Exception as e:
            logger.error(f"Error checking file size for user {user_id}: {e}")
            return False, "Error checking file size limits"
        finally:
            db.close()

    async def track_simulation_start(self, user_id: int, simulation_id: str, iterations: int, engine_type: str) -> bool:
        """Track when a simulation starts"""
        try:
            await self.increment_usage(user_id, "simulations", 1)
            await self.increment_usage(user_id, "iterations", iterations)
            await self.increment_usage(user_id, "api_calls", 1)
            
            if engine_type in ["enhanced", "super", "gpu"]:
                await self.increment_usage(user_id, "gpu_simulations", 1)
            
            logger.info(f"Tracked simulation start for user {user_id}: {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking simulation start for user {user_id}: {e}")
            return False

    async def get_upgrade_recommendation(self, user_id: int) -> Dict[str, Any]:
        """Get personalized upgrade recommendation"""
        comprehensive_info = await self.get_comprehensive_limits_info(user_id)
        
        if "error" in comprehensive_info:
            return comprehensive_info
        
        subscription = comprehensive_info["subscription"]
        usage = comprehensive_info["usage"]
        limits = comprehensive_info["limits"]
        
        if subscription["tier"] == "enterprise":
            return {"upgrade_needed": False, "message": "You have the highest tier available."}
        
        recommendations = []
        target_tier = self._get_next_tier(subscription["tier"])
        
        # Analyze usage patterns
        if limits["simulations_per_month"] > 0:
            usage_percentage = (usage["simulations_this_month"] / limits["simulations_per_month"]) * 100
            if usage_percentage >= 75:
                recommendations.append(f"You've used {usage_percentage:.0f}% of your monthly simulations")
        
        if usage["running_simulations"] >= limits["concurrent_simulations"] * 0.8:
            recommendations.append("You frequently hit concurrent simulation limits")
        
        if usage["total_iterations_this_month"] > 0 and usage["simulations_this_month"] > 0:
            avg_iterations = usage["total_iterations_this_month"] / usage["simulations_this_month"]
            if avg_iterations > limits["iterations_per_simulation"] * 0.8:
                recommendations.append(f"Your simulations average {avg_iterations:.0f} iterations, close to the limit")
        
        if recommendations and target_tier:
            next_tier_limits = TIER_LIMITS.get(target_tier, {})
            return {
                "upgrade_needed": True,
                "current_tier": subscription["tier"],
                "recommended_tier": target_tier,
                "reasons": recommendations,
                "benefits": {
                    "simulations_per_month": next_tier_limits.get("simulations_per_month", "More"),
                    "iterations_per_simulation": next_tier_limits.get("iterations_per_simulation", "More"),
                    "concurrent_simulations": next_tier_limits.get("concurrent_simulations", "More"),
                    "gpu_access": next_tier_limits.get("gpu_access", False),
                    "file_size_mb": next_tier_limits.get("file_size_mb", "More")
                }
            }
        
        return {"upgrade_needed": False, "message": "Your current tier meets your usage patterns."}

    # Legacy methods for backward compatibility
    async def _get_usage(self, user_id: int) -> Dict:
        """Legacy method - use get_usage instead"""
        return await self.get_usage(user_id)
    
    async def _store_usage(self, user_id: int, usage: Dict) -> None:
        """Legacy method - usage is now stored automatically in database"""
        pass
    
    async def _get_user_tier(self, user_id: int) -> str:
        """Get user's subscription tier"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            return subscription.tier
        finally:
            db.close()
    
    def set_redis(self, redis_client):
        """Set Redis client for caching (optional)"""
        self._redis = redis_client
        logger.info("Redis client set for limits service")
    
    async def get_tier_info(self) -> Dict:
        """Get information about all available tiers"""
        return {
            "tiers": {
                tier: {
                    "limits": limits,
                    "price": self._get_tier_price(tier),
                    "features": self._get_tier_features(tier)
                }
                for tier, limits in TIER_LIMITS.items()
            }
        }
    
    def _get_tier_price(self, tier: str) -> str:
        """Get tier pricing information"""
        prices = {
            "free": "$0/month",
            "basic": "$29/month", 
            "pro": "$99/month",
            "enterprise": "Contact Sales"
        }
        return prices.get(tier, "Unknown")
    
    def _get_tier_features(self, tier: str) -> list:
        """Get tier feature list"""
        features = {
            "free": ["Basic Monte Carlo simulations", "Standard engines", "Email support"],
            "basic": ["GPU acceleration", "Advanced engines", "Priority support", "Extended retention"],
            "pro": ["Unlimited simulations*", "All engines", "Advanced analytics", "API access"],
            "enterprise": ["Custom limits", "Dedicated support", "SLA", "Advanced integrations"]
        }
        return features.get(tier, []) 