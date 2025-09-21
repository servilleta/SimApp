"""
Trial Management Service

Handles 7-day trial creation, tracking, and expiration logic.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple
from sqlalchemy.orm import Session

from database import get_db
from models import UserSubscription, User

logger = logging.getLogger(__name__)


class TrialService:
    """Service for managing user trials"""
    
    @staticmethod
    def start_trial_for_user(user_id: int, db: Session) -> Dict[str, any]:
        """
        Start a 7-day trial for a new user
        
        Args:
            user_id: User ID to start trial for
            db: Database session
            
        Returns:
            Dictionary with trial information
        """
        try:
            # Check if user already has a subscription
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if subscription:
                # User already has subscription, don't start trial
                return {
                    "success": False,
                    "message": "User already has an active subscription",
                    "trial_info": None
                }
            
            # Create new trial subscription
            subscription = UserSubscription(user_id=user_id)
            subscription.start_trial(duration_days=7)
            
            db.add(subscription)
            db.commit()
            db.refresh(subscription)
            
            logger.info(f"Started 7-day trial for user {user_id}")
            
            return {
                "success": True,
                "message": "7-day trial started successfully",
                "trial_info": {
                    "trial_start": subscription.trial_start_date.isoformat(),
                    "trial_end": subscription.trial_end_date.isoformat(),
                    "days_remaining": subscription.trial_days_remaining(),
                    "tier": subscription.tier
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start trial for user {user_id}: {e}")
            db.rollback()
            return {
                "success": False,
                "message": f"Failed to start trial: {str(e)}",
                "trial_info": None
            }
    
    @staticmethod
    def check_trial_status(user_id: int, db: Session) -> Dict[str, any]:
        """
        Check current trial status for a user
        
        Args:
            user_id: User ID to check
            db: Database session
            
        Returns:
            Dictionary with trial status information
        """
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if not subscription:
                return {
                    "has_subscription": False,
                    "is_trial": False,
                    "trial_active": False,
                    "trial_expired": False,
                    "days_remaining": 0
                }
            
            if not subscription.is_trial:
                return {
                    "has_subscription": True,
                    "is_trial": False,
                    "trial_active": False,
                    "trial_expired": False,
                    "days_remaining": 0,
                    "current_tier": subscription.tier
                }
            
            # Check if trial has expired
            if subscription.is_trial_expired():
                # Auto-expire the trial
                TrialService.expire_trial(user_id, db)
                return {
                    "has_subscription": True,
                    "is_trial": False,
                    "trial_active": False,
                    "trial_expired": True,
                    "days_remaining": 0,
                    "current_tier": "free"  # Downgraded to free
                }
            
            return {
                "has_subscription": True,
                "is_trial": True,
                "trial_active": subscription.is_trial_active(),
                "trial_expired": False,
                "days_remaining": subscription.trial_days_remaining(),
                "trial_start": subscription.trial_start_date.isoformat() if subscription.trial_start_date else None,
                "trial_end": subscription.trial_end_date.isoformat() if subscription.trial_end_date else None,
                "current_tier": subscription.tier
            }
            
        except Exception as e:
            logger.error(f"Failed to check trial status for user {user_id}: {e}")
            return {
                "has_subscription": False,
                "is_trial": False,
                "trial_active": False,
                "trial_expired": False,
                "days_remaining": 0,
                "error": str(e)
            }
    
    @staticmethod
    def expire_trial(user_id: int, db: Session) -> bool:
        """
        Expire a user's trial and downgrade to free tier
        
        Args:
            user_id: User ID to expire trial for
            db: Database session
            
        Returns:
            True if successful, False otherwise
        """
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if not subscription or not subscription.is_trial:
                logger.warning(f"No active trial found for user {user_id}")
                return False
            
            subscription.expire_trial()
            db.commit()
            
            logger.info(f"Expired trial for user {user_id}, downgraded to free tier")
            return True
            
        except Exception as e:
            logger.error(f"Failed to expire trial for user {user_id}: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def check_and_enforce_trial_limits(user_id: int, db: Session) -> Tuple[bool, str, Dict]:
        """
        Check if user can perform action based on trial status and limits
        
        Args:
            user_id: User ID to check
            db: Database session
            
        Returns:
            Tuple of (allowed, reason, trial_info)
        """
        trial_status = TrialService.check_trial_status(user_id, db)
        
        # If trial has expired, deny access to premium features
        if trial_status.get("trial_expired"):
            return False, "Your 7-day trial has expired. Please upgrade to continue using premium features.", trial_status
        
        # If no trial or subscription, check if they're eligible for trial
        if not trial_status.get("has_subscription"):
            # Start trial automatically for new users
            trial_result = TrialService.start_trial_for_user(user_id, db)
            if trial_result.get("success"):
                return True, "Trial started successfully", trial_result.get("trial_info", {})
            else:
                return False, "Failed to start trial", {}
        
        # Trial is active or user has paid subscription
        return True, "Access granted", trial_status
    
    @staticmethod
    def get_trial_limits() -> Dict[str, any]:
        """Get the limits for trial users"""
        from models import UserSubscription
        
        # Create temporary subscription to get trial limits
        temp_subscription = UserSubscription(tier="trial")
        return temp_subscription.get_limits()


# Global instance
trial_service = TrialService()
