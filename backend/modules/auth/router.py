"""
Authentication router module

Provides authentication endpoints including login, registration, user management,
and user dashboard statistics.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Annotated
import logging

from ..base import AuthServiceProtocol
from modules.auth.schemas import Token, User, UserCreate, UserUpdate
from models import UserSubscription
from database import get_db

logger = logging.getLogger(__name__)


def create_auth_router(auth_service: AuthServiceProtocol) -> APIRouter:
    """Create authentication router with dependency injection"""
    
    router = APIRouter(prefix="/auth", tags=["authentication"])
    
    @router.post("/token", response_model=Token)
    async def login_for_access_token(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
    ):
        """Authenticate user and return access token"""
        token_data = await auth_service.authenticate_user(form_data.username, form_data.password)
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token_data

    @router.post("/register", response_model=User)
    async def register_user(user_in: UserCreate):
        """Register a new user account - DISABLED FOR PRIVATE LAUNCH"""
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="New user registrations are temporarily disabled. SimApp is currently in private launch mode."
        )
        # Original registration logic (disabled for private launch):
        # return await auth_service.create_user(user_in.dict())

    @router.get("/me", response_model=User)
    async def read_users_me(current_user: Annotated[User, Depends(auth_service.get_current_active_user)]):
        """Get current user profile"""
        return current_user

    @router.patch("/me", response_model=User)
    async def update_current_user(
        user_update: UserUpdate,
        current_user: Annotated[User, Depends(auth_service.get_current_active_user)]
    ):
        """Update current user profile"""
        updated_user = await auth_service.update_user(current_user.id, user_update.dict(exclude_unset=True))
        return updated_user
    
    @router.post("/me/revoke-sessions")
    async def revoke_all_sessions(
        current_user: Annotated[User, Depends(auth_service.get_current_active_user)]
    ):
        """Revoke all user sessions (logout from all devices)"""
        # Since we use stateless JWT tokens, this endpoint serves as a placeholder
        # In a production system, you might maintain a blacklist of tokens or use refresh tokens
        logger.info(f"User {current_user.username} requested to revoke all sessions")
        return {
            "message": "All sessions revoked successfully. Please log in again on all devices.",
            "action_required": "re_login"
        }
    
    @router.delete("/me")
    async def delete_current_user_account(
        current_user: Annotated[User, Depends(auth_service.get_current_active_user)],
        db: Session = Depends(get_db)
    ):
        """Delete current user account - PLACEHOLDER"""
        # In a production system, this would:
        # 1. Mark user as deleted/disabled
        # 2. Delete user data according to GDPR requirements  
        # 3. Anonymize or remove simulation data
        # 4. Cancel subscriptions
        logger.warning(f"User {current_user.username} requested account deletion")
        
        # For now, just disable the account
        from models import User as UserModel
        user = db.query(UserModel).filter(UserModel.id == current_user.id).first()
        if user:
            user.disabled = True
            db.commit()
            
        return {
            "message": "Account deletion request processed. Your account has been disabled.",
            "note": "For complete data deletion, please contact support."
        }

    @router.get("/dashboard/stats")
    async def get_user_dashboard_stats(
        current_user: Annotated[User, Depends(auth_service.get_current_active_user)],
        db: Session = Depends(get_db)
    ):
        """Get dashboard statistics for the current user"""
        try:
            from modules.container import get_service_container
            
            # Get services
            container = get_service_container()
            limits_service = container.get_limits_service()
            
            # Get user subscription
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == current_user.id
            ).first()
            
            if not subscription:
                # Create default free subscription
                subscription = UserSubscription(
                    user_id=current_user.id,
                    tier="free",
                    status="active"
                )
                db.add(subscription)
                db.commit()
                db.refresh(subscription)
            
            # Get current usage
            usage_data = await limits_service.get_usage(current_user.id)
            limits_data = await limits_service.get_limits(current_user.id)
            
            # Get current month start for period info
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Generate quota warnings
            warnings = await _get_quota_warnings(current_user.id, usage_data, limits_data)
            
            return {
                "user": {
                    "id": current_user.id,
                    "username": current_user.username,
                    "email": current_user.email,
                    "full_name": current_user.full_name
                },
                "subscription": {
                    "tier": subscription.tier,
                    "status": subscription.status,
                    "current_period_start": subscription.current_period_start.isoformat() if subscription.current_period_start else month_start.isoformat(),
                    "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None
                },
                "usage": {
                    "simulations_this_month": usage_data.get("simulations_this_month", 0),
                    "running_simulations": usage_data.get("running_simulations", 0),
                    "total_iterations_this_month": usage_data.get("total_iterations_this_month", 0),
                    "period_start": usage_data.get("period_start", month_start.isoformat()),
                    "current_month": usage_data.get("current_month", now.strftime("%Y-%m"))
                },
                "limits": limits_data,
                "quota_warnings": warnings,
                "timestamp": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard stats for user {current_user.id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching dashboard statistics: {str(e)}")

    return router


async def _get_quota_warnings(user_id: int, usage: dict, limits: dict) -> list:
    """Generate quota warnings for the user"""
    warnings = []
    
    # Check simulation quota
    simulations_used = usage.get("simulations_this_month", 0)
    simulations_limit = limits.get("simulations_per_month", 100)
    
    if simulations_limit > 0:  # Not unlimited
        usage_percentage = (simulations_used / simulations_limit) * 100
        
        if usage_percentage >= 90:
            warnings.append({
                "type": "quota_critical",
                "message": f"You've used {simulations_used}/{simulations_limit} simulations this month (90%+)",
                "action": "Consider upgrading your plan to continue"
            })
        elif usage_percentage >= 75:
            warnings.append({
                "type": "quota_warning", 
                "message": f"You've used {simulations_used}/{simulations_limit} simulations this month (75%+)",
                "action": "Monitor your usage or consider upgrading"
            })
    
    # Check concurrent simulations
    running_sims = usage.get("running_simulations", 0)
    concurrent_limit = limits.get("concurrent_simulations", 3)
    
    if concurrent_limit > 0 and running_sims >= concurrent_limit:
        warnings.append({
            "type": "concurrent_limit",
            "message": f"You have {running_sims}/{concurrent_limit} simulations running",
            "action": "Wait for simulations to complete before starting new ones"
        })
    
    return warnings 