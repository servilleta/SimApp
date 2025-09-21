"""
Trial Management API endpoints
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import User as UserModel
from auth.auth0_dependencies import get_current_active_auth0_user
from services.trial_service import trial_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trial", tags=["trial"])


@router.get("/status")
async def get_trial_status(
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Get current trial status for the authenticated user
    
    Returns:
        Dictionary with trial status, days remaining, etc.
    """
    try:
        trial_status = trial_service.check_trial_status(current_user.id, db)
        
        return {
            "success": True,
            "user_id": current_user.id,
            "trial_status": trial_status
        }
        
    except Exception as e:
        logger.error(f"Failed to get trial status for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve trial status"
        )


@router.post("/start")
async def start_trial(
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Manually start a trial for the authenticated user
    (Usually trials start automatically on first login)
    
    Returns:
        Dictionary with trial information
    """
    try:
        result = trial_service.start_trial_for_user(current_user.id, db)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Failed to start trial")
            )
        
        return {
            "success": True,
            "message": result.get("message"),
            "trial_info": result.get("trial_info")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start trial for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start trial"
        )


@router.get("/limits")
async def get_trial_limits():
    """
    Get the limits and features available during trial
    
    Returns:
        Dictionary with trial tier limits
    """
    try:
        limits = trial_service.get_trial_limits()
        
        return {
            "success": True,
            "trial_limits": limits
        }
        
    except Exception as e:
        logger.error(f"Failed to get trial limits: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve trial limits"
        )
