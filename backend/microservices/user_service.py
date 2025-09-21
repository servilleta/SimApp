"""
👤 USER SERVICE - Microservice Architecture

Handles all user-related operations:
- Authentication and authorization
- User profile management
- Subscription management
- API key management
- Role-based access control

This service is part of the microservices decomposition from the monolithic application.
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from pydantic import BaseModel, EmailStr

# Import from monolith (during transition)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db, SessionLocal
from models import User, UserSubscription, APIKey, UserUsageMetrics
from auth.auth0_dependencies import get_current_active_auth0_user

logger = logging.getLogger(__name__)

# FastAPI app for User Service
app = FastAPI(
    title="User Service",
    description="Microservice for user management, authentication, and subscriptions",
    version="1.0.0"
)

security = HTTPBearer()

# ===============================
# PYDANTIC MODELS (API SCHEMAS)
# ===============================

class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    auth0_user_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserSubscriptionResponse(BaseModel):
    id: int
    user_id: int
    tier: str
    status: str
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    is_trial: bool
    trial_end_date: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class APIKeyResponse(BaseModel):
    id: int
    key_id: str
    name: str
    subscription_tier: str
    is_active: bool
    monthly_requests: int
    requests_used_this_month: int
    last_used_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserUsageResponse(BaseModel):
    simulations_run: int
    total_iterations: int
    files_uploaded: int
    total_file_size_mb: float
    api_calls: int
    gpu_simulations: int
    period_start: datetime
    period_end: datetime
    
    class Config:
        from_attributes = True

class CreateAPIKeyRequest(BaseModel):
    name: str
    subscription_tier: str = "professional"

class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None

class UserPreferences(BaseModel):
    email_notifications: bool = True
    webhook_notifications: bool = False
    api_rate_limit_notifications: bool = True
    security_alerts: bool = True

# ===============================
# USER SERVICE CLASS
# ===============================

class UserService:
    """
    Core user service handling all user-related operations.
    Designed for microservices architecture with clear boundaries.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def get_user_profile(self, db: Session, user_id: int) -> Optional[UserProfile]:
        """Get user profile by ID."""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            return UserProfile.from_orm(user)
            
        except Exception as e:
            self.logger.error(f"❌ [USER_SERVICE] Failed to get user profile {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user profile")
    
    async def update_user_profile(self, db: Session, user_id: int, updates: UpdateUserRequest) -> UserProfile:
        """Update user profile."""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Apply updates
            update_data = updates.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(user, field, value)
            
            db.commit()
            db.refresh(user)
            
            self.logger.info(f"✅ [USER_SERVICE] Updated user profile {user_id}")
            return UserProfile.from_orm(user)
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"❌ [USER_SERVICE] Failed to update user profile {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update user profile")
    
    async def get_user_subscription(self, db: Session, user_id: int) -> Optional[UserSubscriptionResponse]:
        """Get user subscription details."""
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if not subscription:
                return None
            
            return UserSubscriptionResponse.from_orm(subscription)
            
        except Exception as e:
            self.logger.error(f"❌ [USER_SERVICE] Failed to get subscription for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve subscription")
    
    async def get_user_api_keys(self, db: Session, user_id: int) -> List[APIKeyResponse]:
        """Get all API keys for a user."""
        try:
            api_keys = db.query(APIKey).filter(APIKey.user_id == user_id).all()
            
            return [APIKeyResponse.from_orm(key) for key in api_keys]
            
        except Exception as e:
            self.logger.error(f"❌ [USER_SERVICE] Failed to get API keys for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve API keys")
    
    async def create_api_key(self, db: Session, user_id: int, request: CreateAPIKeyRequest) -> APIKeyResponse:
        """Create a new API key for a user."""
        try:
            # Generate key components
            key_id = f"ak_{uuid.uuid4().hex[:16]}"
            secret_key = f"sk_{uuid.uuid4().hex}"
            
            # Create API key record
            api_key = APIKey(
                user_id=user_id,
                key_id=key_id,
                key_hash=self._hash_key(secret_key),  # Store hashed secret
                name=request.name,
                client_id=f"client_{uuid.uuid4().hex[:8]}",
                subscription_tier=request.subscription_tier,
                is_active=True
            )
            
            db.add(api_key)
            db.commit()
            db.refresh(api_key)
            
            self.logger.info(f"✅ [USER_SERVICE] Created API key {key_id} for user {user_id}")
            
            # Return API key info (without the actual secret)
            response = APIKeyResponse.from_orm(api_key)
            # Add the secret key to response for one-time display
            response.secret_key = secret_key  # Only shown once!
            
            return response
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"❌ [USER_SERVICE] Failed to create API key for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to create API key")
    
    async def revoke_api_key(self, db: Session, user_id: int, key_id: str) -> bool:
        """Revoke an API key."""
        try:
            api_key = db.query(APIKey).filter(
                and_(APIKey.user_id == user_id, APIKey.key_id == key_id)
            ).first()
            
            if not api_key:
                raise HTTPException(status_code=404, detail="API key not found")
            
            api_key.is_active = False
            db.commit()
            
            self.logger.info(f"✅ [USER_SERVICE] Revoked API key {key_id} for user {user_id}")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"❌ [USER_SERVICE] Failed to revoke API key {key_id} for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to revoke API key")
    
    async def get_user_usage_metrics(self, db: Session, user_id: int, period: str = "current") -> Optional[UserUsageResponse]:
        """Get user usage metrics for a specific period."""
        try:
            # Determine period dates
            now = datetime.now(timezone.utc)
            if period == "current":
                # Current month
                period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                next_month = (period_start + timedelta(days=32)).replace(day=1)
                period_end = next_month - timedelta(seconds=1)
            else:
                # Default to current month
                period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                next_month = (period_start + timedelta(days=32)).replace(day=1)
                period_end = next_month - timedelta(seconds=1)
            
            # Get usage metrics
            usage = db.query(UserUsageMetrics).filter(
                and_(
                    UserUsageMetrics.user_id == user_id,
                    UserUsageMetrics.period_start >= period_start,
                    UserUsageMetrics.period_end <= period_end
                )
            ).first()
            
            if not usage:
                # Create default usage record if none exists
                usage = UserUsageMetrics(
                    user_id=user_id,
                    period_start=period_start,
                    period_end=period_end,
                    period_type="monthly"
                )
                db.add(usage)
                db.commit()
                db.refresh(usage)
            
            return UserUsageResponse.from_orm(usage)
            
        except Exception as e:
            self.logger.error(f"❌ [USER_SERVICE] Failed to get usage metrics for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve usage metrics")
    
    async def get_user_preferences(self, db: Session, user_id: int) -> UserPreferences:
        """Get user notification and system preferences."""
        try:
            # For now, return default preferences
            # In the future, store these in a user_preferences table
            return UserPreferences()
            
        except Exception as e:
            self.logger.error(f"❌ [USER_SERVICE] Failed to get preferences for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve preferences")
    
    async def update_user_preferences(self, db: Session, user_id: int, preferences: UserPreferences) -> UserPreferences:
        """Update user preferences."""
        try:
            # For now, just return the preferences
            # In the future, persist these to database
            self.logger.info(f"✅ [USER_SERVICE] Updated preferences for user {user_id}")
            return preferences
            
        except Exception as e:
            self.logger.error(f"❌ [USER_SERVICE] Failed to update preferences for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update preferences")
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage."""
        import hashlib
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def validate_api_key(self, db: Session, key_id: str, secret_key: str) -> Optional[User]:
        """Validate an API key and return the associated user."""
        try:
            # Get API key record
            api_key = db.query(APIKey).filter(
                and_(APIKey.key_id == key_id, APIKey.is_active == True)
            ).first()
            
            if not api_key:
                return None
            
            # Verify secret key
            if api_key.key_hash != self._hash_key(secret_key):
                return None
            
            # Update usage tracking
            api_key.last_used_at = datetime.now(timezone.utc)
            api_key.requests_used_this_month += 1
            db.commit()
            
            # Get associated user
            user = db.query(User).filter(User.id == api_key.user_id).first()
            return user
            
        except Exception as e:
            self.logger.error(f"❌ [USER_SERVICE] Failed to validate API key {key_id}: {e}")
            return None

# Global service instance
user_service = UserService()

# ===============================
# API ENDPOINTS
# ===============================

@app.get("/health")
async def health_check():
    """Service health check."""
    return {"status": "healthy", "service": "user-service", "version": "1.0.0"}

@app.get("/profile", response_model=UserProfile)
async def get_profile(
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Get current user profile."""
    profile = await user_service.get_user_profile(db, current_user.id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    return profile

@app.put("/profile", response_model=UserProfile)
async def update_profile(
    updates: UpdateUserRequest,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Update user profile."""
    return await user_service.update_user_profile(db, current_user.id, updates)

@app.get("/subscription", response_model=UserSubscriptionResponse)
async def get_subscription(
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Get user subscription details."""
    subscription = await user_service.get_user_subscription(db, current_user.id)
    if not subscription:
        raise HTTPException(status_code=404, detail="No subscription found")
    return subscription

@app.get("/api-keys", response_model=List[APIKeyResponse])
async def get_api_keys(
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Get user's API keys."""
    return await user_service.get_user_api_keys(db, current_user.id)

@app.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Create a new API key."""
    return await user_service.create_api_key(db, current_user.id, request)

@app.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Revoke an API key."""
    success = await user_service.revoke_api_key(db, current_user.id, key_id)
    return {"message": "API key revoked successfully", "key_id": key_id}

@app.get("/usage", response_model=UserUsageResponse)
async def get_usage_metrics(
    period: str = "current",
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Get user usage metrics."""
    usage = await user_service.get_user_usage_metrics(db, current_user.id, period)
    if not usage:
        raise HTTPException(status_code=404, detail="Usage metrics not found")
    return usage

@app.get("/preferences", response_model=UserPreferences)
async def get_preferences(
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Get user preferences."""
    return await user_service.get_user_preferences(db, current_user.id)

@app.put("/preferences", response_model=UserPreferences)
async def update_preferences(
    preferences: UserPreferences,
    current_user: User = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """Update user preferences."""
    return await user_service.update_user_preferences(db, current_user.id, preferences)

# ===============================
# SERVICE DISCOVERY ENDPOINTS
# ===============================

@app.get("/service-info")
async def get_service_info():
    """Service discovery information."""
    return {
        "service_name": "user-service",
        "version": "1.0.0",
        "description": "User management, authentication, and subscriptions",
        "endpoints": {
            "profile": "/profile",
            "subscription": "/subscription", 
            "api_keys": "/api-keys",
            "usage": "/usage",
            "preferences": "/preferences"
        },
        "dependencies": ["auth0", "database", "stripe"],
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
