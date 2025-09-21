"""
API Key Management Endpoints

Allows users to create, list, and manage their API keys.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database import get_db
from models import User, APIKey
from services.api_key_service import APIKeyService
from auth.auth0_dependencies import get_current_auth0_user

router = APIRouter(prefix="/api/api-keys", tags=["API Key Management"])

# Pydantic models for API responses
class APIKeyResponse(BaseModel):
    """API Key information for responses (without secret)"""
    key_id: str
    name: str
    subscription_tier: str
    is_active: bool
    monthly_requests: int
    max_iterations: int
    max_file_size_mb: int
    requests_used_this_month: int
    last_used_at: Optional[str] = None
    created_at: str
    expires_at: Optional[str] = None
    
    class Config:
        from_attributes = True

class APIKeyCreateRequest(BaseModel):
    """Request to create a new API key"""
    name: str = Field(..., description="Human-readable name for the API key")
    subscription_tier: str = Field(default="starter", description="Subscription tier")
    expires_in_days: Optional[int] = Field(None, description="Optional expiration in days")

class APIKeyCreateResponse(BaseModel):
    """Response when creating a new API key (includes secret)"""
    api_key_info: APIKeyResponse
    secret_key: str = Field(..., description="Full API key (only shown once)")
    warning: str = "Store this API key securely. It will not be shown again."

@router.get("/", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """
    List all API keys for the current user.
    """
    api_keys = APIKeyService.list_user_api_keys(db, current_user.id)
    
    return [
        APIKeyResponse(
            key_id=key.key_id,
            name=key.name,
            subscription_tier=key.subscription_tier,
            is_active=key.is_active,
            monthly_requests=key.monthly_requests,
            max_iterations=key.max_iterations,
            max_file_size_mb=key.max_file_size_mb,
            requests_used_this_month=key.requests_used_this_month,
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            created_at=key.created_at.isoformat(),
            expires_at=key.expires_at.isoformat() if key.expires_at else None
        )
        for key in api_keys
    ]

@router.post("/", response_model=APIKeyCreateResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Create a new API key for the current user.
    
    **⚠️ Important**: The secret key will only be shown once. Store it securely.
    """
    # Check if user already has maximum number of keys (prevent abuse)
    existing_keys = APIKeyService.list_user_api_keys(db, current_user.id)
    active_keys = [key for key in existing_keys if key.is_active]
    
    if len(active_keys) >= 10:  # Maximum 10 active keys per user
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of API keys reached (10). Please revoke unused keys first."
        )
    
    # Create the API key
    api_key, secret_key = APIKeyService.create_api_key(
        db=db,
        user_id=current_user.id,
        name=request.name,
        subscription_tier=request.subscription_tier,
        expires_in_days=request.expires_in_days
    )
    
    # Format the full key for the user
    full_key = f"{api_key.key_id}_{secret_key}"
    
    return APIKeyCreateResponse(
        api_key_info=APIKeyResponse(
            key_id=api_key.key_id,
            name=api_key.name,
            subscription_tier=api_key.subscription_tier,
            is_active=api_key.is_active,
            monthly_requests=api_key.monthly_requests,
            max_iterations=api_key.max_iterations,
            max_file_size_mb=api_key.max_file_size_mb,
            requests_used_this_month=api_key.requests_used_this_month,
            last_used_at=None,
            created_at=api_key.created_at.isoformat(),
            expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None
        ),
        secret_key=full_key,
        warning="Store this API key securely. It will not be shown again."
    )

@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Revoke (deactivate) an API key.
    
    This will immediately disable the API key and prevent further use.
    """
    success = APIKeyService.revoke_api_key(db, key_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or you don't have permission to revoke it."
        )
    
    return {"message": f"API key {key_id} has been revoked successfully."}

@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific API key.
    """
    api_keys = APIKeyService.list_user_api_keys(db, current_user.id)
    api_key = next((key for key in api_keys if key.key_id == key_id), None)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or you don't have permission to view it."
        )
    
    return APIKeyResponse(
        key_id=api_key.key_id,
        name=api_key.name,
        subscription_tier=api_key.subscription_tier,
        is_active=api_key.is_active,
        monthly_requests=api_key.monthly_requests,
        max_iterations=api_key.max_iterations,
        max_file_size_mb=api_key.max_file_size_mb,
        requests_used_this_month=api_key.requests_used_this_month,
        last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        created_at=api_key.created_at.isoformat(),
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None
    )
