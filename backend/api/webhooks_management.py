"""
Webhook Management API Endpoints

Provides CRUD operations for webhook configurations, delivery tracking,
and webhook testing functionality.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, HttpUrl, validator

from database import get_db
from models import User, WebhookConfiguration, WebhookDelivery
from auth.auth0_dependencies import get_current_auth0_user
from api.v1.auth import verify_api_key, APIKey
from services.webhook_service import (
    webhook_service, 
    WebhookEventType, 
    WebhookRequest,
    WebhookDeliveryStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Pydantic models for API requests/responses
class WebhookCreateRequest(BaseModel):
    """Request model for creating webhooks"""
    name: str
    url: HttpUrl
    events: List[WebhookEventType]
    secret: Optional[str] = None
    enabled: bool = True
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Webhook name cannot be empty')
        if len(v) > 100:
            raise ValueError('Webhook name cannot exceed 100 characters')
        return v.strip()
    
    @validator('events')
    def validate_events(cls, v):
        if not v:
            raise ValueError('At least one event must be selected')
        return v

class WebhookUpdateRequest(BaseModel):
    """Request model for updating webhooks"""
    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    events: Optional[List[WebhookEventType]] = None
    secret: Optional[str] = None
    enabled: Optional[bool] = None

class WebhookResponse(BaseModel):
    """Response model for webhook data"""
    id: int
    name: str
    url: str
    events: List[str]
    enabled: bool
    created_at: datetime
    updated_at: Optional[datetime]
    last_delivery_at: Optional[datetime]
    last_delivery_status: Optional[str]
    total_deliveries: int
    failed_deliveries: int
    
    class Config:
        from_attributes = True

class WebhookDeliveryResponse(BaseModel):
    """Response model for webhook delivery data"""
    id: int
    webhook_id: int
    simulation_id: str
    event_type: str
    attempt: int
    status: str
    response_status: Optional[int]
    response_time_ms: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    delivered_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class WebhookTestResponse(BaseModel):
    """Response model for webhook test results"""
    success: bool
    webhook_id: int
    webhook_name: str
    test_event: str
    timestamp: str
    message: str

# Regular user endpoints (require authentication)
@router.get("/", response_model=List[WebhookResponse])
async def get_user_webhooks(
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """Get all webhooks for the current user"""
    webhooks = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.user_id == current_user.id
    ).all()
    
    return webhooks

@router.post("/", response_model=WebhookResponse)
async def create_user_webhook(
    request: WebhookCreateRequest,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """Create a new webhook for the current user"""
    # Check webhook limit (e.g., max 10 webhooks per user)
    existing_count = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.user_id == current_user.id
    ).count()
    
    if existing_count >= 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of webhooks (10) exceeded"
        )
    
    # Create webhook
    webhook = WebhookConfiguration(
        name=request.name,
        url=str(request.url),
        events=[event.value for event in request.events],
        secret=request.secret,
        enabled=request.enabled,
        user_id=current_user.id
    )
    
    db.add(webhook)
    db.commit()
    db.refresh(webhook)
    
    logger.info(f"Created webhook '{webhook.name}' for user {current_user.id}")
    return webhook

@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_user_webhook(
    webhook_id: int,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """Get a specific webhook for the current user"""
    webhook = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.id == webhook_id,
        WebhookConfiguration.user_id == current_user.id
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    return webhook

@router.put("/{webhook_id}", response_model=WebhookResponse)
async def update_user_webhook(
    webhook_id: int,
    request: WebhookUpdateRequest,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """Update a webhook for the current user"""
    webhook = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.id == webhook_id,
        WebhookConfiguration.user_id == current_user.id
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    # Update fields
    if request.name is not None:
        webhook.name = request.name.strip()
    if request.url is not None:
        webhook.url = str(request.url)
    if request.events is not None:
        webhook.events = [event.value for event in request.events]
    if request.secret is not None:
        webhook.secret = request.secret
    if request.enabled is not None:
        webhook.enabled = request.enabled
    
    webhook.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(webhook)
    
    logger.info(f"Updated webhook '{webhook.name}' for user {current_user.id}")
    return webhook

@router.delete("/{webhook_id}")
async def delete_user_webhook(
    webhook_id: int,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """Delete a webhook for the current user"""
    webhook = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.id == webhook_id,
        WebhookConfiguration.user_id == current_user.id
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    webhook_name = webhook.name
    db.delete(webhook)
    db.commit()
    
    logger.info(f"Deleted webhook '{webhook_name}' for user {current_user.id}")
    return {"message": f"Webhook '{webhook_name}' deleted successfully"}

@router.post("/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_user_webhook(
    webhook_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """Test a webhook by sending a test payload"""
    webhook = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.id == webhook_id,
        WebhookConfiguration.user_id == current_user.id
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    # Test webhook in background
    test_result = await webhook_service.test_webhook(webhook, db)
    
    return WebhookTestResponse(
        success=test_result["success"],
        webhook_id=test_result["webhook_id"],
        webhook_name=test_result["webhook_name"],
        test_event=test_result["test_event"],
        timestamp=test_result["timestamp"],
        message="Test webhook sent successfully" if test_result["success"] else "Test webhook failed"
    )

@router.get("/{webhook_id}/deliveries", response_model=List[WebhookDeliveryResponse])
async def get_webhook_deliveries(
    webhook_id: int,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_auth0_user),
    db: Session = Depends(get_db)
):
    """Get delivery history for a webhook"""
    # Verify webhook ownership
    webhook = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.id == webhook_id,
        WebhookConfiguration.user_id == current_user.id
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    deliveries = db.query(WebhookDelivery).filter(
        WebhookDelivery.webhook_id == webhook_id
    ).order_by(WebhookDelivery.created_at.desc()).offset(offset).limit(limit).all()
    
    return deliveries

# B2B API endpoints (require API key)
@router.get("/b2b/", response_model=List[WebhookResponse])
async def get_b2b_webhooks(
    api_key_info: APIKey = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get all webhooks for the B2B API client"""
    webhooks = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.client_id == api_key_info.client_id
    ).all()
    
    return webhooks

@router.post("/b2b/", response_model=WebhookResponse)
async def create_b2b_webhook(
    request: WebhookCreateRequest,
    api_key_info: APIKey = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Create a new webhook for the B2B API client"""
    # Check webhook limit
    existing_count = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.client_id == api_key_info.client_id
    ).count()
    
    if existing_count >= 20:  # Higher limit for B2B clients
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of webhooks (20) exceeded"
        )
    
    # Create webhook
    webhook = WebhookConfiguration(
        name=request.name,
        url=str(request.url),
        events=[event.value for event in request.events],
        secret=request.secret,
        enabled=request.enabled,
        client_id=api_key_info.client_id
    )
    
    db.add(webhook)
    db.commit()
    db.refresh(webhook)
    
    logger.info(f"Created B2B webhook '{webhook.name}' for client {api_key_info.client_id}")
    return webhook

@router.put("/b2b/{webhook_id}", response_model=WebhookResponse)
async def update_b2b_webhook(
    webhook_id: int,
    request: WebhookUpdateRequest,
    api_key_info: APIKey = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Update a webhook for the B2B API client"""
    webhook = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.id == webhook_id,
        WebhookConfiguration.client_id == api_key_info.client_id
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    # Update fields (same logic as user webhook)
    if request.name is not None:
        webhook.name = request.name.strip()
    if request.url is not None:
        webhook.url = str(request.url)
    if request.events is not None:
        webhook.events = [event.value for event in request.events]
    if request.secret is not None:
        webhook.secret = request.secret
    if request.enabled is not None:
        webhook.enabled = request.enabled
    
    webhook.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(webhook)
    
    logger.info(f"Updated B2B webhook '{webhook.name}' for client {api_key_info.client_id}")
    return webhook

@router.delete("/b2b/{webhook_id}")
async def delete_b2b_webhook(
    webhook_id: int,
    api_key_info: APIKey = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Delete a webhook for the B2B API client"""
    webhook = db.query(WebhookConfiguration).filter(
        WebhookConfiguration.id == webhook_id,
        WebhookConfiguration.client_id == api_key_info.client_id
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    webhook_name = webhook.name
    db.delete(webhook)
    db.commit()
    
    logger.info(f"Deleted B2B webhook '{webhook_name}' for client {api_key_info.client_id}")
    return {"message": f"Webhook '{webhook_name}' deleted successfully"}

# Utility endpoints
@router.get("/events/types")
async def get_webhook_event_types():
    """Get all available webhook event types"""
    return {
        "events": [
            {
                "type": event.value,
                "description": {
                    "simulation.started": "Triggered when a simulation begins processing",
                    "simulation.progress": "Triggered for simulation progress updates (every 25%)",
                    "simulation.completed": "Triggered when a simulation finishes successfully",
                    "simulation.failed": "Triggered when a simulation encounters an error",
                    "simulation.cancelled": "Triggered when a simulation is cancelled by user"
                }.get(event.value, "No description available")
            }
            for event in WebhookEventType
        ]
    }
