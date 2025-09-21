"""
ENTERPRISE AUTHENTICATION & AUTHORIZATION ROUTER
Phase 2 Week 6-7: Enhanced OAuth 2.0 + RBAC Router

This router provides enterprise authentication endpoints with:
- Enhanced user context
- Organization management
- Role-based access control
- Permission checking
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth.auth0_dependencies import get_current_active_auth0_user
from models import User
from database import get_db
from sqlalchemy.orm import Session
from enterprise.auth_service import (
    EnterpriseAuthService, 
    EnterpriseUser, 
    UserRole, 
    UserTier,
    enterprise_auth_service,
    quota_manager,
    require_permission
)
from enterprise.organization_service import (
    Organization,
    OrganizationSettings,
    OrganizationUsage,
    organization_service
)

logger = logging.getLogger(__name__)

# Pydantic models for API
class EnterpriseUserResponse(BaseModel):
    """Enterprise user response model"""
    id: int
    email: str
    full_name: str
    organization_name: str
    organization_tier: str
    roles: List[str]
    permissions: List[str]
    quotas: Dict[str, Any]
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True

class OrganizationResponse(BaseModel):
    """Organization response model"""
    id: int
    name: str
    domain: str
    tier: str
    max_users: int
    max_simulations_per_month: int
    max_storage_gb: int
    sso_enabled: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class OrganizationUsageResponse(BaseModel):
    """Organization usage response model"""
    organization_id: int
    month: int
    year: int
    total_simulations: int
    total_compute_units: float
    total_storage_gb: float
    unique_active_users: int
    simulation_success_rate: float
    estimated_cost_usd: float
    
    class Config:
        from_attributes = True

class PermissionCheckRequest(BaseModel):
    """Permission check request model"""
    permission: str = Field(..., description="Permission to check (e.g., 'simulation.create')")

class PermissionCheckResponse(BaseModel):
    """Permission check response model"""
    has_permission: bool
    user_roles: List[str]
    required_permission: str

# Create router
router = APIRouter(prefix="/enterprise/auth", tags=["Enterprise Authentication"])

@router.get("/me", response_model=EnterpriseUserResponse)
async def get_current_enterprise_user(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get current user with enterprise context"""
    try:
        # Convert Auth0 user to enterprise user
        enterprise_user = await enterprise_auth_service.authenticate_enterprise_user(current_user)
        
        return EnterpriseUserResponse(
            id=enterprise_user.id,
            email=enterprise_user.email,
            full_name=enterprise_user.full_name,
            organization_name=enterprise_user.organization.name,
            organization_tier=enterprise_user.organization.tier.value,
            roles=[role.value for role in enterprise_user.roles],
            permissions=enterprise_user.permissions,
            quotas=enterprise_user.quotas,
            last_login=enterprise_user.last_login
        )
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_AUTH] Failed to get enterprise user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load enterprise user context"
        )

@router.post("/check-permission", response_model=PermissionCheckResponse)
async def check_user_permission(
    request: PermissionCheckRequest,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Check if current user has specific permission"""
    try:
        # Get enterprise user context
        enterprise_user = await enterprise_auth_service.authenticate_enterprise_user(current_user)
        
        # Check permission
        has_permission = enterprise_auth_service.check_user_permission(
            enterprise_user, 
            request.permission
        )
        
        return PermissionCheckResponse(
            has_permission=has_permission,
            user_roles=[role.value for role in enterprise_user.roles],
            required_permission=request.permission
        )
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_AUTH] Failed to check permission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check user permission"
        )

@router.get("/organization", response_model=OrganizationResponse)
async def get_user_organization(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get current user's organization information"""
    try:
        # Get enterprise user context
        enterprise_user = await enterprise_auth_service.authenticate_enterprise_user(current_user)
        
        return OrganizationResponse(
            id=enterprise_user.organization.id,
            name=enterprise_user.organization.name,
            domain=enterprise_user.organization.domain,
            tier=enterprise_user.organization.tier.value,
            max_users=enterprise_user.organization.max_users,
            max_simulations_per_month=enterprise_user.organization.max_simulations_per_month,
            max_storage_gb=enterprise_user.organization.max_storage_gb,
            sso_enabled=enterprise_user.organization.sso_enabled,
            created_at=enterprise_user.organization.created_at
        )
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_AUTH] Failed to get organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load organization information"
        )

@router.get("/organization/usage", response_model=OrganizationUsageResponse)
@require_permission("organization.view")
async def get_organization_usage(
    month: Optional[int] = None,
    year: Optional[int] = None,
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Get organization usage statistics"""
    try:
        # Default to current month/year
        if month is None:
            month = datetime.utcnow().month
        if year is None:
            year = datetime.utcnow().year
        
        # Get usage data
        usage = await organization_service.get_organization_usage(
            enterprise_user.organization.id, 
            month, 
            year
        )
        
        return OrganizationUsageResponse(
            organization_id=usage.organization_id,
            month=usage.month,
            year=usage.year,
            total_simulations=usage.total_simulations,
            total_compute_units=usage.total_compute_units,
            total_storage_gb=usage.total_storage_gb,
            unique_active_users=usage.unique_active_users,
            simulation_success_rate=usage.simulation_success_rate,
            estimated_cost_usd=usage.estimated_cost_usd
        )
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_AUTH] Failed to get organization usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load organization usage"
        )

@router.get("/quotas")
async def get_user_quotas(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get current user's quotas and usage"""
    try:
        # Get enterprise user context
        enterprise_user = await enterprise_auth_service.authenticate_enterprise_user(current_user)
        
        # Get current usage
        current_simulations = await quota_manager.get_user_active_simulations_count(current_user.id)
        current_storage = await quota_manager.get_user_storage_usage_gb(current_user.id)
        
        return {
            "quotas": enterprise_user.quotas,
            "current_usage": {
                "active_simulations": current_simulations,
                "storage_gb": current_storage
            },
            "quota_status": {
                "simulations": {
                    "used": current_simulations,
                    "limit": enterprise_user.quotas.get('max_concurrent_simulations', 1),
                    "percentage": (current_simulations / enterprise_user.quotas.get('max_concurrent_simulations', 1)) * 100
                },
                "storage": {
                    "used_gb": current_storage,
                    "limit_gb": enterprise_user.quotas.get('max_storage_gb', 1),
                    "percentage": (current_storage / enterprise_user.quotas.get('max_storage_gb', 1)) * 100
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_AUTH] Failed to get user quotas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load user quotas"
        )

@router.get("/roles")
async def get_available_roles(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get available roles and their permissions"""
    try:
        # Get enterprise user context (to check if user can view roles)
        enterprise_user = await enterprise_auth_service.authenticate_enterprise_user(current_user)
        
        # Only admins and power users can view role definitions
        if not enterprise_auth_service.check_user_permission(enterprise_user, "organization.view"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view roles"
            )
        
        # Return role definitions
        role_definitions = {}
        for role, permissions in enterprise_auth_service.rbac.PERMISSIONS.items():
            role_definitions[role.value] = {
                "name": role.value.replace('_', ' ').title(),
                "permissions": permissions,
                "description": {
                    UserRole.ADMIN: "Full administrative access to all features",
                    UserRole.POWER_USER: "Advanced user with most permissions",
                    UserRole.ANALYST: "Standard user for simulation and analysis",
                    UserRole.VIEWER: "Read-only access to simulations and results"
                }.get(role, "Standard user role")
            }
        
        return {
            "roles": role_definitions,
            "user_roles": [role.value for role in enterprise_user.roles]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_AUTH] Failed to get roles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load role information"
        )

@router.get("/health")
async def enterprise_auth_health(db: Session = Depends(get_db)):
    """Health check for enterprise authentication service"""
    try:
        # Test database connectivity
        db.execute("SELECT 1")
        
        # Test enterprise auth service
        auth_service_status = enterprise_auth_service is not None
        
        return {
            "status": "healthy",
            "service": "Enterprise Authentication Service",
            "database": "connected",
            "auth_service": "active" if auth_service_status else "inactive",
            "features": {
                "rbac": True,
                "organization_management": True,
                "quota_management": True,
                "permission_checking": True
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_AUTH] Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Enterprise authentication service unhealthy: {str(e)}"
        )
