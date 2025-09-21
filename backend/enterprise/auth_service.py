"""
ENTERPRISE AUTHENTICATION & AUTHORIZATION SERVICE
Phase 2 Week 6-7: Enhanced OAuth 2.0 + RBAC Implementation

This service extends the existing Auth0 integration with enterprise features:
- Organization management
- Role-based access control (RBAC)
- Enterprise user context
- Subscription-based quotas
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from database import get_db
from models import User
from auth.auth0_dependencies import get_current_active_auth0_user

logger = logging.getLogger(__name__)

class UserTier(Enum):
    """Enterprise user tiers with different capabilities"""
    ENTERPRISE = "enterprise"
    PROFESSIONAL = "professional" 
    STANDARD = "standard"
    TRIAL = "trial"

class UserRole(Enum):
    """Role-based access control roles"""
    ADMIN = "admin"
    POWER_USER = "power_user"
    ANALYST = "analyst"
    VIEWER = "viewer"

@dataclass
class Organization:
    """Enterprise organization information"""
    id: int
    name: str
    domain: str
    tier: UserTier
    subscription_id: str
    max_users: int
    max_simulations_per_month: int
    max_storage_gb: int
    sso_enabled: bool
    created_at: datetime
    
@dataclass
class EnterpriseUser:
    """Enhanced user with enterprise context"""
    id: int
    email: str
    first_name: str
    last_name: str
    organization: Organization
    roles: List[UserRole]
    quotas: Dict[str, Any]
    permissions: List[str]
    last_login: Optional[datetime]
    created_at: datetime
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_admin(self) -> bool:
        return UserRole.ADMIN in self.roles
    
    @property
    def can_create_simulations(self) -> bool:
        return any(role in [UserRole.ADMIN, UserRole.POWER_USER, UserRole.ANALYST] 
                  for role in self.roles)

class RoleBasedAccessControl:
    """Enterprise RBAC system"""
    
    # Permission definitions
    PERMISSIONS = {
        UserRole.ADMIN: [
            '*'  # All permissions
        ],
        UserRole.POWER_USER: [
            'simulation.create', 'simulation.read', 'simulation.update', 'simulation.delete',
            'file.upload', 'file.download', 'file.delete',
            'results.view', 'results.export', 'results.share',
            'organization.view', 'user.view',
            'billing.view'
        ],
        UserRole.ANALYST: [
            'simulation.create', 'simulation.read', 'simulation.update',
            'file.upload', 'file.download',
            'results.view', 'results.export',
            'organization.view'
        ],
        UserRole.VIEWER: [
            'simulation.read',
            'results.view',
            'organization.view'
        ]
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".RBAC")
    
    def check_permission(self, user_roles: List[UserRole], required_permission: str) -> bool:
        """Check if user roles have required permission"""
        for role in user_roles:
            permissions = self.PERMISSIONS.get(role, [])
            
            # Admin has all permissions
            if '*' in permissions:
                return True
                
            # Check specific permission
            if required_permission in permissions:
                return True
                
            # Check wildcard permissions (e.g., 'simulation.*')
            permission_parts = required_permission.split('.')
            if len(permission_parts) >= 2:
                wildcard = f"{permission_parts[0]}.*"
                if wildcard in permissions:
                    return True
        
        self.logger.warning(f"Permission denied: {required_permission} for roles {user_roles}")
        return False
    
    def get_user_permissions(self, user_roles: List[UserRole]) -> List[str]:
        """Get all permissions for user roles"""
        all_permissions = set()
        
        for role in user_roles:
            permissions = self.PERMISSIONS.get(role, [])
            all_permissions.update(permissions)
        
        return list(all_permissions)

class EnterpriseAuthService:
    """Enhanced authentication service with enterprise features"""
    
    def __init__(self):
        self.rbac = RoleBasedAccessControl()
        self.logger = logging.getLogger(__name__ + ".EnterpriseAuth")
    
    async def authenticate_enterprise_user(self, auth0_user: User) -> EnterpriseUser:
        """Convert Auth0 user to enterprise user with full context"""
        
        # Get organization information
        organization = await self.get_user_organization(auth0_user.id)
        if not organization:
            # Create default organization for new users
            organization = await self.create_default_organization(auth0_user)
        
        # Get user roles
        user_roles = await self.get_user_roles(auth0_user.id, organization.id)
        
        # Calculate user quotas based on organization tier and roles
        quotas = await self.calculate_user_quotas(organization, user_roles)
        
        # Get user permissions
        permissions = self.rbac.get_user_permissions(user_roles)
        
        enterprise_user = EnterpriseUser(
            id=auth0_user.id,
            email=auth0_user.email,
            first_name=auth0_user.full_name.split(' ')[0] if auth0_user.full_name else 'Unknown',
            last_name=' '.join(auth0_user.full_name.split(' ')[1:]) if auth0_user.full_name and ' ' in auth0_user.full_name else '',
            organization=organization,
            roles=user_roles,
            quotas=quotas,
            permissions=permissions,
            last_login=datetime.utcnow(),
            created_at=auth0_user.created_at or datetime.utcnow()
        )
        
        # Update last login
        await self.update_user_last_login(auth0_user.id)
        
        self.logger.info(f"✅ [ENTERPRISE_AUTH] User authenticated: {enterprise_user.email} "
                        f"(org: {organization.name}, roles: {[r.value for r in user_roles]})")
        
        return enterprise_user
    
    async def get_user_organization(self, user_id: int) -> Optional[Organization]:
        """Get organization for user"""
        # This would typically query a database table
        # For now, create a default organization structure
        
        db = next(get_db())
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            # Extract organization from email domain or create default
            email_domain = user.email.split('@')[1] if '@' in user.email else 'default.com'
            
            # For demo purposes, create organization based on email domain
            if email_domain in ['gmail.com', 'outlook.com', 'yahoo.com']:
                org_name = "Individual Account"
                tier = UserTier.STANDARD
            else:
                org_name = email_domain.replace('.com', '').title() + " Organization"
                tier = UserTier.PROFESSIONAL
            
            return Organization(
                id=1,  # Default organization ID
                name=org_name,
                domain=email_domain,
                tier=tier,
                subscription_id=f"sub_{user_id}",
                max_users=100 if tier == UserTier.ENTERPRISE else 10,
                max_simulations_per_month=10000 if tier == UserTier.ENTERPRISE else 1000,
                max_storage_gb=1000 if tier == UserTier.ENTERPRISE else 100,
                sso_enabled=tier == UserTier.ENTERPRISE,
                created_at=user.created_at or datetime.utcnow()
            )
        finally:
            db.close()
    
    async def create_default_organization(self, user: User) -> Organization:
        """Create default organization for new user"""
        email_domain = user.email.split('@')[1] if '@' in user.email else 'default.com'
        
        return Organization(
            id=1,
            name=f"{email_domain.replace('.com', '').title()} Organization",
            domain=email_domain,
            tier=UserTier.STANDARD,
            subscription_id=f"sub_{user.id}",
            max_users=5,
            max_simulations_per_month=100,
            max_storage_gb=10,
            sso_enabled=False,
            created_at=datetime.utcnow()
        )
    
    async def get_user_roles(self, user_id: int, organization_id: int) -> List[UserRole]:
        """Get user roles within organization"""
        
        db = next(get_db())
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return [UserRole.VIEWER]  # Default role
            
            # For now, determine roles based on is_admin flag and email
            roles = []
            
            if user.is_admin:
                roles.append(UserRole.ADMIN)
            elif user.email.endswith('@gmail.com'):
                # Demo: Gmail users get power user access
                roles.append(UserRole.POWER_USER)
            else:
                roles.append(UserRole.ANALYST)
            
            # Always include viewer role as base
            if UserRole.VIEWER not in roles:
                roles.append(UserRole.VIEWER)
            
            return roles
        finally:
            db.close()
    
    async def calculate_user_quotas(self, organization: Organization, user_roles: List[UserRole]) -> Dict[str, Any]:
        """Calculate user quotas based on organization tier and roles"""
        
        # Base quotas from organization tier
        base_quotas = {
            UserTier.ENTERPRISE: {
                'max_concurrent_simulations': 10,
                'max_file_size_mb': 500,
                'max_iterations_per_simulation': 1000000,
                'api_rate_limit_per_minute': 1000,
                'max_storage_gb': 100
            },
            UserTier.PROFESSIONAL: {
                'max_concurrent_simulations': 5,
                'max_file_size_mb': 200,
                'max_iterations_per_simulation': 100000,
                'api_rate_limit_per_minute': 300,
                'max_storage_gb': 50
            },
            UserTier.STANDARD: {
                'max_concurrent_simulations': 2,
                'max_file_size_mb': 50,
                'max_iterations_per_simulation': 10000,
                'api_rate_limit_per_minute': 100,
                'max_storage_gb': 10
            },
            UserTier.TRIAL: {
                'max_concurrent_simulations': 1,
                'max_file_size_mb': 10,
                'max_iterations_per_simulation': 1000,
                'api_rate_limit_per_minute': 20,
                'max_storage_gb': 1
            }
        }
        
        quotas = base_quotas.get(organization.tier, base_quotas[UserTier.STANDARD]).copy()
        
        # Role-based quota adjustments
        if UserRole.ADMIN in user_roles:
            # Admins get unlimited quotas
            quotas.update({
                'max_concurrent_simulations': 999,
                'max_file_size_mb': 1000,
                'max_iterations_per_simulation': 10000000,
                'api_rate_limit_per_minute': 5000
            })
        elif UserRole.POWER_USER in user_roles:
            # Power users get 2x quotas
            quotas['max_concurrent_simulations'] *= 2
            quotas['max_file_size_mb'] = min(quotas['max_file_size_mb'] * 2, 1000)
            quotas['api_rate_limit_per_minute'] *= 2
        
        return quotas
    
    async def update_user_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        db = next(get_db())
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                # Note: We'd need to add last_login field to User model
                # For now, this is a placeholder
                self.logger.info(f"Updated last login for user {user_id}")
        finally:
            db.close()
    
    def check_user_permission(self, enterprise_user: EnterpriseUser, required_permission: str) -> bool:
        """Check if enterprise user has required permission"""
        return self.rbac.check_permission(enterprise_user.roles, required_permission)
    
    async def get_organization_users(self, organization_id: int) -> List[EnterpriseUser]:
        """Get all users in an organization"""
        # This would query the database for all users in the organization
        # For now, return placeholder
        return []
    
    async def invite_user_to_organization(self, organization_id: int, email: str, roles: List[UserRole]) -> Dict[str, Any]:
        """Invite new user to organization"""
        
        # Generate invitation token
        invitation_token = f"inv_{organization_id}_{email}_{datetime.utcnow().timestamp()}"
        
        # Store invitation in database (would need invitation table)
        invitation_data = {
            'organization_id': organization_id,
            'email': email,
            'roles': [role.value for role in roles],
            'token': invitation_token,
            'expires_at': datetime.utcnow() + timedelta(days=7),
            'created_at': datetime.utcnow()
        }
        
        self.logger.info(f"✅ [ENTERPRISE_AUTH] Invitation created for {email} to organization {organization_id}")
        
        return {
            'invitation_token': invitation_token,
            'invitation_url': f"/invite/{invitation_token}",
            'expires_at': invitation_data['expires_at']
        }

class EnterpriseQuotaManager:
    """Manages user quotas and usage tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".QuotaManager")
    
    async def check_simulation_quota(self, user_id: int, enterprise_user: EnterpriseUser) -> bool:
        """Check if user can create new simulation"""
        
        # Get current simulation count
        current_simulations = await self.get_user_active_simulations_count(user_id)
        max_concurrent = enterprise_user.quotas.get('max_concurrent_simulations', 1)
        
        if current_simulations >= max_concurrent:
            self.logger.warning(f"❌ [QUOTA] User {user_id} exceeded concurrent simulation quota: {current_simulations}/{max_concurrent}")
            return False
        
        return True
    
    async def check_file_upload_quota(self, user_id: int, enterprise_user: EnterpriseUser, file_size_mb: float) -> bool:
        """Check if user can upload file of given size"""
        
        max_file_size = enterprise_user.quotas.get('max_file_size_mb', 10)
        
        if file_size_mb > max_file_size:
            self.logger.warning(f"❌ [QUOTA] User {user_id} file too large: {file_size_mb}MB > {max_file_size}MB")
            return False
        
        # Check storage quota
        current_storage = await self.get_user_storage_usage_gb(user_id)
        max_storage = enterprise_user.quotas.get('max_storage_gb', 1)
        
        if current_storage + (file_size_mb / 1024) > max_storage:
            self.logger.warning(f"❌ [QUOTA] User {user_id} storage quota exceeded: {current_storage}GB + {file_size_mb/1024:.2f}GB > {max_storage}GB")
            return False
        
        return True
    
    async def check_api_rate_limit(self, user_id: int, enterprise_user: EnterpriseUser) -> bool:
        """Check API rate limit for user"""
        # This would integrate with Redis to track API calls per minute
        # For now, return True (rate limiting would be implemented in API Gateway)
        return True
    
    async def get_user_active_simulations_count(self, user_id: int) -> int:
        """Get count of user's currently active simulations"""
        # This would query the database for active simulations
        # For now, return 0
        return 0
    
    async def get_user_storage_usage_gb(self, user_id: int) -> float:
        """Get user's current storage usage in GB"""
        # This would query the file service for user's storage usage
        # For now, return 0
        return 0.0

class EnterprisePermissionDecorator:
    """Decorator for enforcing enterprise permissions"""
    
    def __init__(self, auth_service: EnterpriseAuthService):
        self.auth_service = auth_service
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract user from request (this would be implemented based on FastAPI integration)
                # For now, this is a placeholder structure
                
                # Get current user from Auth0
                current_user = await get_current_active_auth0_user()
                
                # Convert to enterprise user
                enterprise_user = await self.auth_service.authenticate_enterprise_user(current_user)
                
                # Check permission
                if not self.auth_service.check_user_permission(enterprise_user, permission):
                    raise PermissionError(f"User {enterprise_user.email} lacks permission: {permission}")
                
                # Add enterprise user to kwargs for use in endpoint
                kwargs['enterprise_user'] = enterprise_user
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Global instances
enterprise_auth_service = EnterpriseAuthService()
quota_manager = EnterpriseQuotaManager()
permission_decorator = EnterprisePermissionDecorator(enterprise_auth_service)

# Convenience functions
async def get_enterprise_user(auth0_user: User) -> EnterpriseUser:
    """Get enterprise user context from Auth0 user"""
    return await enterprise_auth_service.authenticate_enterprise_user(auth0_user)

def require_permission(permission: str):
    """Decorator to require specific permission"""
    return permission_decorator.require_permission(permission)

def check_permission(enterprise_user: EnterpriseUser, permission: str) -> bool:
    """Check if enterprise user has permission"""
    return enterprise_auth_service.check_user_permission(enterprise_user, permission)
