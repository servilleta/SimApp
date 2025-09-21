"""
ENTERPRISE ORGANIZATION MANAGEMENT SERVICE
Phase 2 Week 6-7: Organization Management System

This service manages enterprise organizations with:
- Multi-tenant organization structure
- Subscription management
- User provisioning and management
- Organization-level settings and quotas
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from database import get_db
from models import User
from enterprise.auth_service import Organization, UserTier, UserRole, EnterpriseUser

logger = logging.getLogger(__name__)

@dataclass
class OrganizationSettings:
    """Organization-specific settings and configurations"""
    id: int
    organization_id: int
    
    # Security settings
    require_mfa: bool = False
    sso_enabled: bool = False
    sso_provider: Optional[str] = None
    sso_config: Dict[str, Any] = None
    
    # Simulation settings
    default_engine: str = "ultra"
    max_iterations_default: int = 10000
    allow_gpu_acceleration: bool = True
    
    # File settings
    allowed_file_types: List[str] = None
    max_file_size_mb: int = 100
    file_retention_days: int = 365
    
    # Billing settings
    billing_contact_email: str = ""
    billing_address: Dict[str, str] = None
    payment_method_id: Optional[str] = None
    
    # Notification settings
    notification_webhooks: List[str] = None
    email_notifications: bool = True
    slack_webhook: Optional[str] = None
    
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.xlsx', '.xls', '.csv']
        if self.billing_address is None:
            self.billing_address = {}
        if self.notification_webhooks is None:
            self.notification_webhooks = []
        if self.sso_config is None:
            self.sso_config = {}

@dataclass
class OrganizationUsage:
    """Organization usage statistics"""
    organization_id: int
    month: int
    year: int
    
    # Usage metrics
    total_simulations: int = 0
    total_compute_units: float = 0.0
    total_storage_gb: float = 0.0
    total_api_calls: int = 0
    unique_active_users: int = 0
    
    # Performance metrics
    avg_simulation_duration_seconds: float = 0.0
    simulation_success_rate: float = 100.0
    
    # Cost metrics
    estimated_cost_usd: float = 0.0
    
    last_updated: datetime = None

class EnterpriseOrganizationService:
    """Complete organization management service"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".OrganizationService")
    
    async def create_organization(self, 
                                name: str, 
                                domain: str, 
                                tier: UserTier,
                                admin_email: str) -> Organization:
        """Create new enterprise organization"""
        
        # Generate unique IDs
        org_id = await self.generate_organization_id()
        subscription_id = f"sub_{uuid.uuid4().hex[:12]}"
        
        organization = Organization(
            id=org_id,
            name=name,
            domain=domain,
            tier=tier,
            subscription_id=subscription_id,
            max_users=self.get_tier_limits(tier)['max_users'],
            max_simulations_per_month=self.get_tier_limits(tier)['max_simulations'],
            max_storage_gb=self.get_tier_limits(tier)['max_storage_gb'],
            sso_enabled=tier == UserTier.ENTERPRISE,
            created_at=datetime.utcnow()
        )
        
        # Create default organization settings
        settings = OrganizationSettings(
            id=await self.generate_settings_id(),
            organization_id=org_id,
            billing_contact_email=admin_email,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Save to database (placeholder)
        await self.save_organization(organization)
        await self.save_organization_settings(settings)
        
        self.logger.info(f"✅ [ORG_SERVICE] Created organization: {name} (ID: {org_id}, Tier: {tier.value})")
        
        return organization
    
    async def get_organization(self, organization_id: int) -> Optional[Organization]:
        """Get organization by ID"""
        # This would query the database
        # For demo purposes, return a mock organization
        
        if organization_id == 1:
            return Organization(
                id=1,
                name="Demo Organization",
                domain="demo.com",
                tier=UserTier.PROFESSIONAL,
                subscription_id="sub_demo_123",
                max_users=50,
                max_simulations_per_month=5000,
                max_storage_gb=500,
                sso_enabled=False,
                created_at=datetime.utcnow() - timedelta(days=30)
            )
        
        return None
    
    async def get_organization_by_domain(self, domain: str) -> Optional[Organization]:
        """Get organization by email domain"""
        # This would query the database by domain
        return await self.get_organization(1)  # Placeholder
    
    async def update_organization_tier(self, organization_id: int, new_tier: UserTier) -> Organization:
        """Update organization tier (upgrade/downgrade)"""
        
        organization = await self.get_organization(organization_id)
        if not organization:
            raise ValueError(f"Organization {organization_id} not found")
        
        # Update tier and related quotas
        old_tier = organization.tier
        organization.tier = new_tier
        
        # Update quotas based on new tier
        tier_limits = self.get_tier_limits(new_tier)
        organization.max_users = tier_limits['max_users']
        organization.max_simulations_per_month = tier_limits['max_simulations']
        organization.max_storage_gb = tier_limits['max_storage_gb']
        organization.sso_enabled = new_tier == UserTier.ENTERPRISE
        
        await self.save_organization(organization)
        
        self.logger.info(f"✅ [ORG_SERVICE] Updated organization {organization_id} tier: {old_tier.value} → {new_tier.value}")
        
        return organization
    
    async def get_organization_usage(self, organization_id: int, month: int, year: int) -> OrganizationUsage:
        """Get organization usage statistics for billing"""
        
        # This would aggregate usage data from various services
        # For demo purposes, return mock data
        
        usage = OrganizationUsage(
            organization_id=organization_id,
            month=month,
            year=year,
            total_simulations=150,
            total_compute_units=1250.5,
            total_storage_gb=45.2,
            total_api_calls=15000,
            unique_active_users=8,
            avg_simulation_duration_seconds=45.2,
            simulation_success_rate=98.7,
            estimated_cost_usd=2847.50,
            last_updated=datetime.utcnow()
        )
        
        return usage
    
    async def get_organization_settings(self, organization_id: int) -> OrganizationSettings:
        """Get organization settings"""
        
        # This would query the organization_settings table
        # For demo purposes, return default settings
        
        return OrganizationSettings(
            id=1,
            organization_id=organization_id,
            require_mfa=False,
            sso_enabled=False,
            default_engine="ultra",
            max_iterations_default=10000,
            allow_gpu_acceleration=True,
            allowed_file_types=['.xlsx', '.xls', '.csv'],
            max_file_size_mb=200,
            file_retention_days=365,
            billing_contact_email="admin@demo.com",
            email_notifications=True,
            created_at=datetime.utcnow() - timedelta(days=30),
            updated_at=datetime.utcnow()
        )
    
    async def update_organization_settings(self, organization_id: int, settings_update: Dict[str, Any]) -> OrganizationSettings:
        """Update organization settings"""
        
        current_settings = await self.get_organization_settings(organization_id)
        
        # Update settings
        for key, value in settings_update.items():
            if hasattr(current_settings, key):
                setattr(current_settings, key, value)
        
        current_settings.updated_at = datetime.utcnow()
        
        await self.save_organization_settings(current_settings)
        
        self.logger.info(f"✅ [ORG_SERVICE] Updated settings for organization {organization_id}: {list(settings_update.keys())}")
        
        return current_settings
    
    def get_tier_limits(self, tier: UserTier) -> Dict[str, int]:
        """Get limits for organization tier"""
        
        tier_limits = {
            UserTier.ENTERPRISE: {
                'max_users': 1000,
                'max_simulations': 100000,
                'max_storage_gb': 10000,
                'api_rate_limit': 10000
            },
            UserTier.PROFESSIONAL: {
                'max_users': 100,
                'max_simulations': 10000,
                'max_storage_gb': 1000,
                'api_rate_limit': 2000
            },
            UserTier.STANDARD: {
                'max_users': 10,
                'max_simulations': 1000,
                'max_storage_gb': 100,
                'api_rate_limit': 500
            },
            UserTier.TRIAL: {
                'max_users': 3,
                'max_simulations': 50,
                'max_storage_gb': 10,
                'api_rate_limit': 100
            }
        }
        
        return tier_limits.get(tier, tier_limits[UserTier.STANDARD])
    
    async def generate_organization_id(self) -> int:
        """Generate unique organization ID"""
        # This would query the database for the next available ID
        return int(datetime.utcnow().timestamp()) % 1000000
    
    async def generate_settings_id(self) -> int:
        """Generate unique settings ID"""
        return int(datetime.utcnow().timestamp()) % 1000000
    
    async def save_organization(self, organization: Organization):
        """Save organization to database"""
        # Placeholder for database save
        self.logger.info(f"💾 [ORG_SERVICE] Saved organization: {organization.name}")
    
    async def save_organization_settings(self, settings: OrganizationSettings):
        """Save organization settings to database"""
        # Placeholder for database save
        self.logger.info(f"💾 [ORG_SERVICE] Saved settings for organization: {settings.organization_id}")
    
    async def get_organization_users_count(self, organization_id: int) -> int:
        """Get count of users in organization"""
        # This would query the database
        return 1  # Placeholder
    
    async def check_organization_limits(self, organization_id: int) -> Dict[str, Any]:
        """Check if organization is within its limits"""
        
        organization = await self.get_organization(organization_id)
        if not organization:
            return {"error": "Organization not found"}
        
        current_users = await self.get_organization_users_count(organization_id)
        current_usage = await self.get_organization_usage(
            organization_id, 
            datetime.utcnow().month, 
            datetime.utcnow().year
        )
        
        limits_check = {
            "users": {
                "current": current_users,
                "limit": organization.max_users,
                "percentage": (current_users / organization.max_users) * 100,
                "within_limit": current_users <= organization.max_users
            },
            "simulations": {
                "current": current_usage.total_simulations,
                "limit": organization.max_simulations_per_month,
                "percentage": (current_usage.total_simulations / organization.max_simulations_per_month) * 100,
                "within_limit": current_usage.total_simulations <= organization.max_simulations_per_month
            },
            "storage": {
                "current_gb": current_usage.total_storage_gb,
                "limit_gb": organization.max_storage_gb,
                "percentage": (current_usage.total_storage_gb / organization.max_storage_gb) * 100,
                "within_limit": current_usage.total_storage_gb <= organization.max_storage_gb
            }
        }
        
        return limits_check

# Global service instance
organization_service = EnterpriseOrganizationService()

# Convenience functions
async def get_organization(organization_id: int) -> Optional[Organization]:
    """Get organization by ID"""
    return await organization_service.get_organization(organization_id)

async def get_organization_settings(organization_id: int) -> OrganizationSettings:
    """Get organization settings"""
    return await organization_service.get_organization_settings(organization_id)

async def check_organization_limits(organization_id: int) -> Dict[str, Any]:
    """Check organization limits"""
    return await organization_service.check_organization_limits(organization_id)
