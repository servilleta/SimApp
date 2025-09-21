"""
ENTERPRISE AUTHENTICATION DEMO
Phase 2 Week 6-7: Enhanced OAuth 2.0 + RBAC Demo

This script demonstrates the new enterprise authentication features:
- Enhanced user context with organization information
- Role-based access control (RBAC)
- Permission checking
- Quota management
"""

import sys
import os
sys.path.append('/app')
os.chdir('/app')

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_enterprise_authentication():
    """Demonstrate enterprise authentication features"""
    
    print("🏢 ENTERPRISE AUTHENTICATION DEMO")
    print("=" * 50)
    
    try:
        # Import enterprise services
        from enterprise.auth_service import (
            EnterpriseAuthService, 
            RoleBasedAccessControl,
            UserRole, 
            UserTier,
            enterprise_auth_service
        )
        from enterprise.organization_service import organization_service
        from models import User
        from database import get_db
        
        print("✅ Enterprise authentication services imported successfully")
        
        # Get a test user from database
        db = next(get_db())
        test_user = db.query(User).first()
        db.close()
        
        if not test_user:
            print("❌ No users found in database")
            return
        
        print(f"🔍 Testing with user: {test_user.email}")
        
        # Test 1: Convert Auth0 user to Enterprise user
        print("\n1️⃣ TESTING: Auth0 to Enterprise User Conversion")
        print("-" * 40)
        
        enterprise_user = await enterprise_auth_service.authenticate_enterprise_user(test_user)
        
        print(f"✅ Enterprise User Created:")
        print(f"   Email: {enterprise_user.email}")
        print(f"   Full Name: {enterprise_user.full_name}")
        print(f"   Organization: {enterprise_user.organization.name}")
        print(f"   Tier: {enterprise_user.organization.tier.value}")
        print(f"   Roles: {[role.value for role in enterprise_user.roles]}")
        print(f"   Permissions: {len(enterprise_user.permissions)} permissions")
        
        # Test 2: Role-Based Access Control
        print("\n2️⃣ TESTING: Role-Based Access Control")
        print("-" * 40)
        
        rbac = RoleBasedAccessControl()
        
        test_permissions = [
            'simulation.create',
            'simulation.delete', 
            'organization.manage',
            'billing.view',
            'admin.users'
        ]
        
        for permission in test_permissions:
            has_permission = rbac.check_permission(enterprise_user.roles, permission)
            status_icon = "✅" if has_permission else "❌"
            print(f"   {status_icon} {permission}: {'ALLOWED' if has_permission else 'DENIED'}")
        
        # Test 3: Quota Management
        print("\n3️⃣ TESTING: Quota Management")
        print("-" * 40)
        
        print(f"✅ User Quotas:")
        for quota_name, quota_value in enterprise_user.quotas.items():
            print(f"   {quota_name}: {quota_value}")
        
        # Test 4: Organization Information
        print("\n4️⃣ TESTING: Organization Information")
        print("-" * 40)
        
        org = enterprise_user.organization
        print(f"✅ Organization Details:")
        print(f"   ID: {org.id}")
        print(f"   Name: {org.name}")
        print(f"   Domain: {org.domain}")
        print(f"   Tier: {org.tier.value}")
        print(f"   Max Users: {org.max_users}")
        print(f"   Max Simulations/Month: {org.max_simulations_per_month}")
        print(f"   Max Storage: {org.max_storage_gb}GB")
        print(f"   SSO Enabled: {org.sso_enabled}")
        
        # Test 5: Organization Usage
        print("\n5️⃣ TESTING: Organization Usage Statistics")
        print("-" * 40)
        
        current_month = datetime.utcnow().month
        current_year = datetime.utcnow().year
        
        usage = await organization_service.get_organization_usage(org.id, current_month, current_year)
        
        print(f"✅ Usage Statistics ({current_month}/{current_year}):")
        print(f"   Total Simulations: {usage.total_simulations}")
        print(f"   Compute Units: {usage.total_compute_units}")
        print(f"   Storage Used: {usage.total_storage_gb}GB")
        print(f"   Active Users: {usage.unique_active_users}")
        print(f"   Success Rate: {usage.simulation_success_rate}%")
        print(f"   Estimated Cost: ${usage.estimated_cost_usd}")
        
        # Test 6: Organization Limits Check
        print("\n6️⃣ TESTING: Organization Limits Check")
        print("-" * 40)
        
        limits_check = await organization_service.check_organization_limits(org.id)
        
        for category, data in limits_check.items():
            if isinstance(data, dict) and 'within_limit' in data:
                status_icon = "✅" if data['within_limit'] else "⚠️"
                current = data.get('current', data.get('current_gb', 'N/A'))
                limit = data.get('limit', data.get('limit_gb', 'N/A'))
                percentage = data.get('percentage', 0)
                print(f"   {status_icon} {category.title()}: {current}/{limit} ({percentage:.1f}%)")
        
        print("\n🎉 ENTERPRISE AUTHENTICATION DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        return {
            "success": True,
            "enterprise_user": enterprise_user,
            "organization": org,
            "usage": usage,
            "limits_check": limits_check
        }
        
    except Exception as e:
        print(f"\n❌ ENTERPRISE AUTHENTICATION DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(demo_enterprise_authentication())
    
    if result["success"]:
        print("\n🚀 Enterprise authentication system is ready for production!")
    else:
        print(f"\n💥 Demo failed: {result['error']}")
