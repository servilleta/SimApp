#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

from database import get_db
from models import User as UserModel
from modules.auth.service import AuthService

def main():
    try:
        # Get database session
        db = next(get_db())
        
        # Query admin user
        admin = db.query(UserModel).filter(UserModel.username == 'admin').first()
        
        if admin:
            print("✅ Admin user found in database:")
            print(f"  ID: {admin.id}")
            print(f"  Username: {admin.username}")
            print(f"  Email: {admin.email}")
            print(f"  Full Name: {admin.full_name}")
            print(f"  Is Admin: {admin.is_admin}")
            print(f"  Disabled: {admin.disabled}")
            print(f"  Hashed Password: {admin.hashed_password}")
            print(f"  Hash Length: {len(admin.hashed_password) if admin.hashed_password else 0}")
            
            # Test password verification
            auth_service = AuthService(secret_key='your-secret-key-needs-to-be-changed')
            test_password = 'Demo123!MonteCarlo'
            
            print(f"\n🔍 Testing password verification:")
            print(f"  Test Password: {test_password}")
            
            # Test with modular service
            is_valid = auth_service._verify_password(test_password, admin.hashed_password)
            print(f"  Modular AuthService verification: {is_valid}")
            
            # Test with legacy service for comparison
            try:
                from auth.service import verify_password
                legacy_valid = verify_password(test_password, admin.hashed_password)
                print(f"  Legacy auth verification: {legacy_valid}")
            except Exception as e:
                print(f"  Legacy auth verification failed: {e}")
            
            # Test authentication flow
            print(f"\n🔍 Testing full authentication flow:")
            auth_result = auth_service.authenticate_user('admin', test_password)
            print(f"  AuthService.authenticate_user result: {auth_result}")
            
        else:
            print("❌ No admin user found in database")
            
            # List all users
            all_users = db.query(UserModel).all()
            print(f"\n📋 All users in database ({len(all_users)}):")
            for user in all_users:
                print(f"  - {user.username} (admin: {user.is_admin}, disabled: {user.disabled})")
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 