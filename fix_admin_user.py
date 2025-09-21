#!/usr/bin/env python3
"""
Fix admin user Auth0 linking
This script will help link your Auth0 authentication to your existing admin user.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from database import SessionLocal
from models import User as UserModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_admin_user():
    """Fix the admin user authentication issue"""
    db = SessionLocal()
    try:
        print("🔍 Checking current user situation...")
        
        # Find the original admin user
        admin_user = db.query(UserModel).filter(
            UserModel.email == 'mredard@gmail.com',
            UserModel.is_admin == True
        ).first()
        
        if admin_user:
            print(f"✅ Found admin user: {admin_user.username} (ID: {admin_user.id})")
            print(f"   - Email: {admin_user.email}")
            print(f"   - Is Admin: {admin_user.is_admin}")
            print(f"   - Auth0 ID: {admin_user.auth0_user_id}")
        else:
            print("❌ Admin user not found!")
            return
        
        # Check for any duplicate users created by Auth0
        all_mredard_users = db.query(UserModel).filter(
            UserModel.email == 'mredard@gmail.com'
        ).all()
        
        print(f"\n🔍 Found {len(all_mredard_users)} users with email 'mredard@gmail.com':")
        for i, user in enumerate(all_mredard_users):
            print(f"   {i+1}. ID: {user.id}, Username: {user.username}, Admin: {user.is_admin}, Auth0 ID: {user.auth0_user_id}")
        
        # If there are multiple users, merge them
        if len(all_mredard_users) > 1:
            print("\n🔧 Multiple users found - merging...")
            
            # Find which one has Auth0 ID and which is admin
            auth0_user = None
            admin_user_final = None
            
            for user in all_mredard_users:
                if user.auth0_user_id:
                    auth0_user = user
                if user.is_admin:
                    admin_user_final = user
            
            if auth0_user and admin_user_final and auth0_user.id != admin_user_final.id:
                print(f"🔄 Merging Auth0 user (ID: {auth0_user.id}) into admin user (ID: {admin_user_final.id})")
                
                # Transfer Auth0 ID to admin user
                admin_user_final.auth0_user_id = auth0_user.auth0_user_id
                admin_user_final.full_name = auth0_user.full_name or admin_user_final.full_name
                
                # Delete the duplicate Auth0 user
                db.delete(auth0_user)
                db.commit()
                
                print(f"✅ Successfully merged users! Admin user now has Auth0 ID: {admin_user_final.auth0_user_id}")
                
            elif admin_user_final and not admin_user_final.auth0_user_id:
                print("⚠️ Admin user exists but has no Auth0 ID. You need to authenticate once more to link accounts.")
                
        else:
            print("✅ Only one user found - this is good!")
            if not admin_user.auth0_user_id:
                print("⚠️ Admin user has no Auth0 ID. After the next login, the accounts should be linked automatically.")
        
        print("\n" + "="*60)
        print("🎯 SOLUTION:")
        print("1. Refresh your browser page completely (Ctrl+F5)")
        print("2. If you still see 403 Forbidden, log out and log back in")
        print("3. The system should now link your Auth0 account to your admin user")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fix_admin_user()
