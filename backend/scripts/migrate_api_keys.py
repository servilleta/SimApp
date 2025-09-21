#!/usr/bin/env python3
"""
API Key Migration Script

Migrates from hardcoded API keys to the new secure database-backed system.
Creates replacement API keys in the database and shows the new format.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, User
from services.api_key_service import APIKeyService

def migrate_hardcoded_keys():
    """Migrate from hardcoded API keys to secure database keys."""
    
    print("🔄 Starting API Key Migration...")
    print("=" * 60)
    
    # Create database tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Old hardcoded keys to replace
        old_keys = [
            {
                "old_key": "mc_test_demo123456789012345678901234",
                "client_name": "Demo Client",
                "tier": "starter"
            },
            {
                "old_key": "mc_live_prod123456789012345678901234", 
                "client_name": "Production Client",
                "tier": "professional"
            }
        ]
        
        migration_results = []
        
        for key_info in old_keys:
            print(f"\n📋 Migrating: {key_info['client_name']}")
            print(f"   Old Key: {key_info['old_key'][:20]}...")
            
            # Get or create user for this client
            username = f"api_client_{key_info['client_name'].lower().replace(' ', '_')}"
            user = db.query(User).filter(User.username == username).first()
            
            if not user:
                print(f"   Creating user: {username}")
                user = User(
                    username=username,
                    email=f"{username}@api.example.com",
                    full_name=key_info['client_name'],
                    hashed_password="api_key_auth_only",  # Not used for API auth
                    is_admin=False
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            
            # Create new secure API key
            api_key, secret_key = APIKeyService.create_api_key(
                db=db,
                user_id=user.id,
                name=f"{key_info['client_name']} - Migrated Key",
                subscription_tier=key_info['tier']
            )
            
            full_key = f"{api_key.key_id}_{secret_key}"
            
            migration_results.append({
                "old_key": key_info['old_key'],
                "new_key": full_key,
                "client_name": key_info['client_name'],
                "tier": key_info['tier'],
                "key_id": api_key.key_id
            })
            
            print(f"   ✅ Created new key: {api_key.key_id}")
        
        # Print migration summary
        print("\n" + "=" * 80)
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        for result in migration_results:
            print(f"\n📋 {result['client_name']} ({result['tier']} tier)")
            print(f"   OLD: {result['old_key']}")
            print(f"   NEW: {result['new_key']}")
            print(f"   Key ID: {result['key_id']}")
        
        print("\n" + "=" * 80)
        print("⚠️  IMPORTANT NEXT STEPS:")
        print("=" * 80)
        print("1. 🔒 SECURITY: Remove hardcoded keys from backend/api/v1/auth.py")
        print("2. 📧 NOTIFY: Send new API keys to clients securely")
        print("3. 🗑️  CLEANUP: The old keys are now invalid")
        print("4. ✅ TEST: Verify new keys work with the API")
        print("=" * 80)
        
        print("\n📝 Test the new keys:")
        for result in migration_results:
            print(f'curl -H "Authorization: Bearer {result["new_key"]}" \\')
            print('     http://209.51.170.185:8000/monte-carlo-api/health')
            print()
        
        return migration_results
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.rollback()
        return None
    finally:
        db.close()

if __name__ == "__main__":
    migrate_hardcoded_keys()
