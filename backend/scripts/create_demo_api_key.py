#!/usr/bin/env python3
"""
Create Demo API Key Script

Creates a demo API key for testing the new secure API key system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, User
from services.api_key_service import APIKeyService

def create_demo_api_key():
    """Create a demo API key for testing."""
    
    # Create database tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Get or create demo user
        demo_user = db.query(User).filter(User.username == "demo_user").first()
        
        if not demo_user:
            print("Creating demo user...")
            demo_user = User(
                username="demo_user",
                email="demo@example.com",
                full_name="Demo User",
                hashed_password="dummy_hash",  # Not used for API keys
                is_admin=False
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)
            print(f"✅ Created demo user: {demo_user.username} (ID: {demo_user.id})")
        else:
            print(f"✅ Found existing demo user: {demo_user.username} (ID: {demo_user.id})")
        
        # Create API key for demo user
        print("Creating demo API key...")
        api_key, secret_key = APIKeyService.create_api_key(
            db=db,
            user_id=demo_user.id,
            name="Demo API Key",
            subscription_tier="professional"
        )
        
        # Format the full key
        full_key = f"{api_key.key_id}_{secret_key}"
        
        print("\n" + "="*60)
        print("🎉 DEMO API KEY CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"Key ID: {api_key.key_id}")
        print(f"Full API Key: {full_key}")
        print(f"Subscription Tier: {api_key.subscription_tier}")
        print(f"Monthly Requests: {api_key.monthly_requests}")
        print(f"Max Iterations: {api_key.max_iterations}")
        print(f"Max File Size: {api_key.max_file_size_mb}MB")
        print("="*60)
        print("⚠️  IMPORTANT: Store this API key securely!")
        print("   This is the only time the full key will be shown.")
        print("="*60)
        print("\n📝 Usage Example:")
        print(f'curl -H "Authorization: Bearer {full_key}" \\')
        print('     http://209.51.170.185:8000/monte-carlo-api/health')
        print()
        
        return full_key
        
    except Exception as e:
        print(f"❌ Error creating demo API key: {e}")
        db.rollback()
        return None
    finally:
        db.close()

if __name__ == "__main__":
    create_demo_api_key()
