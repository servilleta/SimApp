#!/usr/bin/env python3
"""
Quick script to manually update user subscription when Stripe webhooks aren't working
"""

import sys
import os
sys.path.append('/app')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import UserSubscription, User
from config import settings
from datetime import datetime, timezone, timedelta

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

def update_user_subscription(email: str, plan: str = "professional"):
    """Update user subscription manually"""
    try:
        # Find user by email
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"❌ User not found: {email}")
            return
        
        print(f"👤 Found user: {user.username} (ID: {user.id})")
        
        # Find or create subscription
        subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == user.id
        ).first()
        
        if not subscription:
            subscription = UserSubscription(user_id=user.id)
            db.add(subscription)
            print(f"🆕 Created new subscription for user {user.id}")
        else:
            print(f"📋 Found existing subscription: {subscription.tier}")
        
        # Update subscription
        subscription.tier = plan
        subscription.status = "active"
        subscription.current_period_start = datetime.now(timezone.utc)
        subscription.current_period_end = datetime.now(timezone.utc) + timedelta(days=30)
        
        db.commit()
        print(f"✅ Updated subscription for {email} to '{plan}' plan")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        email = sys.argv[1]
        plan = sys.argv[2] if len(sys.argv) > 2 else "professional"
        update_user_subscription(email, plan)
    else:
        update_user_subscription("mredard@gmx.com", "professional")
