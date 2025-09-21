#!/usr/bin/env python3
"""
Script to update user email and full name in the database
"""
import sys
import os
sys.path.append('/app')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import User

# Get the email from command line argument
if len(sys.argv) < 2:
    print("Usage: python update_user_email.py <your-email@example.com> [optional-full-name]")
    sys.exit(1)

new_email = sys.argv[1]
new_full_name = sys.argv[2] if len(sys.argv) > 2 else None

# Create database connection
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/montecarlo')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

try:
    # Find the user
    user = db.query(User).filter(User.username == 'user_103991281505278778734').first()
    
    if user:
        print(f"Found user: {user.username}")
        print(f"Current email: {user.email}")
        print(f"Current full name: {user.full_name}")
        
        # Update the user
        user.email = new_email
        if new_full_name:
            user.full_name = new_full_name
        
        db.commit()
        
        print(f"\nUser updated successfully!")
        print(f"New email: {user.email}")
        print(f"New full name: {user.full_name}")
    else:
        print("User not found!")
        
except Exception as e:
    print(f"Error updating user: {e}")
    db.rollback()
finally:
    db.close() 