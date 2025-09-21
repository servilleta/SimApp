#!/usr/bin/env python3
"""
Script to delete a user from the database
"""
import sys
import os
sys.path.append('/app')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import User

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
        print(f"Email: {user.email}")
        print(f"Auth0 ID: {user.auth0_user_id}")
        
        # Confirm deletion
        confirm = input("\nAre you sure you want to delete this user? (yes/no): ")
        
        if confirm.lower() == 'yes':
            db.delete(user)
            db.commit()
            print("\nUser deleted successfully!")
            print("You can now log in again and a new user will be created with proper email.")
        else:
            print("\nDeletion cancelled.")
    else:
        print("User not found!")
        
except Exception as e:
    print(f"Error deleting user: {e}")
    db.rollback()
finally:
    db.close() 