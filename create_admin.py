#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

from auth.service import create_user
from auth.schemas import UserCreate
from database import get_db

def main():
    try:
        # Get database session
        db = next(get_db())
        
        # Create admin user
        user_data = UserCreate(
            username='admin',
            email='admin@example.com',
            password='Demo123!MonteCarlo',
            password_confirm='Demo123!MonteCarlo'
        )
        
        user = create_user(db, user_data)
        print(f'✅ Admin user created successfully: {user.username} ({user.email})')
        
    except Exception as e:
        print(f'❌ Error creating admin user: {e}')
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 