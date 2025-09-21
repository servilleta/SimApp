#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

from modules.auth.service import AuthService
from database import get_db
from models import User as UserModel

def main():
    try:
        # Get database session
        db = next(get_db())
        
        # Delete old admin user if exists
        old_admin = db.query(UserModel).filter(UserModel.username == 'admin').first()
        if old_admin:
            db.delete(old_admin)
            db.commit()
            print('🗑️ Old admin user deleted')
        
        # Create new admin user using modular AuthService
        auth_service = AuthService(secret_key='your-secret-key-needs-to-be-changed')
        new_admin_data = {
            'username': 'admin',
            'email': 'admin@example.com',
            'full_name': 'Admin User',
            'password': 'admin123',
            'is_admin': True
        }
        new_admin = db.query(UserModel).filter(UserModel.username == 'admin').first()
        if not new_admin:
            # Use the modular service's password hashing
            hashed_password = auth_service._get_password_hash(new_admin_data['password'])
            db_user = UserModel(
                username=new_admin_data['username'],
                email=new_admin_data['email'],
                full_name=new_admin_data['full_name'],
                hashed_password=hashed_password,
                disabled=False,
                is_admin=True
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            print('✅ New admin user created successfully')
            print(f"Username: {db_user.username}")
            print(f"Password: {new_admin_data['password']}")
        else:
            print('⚠️ Admin user already exists after attempted recreation')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 