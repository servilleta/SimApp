#!/usr/bin/env python3

import sys
import os
sys.path.append('./backend')

from auth.service import get_user, get_password_hash, verify_password
from database import get_db
from models import User as UserModel

def update_admin_password():
    """Update admin password to the new secure one"""
    try:
        # Get database session
        db = next(get_db())
        
        # Get admin user
        admin_user = db.query(UserModel).filter(UserModel.username == 'admin').first()
        if not admin_user:
            print('❌ Admin user not found')
            return 1
        
        print(f'✅ Found admin user: {admin_user.username}')
        
        # New secure password
        new_password = 'NewSecurePassword123!'
        
        # Hash the new password
        new_hash = get_password_hash(new_password)
        print(f'🔐 Generated new password hash')
        
        # Update the password in database
        admin_user.hashed_password = new_hash
        db.commit()
        print('✅ Password updated in database')
        
        # Test the new password
        test_verify = verify_password(new_password, admin_user.hashed_password)
        print(f'🧪 Password verification test: {test_verify}')
        
        if test_verify:
            print('🎉 SUCCESS! Admin password updated successfully')
            print(f'👤 Username: admin')
            print(f'🔑 New Password: {new_password}')
            return 0
        else:
            print('❌ Password verification failed')
            return 1
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(update_admin_password())
