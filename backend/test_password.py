#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

from auth.service import get_user, get_password_hash, verify_password
from database import get_db

def main():
    try:
        # Get database session
        db = next(get_db())
        
        # Get admin user
        admin_user = get_user(db, 'admin')
        if not admin_user:
            print('❌ Admin user not found')
            return 1
        
        print(f'Admin user found: {admin_user.username}')
        print(f'Current hash: {admin_user.hashed_password[:20]}...')
        
        # Test password
        test_password = 'Demo123!MonteCarlo'
        
        # Generate new hash
        new_hash = get_password_hash(test_password)
        print(f'New hash: {new_hash[:20]}...')
        
        # Test verification with current hash
        current_verify = verify_password(test_password, admin_user.hashed_password)
        print(f'Current hash verification: {current_verify}')
        
        # Test verification with new hash
        new_verify = verify_password(test_password, new_hash)
        print(f'New hash verification: {new_verify}')
        
        # Update the password
        admin_user.hashed_password = new_hash
        db.commit()
        print('✅ Password updated in database')
        
        # Test verification again
        final_verify = verify_password(test_password, admin_user.hashed_password)
        print(f'Final verification: {final_verify}')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 