from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging

from auth import service as auth_service # Use absolute import
from auth import schemas as auth_schemas # Use absolute import
from typing import Optional # Ensure Optional is imported
from database import get_db # Added: Import get_db
from models import User as UserModel

# Point to your actual token URL. Assuming auth_router is prefixed with /api/auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Security scheme for JWT tokens
security = HTTPBearer()

logger = logging.getLogger(__name__)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> UserModel:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Args:
        credentials: JWT token from Authorization header
        db: Database session
        
    Returns:
        UserModel: The authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    logger.info(f"get_current_user: Token validation starting, token length: {len(credentials.credentials) if credentials.credentials else 0}")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decode the token
    token_data = auth_service.decode_token(credentials.credentials)
    logger.info(f"get_current_user: Token decoded, username from token: {token_data.username if token_data else 'None'}")
    if not token_data or not token_data.username:
        logger.warning("get_current_user: Invalid token data or missing username")
        raise credentials_exception
    
    # Get user from database
    user = auth_service.get_user(db, username=token_data.username)
    logger.info(f"get_current_user: User lookup complete, found: {user is not None}")
    if user is None:
        logger.warning(f"get_current_user: User '{token_data.username}' not found in database")
        raise credentials_exception
    logger.info(f"get_current_user: Returning user '{user.username}' (id: {user.id}, is_admin: {user.is_admin})")
    return user

async def get_current_active_user(current_user: UserModel = Depends(get_current_user)) -> UserModel:
    """
    Dependency to get the current authenticated and active user.
    
    Args:
        current_user: The current user from get_current_user
        
    Returns:
        UserModel: The active user
        
    Raises:
        HTTPException: If user is disabled
    """
    logger.info(f"get_current_active_user: Checking if user '{current_user.username}' is active (disabled: {current_user.disabled})")
    if current_user.disabled:
        logger.warning(f"get_current_active_user: User '{current_user.username}' is disabled")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    logger.info(f"get_current_active_user: User '{current_user.username}' is active and authenticated")
    return current_user

async def get_current_admin_user(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """
    Dependency to get the current authenticated admin user.
    
    Args:
        current_user: The current active user
        
    Returns:
        UserModel: The admin user
        
    Raises:
        HTTPException: If user is not an admin
    """
    logger.info(f"require_admin: Checking admin privileges for user '{current_user.username}' (is_admin: {current_user.is_admin})")
    if not current_user.is_admin:
        logger.warning(f"require_admin: User '{current_user.username}' attempted admin action but is not admin")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin access required."
        )
    logger.info(f"require_admin: User '{current_user.username}' has admin privileges")
    return current_user 