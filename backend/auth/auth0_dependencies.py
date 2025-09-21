from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import jwt, JWTError
import requests
import logging
from typing import Optional
from datetime import datetime, timezone
from functools import lru_cache
import time

from config import settings
from database import get_db
from models import User as UserModel
from auth import service as auth_service

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer()

def get_auth0_public_key():
    """Fetch Auth0 public key for JWT verification"""
    try:
        jwks_url = f"https://{settings.AUTH0_DOMAIN}/.well-known/jwks.json"
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch Auth0 public key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable"
        )

def verify_auth0_token(token: str):
    """Verify Auth0 JWT token and return payload"""
    try:
        # Get the public key
        jwks = get_auth0_public_key()
        
        # Get the token header
        unverified_header = jwt.get_unverified_header(token)
        
        # Find the correct key
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break
        
        if not rsa_key:
            logger.error("Unable to find appropriate key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token header"
            )
        
        # Verify the token
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=settings.AUTH0_ALGORITHMS,
            audience=settings.AUTH0_AUDIENCE,
            issuer=f"https://{settings.AUTH0_DOMAIN}/"
        )
        
        return payload
        
    except JWTError as e:
        logger.error(f"🚨 AUTH FAILURE - JWT verification failed: {e}")
        logger.error(f"🚨 AUTH FAILURE - Token (first 50 chars): {token[:50]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"🚨 AUTH FAILURE - Token verification error: {e}")
        logger.error(f"🚨 AUTH FAILURE - Token (first 50 chars): {token[:50]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed"
        )

# Cache userinfo for 5 minutes to avoid rate limiting
@lru_cache(maxsize=128)
def get_cached_auth0_user_info(token: str, cache_key: str):
    """Cached version of get_auth0_user_info to avoid rate limiting"""
    try:
        userinfo_url = f"https://{settings.AUTH0_DOMAIN}/userinfo"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(userinfo_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch user info from Auth0: {e}")
        return {}

def get_auth0_user_info(token: str):
    """Fetch user info from Auth0 userinfo endpoint with caching"""
    # Create a cache key based on current 5-minute window
    cache_key = str(int(time.time() / 300))  # 5-minute windows
    return get_cached_auth0_user_info(token, cache_key)

def get_or_create_user_from_auth0(db: Session, auth0_payload: dict) -> UserModel:
    """Get or create user from Auth0 payload - RESTRICTED FOR PRIVATE LAUNCH"""
    try:
        # Extract user info from Auth0 token
        auth0_user_id = auth0_payload.get("sub")  # Auth0 user ID
        email = auth0_payload.get("email")
        name = auth0_payload.get("name")
        nickname = auth0_payload.get("nickname")
        
        if not auth0_user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID"
            )
        
        # Try to find existing user by Auth0 ID
        user = db.query(UserModel).filter(UserModel.auth0_user_id == auth0_user_id).first()
        
        if not user:
            # Try to find by email if provided
            if email:
                user = db.query(UserModel).filter(UserModel.email == email).first()
                if user:
                    # Update existing user with Auth0 ID (preserve existing admin status)
                    user.auth0_user_id = auth0_user_id
                    # Preserve existing admin status and other important fields
                    db.commit()
                    logger.info(f"Updated existing user {email} with Auth0 ID (preserved admin status: {user.is_admin})")
        
        if not user:
            # Create new user from Auth0 (enabled for normal operations)
            username = nickname or email or f"user_{auth0_user_id.split('|')[-1]}"
            
            # Ensure username is unique
            counter = 1
            original_username = username
            while db.query(UserModel).filter(UserModel.username == username).first():
                username = f"{original_username}_{counter}"
                counter += 1
            
            user = UserModel(
                username=username,
                email=email or f"{username}@auth0.local",
                auth0_user_id=auth0_user_id,
                full_name=name,
                disabled=False,
                is_admin=False,  # New users are not admin by default
                hashed_password=""  # Not needed for Auth0 users
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new user from Auth0: {username} ({email})")
        
        return user
        
    except HTTPException:
        # Re-raise HTTP exceptions (like the 403 above)
        raise
    except Exception as e:
        logger.error(f"Error getting/creating user from Auth0: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User authentication failed"
        )

async def get_current_auth0_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> UserModel:
    """
    Dependency to get the current authenticated user from Auth0 JWT token.
    
    Args:
        credentials: JWT token from Authorization header
        db: Database session
        
    Returns:
        UserModel: The authenticated user
        
    Raises:
        HTTPException: If token is invalid or user creation fails
    """
    logger.info(f"🔐 AUTH START - Token validation starting")
    logger.info(f"🔐 AUTH START - Token present: {bool(credentials and credentials.credentials)}")
    
    if not credentials or not credentials.credentials:
        logger.error(f"🚨 AUTH FAILURE - No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication token provided"
        )
    
    try:
        # Verify Auth0 token
        auth0_payload = verify_auth0_token(credentials.credentials)
        
        # Get additional user info from Auth0 userinfo endpoint
        userinfo = get_auth0_user_info(credentials.credentials)
        
        # Merge userinfo with token payload (userinfo takes precedence)
        auth0_payload.update(userinfo)
        
        logger.info(f"Auth0 token verified for user: {auth0_payload.get('email', 'unknown')}")
        logger.info(f"Full Auth0 payload: {auth0_payload}")
        
        # Get or create user
        user = get_or_create_user_from_auth0(db, auth0_payload)
        logger.info(f"✅ AUTH SUCCESS - User authenticated: {user.username} (id: {user.id}, is_admin: {user.is_admin})")
        
        return user
        
    except HTTPException as e:
        logger.error(f"🚨 AUTH FAILURE - HTTPException: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        logger.error(f"🚨 AUTH FAILURE - Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def get_current_active_auth0_user(
    current_user: UserModel = Depends(get_current_auth0_user)
) -> UserModel:
    """
    Dependency to get the current authenticated and active Auth0 user.
    
    Args:
        current_user: The current user from get_current_auth0_user
        
    Returns:
        UserModel: The active user
        
    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        logger.warning(f"Auth0 user '{current_user.username}' is disabled")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Account disabled"
        )
    
    return current_user

async def get_current_admin_auth0_user(
    current_user: UserModel = Depends(get_current_active_auth0_user)
) -> UserModel:
    """
    Dependency to get the current authenticated Auth0 admin user.
    
    Args:
        current_user: The current active user
        
    Returns:
        UserModel: The admin user
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        logger.warning(f"Auth0 user '{current_user.username}' attempted admin action but is not admin")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user

def get_db_user_by_auth0_sub(db: Session, auth0_sub: str) -> Optional[UserModel]:
    """Get a database user by their Auth0 sub ID"""
    return db.query(UserModel).filter(UserModel.auth0_sub == auth0_sub).first() 