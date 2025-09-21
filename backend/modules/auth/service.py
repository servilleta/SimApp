"""
Authentication service module - implements AuthServiceProtocol

Handles user authentication, JWT tokens, user management.
This service can be easily extracted to a microservice later.
"""

import logging
from typing import Optional, Dict
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..base import BaseService, AuthServiceProtocol
from models import User as UserModel
from modules.auth.schemas import UserCreate, TokenData


logger = logging.getLogger(__name__)


class AuthService(BaseService, AuthServiceProtocol):
    """Authentication service implementation"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256", 
                 access_token_expire_minutes: int = 30):
        super().__init__("auth")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        
        # Password hashing
        try:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        except Exception as e:
            logger.warning(f"Bcrypt setup issue: {e}. Falling back to simpler bcrypt configuration.")
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__min_rounds=4)
        
        if secret_key == "your-secret-key-needs-to-be-changed":
            logger.warning("Using default SECRET_KEY. Please change this in your configuration for security.")
    
    async def initialize(self) -> None:
        """Initialize the auth service"""
        await super().initialize()
        logger.info("Auth service initialized")
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def _get_password_hash(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    async def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user with email/password"""
        # This will be injected by the container
        db = self._get_db_session()
        
        # Look up user by email or username
        user = db.query(UserModel).filter(
            (UserModel.email == email) | (UserModel.username == email)
        ).first()
        
        if not user or user.disabled:
            logger.info(f"Authentication failed for {email}: user not found or disabled")
            return None
        
        if not self._verify_password(password, user.hashed_password):
            logger.info(f"Authentication failed for {email}: invalid password")
            return None
        
        logger.info(f"Authentication successful for user {user.username}")
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin
        }
    
    async def create_access_token(self, data: dict, expires_delta: Optional[int] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + timedelta(minutes=expires_delta)
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    async def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: Optional[str] = payload.get("sub")
            
            if username is None:
                logger.warning("No username found in token payload")
                return None
            
            # Get user details
            db = self._get_db_session()
            user = db.query(UserModel).filter(UserModel.username == username).first()
            
            if not user or user.disabled:
                logger.warning(f"Token verification failed: user {username} not found or disabled")
                return None
            
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_admin": user.is_admin
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except JWTError as e:
            logger.error(f"JWT decode error: {str(e)}")
            return None
    
    async def create_user(self, user_data: dict) -> Dict:
        """Create new user account"""
        db = self._get_db_session()
        
        # Check if username already exists
        existing_user = db.query(UserModel).filter(UserModel.username == user_data["username"]).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists (if provided)
        if user_data.get("email"):
            existing_email = db.query(UserModel).filter(UserModel.email == user_data["email"]).first()
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        # Hash password
        hashed_password = self._get_password_hash(user_data["password"])
        
        # Create user
        db_user = UserModel(
            username=user_data["username"],
            email=user_data.get("email"),
            full_name=user_data.get("full_name"),
            hashed_password=hashed_password,
            disabled=user_data.get("disabled", False),
            is_admin=user_data.get("is_admin", False)
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"User '{user_data['username']}' created successfully with id {db_user.id}")
        
        return {
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email,
            "full_name": db_user.full_name,
            "is_admin": db_user.is_admin,
            "disabled": db_user.disabled
        }
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        db = self._get_db_session()
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        
        if not user:
            return None
        
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
            "disabled": user.disabled
        }
    
    async def update_user(self, user_id: int, update_data: dict) -> Dict:
        """Update user information"""
        db = self._get_db_session()
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields
        for key, value in update_data.items():
            if key == "password":
                user.hashed_password = self._get_password_hash(value)
            elif hasattr(user, key):
                setattr(user, key, value)
        
        db.commit()
        db.refresh(user)
        
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
            "disabled": user.disabled
        }
    
    def _get_db_session(self) -> Session:
        """Get database session - will be injected by container"""
        # This will be set by the service container
        if not hasattr(self, '_db_session'):
            raise RuntimeError("Database session not injected")
        return self._db_session
    
    def set_db_session(self, db_session: Session):
        """Set database session (called by container)"""
        self._db_session = db_session 