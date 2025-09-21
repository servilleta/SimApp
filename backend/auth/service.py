import logging
# Authentication business logic will go here.
# e.g., functions to create users, verify passwords, generate JWT tokens.

from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone # Added timezone
from fastapi import HTTPException, status, Depends # Added Depends
from sqlalchemy.orm import Session # Added Session

from auth.schemas import User as UserSchema, UserInDB as UserInDBSchema, UserCreate, TokenData # Aliased Pydantic models
from models import User as UserModel # Import from main models.py
from database import get_db # Added get_db
from config import settings # Assuming you have a settings module for SECRET_KEY etc.

# --- Security Settings (Ideally from config.py or environment variables) ---
# Make sure to set a strong, unique SECRET_KEY in your environment or config
SECRET_KEY = getattr(settings, "SECRET_KEY", "your-secret-key-needs-to-be-changed")
ALGORITHM = getattr(settings, "ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 30)

# Support both local (HS256) and Auth0 (RS256) algorithms
SUPPORTED_ALGORITHMS = ["HS256", "RS256"]

logger = logging.getLogger(__name__)

if SECRET_KEY == "your-secret-key-needs-to-be-changed":
    logger.warning("Using default SECRET_KEY. Please change this in your configuration for security.")

# Password Hashing
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception as e:
    logger.warning(f"Bcrypt setup issue: {e}. Falling back to simpler bcrypt configuration.")
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__min_rounds=4)

# In-memory user store (Replace with a database in production)
# Structure: {username: UserInDB_instance}
# fake_users_db: dict[str, UserInDB] = {} # REMOVED


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user(db: Session, username: str) -> Optional[UserModel]:
    user = db.query(UserModel).filter(UserModel.username == username).first()
    logger.info(f"get_user: Looked up username '{username}', found: {user is not None}")
    return user

def create_user(db: Session, user_in: UserCreate) -> UserModel:
    logger.info(f"create_user: Attempting to create user: {user_in.username}, email: {user_in.email}")
    db_user_by_username = get_user(db, username=user_in.username)
    if db_user_by_username:
        logger.warning(f"create_user: Username '{user_in.username}' already exists.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    if user_in.email:
        db_user_by_email = db.query(UserModel).filter(UserModel.email == user_in.email).first()
        logger.info(f"create_user: Email lookup for '{user_in.email}', found: {db_user_by_email is not None}")
        if db_user_by_email:
            logger.warning(f"create_user: Email '{user_in.email}' already exists.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
        )
    hashed_password = get_password_hash(user_in.password)
    db_user = UserModel(
        username=user_in.username,
        email=user_in.email,
        full_name=getattr(user_in, 'full_name', None),
        hashed_password=hashed_password,
        disabled=getattr(user_in, 'disabled', False),
        is_admin=getattr(user_in, 'is_admin', False)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"create_user: User '{user_in.username}' created successfully with id {db_user.id}")
    return db_user

def update_user(db: Session, user: UserModel, update_data: dict) -> UserModel:
    for key, value in update_data.items():
        if key == "password":
            user.hashed_password = get_password_hash(value)
        elif hasattr(user, key):
            setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, username: str, password: str) -> Optional[UserModel]: # Changed signature, return UserModel
    user = get_user(db, username=username)
    if not user or user.disabled: # Check if user.disabled is True
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[TokenData]:
    logger.info(f"decode_token: Starting token decode, token length: {len(token) if token else 0}")
    try:
        # Try to decode with all supported algorithms
        payload = jwt.decode(token, SECRET_KEY, algorithms=SUPPORTED_ALGORITHMS)
        logger.info(f"decode_token: JWT payload decoded successfully: {payload}")
        username: Optional[str] = payload.get("sub")
        logger.info(f"decode_token: Username extracted from token: '{username}'")
        if username is None:
            logger.warning("decode_token: No username found in token payload")
            return None
        logger.info(f"decode_token: Returning TokenData for username: '{username}'")
        return TokenData(username=username)
    except jwt.ExpiredSignatureError:
        logger.warning("decode_token: JWT token has expired")
        return None
    except JWTError as e:
        logger.error(f"decode_token: JWT decode error: {str(e)}")
        return None 