"""
Secure API Key Management Service

Provides secure generation, validation, and management of API keys.
"""

import secrets
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from models import APIKey, User
import logging

logger = logging.getLogger(__name__)

class APIKeyService:
    """Service for secure API key management"""
    
    @staticmethod
    def generate_api_key() -> Tuple[str, str]:
        """
        Generate a new API key pair.
        
        Returns:
            Tuple[str, str]: (key_id, secret_key)
            
        Example:
            key_id: "ak_1a2b3c4d5e6f7g8h"
            secret_key: "sk_9i8h7g6f5e4d3c2b1a0z9y8x7w6v5u4t3s2r1q0p"
        """
        # Generate a unique key ID (public identifier)
        key_id = "ak_" + secrets.token_hex(16)
        
        # Generate a secure secret key (private part)
        secret_key = "sk_" + secrets.token_hex(32)
        
        return key_id, secret_key
    
    @staticmethod
    def hash_secret_key(secret_key: str, salt: Optional[str] = None) -> str:
        """
        Securely hash the secret key for database storage.
        
        Args:
            secret_key: The raw secret key
            salt: Optional salt (will generate if not provided)
            
        Returns:
            str: Hashed key with salt
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 with SHA-256 for secure hashing
        key_hash = hashlib.pbkdf2_hmac(
            'sha256',
            secret_key.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100k iterations
        )
        
        # Combine salt and hash for storage
        return f"{salt}:{key_hash.hex()}"
    
    @staticmethod
    def verify_secret_key(secret_key: str, stored_hash: str) -> bool:
        """
        Verify a secret key against its stored hash.
        
        Args:
            secret_key: The raw secret key to verify
            stored_hash: The stored hash from database
            
        Returns:
            bool: True if key is valid
        """
        try:
            salt, key_hash = stored_hash.split(':', 1)
            
            # Hash the provided key with the same salt
            test_hash = hashlib.pbkdf2_hmac(
                'sha256',
                secret_key.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            
            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(key_hash, test_hash.hex())
            
        except (ValueError, AttributeError):
            logger.warning("Invalid hash format in API key verification")
            return False
    
    @staticmethod
    def validate_api_key(db: Session, full_api_key: str) -> Optional[APIKey]:
        """
        Validate a full API key and return the API key record if valid.
        
        Args:
            db: Database session
            full_api_key: Full API key in format "ak_keyid_sk_secret"
            
        Returns:
            APIKey record if valid, None otherwise
        """
        try:
            # Parse the API key
            if not full_api_key.startswith("ak_") or "_sk_" not in full_api_key:
                return None
                
            parts = full_api_key.split("_sk_")
            if len(parts) != 2:
                return None
                
            key_id = parts[0]  # e.g., "ak_1a2b3c4d5e6f7g8h"
            secret_key = "sk_" + parts[1]  # e.g., "sk_9i8h7g6f5e4d3c2b1a0z9y8x7w6v5u4t3s2r1q0p"
            
            # Find the API key record by key_id
            api_key_record = db.query(APIKey).filter(APIKey.key_id == key_id).first()
            
            if not api_key_record:
                return None
                
            # Verify the secret key
            if not APIKeyService.verify_secret_key(secret_key, api_key_record.key_hash):
                return None
                
            # Update last used timestamp
            api_key_record.last_used_at = datetime.now(timezone.utc)
            db.commit()
            
            return api_key_record
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    @staticmethod
    def create_api_key(
        db: Session,
        user_id: int,
        name: str,
        subscription_tier: str = "starter",
        expires_in_days: Optional[int] = None
    ) -> Tuple[APIKey, str]:
        """
        Create a new API key for a user.
        
        Args:
            db: Database session
            user_id: ID of the user
            name: Human-readable name for the key
            subscription_tier: Subscription tier (starter, professional, enterprise)
            expires_in_days: Optional expiration in days
            
        Returns:
            Tuple[APIKey, str]: (API key record, raw secret key)
        """
        # Generate the key pair
        key_id, secret_key = APIKeyService.generate_api_key()
        
        # Hash the secret key
        key_hash = APIKeyService.hash_secret_key(secret_key)
        
        # Set expiration if specified
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        # Set limits based on subscription tier
        tier_limits = {
            "starter": {
                "monthly_requests": 1000,
                "max_iterations": 10000,
                "max_file_size_mb": 10
            },
            "professional": {
                "monthly_requests": 10000,
                "max_iterations": 100000,
                "max_file_size_mb": 50
            },
            "enterprise": {
                "monthly_requests": 100000,
                "max_iterations": 1000000,
                "max_file_size_mb": 500
            }
        }
        
        limits = tier_limits.get(subscription_tier, tier_limits["starter"])
        
        # Generate a client_id based on user and key
        client_id = f"client_{user_id}_{key_id.split('_')[1][:8]}"
        
        # Create the API key record
        api_key = APIKey(
            user_id=user_id,
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            client_id=client_id,
            subscription_tier=subscription_tier,
            expires_at=expires_at,
            **limits
        )
        
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        logger.info(f"Created API key {key_id} for user {user_id}")
        
        # Return the record and the raw secret key (only time it's available)
        return api_key, secret_key
    
    @staticmethod
    def verify_api_key(db: Session, full_key: str) -> Optional[APIKey]:
        """
        Verify a full API key and return the API key record if valid.
        
        Args:
            db: Database session
            full_key: The full API key (e.g., "ak_123...sk_456...")
            
        Returns:
            Optional[APIKey]: API key record if valid, None otherwise
        """
        try:
            # Expected format: "ak_16chars_sk_64chars"
            if not full_key.startswith("ak_") or "_sk_" not in full_key:
                logger.warning(f"Invalid API key format: {full_key[:20]}...")
                return None
            
            # Split key_id and secret_key
            parts = full_key.split("_sk_")
            if len(parts) != 2:
                logger.warning(f"Invalid API key structure: {full_key[:20]}...")
                return None
            
            key_id = parts[0]  # "ak_16chars"
            secret_key = "sk_" + parts[1]  # "sk_64chars"
            
            # Find the API key by key_id
            api_key = db.query(APIKey).filter(
                and_(
                    APIKey.key_id == key_id,
                    APIKey.is_active == True
                )
            ).first()
            
            if not api_key:
                logger.warning(f"API key not found or inactive: {key_id}")
                return None
            
            # Check if key has expired
            if api_key.expires_at and datetime.now(timezone.utc) > api_key.expires_at:
                logger.warning(f"API key expired: {key_id}")
                return None
            
            # Verify the secret key
            if not APIKeyService.verify_secret_key(secret_key, api_key.key_hash):
                logger.warning(f"Invalid secret key for: {key_id}")
                return None
            
            # Update last used timestamp
            api_key.last_used_at = datetime.now(timezone.utc)
            db.commit()
            
            logger.info(f"Successfully verified API key: {key_id}")
            return api_key
            
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return None
    
    @staticmethod
    def list_user_api_keys(db: Session, user_id: int) -> List[APIKey]:
        """
        List all API keys for a user.
        
        Args:
            db: Database session
            user_id: ID of the user
            
        Returns:
            List[APIKey]: List of API key records
        """
        return db.query(APIKey).filter(APIKey.user_id == user_id).all()
    
    @staticmethod
    def revoke_api_key(db: Session, key_id: str, user_id: int) -> bool:
        """
        Revoke (deactivate) an API key.
        
        Args:
            db: Database session
            key_id: The key ID to revoke
            user_id: ID of the user (for security)
            
        Returns:
            bool: True if key was revoked
        """
        api_key = db.query(APIKey).filter(
            and_(
                APIKey.key_id == key_id,
                APIKey.user_id == user_id
            )
        ).first()
        
        if api_key:
            api_key.is_active = False
            db.commit()
            logger.info(f"Revoked API key: {key_id}")
            return True
        
        logger.warning(f"API key not found for revocation: {key_id}")
        return False
    
    @staticmethod
    def increment_usage(db: Session, api_key: APIKey) -> None:
        """
        Increment the usage counter for an API key.
        
        Args:
            db: Database session
            api_key: The API key record
        """
        api_key.requests_used_this_month += 1
        api_key.last_used_at = datetime.now(timezone.utc)
        db.commit()
    
    @staticmethod
    def check_rate_limits(api_key: APIKey, iterations: int = 0, file_size_mb: float = 0) -> Optional[str]:
        """
        Check if the API key is within rate limits.
        
        Args:
            api_key: The API key record
            iterations: Number of iterations requested
            file_size_mb: Size of file being uploaded
            
        Returns:
            Optional[str]: Error message if limit exceeded, None if OK
        """
        # Check monthly requests
        if api_key.requests_used_this_month >= api_key.monthly_requests:
            return f"Monthly request limit exceeded ({api_key.monthly_requests})"
        
        # Check iteration limit
        if iterations > api_key.max_iterations:
            return f"Iteration limit exceeded (max: {api_key.max_iterations})"
        
        # Check file size limit
        if file_size_mb > api_key.max_file_size_mb:
            return f"File size limit exceeded (max: {api_key.max_file_size_mb}MB)"
        
        return None
