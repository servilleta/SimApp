"""
B2B API v1 - Authentication and authorization for API clients

Secure database-backed API key authentication system.
"""
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging

from database import get_db
from models import APIKey
from services.api_key_service import APIKeyService

logger = logging.getLogger(__name__)

# Initialize HTTP Bearer security scheme
security = HTTPBearer()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> APIKey:
    """
    Verify API key and return API key record.
    
    Expected format: Bearer ak_[key_id]_sk_[secret_key]
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key_full = credentials.credentials
    
    # Validate the API key using the secure service
    api_key_record = APIKeyService.validate_api_key(db, api_key_full)
    
    if not api_key_record:
        logger.warning(f"Invalid API key attempted: {api_key_full[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not api_key_record.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"API key authenticated: {api_key_record.key_id} (User: {api_key_record.user_id})")
    return api_key_record


async def check_rate_limits(api_key: APIKey, iterations: int, file_size_mb: float = 0):
    """
    Check if request is within subscription limits.
    
    Args:
        api_key: API key record with limits
        iterations: Number of Monte Carlo iterations requested
        file_size_mb: Size of uploaded file in MB
        
    Raises:
        HTTPException: If any limits are exceeded
    """
    error_message = APIKeyService.check_rate_limits(api_key, iterations, file_size_mb)
    
    if error_message:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error_message
        )


def calculate_credits(iterations: int, file_size_mb: float, formulas_count: int) -> int:
    """
    Calculate credits required for a simulation.
    
    Args:
        iterations: Number of Monte Carlo iterations
        file_size_mb: Size of the Excel file in MB
        formulas_count: Number of formulas in the Excel model
        
    Returns:
        int: Credits required for the simulation
    """
    # Basic credit calculation formula
    base_credits = max(1, iterations // 1000)  # 1 credit per 1000 iterations
    file_credits = int(file_size_mb * 0.1)     # 0.1 credits per MB
    formula_credits = max(1, formulas_count // 100)  # 1 credit per 100 formulas
    
    total_credits = base_credits + file_credits + formula_credits
    return max(1, total_credits)  # Minimum 1 credit


# Dependency for API key verification  
def get_api_key() -> APIKey:
    """Dependency to get API key information."""
    return Depends(verify_api_key)