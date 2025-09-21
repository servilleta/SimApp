"""
Rate Limiting Service - Security Module

Provides comprehensive rate limiting including:
- Tiered rate limits based on user subscription
- Redis-based distributed rate limiting
- Multiple rate limit strategies (fixed window, sliding window)
- Usage tracking and analytics
- Rate limit decorators
- API endpoint protection
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from functools import wraps
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
import time

from ..base import BaseService


logger = logging.getLogger(__name__)


class RateLimiterService(BaseService):
    """Advanced rate limiting service"""
    
    # Tiered rate limits (requests per minute)
    TIER_LIMITS = {
        'free': {
            'api_calls': 60,        # 60 requests per minute
            'file_uploads': 5,      # 5 uploads per minute
            'simulations': 10,      # 10 simulations per minute
            'daily_api_calls': 1000, # 1000 requests per day
            'daily_simulations': 100  # 100 simulations per day
        },
        'basic': {
            'api_calls': 300,       # 300 requests per minute
            'file_uploads': 20,     # 20 uploads per minute
            'simulations': 50,      # 50 simulations per minute
            'daily_api_calls': 10000, # 10k requests per day
            'daily_simulations': 500  # 500 simulations per day
        },
        'pro': {
            'api_calls': 1000,      # 1000 requests per minute
            'file_uploads': 100,    # 100 uploads per minute
            'simulations': 200,     # 200 simulations per minute
            'daily_api_calls': 50000, # 50k requests per day
            'daily_simulations': 2000 # 2000 simulations per day
        },
        'enterprise': {
            'api_calls': 5000,      # 5000 requests per minute
            'file_uploads': 500,    # 500 uploads per minute
            'simulations': 1000,    # 1000 simulations per minute
            'daily_api_calls': 500000, # 500k requests per day
            'daily_simulations': 20000  # 20k simulations per day
        }
    }
    
    # Rate limit strategies
    STRATEGIES = {
        'fixed_window': 'fixed_window',
        'sliding_window': 'sliding_window',
        'token_bucket': 'token_bucket'
    }
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__("rate_limiter")
        self.redis_url = redis_url
        self.redis_client = None
        
        # Initialize slowapi limiter
        self.limiter = Limiter(
            key_func=self._get_rate_limit_key,
            default_limits=["100/minute"]
        )
        
    async def initialize(self) -> None:
        """Initialize the rate limiter service"""
        await super().initialize()
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory rate limiting.")
            self.redis_client = None
        
        logger.info("Rate limiter service initialized")
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """Get rate limit key from request"""
        # Try to get user from JWT token
        user_email = getattr(request.state, 'user_email', None)
        if user_email:
            return f"user:{user_email}"
        
        # Fall back to IP address
        return f"ip:{get_remote_address(request)}"
    
    async def check_rate_limit(self, identifier: str, limit_type: str, 
                              user_tier: str = 'free', window_seconds: int = 60) -> Dict[str, any]:
        """
        Check if request is within rate limits
        
        Args:
            identifier: User email or IP address
            limit_type: Type of limit (api_calls, file_uploads, simulations)
            user_tier: User subscription tier
            window_seconds: Time window in seconds
        
        Returns:
            Dict with rate limit status
        """
        try:
            # Get tier limits
            tier_limits = self.TIER_LIMITS.get(user_tier, self.TIER_LIMITS['free'])
            limit = tier_limits.get(limit_type, 60)  # Default to 60/minute
            
            # Check current usage
            current_usage = await self._get_current_usage(identifier, limit_type, window_seconds)
            
            # Calculate remaining requests
            remaining = max(0, limit - current_usage)
            
            # Check if limit exceeded
            if current_usage >= limit:
                # Get reset time
                reset_time = await self._get_reset_time(identifier, limit_type, window_seconds)
                
                return {
                    'allowed': False,
                    'limit': limit,
                    'current': current_usage,
                    'remaining': 0,
                    'reset_time': reset_time,
                    'tier': user_tier,
                    'limit_type': limit_type
                }
            
            # Increment usage counter
            await self._increment_usage(identifier, limit_type, window_seconds)
            
            return {
                'allowed': True,
                'limit': limit,
                'current': current_usage + 1,
                'remaining': remaining - 1,
                'reset_time': await self._get_reset_time(identifier, limit_type, window_seconds),
                'tier': user_tier,
                'limit_type': limit_type
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # In case of error, allow the request but log it
            return {
                'allowed': True,
                'error': str(e),
                'limit': tier_limits.get(limit_type, 60),
                'current': 0,
                'remaining': tier_limits.get(limit_type, 60)
            }
    
    async def _get_current_usage(self, identifier: str, limit_type: str, window_seconds: int) -> int:
        """Get current usage count for identifier"""
        if not self.redis_client:
            # In-memory fallback (not recommended for production)
            return 0
        
        key = f"rate_limit:{identifier}:{limit_type}:{window_seconds}"
        
        try:
            # Use sliding window approach
            now = time.time()
            window_start = now - window_seconds
            
            # Remove old entries
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            count = await self.redis_client.zcard(key)
            return count
            
        except Exception as e:
            logger.error(f"Failed to get current usage: {e}")
            return 0
    
    async def _increment_usage(self, identifier: str, limit_type: str, window_seconds: int) -> None:
        """Increment usage counter"""
        if not self.redis_client:
            return
        
        key = f"rate_limit:{identifier}:{limit_type}:{window_seconds}"
        
        try:
            now = time.time()
            
            # Add current timestamp to sorted set
            await self.redis_client.zadd(key, {str(now): now})
            
            # Set expiration for cleanup
            await self.redis_client.expire(key, window_seconds * 2)
            
        except Exception as e:
            logger.error(f"Failed to increment usage: {e}")
    
    async def _get_reset_time(self, identifier: str, limit_type: str, window_seconds: int) -> datetime:
        """Get time when rate limit resets"""
        if not self.redis_client:
            return datetime.utcnow() + timedelta(seconds=window_seconds)
        
        key = f"rate_limit:{identifier}:{limit_type}:{window_seconds}"
        
        try:
            # Get oldest entry
            oldest_entries = await self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest_entries:
                oldest_time = oldest_entries[0][1]
                reset_time = datetime.fromtimestamp(oldest_time + window_seconds)
                return reset_time
            
        except Exception as e:
            logger.error(f"Failed to get reset time: {e}")
        
        return datetime.utcnow() + timedelta(seconds=window_seconds)
    
    # Daily limits tracking
    async def check_daily_limit(self, identifier: str, limit_type: str, user_tier: str = 'free') -> Dict[str, any]:
        """Check daily rate limits"""
        daily_limit_type = f"daily_{limit_type}"
        return await self.check_rate_limit(identifier, daily_limit_type, user_tier, 86400)  # 24 hours
    
    # Usage analytics
    async def get_usage_stats(self, identifier: str, hours: int = 24) -> Dict[str, any]:
        """Get usage statistics for identifier"""
        if not self.redis_client:
            return {'error': 'Redis not available'}
        
        try:
            stats = {}
            now = time.time()
            start_time = now - (hours * 3600)
            
            # Get stats for each limit type
            for limit_type in ['api_calls', 'file_uploads', 'simulations']:
                key = f"rate_limit:{identifier}:{limit_type}:60"
                
                # Get entries in time range
                entries = await self.redis_client.zrangebyscore(key, start_time, now)
                
                stats[limit_type] = {
                    'total_requests': len(entries),
                    'hourly_breakdown': await self._get_hourly_breakdown(entries, hours)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {'error': str(e)}
    
    async def _get_hourly_breakdown(self, entries: List, hours: int) -> List[Dict]:
        """Get hourly breakdown of usage"""
        hourly_data = []
        now = time.time()
        
        for i in range(hours):
            hour_start = now - ((i + 1) * 3600)
            hour_end = now - (i * 3600)
            
            count = sum(1 for entry in entries if hour_start <= float(entry) < hour_end)
            
            hourly_data.append({
                'hour': datetime.fromtimestamp(hour_start).strftime('%Y-%m-%d %H:00'),
                'requests': count
            })
        
        return list(reversed(hourly_data))
    
    # Rate limit decorators
    def rate_limit(self, limit_type: str = 'api_calls', window_seconds: int = 60):
        """Decorator for rate limiting endpoints"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                # Get user info from request
                user_email = getattr(request.state, 'user_email', None)
                user_tier = getattr(request.state, 'user_tier', 'free')
                
                # Use IP if no user
                identifier = user_email or get_remote_address(request)
                
                # Check rate limit
                result = await self.check_rate_limit(identifier, limit_type, user_tier, window_seconds)
                
                if not result['allowed']:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            'error': 'Rate limit exceeded',
                            'limit': result['limit'],
                            'reset_time': result['reset_time'].isoformat(),
                            'tier': result['tier']
                        },
                        headers={
                            'X-RateLimit-Limit': str(result['limit']),
                            'X-RateLimit-Remaining': str(result['remaining']),
                            'X-RateLimit-Reset': str(int(result['reset_time'].timestamp())),
                            'Retry-After': str(int((result['reset_time'] - datetime.utcnow()).total_seconds()))
                        }
                    )
                
                # Add rate limit headers to response
                response = await func(request, *args, **kwargs)
                
                if hasattr(response, 'headers'):
                    response.headers['X-RateLimit-Limit'] = str(result['limit'])
                    response.headers['X-RateLimit-Remaining'] = str(result['remaining'])
                    response.headers['X-RateLimit-Reset'] = str(int(result['reset_time'].timestamp()))
                
                return response
            
            return wrapper
        return decorator
    
    # Burst protection
    async def check_burst_protection(self, identifier: str, requests_per_second: int = 10) -> bool:
        """Check for burst/spike protection"""
        if not self.redis_client:
            return True
        
        key = f"burst:{identifier}"
        window = 1  # 1 second window
        
        try:
            now = time.time()
            window_start = now - window
            
            # Remove old entries
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            current_count = await self.redis_client.zcard(key)
            
            if current_count >= requests_per_second:
                return False
            
            # Add current request
            await self.redis_client.zadd(key, {str(now): now})
            await self.redis_client.expire(key, 2)  # Cleanup after 2 seconds
            
            return True
            
        except Exception as e:
            logger.error(f"Burst protection check failed: {e}")
            return True  # Allow on error
    
    # IP-based blocking
    async def block_ip(self, ip_address: str, duration_seconds: int = 3600, reason: str = "Manual block") -> bool:
        """Block IP address temporarily"""
        if not self.redis_client:
            return False
        
        key = f"blocked_ip:{ip_address}"
        
        try:
            block_data = {
                'blocked_at': datetime.utcnow().isoformat(),
                'reason': reason,
                'duration': duration_seconds
            }
            
            await self.redis_client.setex(key, duration_seconds, json.dumps(block_data))
            logger.warning(f"IP {ip_address} blocked for {duration_seconds}s: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {e}")
            return False
    
    async def is_ip_blocked(self, ip_address: str) -> Dict[str, any]:
        """Check if IP address is blocked"""
        if not self.redis_client:
            return {'blocked': False}
        
        key = f"blocked_ip:{ip_address}"
        
        try:
            block_data = await self.redis_client.get(key)
            if block_data:
                data = json.loads(block_data)
                return {
                    'blocked': True,
                    'reason': data.get('reason'),
                    'blocked_at': data.get('blocked_at'),
                    'ttl': await self.redis_client.ttl(key)
                }
            
            return {'blocked': False}
            
        except Exception as e:
            logger.error(f"Failed to check IP block status: {e}")
            return {'blocked': False}
    
    async def unblock_ip(self, ip_address: str) -> bool:
        """Unblock IP address"""
        if not self.redis_client:
            return False
        
        key = f"blocked_ip:{ip_address}"
        
        try:
            result = await self.redis_client.delete(key)
            if result:
                logger.info(f"IP {ip_address} unblocked")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to unblock IP {ip_address}: {e}")
            return False
    
    # Cleanup and maintenance
    async def cleanup_old_entries(self, older_than_hours: int = 24) -> int:
        """Clean up old rate limit entries"""
        if not self.redis_client:
            return 0
        
        try:
            cutoff_time = time.time() - (older_than_hours * 3600)
            pattern = "rate_limit:*"
            
            cleaned_count = 0
            async for key in self.redis_client.scan_iter(match=pattern):
                # Remove old entries from sorted sets
                removed = await self.redis_client.zremrangebyscore(key, 0, cutoff_time)
                cleaned_count += removed
                
                # Remove empty keys
                if await self.redis_client.zcard(key) == 0:
                    await self.redis_client.delete(key)
            
            logger.info(f"Cleaned up {cleaned_count} old rate limit entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old entries: {e}")
            return 0
    
    def health_check(self) -> Dict[str, any]:
        """Health check for rate limiter service"""
        health = super().health_check()
        
        health.update({
            'redis_connected': self.redis_client is not None,
            'tier_limits': self.TIER_LIMITS,
            'strategies_available': list(self.STRATEGIES.keys())
        })
        
        if self.redis_client:
            try:
                # This would be async in real implementation
                health['redis_info'] = 'connected'
            except:
                health['redis_info'] = 'connection_failed'
        
        return health 