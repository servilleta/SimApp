"""
🔧 REDIS THREAD MANAGER: Advanced solution for Redis connections across threads
This module manages Redis connections that work properly across different event loops
"""

import asyncio
import threading
import redis.asyncio as redis
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RedisThreadManager:
    """
    Manages Redis connections for different threads and event loops
    Uses thread-local storage to ensure each thread gets its own connection
    """
    
    def __init__(self):
        self._thread_local = threading.local()
        self._redis_config = {
            'host': os.getenv('REDIS_HOST', 'redis'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'max_connections': 50,
            'retry_on_timeout': False,
            'socket_connect_timeout': 2.0,
            'socket_timeout': 5.0,
            'health_check_interval': 30,
            'decode_responses': True
        }
    
    async def get_redis_client(self) -> Optional[redis.Redis]:
        """
        Get Redis client for current thread/event loop
        Creates a new connection if needed for this thread
        """
        try:
            # Check if current thread has a Redis client
            if not hasattr(self._thread_local, 'redis_client'):
                logger.info(f"🔧 [REDIS_THREAD] Creating new Redis connection for thread: {threading.current_thread().name}")
                
                # Create connection pool for this thread
                pool = redis.ConnectionPool(**self._redis_config)
                
                # Create Redis client
                client = redis.Redis(connection_pool=pool)
                
                # Test connection
                await client.ping()
                
                # Store in thread-local storage
                self._thread_local.redis_client = client
                self._thread_local.redis_pool = pool
                
                logger.info(f"✅ [REDIS_THREAD] Successfully created Redis connection for thread: {threading.current_thread().name}")
            
            return self._thread_local.redis_client
            
        except Exception as e:
            logger.error(f"❌ [REDIS_THREAD] Failed to create Redis connection: {e}")
            return None
    
    async def close_thread_connection(self):
        """Close Redis connection for current thread"""
        try:
            if hasattr(self._thread_local, 'redis_client'):
                await self._thread_local.redis_client.close()
                delattr(self._thread_local, 'redis_client')
                delattr(self._thread_local, 'redis_pool')
                logger.info(f"🔧 [REDIS_THREAD] Closed Redis connection for thread: {threading.current_thread().name}")
        except Exception as e:
            logger.warning(f"⚠️ [REDIS_THREAD] Error closing Redis connection: {e}")

# Global instance
_redis_thread_manager = RedisThreadManager()

async def get_thread_redis_client() -> Optional[redis.Redis]:
    """Get Redis client that works in current thread/event loop"""
    return await _redis_thread_manager.get_redis_client()

async def close_thread_redis():
    """Close Redis connection for current thread"""
    await _redis_thread_manager.close_thread_connection()
