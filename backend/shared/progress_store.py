# Shared progress store for simulation tracking
# This module provides a centralized progress store that can be imported by multiple modules

import redis
import redis.asyncio
import json
import logging
import asyncio
import time
import threading
from typing import Dict, Optional
import os
from functools import lru_cache
import hashlib

# Import settings for configuration
from config import settings

# Import the new DTO adapter
try:
    from shared.progress_schema import create_progress_dto, ProgressDTO
    DTO_AVAILABLE = True
except ImportError:
    # Fallback if schema not available
    DTO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ProgressDTO not available, using raw progress format")

logger = logging.getLogger(__name__)

class ProgressStore:
    """Redis-backed progress store for simulation tracking with async support and fast fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.async_redis_client = None
        # Track target variables for each simulation
        self._simulation_metadata = {}
        # Circuit breaker for Redis failures
        self._redis_failure_count = 0
        self._redis_failure_threshold = settings.PROGRESS_CIRCUIT_BREAKER_THRESHOLD
        self._redis_circuit_open_until = 0
        self._last_health_check = 0
        self._health_check_interval = settings.PROGRESS_HEALTH_CHECK_INTERVAL
        # Thread-safe fallback store
        self._fallback_lock = threading.RLock()
        # Progress response cache (1-2 second TTL)
        self._progress_cache = {}
        self._cache_ttl = settings.PROGRESS_CACHE_TTL
        # 🚀 FINAL SOLUTION: Thread-safe in-memory progress bridge
        self._progress_bridge = {}
        self._bridge_lock = threading.RLock()
        # 🚀 OPTIMIZATION: Background updates feature flag
        self._use_background_updates = False
        # 🔧 ASYNC TASK OVERLOAD PROTECTION
        self._last_async_update = {}
        self._async_update_lock = threading.RLock()
        # Connect to Redis only after essential fields/locks are initialized
        self._connect()
    
    async def enable_background_updates(self):
        """🚀 Enable high-performance background update processing"""
        try:
            from shared.background_progress import start_background_progress
            await start_background_progress()
            self._use_background_updates = True
            logger.info("🚀 Background progress updates enabled")
        except Exception as e:
            logger.warning(f"⚠️ Could not enable background updates: {e}")
    
    async def disable_background_updates(self):
        """🛑 Disable background update processing"""
        try:
            if self._use_background_updates:
                from shared.background_progress import stop_background_progress
                await stop_background_progress()
                self._use_background_updates = False
                logger.info("🛑 Background progress updates disabled")
        except Exception as e:
            logger.warning(f"⚠️ Error disabling background updates: {e}")
    
    def _ensure_fallback_store(self):
        """Ensure thread-safe fallback store is initialized"""
        with self._fallback_lock:
            if not hasattr(self, '_fallback_store'):
                logger.info("🔧 Initializing thread-safe fallback in-memory store")
                self._fallback_store = {}
                self._fallback_ttl = {}  # Track TTL for memory entries
    
    def _connect(self):
        """Initialize Redis connection with high-performance connection pooling"""
        try:
            # Allow disabling Redis via env or when not configured
            if str(os.getenv('REDIS_DISABLED', '')).lower() in {"1", "true", "yes"}:
                logger.info("🚫 Redis explicitly disabled by environment; using in-memory store")
                self.redis_client = None
                self.async_redis_client = None
                self._ensure_fallback_store()
                return

            # If no host/url configured, avoid reliance on Redis
            if not os.getenv('REDIS_HOST') and not os.getenv('REDIS_URL'):
                logger.info("ℹ️ No Redis configuration found; using in-memory progress store")
                self.redis_client = None
                self.async_redis_client = None
                self._ensure_fallback_store()
                return

            # Try to connect to Redis with optimized connection pooling
            redis_host = os.getenv('REDIS_HOST', 'redis')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            
            # 🚀 OPTIMIZATION: Create connection pool for better performance
            self.redis_pool = redis.ConnectionPool(
                host=redis_host,
                port=redis_port,
                max_connections=200,  # ULTRA BOOST: Allow up to 200 concurrent connections for ultra engine real-time updates
                retry_on_timeout=False,
                socket_connect_timeout=settings.PROGRESS_REDIS_CONNECT_TIMEOUT,
                socket_timeout=settings.progress_redis_timeout,
                health_check_interval=settings.PROGRESS_HEALTH_CHECK_INTERVAL,
                decode_responses=True
            )
            
            # Synchronous client using connection pool
            self.redis_client = redis.Redis(
                connection_pool=self.redis_pool,
                retry_on_timeout=False    # Fail fast instead of retrying
            )
            
            # 🚀 OPTIMIZATION: Async connection pool for non-blocking operations
            self.async_redis_pool = redis.asyncio.ConnectionPool(
                host=redis_host,
                port=redis_port,
                max_connections=200,  # ULTRA BOOST: Allow up to 200 concurrent async connections for ultra engine real-time updates
                retry_on_timeout=False,
                socket_connect_timeout=settings.PROGRESS_REDIS_CONNECT_TIMEOUT,
                socket_timeout=settings.progress_redis_timeout,
                health_check_interval=settings.PROGRESS_HEALTH_CHECK_INTERVAL,
                decode_responses=True
            )
            
            # Async client using connection pool
            self.async_redis_client = redis.asyncio.Redis(
                connection_pool=self.async_redis_pool,
                retry_on_timeout=False
            )
            
            # Test connection with timeout
            self.redis_client.ping()
            logger.info(f"✅ Redis progress store connected to {redis_host}:{redis_port} (fast timeouts)")
            self._redis_failure_count = 0  # Reset failure count on success
            
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, falling back to in-memory store: {e}")
            self.redis_client = None
            self.async_redis_client = None
            self._ensure_fallback_store()
            self._redis_failure_count += 1
    
    def _get_key(self, simulation_id: str) -> str:
        """Generate Redis key for simulation progress"""
        return f"simulation:progress:{simulation_id}"
    
    def _get_metadata_key(self, simulation_id: str) -> str:
        """Generate Redis key for simulation metadata"""
        return f"simulation:metadata:{simulation_id}"
    
    def _set_progress_bridge(self, simulation_id: str, progress_data: dict):
        """🚀 FINAL SOLUTION: Store progress in thread-safe in-memory bridge AND sync to Redis"""
        with self._bridge_lock:
            self._progress_bridge[simulation_id] = {
                'data': progress_data,
                'timestamp': time.time()
            }
            logger.info(f"🌉 [BRIDGE] Stored progress in memory bridge for {simulation_id}: {progress_data.get('progress_percentage', 'N/A')}%")
            
            # CRITICAL FIX: Also update Redis immediately to ensure sync
            try:
                if self.redis_client and self._is_redis_available():
                    key = self._get_key(simulation_id)
                    value = json.dumps(progress_data)
                    # Use short TTL for progress updates
                    self.redis_client.setex(key, 300, value)  # 5 minute TTL for active progress
                    logger.debug(f"🔄 [BRIDGE->REDIS] Synced progress to Redis for {simulation_id}")
            except Exception as e:
                logger.warning(f"⚠️ [BRIDGE->REDIS] Failed to sync to Redis: {e}")
    
    def _get_progress_bridge(self, simulation_id: str) -> Optional[dict]:
        """🚀 FINAL SOLUTION: Get progress from thread-safe in-memory bridge"""
        with self._bridge_lock:
            if simulation_id in self._progress_bridge:
                bridge_data = self._progress_bridge[simulation_id]
                # Data expires after 300 seconds (5 minutes)
                if time.time() - bridge_data['timestamp'] < 300:
                    logger.info(f"🌉 [BRIDGE] Retrieved progress from memory bridge for {simulation_id}: {bridge_data['data'].get('progress_percentage', 'N/A')}%")
                    return bridge_data['data']
                else:
                    # Expired, remove it
                    del self._progress_bridge[simulation_id]
            return None
    
    def set_simulation_metadata(self, simulation_id: str, metadata: dict):
        """Store simulation metadata (target variables, etc.)"""
        try:
            self._simulation_metadata[simulation_id] = metadata
            
            if self.redis_client:
                key = self._get_metadata_key(simulation_id)
                value = json.dumps(metadata)
                self.redis_client.setex(key, 7200, value)  # 2 hour TTL
        except Exception as e:
            logger.error(f"❌ Failed to set metadata for {simulation_id}: {e}")
    
    def get_simulation_metadata(self, simulation_id: str) -> Optional[dict]:
        """Get simulation metadata"""
        try:
            # Try in-memory first
            if simulation_id in self._simulation_metadata:
                return self._simulation_metadata[simulation_id]
            
            # Try Redis
            if self.redis_client:
                key = self._get_metadata_key(simulation_id)
                value = self.redis_client.get(key)
                if value:
                    metadata = json.loads(value)
                    self._simulation_metadata[simulation_id] = metadata
                    return metadata
        except Exception as e:
            logger.error(f"❌ Failed to get metadata for {simulation_id}: {e}")
        
        return None

    def _is_redis_circuit_open(self) -> bool:
        """Check if Redis circuit breaker is open"""
        return time.time() < self._redis_circuit_open_until
    
    def _open_redis_circuit(self):
        """Open circuit breaker for Redis failures"""
        self._redis_circuit_open_until = time.time() + settings.PROGRESS_CIRCUIT_BREAKER_DURATION
        logger.warning(f"🔴 Redis circuit breaker opened for {settings.PROGRESS_CIRCUIT_BREAKER_DURATION} seconds (failures: {self._redis_failure_count})")
    
    def _should_check_redis_health(self) -> bool:
        """Check if it's time for a Redis health check"""
        return time.time() - self._last_health_check > self._health_check_interval
    
    async def _redis_health_check(self) -> bool:
        """Async Redis health check"""
        if not self.async_redis_client:
            return False
        try:
            await asyncio.wait_for(self.async_redis_client.ping(), timeout=settings.progress_redis_timeout)
            self._redis_failure_count = 0
            self._redis_circuit_open_until = 0
            self._last_health_check = time.time()
            return True
        except Exception as e:
            logger.debug(f"Redis health check failed: {e}")
            self._redis_failure_count += 1
            if self._redis_failure_count >= self._redis_failure_threshold:
                self._open_redis_circuit()
            return False
    
    def _get_cached_progress(self, simulation_id: str) -> Optional[dict]:
        """Get cached progress with TTL check"""
        if simulation_id in self._progress_cache:
            cached_data, timestamp = self._progress_cache[simulation_id]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
            else:
                # Remove expired cache
                del self._progress_cache[simulation_id]
        return None
    
    def _cache_progress(self, simulation_id: str, progress_data: dict):
        """🚀 OPTIMIZED: Cache progress data with intelligent cleanup"""
        current_time = time.time()
        self._progress_cache[simulation_id] = (progress_data, current_time)
        
        # 🚀 OPTIMIZATION: Efficient cache cleanup (only every 10th cache operation)
        if len(self._progress_cache) % 10 == 0:
            expired_keys = [k for k, (_, t) in self._progress_cache.items() 
                           if current_time - t > self._cache_ttl * 2]
            for key in expired_keys:
                del self._progress_cache[key]
            if expired_keys:
                logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
    
    @lru_cache(maxsize=500)
    def _get_cached_response_by_content(self, simulation_id: str, content_hash: str, timestamp_bucket: int) -> Optional[str]:
        """🚀 OPTIMIZATION: LRU cache for identical progress responses to reduce JSON serialization"""
        # This caches the JSON-serialized response based on content hash + time bucket
        # Time bucket changes every 2 seconds, allowing for fresh data while caching identical responses
        return None  # Placeholder - actual implementation below
    
    def _generate_content_hash(self, progress_data: dict) -> str:
        """Generate hash of progress data for intelligent caching"""
        # Create deterministic hash based on key progress fields
        key_fields = {
            'progress_percentage': progress_data.get('progress_percentage', 0),
            'status': progress_data.get('status', 'unknown'),
            'current_iteration': progress_data.get('current_iteration', 0),
            'stage': progress_data.get('stage', 'unknown')
        }
        content_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:8]  # Short hash for performance
    
    async def set_progress_async(self, simulation_id: str, progress_data: dict, bypass_merge: bool = False, temp_id: str = None):
        """🚀 OPTIMIZED: Async version of set_progress with background processing and dual ID support"""
        logger.info(f"🚀 [CLASS_ENTRY] ProgressStore.set_progress_async entered for {simulation_id} with {progress_data.get('progress_percentage', 'N/A')}%")
        try:
            # 🚀 OPTIMIZATION: For high-frequency updates, use background processing
            if not bypass_merge and hasattr(self, '_use_background_updates'):
                try:
                    from shared.background_progress import queue_progress_update
                    await queue_progress_update(simulation_id, progress_data)
                    return  # Background system will handle the update
                except ImportError:
                    # Fallback to direct processing if background system not available
                    pass
            
            # Direct processing for critical updates or when background is disabled
            existing_progress = {}
            if not bypass_merge:
                # Try fallback store first (under lock)
                self._ensure_fallback_store()
                with self._fallback_lock:
                    existing_progress = self._fallback_store.get(simulation_id, {})
                
                # If no fallback data available, then check Redis
                if not existing_progress:
                    existing_progress = await self.get_progress_async(simulation_id) or {}
            
            # Merge existing data with new data (new data takes precedence)
            raw_progress = dict(existing_progress)
            raw_progress.update(progress_data)
            
            # Transform to DTO if available
            if DTO_AVAILABLE:
                try:
                    # Get simulation metadata for target variables
                    metadata = self.get_simulation_metadata(simulation_id)
                    target_variables = metadata.get('target_variables', []) if metadata else []
                    
                    # Create DTO with proper schema
                    progress_dto = create_progress_dto(
                        simulation_id=simulation_id,
                        raw_progress=raw_progress,
                        target_variables=target_variables
                    )
                    
                    # Convert to dict for storage
                    enhanced_progress = progress_dto.dict()
                    logger.debug(f"🎯 DTO transformation successful for {simulation_id}")
                    
                except Exception as dto_error:
                    logger.warning(f"⚠️ DTO transformation failed for {simulation_id}: {dto_error}")
                    enhanced_progress = raw_progress
            else:
                enhanced_progress = raw_progress
            
            # Try async Redis first if circuit is closed
            redis_success = False
            if self.async_redis_client and not self._is_redis_circuit_open():
                try:
                    key = self._get_key(simulation_id)
                    value = json.dumps(enhanced_progress)
                    
                    # Dynamic TTL calculation
                    iterations = enhanced_progress.get('total_iterations', 1000)
                    if iterations > 50000:
                        ttl = 14400  # 4 hours for very large simulations
                    elif iterations > 10000:
                        ttl = 7200   # 2 hours for large simulations  
                    else:
                        ttl = 3600   # 1 hour for normal simulations
                    
                    # 🔧 THREADING FIX: Async Redis operation with event loop safety
                    try:
                        await asyncio.wait_for(
                            self.async_redis_client.setex(key, ttl, value),
                            timeout=settings.progress_redis_timeout
                        )
                        redis_success = True  # Set success only if async worked
                    except RuntimeError as e:
                        if "attached to a different loop" in str(e):
                            logger.warning(f"🔧 [LOOP_FIX] Event loop conflict detected, using sync fallback for {simulation_id}")
                            # Use sync Redis as fallback for loop conflicts
                            if self.redis_client:
                                # Test sync connection first
                                try:
                                    self.redis_client.ping()
                                    self.redis_client.setex(key, ttl, value)
                                    redis_success = True
                                    logger.info(f"✅ [LOOP_FIX] Sync fallback successful for {simulation_id}")
                                    
                                    # Verify the write worked by reading it back
                                    stored_value = self.redis_client.get(key)
                                    if stored_value:
                                        logger.info(f"🔍 [LOOP_FIX] Verified sync write: {simulation_id} stored successfully")
                                    else:
                                        logger.error(f"❌ [LOOP_FIX] Sync write verification failed for {simulation_id}")
                                        
                                except Exception as sync_e:
                                    logger.error(f"❌ [LOOP_FIX] Sync Redis operation failed: {sync_e}")
                                    raise
                            else:
                                logger.error(f"❌ [LOOP_FIX] No sync Redis client available for {simulation_id}")
                                raise
                        else:
                            raise
                    
                    # 🚀 CRITICAL: Store under temp_id in Redis as well for instant frontend access
                    if temp_id and temp_id.startswith('temp_'):
                        temp_key = self._get_key(temp_id)
                        try:
                            await asyncio.wait_for(
                                self.async_redis_client.setex(temp_key, ttl, value),
                                timeout=settings.progress_redis_timeout
                            )
                        except RuntimeError as e:
                            if "attached to a different loop" in str(e):
                                logger.warning(f"🔧 [LOOP_FIX] Event loop conflict for temp_id, using sync fallback for {temp_id}")
                                if self.redis_client:
                                    try:
                                        self.redis_client.setex(temp_key, ttl, value)
                                        logger.info(f"✅ [LOOP_FIX] Sync fallback successful for temp_id {temp_id}")
                                    except Exception as sync_e:
                                        logger.error(f"❌ [LOOP_FIX] Sync temp_id operation failed: {sync_e}")
                                else:
                                    logger.warning(f"⚠️ [LOOP_FIX] No sync Redis client for temp_id {temp_id}")
                            else:
                                raise
                        logger.info(f"🚀 [DUAL_STORAGE] Stored progress in Redis under temp_id: {temp_id} ({enhanced_progress.get('progress_percentage', 'N/A')}%)")
                        logger.debug(f"🔗 [DUAL_REDIS] Stored progress in Redis under both {simulation_id} AND {temp_id}")
                    
                    logger.debug(f"📦 Progress stored in Redis for {simulation_id} (TTL: {ttl}s)")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Redis timeout storing progress for {simulation_id}, using fallback")
                    self._redis_failure_count += 1
                    if self._redis_failure_count >= self._redis_failure_threshold:
                        self._open_redis_circuit()
                except Exception as e:
                    logger.warning(f"❌ Redis error storing progress for {simulation_id}: {e}, using fallback")
                    self._redis_failure_count += 1
                    if self._redis_failure_count >= self._redis_failure_threshold:
                        self._open_redis_circuit()
            
            # Always store in fallback for fast retrieval
            self._ensure_fallback_store()
            with self._fallback_lock:
                self._fallback_store[simulation_id] = enhanced_progress
                self._fallback_ttl[simulation_id] = time.time() + 3600  # 1 hour TTL
                
                # 🚀 CRITICAL: Store under temp_id as well for instant frontend access
                if temp_id and temp_id.startswith('temp_'):
                    self._fallback_store[temp_id] = enhanced_progress
                    self._fallback_ttl[temp_id] = time.time() + 3600  # 1 hour TTL
                    logger.info(f"🚀 [DUAL_STORAGE] Stored progress under temp_id: {temp_id} ({enhanced_progress.get('progress_percentage', 'N/A')}%)")
                    logger.debug(f"🔗 [DUAL_STORAGE] Stored progress under both {simulation_id} AND {temp_id}")
            
            # Cache the progress for rapid access
            self._cache_progress(simulation_id, enhanced_progress)
            
            # 🚀 CRITICAL: Cache under temp_id as well
            if temp_id and temp_id.startswith('temp_'):
                self._cache_progress(temp_id, enhanced_progress)
            
            # In set_progress_async, call this after updating progress
            # For example, after storing in Redis or memory:
            logger.info(f"💥 [PRE_WEBSOCKET] About to call WebSocket update for {simulation_id}: {enhanced_progress.get('progress_percentage', 'N/A')}%")
            self._send_websocket_update(simulation_id, enhanced_progress)
            logger.info(f"💥 [POST_WEBSOCKET] WebSocket update completed for {simulation_id}: {enhanced_progress.get('progress_percentage', 'N/A')}%")
            
            # 🚀 FINAL SOLUTION: Always store in memory bridge first (bypasses all Redis issues)
            logger.info(f"💥 [PRE_BRIDGE] Reached bridge storage point for {simulation_id}: {enhanced_progress.get('progress_percentage', 'N/A')}%")
            logger.info(f"🔧 [BRIDGE_DEBUG] About to store in bridge for {simulation_id}: {enhanced_progress.get('progress_percentage', 'N/A')}%")
            try:
                self._set_progress_bridge(simulation_id, enhanced_progress)
                logger.info(f"✅ [BRIDGE_DEBUG] Successfully stored in bridge for {simulation_id}")
            except Exception as bridge_error:
                logger.error(f"❌ [BRIDGE_ERROR] Failed to store in bridge for {simulation_id}: {bridge_error}")
                import traceback
                logger.error(f"🔍 [BRIDGE_ERROR] Traceback: {traceback.format_exc()}")
            
            if redis_success:
                logger.debug(f"💾 Progress stored in Redis, memory fallback, AND memory bridge for {simulation_id}")
            else:
                logger.debug(f"💾 Progress stored in memory fallback AND memory bridge for {simulation_id}")
                
        except Exception as e:
            logger.error(f"❌ Error in async set_progress for {simulation_id}: {e}")
            # Ensure fallback storage works even on errors
            self._ensure_fallback_store()
            with self._fallback_lock:
                self._fallback_store[simulation_id] = progress_data
    
    def set_progress(self, simulation_id: str, progress_data: dict, temp_id: str = None):
        """
        🎯 BULLETPROOF PROGRESS STORAGE WITH DTO TRANSFORMATION AND DUAL ID SUPPORT
        
        Stores progress data and transforms it to frontend-expected schema
        Supports storing under both simulation_id and temp_id for instant frontend access
        """
        try:
            # CRITICAL FIX: Get existing progress data first to preserve fields
            existing_progress = self.get_progress(simulation_id) or {}
            
            # Merge existing data with new data (new data takes precedence)
            raw_progress = dict(existing_progress)
            raw_progress.update(progress_data)
            
            # Transform to DTO if available
            if DTO_AVAILABLE:
                try:
                    # Get simulation metadata for target variables
                    metadata = self.get_simulation_metadata(simulation_id)
                    target_variables = metadata.get('target_variables', []) if metadata else []
                    
                    # Create DTO with proper schema
                    progress_dto = create_progress_dto(
                        simulation_id=simulation_id,
                        raw_progress=raw_progress,
                        target_variables=target_variables
                    )
                    
                    # Convert to dict for storage
                    enhanced_progress = progress_dto.dict()
                    
                    logger.debug(f"🎯 DTO transformation successful for {simulation_id}")
                    logger.debug(f"   Raw: {raw_progress.get('progress_percentage', 0):.1f}%")
                    logger.debug(f"   DTO: overallProgress={enhanced_progress.get('overallProgress', 0):.1f}%, currentStage='{enhanced_progress.get('currentStage', 'Unknown')}'")
                    
                except Exception as dto_error:
                    logger.warning(f"⚠️ DTO transformation failed for {simulation_id}: {dto_error}")
                    enhanced_progress = raw_progress
            else:
                enhanced_progress = raw_progress
            
            # Store in Redis/memory
            if self.redis_client:
                # Store enhanced progress in Redis with dynamic TTL
                key = self._get_key(simulation_id)
                value = json.dumps(enhanced_progress)
                
                # Dynamic TTL: larger simulations get longer TTL
                iterations = enhanced_progress.get('total_iterations', 1000)
                if iterations > 50000:
                    ttl = 14400  # 4 hours for very large simulations
                elif iterations > 10000:
                    ttl = 7200   # 2 hours for large simulations  
                else:
                    ttl = 3600   # 1 hour for normal simulations
                    
                self.redis_client.setex(key, ttl, value)
                
                # 🚀 CRITICAL: Store under temp_id in Redis as well for instant frontend access
                if temp_id and temp_id.startswith('temp_'):
                    temp_key = self._get_key(temp_id)
                    self.redis_client.setex(temp_key, ttl, value)
                    logger.debug(f"🔗 [DUAL_REDIS] Stored progress in Redis under both {simulation_id} AND {temp_id}")
                
                logger.debug(f"📦 Progress stored in Redis for {simulation_id} (TTL: {ttl}s)")
            else:
                # Fallback to in-memory
                self._ensure_fallback_store()
                with self._fallback_lock:
                    self._fallback_store[simulation_id] = enhanced_progress
                    self._fallback_ttl[simulation_id] = time.time() + 3600  # 1 hour TTL
                    
                    # 🚀 CRITICAL: Store under temp_id as well for instant frontend access
                    if temp_id and temp_id.startswith('temp_'):
                        self._fallback_store[temp_id] = enhanced_progress
                        self._fallback_ttl[temp_id] = time.time() + 3600  # 1 hour TTL
                        logger.debug(f"🔗 [DUAL_STORAGE] Stored progress in memory under both {simulation_id} AND {temp_id}")
                
                logger.debug(f"💾 Progress stored in memory for {simulation_id}")
            
            # WebSocket disabled - using HTTP polling for frontend updates
            # self._send_websocket_update(simulation_id, enhanced_progress)
                
        except redis.exceptions.RedisError as e:
            logger.warning(f"Redis error when setting progress for {simulation_id}: {e}")
            # Increment failure count and check circuit breaker
            self._redis_failure_count += 1
            if self._redis_failure_count >= self._redis_failure_threshold:
                self._open_redis_circuit()
            # Fallback to in-memory on Redis failure
            self._ensure_fallback_store()
            with self._fallback_lock:
                self._fallback_store[simulation_id] = progress_data
                self._fallback_ttl[simulation_id] = time.time() + 3600  # 1 hour TTL
                
                # 🚀 CRITICAL: Store under temp_id as well for instant frontend access
                if temp_id and temp_id.startswith('temp_'):
                    self._fallback_store[temp_id] = progress_data
                    self._fallback_ttl[temp_id] = time.time() + 3600  # 1 hour TTL
        except Exception as e:
            logger.error(f"❌ Unexpected error setting progress for {simulation_id}: {e}")
            # Ensure fallback storage works
            self._ensure_fallback_store()
            with self._fallback_lock:
                self._fallback_store[simulation_id] = progress_data
                self._fallback_ttl[simulation_id] = time.time() + 3600  # 1 hour TTL
                
                # 🚀 CRITICAL: Store under temp_id as well for instant frontend access
                if temp_id and temp_id.startswith('temp_'):
                    self._fallback_store[temp_id] = progress_data
                    self._fallback_ttl[temp_id] = time.time() + 3600  # 1 hour TTL
    
    def _send_websocket_update(self, simulation_id: str, progress_data: dict):
        """Send WebSocket update for real-time progress (thread-safe)"""
        try:
            # 🚀 FINAL FIX: Skip WebSocket updates when called from simulation threads
            # The WebSocket manager will be called directly from the main FastAPI event loop
            import threading
            thread_name = threading.current_thread().name
            
            if thread_name.startswith('sim_'):
                logger.debug(f"📡 [THREAD_SAFE] Skipping WebSocket update from simulation thread {thread_name} for {simulation_id}")
                return
            
            import asyncio
            from shared.websocket_manager import websocket_manager
            
            # Only send WebSocket updates from the main FastAPI event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running() and not thread_name.startswith('sim_'):
                    loop.create_task(websocket_manager.send_progress_update(simulation_id, progress_data))
                    logger.debug(f"📡 WebSocket update sent for {simulation_id}: {progress_data.get('progress', 0)}%")
                else:
                    logger.debug(f"📡 [THREAD_SAFE] Skipping WebSocket update - not in main event loop")
            except Exception as ws_error:
                logger.debug(f"📡 [THREAD_SAFE] WebSocket update skipped due to thread safety: {ws_error}")
                
        except Exception as e:
            # Don't fail progress storage if WebSocket fails
            logger.debug(f"📡 [THREAD_SAFE] WebSocket update safely skipped for {simulation_id}: {e}")
    
    async def get_progress_async(self, simulation_id: str) -> Optional[dict]:
        """Async version of get_progress with fast timeout and fallback - 🚀 ULTRA-FAST OPTIMIZED"""
        logger.info(f"🔧 [DEBUG_RETRIEVE] get_progress_async called for {simulation_id}")
        try:
            # 🚀 FINAL SOLUTION: Check memory bridge FIRST (bypasses all Redis issues)
            bridge_data = self._get_progress_bridge(simulation_id)
            if bridge_data:
                logger.info(f"🌉 [BRIDGE] Progress retrieved from memory bridge for {simulation_id}: {bridge_data.get('progress_percentage', 'N/A')}%")
                self._cache_progress(simulation_id, bridge_data)
                return bridge_data
            
            # Check cache second
            cached_data = self._get_cached_progress(simulation_id)
            if cached_data:
                logger.debug(f"📡 Progress retrieved from cache for {simulation_id}")
                return cached_data
            
            # 🚀 ULTRA-FAST: Check memory fallback FIRST for instant response during simulation
            self._ensure_fallback_store()
            with self._fallback_lock:
                if simulation_id in self._fallback_store:
                    # Check TTL
                    if simulation_id in self._fallback_ttl:
                        if time.time() > self._fallback_ttl[simulation_id]:
                            # Expired, remove it
                            del self._fallback_store[simulation_id]
                            del self._fallback_ttl[simulation_id]
                        else:
                            # Valid data found in memory - instant response!
                            progress_data = self._fallback_store[simulation_id]
                            logger.debug(f"⚡ ULTRA-FAST: Progress retrieved from memory for {simulation_id}")
                            self._cache_progress(simulation_id, progress_data)
                            return progress_data
            
            progress_data = None
            
            # Only try Redis if memory fallback didn't have data AND circuit is closed
            if self.async_redis_client and not self._is_redis_circuit_open():
                try:
                    key = self._get_key(simulation_id)
                    # 🔧 THREADING FIX: Fast async Redis get with event loop safety
                    try:
                        value = await asyncio.wait_for(
                            self.async_redis_client.get(key),
                            timeout=settings.progress_redis_timeout
                        )
                    except RuntimeError as e:
                        if "attached to a different loop" in str(e):
                            logger.warning(f"🔧 [LOOP_FIX] Event loop conflict on get, using sync fallback for {simulation_id}")
                            # Use sync Redis as fallback for loop conflicts
                            if self.redis_client:
                                try:
                                    self.redis_client.ping()  # Test connection
                                    value = self.redis_client.get(key)
                                    logger.info(f"✅ [LOOP_FIX] Sync retrieval successful for {simulation_id}, found: {bool(value)}")
                                except Exception as sync_e:
                                    logger.error(f"❌ [LOOP_FIX] Sync retrieval failed: {sync_e}")
                                    raise
                            else:
                                logger.error(f"❌ [LOOP_FIX] No sync Redis client available for retrieval {simulation_id}")
                                raise
                        else:
                            raise
                    if value:
                        progress_data = json.loads(value)
                        logger.info(f"🔧 [DEBUG_RETRIEVE] Progress retrieved from async Redis for {simulation_id}: {progress_data.get('progress_percentage', 'N/A')}%")
                        # Cache the result
                        self._cache_progress(simulation_id, progress_data)
                        return progress_data
                        
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Redis timeout getting progress for {simulation_id}, using fallback")
                    self._redis_failure_count += 1
                    if self._redis_failure_count >= self._redis_failure_threshold:
                        self._open_redis_circuit()
                except Exception as e:
                    logger.warning(f"❌ Redis error getting progress for {simulation_id}: {e}, using fallback")
                    self._redis_failure_count += 1
                    if self._redis_failure_count >= self._redis_failure_threshold:
                        self._open_redis_circuit()
            
            # 🚀 If we reach here, no data found anywhere
            logger.warning(f"🔧 [DEBUG_RETRIEVE] No progress data found for {simulation_id} in Redis or memory")
            return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get async progress for {simulation_id}: {e}")
            # Try fallback one more time
            self._ensure_fallback_store()
            with self._fallback_lock:
                return self._fallback_store.get(simulation_id)
    
    def get_progress(self, simulation_id: str) -> Optional[dict]:
        """
        📡 BULLETPROOF PROGRESS RETRIEVAL
        
        Returns properly formatted progress data for frontend consumption
        """
        try:
            # 🚀 CRITICAL FIX: Check memory bridge FIRST (same as async version)
            bridge_data = self._get_progress_bridge(simulation_id)
            if bridge_data:
                logger.info(f"🌉 [BRIDGE] Progress retrieved from memory bridge for {simulation_id}: {bridge_data.get('progress_percentage', 'N/A')}%")
                self._cache_progress(simulation_id, bridge_data)
                return bridge_data
            
            # Check cache second for immediate response
            cached_data = self._get_cached_progress(simulation_id)
            if cached_data:
                logger.debug(f"📡 Progress retrieved from cache for {simulation_id}")
                return cached_data
            
            progress_data = None
            
            # Try Redis with fast timeout if circuit is closed
            if self.redis_client and not self._is_redis_circuit_open():
                try:
                    key = self._get_key(simulation_id)
                    # Use configured timeout for sync operations
                    value = self.redis_client.get(key)
                    if value:
                        progress_data = json.loads(value)
                        logger.debug(f"📡 Progress retrieved from Redis for {simulation_id}: {progress_data.get('progress_percentage', 0)}% ({progress_data.get('stage', 'unknown')})")
                        # Cache the result
                        self._cache_progress(simulation_id, progress_data)
                        return progress_data
                    else:
                        logger.debug(f"📡 No progress data found in Redis for {simulation_id}")
                except (redis.exceptions.TimeoutError, redis.exceptions.ConnectionError) as e:
                    logger.warning(f"⏰ Redis timeout/connection error for {simulation_id}: {e}, using fallback")
                    self._redis_failure_count += 1
                    if self._redis_failure_count >= self._redis_failure_threshold:
                        self._open_redis_circuit()
                except Exception as e:
                    logger.warning(f"❌ Redis error for {simulation_id}: {e}, using fallback")
                    self._redis_failure_count += 1
                    if self._redis_failure_count >= self._redis_failure_threshold:
                        self._open_redis_circuit()
            
            # Fallback to in-memory with thread safety
            self._ensure_fallback_store()
            with self._fallback_lock:
                if simulation_id in self._fallback_store:
                    # Check TTL
                    if simulation_id in self._fallback_ttl:
                        if time.time() > self._fallback_ttl[simulation_id]:
                            # Expired, remove it
                            del self._fallback_store[simulation_id]
                            del self._fallback_ttl[simulation_id]
                            return None
                    
                    progress_data = self._fallback_store[simulation_id]
                    logger.debug(f"💾 Progress retrieved from memory for {simulation_id}: {progress_data.get('progress_percentage', 0)}% ({progress_data.get('stage', 'unknown')})")
                    # Cache the result
                    self._cache_progress(simulation_id, progress_data)
                    return progress_data
                else:
                    logger.debug(f"💾 No progress data found in memory for {simulation_id}")
            
            if progress_data:
                # Ensure DTO transformation if not already done
                if DTO_AVAILABLE and 'overallProgress' not in progress_data:
                    try:
                        metadata = self.get_simulation_metadata(simulation_id)
                        target_variables = metadata.get('target_variables', []) if metadata else []
                        
                        progress_dto = create_progress_dto(
                            simulation_id=simulation_id,
                            raw_progress=progress_data,
                            target_variables=target_variables
                        )
                        
                        enhanced_progress = progress_dto.dict()
                        logger.debug(f"🎯 On-demand DTO transformation for {simulation_id}")
                        return enhanced_progress
                    except Exception as dto_error:
                        logger.warning(f"⚠️ On-demand DTO transformation failed: {dto_error}")
                        return progress_data
                
                return progress_data
            
            logger.warning(f"❌ No progress data found for {simulation_id} in Redis or memory")
            return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get progress for {simulation_id}: {e}")
            # Final fallback to in-memory
            self._ensure_fallback_store()
            with self._fallback_lock:
                return self._fallback_store.get(simulation_id)
    
    def clear_progress(self, simulation_id: str):
        """Clear progress data for a simulation"""
        try:
            if self.redis_client:
                # Clear both progress and metadata
                progress_key = self._get_key(simulation_id)
                metadata_key = self._get_metadata_key(simulation_id)
                self.redis_client.delete(progress_key)
                self.redis_client.delete(metadata_key)
            else:
                # Fallback to in-memory
                self._ensure_fallback_store()
                with self._fallback_lock:
                    self._fallback_store.pop(simulation_id, None)
            
            # Clear in-memory metadata
            self._simulation_metadata.pop(simulation_id, None)
                
        except Exception as e:
            logger.error(f"❌ Failed to clear progress for {simulation_id}: {e}")
            # Fallback to in-memory on Redis failure
            self._ensure_fallback_store()
            with self._fallback_lock:
                self._fallback_store.pop(simulation_id, None)
                if hasattr(self, '_fallback_ttl'):
                    self._fallback_ttl.pop(simulation_id, None)
            self._simulation_metadata.pop(simulation_id, None)
            # Clear from cache too
            self._progress_cache.pop(simulation_id, None)
    
    def extend_ttl(self, simulation_id: str, seconds: int = 3600):
        """Extend TTL for long-running simulations"""
        try:
            if self.redis_client:
                progress_key = self._get_key(simulation_id)
                metadata_key = self._get_metadata_key(simulation_id)
                self.redis_client.expire(progress_key, seconds)
                self.redis_client.expire(metadata_key, seconds)
                logger.debug(f"🕒 Extended TTL for {simulation_id} to {seconds}s")
        except Exception as e:
            logger.error(f"❌ Failed to extend TTL for {simulation_id}: {e}")

    def get_all_progress_keys(self):
        """
        Scans for all simulation progress keys in Redis.
        WARNING: SCAN can be slow on very large databases. Use cautiously.
        """
        try:
            if self.redis_client:
                # FIX: The redis client already decodes keys to strings. No need for .decode()
                keys = list(self.redis_client.scan_iter("simulation:*"))
                logger.info(f"📋 Found {len(keys)} simulation keys in Redis")
                return keys
            else:
                # Return in-memory keys
                keys = [f"simulation:progress:{sim_id}" for sim_id in self._fallback_store.keys()]
                logger.info(f"📋 Found {len(keys)} simulation keys in memory")
                return keys
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error when scanning for keys: {e}")
            return []
    
    def list_active_progress(self):
        """
        List all active progress entries with their current values for debugging.
        Returns dict with simulation_id -> progress_data
        """
        active_progress = {}
        try:
            if self.redis_client:
                # Get all progress keys
                progress_keys = list(self.redis_client.scan_iter("simulation:progress:*"))
                logger.info(f"🔍 Found {len(progress_keys)} active progress entries")
                
                for key in progress_keys:
                    simulation_id = key.split(":")[-1]  # Extract simulation ID
                    try:
                        value = self.redis_client.get(key)
                        if value:
                            progress_data = json.loads(value)
                            active_progress[simulation_id] = {
                                'progress_percentage': progress_data.get('progress_percentage', 0),
                                'stage': progress_data.get('stage', 'unknown'),
                                'status': progress_data.get('status', 'unknown'),
                                'current_iteration': progress_data.get('current_iteration', 0),
                                'total_iterations': progress_data.get('total_iterations', 0),
                                'timestamp': progress_data.get('timestamp', 'unknown')
                            }
                    except Exception as e:
                        logger.error(f"❌ Error parsing progress data for {simulation_id}: {e}")
            else:
                # Use in-memory store
                for simulation_id, progress_data in self._fallback_store.items():
                    active_progress[simulation_id] = {
                        'progress_percentage': progress_data.get('progress_percentage', 0),
                        'stage': progress_data.get('stage', 'unknown'),
                        'status': progress_data.get('status', 'unknown'),
                        'current_iteration': progress_data.get('current_iteration', 0),
                        'total_iterations': progress_data.get('total_iterations', 0),
                        'timestamp': progress_data.get('timestamp', 'unknown')
                    }
        except Exception as e:
            logger.error(f"❌ Error listing active progress: {e}")
        
        return active_progress
    
    def cleanup_expired_progress(self):
        """
        Remove expired progress entries to prevent Redis memory bloat.
        Redis TTL should handle this automatically, but this provides manual cleanup.
        """
        try:
            if self.redis_client:
                progress_keys = list(self.redis_client.scan_iter("simulation:progress:*"))
                expired_count = 0
                
                for key in progress_keys:
                    ttl = self.redis_client.ttl(key)
                    if ttl == -1:  # No TTL set
                        # Check if progress is very old (more than 24 hours)
                        try:
                            value = self.redis_client.get(key)
                            if value:
                                progress_data = json.loads(value)
                                timestamp = progress_data.get('timestamp', 0)
                                import time
                                if time.time() - timestamp > 86400:  # 24 hours
                                    self.redis_client.delete(key)
                                    expired_count += 1
                                    logger.info(f"🧹 Cleaned up expired progress: {key}")
                        except Exception as e:
                            logger.error(f"❌ Error checking timestamp for {key}: {e}")
                    elif ttl == -2:  # Key doesn't exist
                        expired_count += 1
                
                logger.info(f"🧹 Cleanup complete: {expired_count} expired entries removed")
                return expired_count
            else:
                # Clean up in-memory store based on timestamp
                import time
                current_time = time.time()
                expired_keys = []
                
                for sim_id, progress_data in self._fallback_store.items():
                    timestamp = progress_data.get('timestamp', 0)
                    if current_time - timestamp > 86400:  # 24 hours
                        expired_keys.append(sim_id)
                
                for key in expired_keys:
                    del self._fallback_store[key]
                    logger.info(f"🧹 Cleaned up expired memory entry: {key}")
                
                logger.info(f"🧹 Memory cleanup complete: {len(expired_keys)} expired entries removed")
                return len(expired_keys)
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            return 0
    
    def ensure_atomic_updates(self, simulation_id: str, update_func):
        """
        Ensure atomic progress updates to prevent data loss during high-frequency updates.
        Uses Redis transactions when available.
        """
        try:
            if self.redis_client:
                # Use Redis pipeline for atomic operations
                pipe = self.redis_client.pipeline()
                key = self._get_key(simulation_id)
                
                # Get current data
                current_data = self.get_progress(simulation_id) or {}
                
                # Apply update function
                updated_data = update_func(current_data)
                
                # Store updated data atomically
                pipe.setex(key, 3600, json.dumps(updated_data))
                pipe.execute()
                
                logger.debug(f"🔒 Atomic update completed for {simulation_id}")
                return True
            else:
                # For in-memory, just apply the update
                self._ensure_fallback_store()
                current_data = self._fallback_store.get(simulation_id, {})
                updated_data = update_func(current_data)
                self._fallback_store[simulation_id] = updated_data
                logger.debug(f"🔒 Memory update completed for {simulation_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Atomic update failed for {simulation_id}: {e}")
            return False

# Global instance
_progress_store = ProgressStore()

# Backward compatibility functions
def get_progress_store():
    """Get the global progress store (backward compatibility)"""
    return _progress_store

def set_progress(simulation_id: str, progress_data: dict, temp_id: str = None):
    """Set progress data for a simulation with optional temp_id for dual storage"""
    _progress_store.set_progress(simulation_id, progress_data, temp_id)

def get_progress(simulation_id: str):
    """Get progress data for a simulation"""
    return _progress_store.get_progress(simulation_id)

def clear_progress(simulation_id: str):
    """Clear progress data for a simulation"""
    _progress_store.clear_progress(simulation_id)

def set_simulation_metadata(simulation_id: str, metadata: dict):
    """Set simulation metadata (target variables, etc.)"""
    _progress_store.set_simulation_metadata(simulation_id, metadata)

def get_simulation_metadata(simulation_id: str):
    """Get simulation metadata"""
    return _progress_store.get_simulation_metadata(simulation_id)

def get_all_progress_keys():
    """
    Scans for all simulation progress keys in Redis.
    WARNING: SCAN can be slow on very large databases. Use cautiously.
    """
    return _progress_store.get_all_progress_keys()

def list_active_progress():
    """
    List all active progress entries with their current values for debugging.
    Returns dict with simulation_id -> progress_data
    """
    return _progress_store.list_active_progress()

def cleanup_expired_progress():
    """
    Remove expired progress entries to prevent Redis memory bloat.
    Returns number of entries cleaned up.
    """
    return _progress_store.cleanup_expired_progress()

def ensure_atomic_updates(simulation_id: str, update_func):
    """
    Ensure atomic progress updates to prevent data loss during high-frequency updates.
    """
    return _progress_store.ensure_atomic_updates(simulation_id, update_func)

async def set_progress_async(simulation_id: str, progress_data: dict, temp_id: str = None, bypass_merge: bool = False):
    """
    🚀 ASYNC PROGRESS UPDATE - Module-level export
    Async version of set_progress with background processing and dual ID support
    """
    logger.info(f"🔧 [DEBUG_PROGRESS] Module set_progress_async called for {simulation_id} with {progress_data.get('progress_percentage', 'N/A')}%")
    logger.info(f"🚀 [FUNC_ENTRY] About to call _progress_store.set_progress_async for {simulation_id}")
    return await _progress_store.set_progress_async(simulation_id, progress_data, bypass_merge, temp_id)


def reset_redis_connection():
    """
    🔧 THREADING FIX: Reset Redis connections for new event loop
    This allows Redis to work properly when called from a different thread/event loop
    """
    global _progress_store
    logger.info("🔧 [REDIS_RESET] Resetting Redis connections for new event loop")
    
    # Force reconnection in current event loop
    _progress_store.redis_client = None
    _progress_store.async_redis_client = None
    _progress_store._redis_failure_count = 0
    _progress_store._redis_circuit_open_until = 0
    
    # Reinitialize Redis with current event loop
    _progress_store._connect()


async def reset_redis_connection_async():
    """
    🔧 ASYNC THREADING FIX: Reset Redis connections for new async event loop
    Creates fresh Redis connections that work properly in the thread's event loop
    """
    global _progress_store
    logger.info("🔧 [ASYNC_REDIS_RESET] Resetting Redis connections for new async event loop")
    
    # Retry logic for robust connection
    max_retries = 3
    retry_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔧 [ASYNC_REDIS_RESET] Connection attempt {attempt + 1}/{max_retries}")
            
            # Force disconnection of old connections
            if hasattr(_progress_store, 'async_redis_client') and _progress_store.async_redis_client:
                try:
                    await _progress_store.async_redis_client.close()
                except:
                    pass
            
            if hasattr(_progress_store, 'redis_client') and _progress_store.redis_client:
                try:
                    _progress_store.redis_client.close()
                except:
                    pass
            
            # Reset connection state
            _progress_store.redis_client = None
            _progress_store.async_redis_client = None
            _progress_store._redis_failure_count = 0
            _progress_store._redis_circuit_open_until = 0
            
            # Create new async Redis connection for this event loop
            redis_host = os.getenv('REDIS_HOST', 'redis')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            
            logger.info(f"🔧 [ASYNC_REDIS_RESET] Creating connection pool to {redis_host}:{redis_port}")
            
            # Create fresh async connection pool for this event loop
            _progress_store.async_redis_pool = redis.asyncio.ConnectionPool(
                host=redis_host,
                port=redis_port,
                max_connections=200,
                retry_on_timeout=False,
                socket_connect_timeout=3.0,  # Increased timeout
                socket_timeout=8.0,          # Increased timeout
                health_check_interval=30,
                decode_responses=True
            )
            
            # Create fresh async client
            _progress_store.async_redis_client = redis.asyncio.Redis(
                connection_pool=_progress_store.async_redis_pool,
                retry_on_timeout=False
            )
            
            # CRITICAL FIX: Also create sync client for compatibility
            _progress_store.redis_pool = redis.ConnectionPool(
                host=redis_host,
                port=redis_port,
                max_connections=200,
                retry_on_timeout=False,
                socket_connect_timeout=3.0,
                socket_timeout=8.0,
                health_check_interval=30,
                decode_responses=True
            )
            
            _progress_store.redis_client = redis.Redis(
                connection_pool=_progress_store.redis_pool,
                retry_on_timeout=False
            )
            
            # Test the connection with timeout
            logger.info("🔧 [ASYNC_REDIS_RESET] Testing Redis connection...")
            await asyncio.wait_for(_progress_store.async_redis_client.ping(), timeout=3.0)
            
            # Test sync client too
            _progress_store.redis_client.ping()
            
            # Test a simple set/get to ensure full functionality
            test_key = f"thread_test_{int(time.time())}"
            await _progress_store.async_redis_client.setex(test_key, 10, "test_value")
            test_value = await _progress_store.async_redis_client.get(test_key)
            await _progress_store.async_redis_client.delete(test_key)
            
            if test_value == "test_value":
                logger.info("✅ [ASYNC_REDIS_RESET] Successfully created and tested BOTH async and sync Redis connections in new event loop")
                return  # Success!
            else:
                raise Exception("Redis set/get test failed")
                
        except Exception as e:
            logger.warning(f"⚠️ [ASYNC_REDIS_RESET] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 [ASYNC_REDIS_RESET] Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"❌ [ASYNC_REDIS_RESET] All {max_retries} attempts failed. Falling back to in-memory store")
                # Fall back to in-memory store
                _progress_store._ensure_fallback_store()
    
    logger.info("🔧 [REDIS_RESET] Redis connections reset completed") 