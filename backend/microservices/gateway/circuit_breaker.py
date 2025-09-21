"""
🔴 Circuit Breaker Pattern Implementation
Enterprise API Gateway Component - Fault Tolerance
"""

import asyncio
import logging
from typing import Callable, Any
from datetime import datetime
from enum import Enum
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit Breaker pattern implementation for fault tolerance"""
    
    def __init__(self, service_name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, timeout: int = 30):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"🔄 [CIRCUIT_BREAKER] {self.service_name} attempting recovery (HALF_OPEN)")
            else:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Circuit breaker OPEN for {self.service_name} - failing fast"
                )
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            await self._on_success()
            return result
            
        except asyncio.TimeoutError:
            await self._on_failure("Timeout")
            raise HTTPException(status_code=504, detail=f"Timeout calling {self.service_name}")
            
        except Exception as e:
            await self._on_failure(str(e))
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"✅ [CIRCUIT_BREAKER] {self.service_name} recovered (CLOSED)")
            
        self.failure_count = 0
        
    async def _on_failure(self, error: str):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"🔴 [CIRCUIT_BREAKER] {self.service_name} OPEN - {error}")
        else:
            logger.warning(f"⚠️ [CIRCUIT_BREAKER] {self.service_name} failure {self.failure_count}/{self.failure_threshold}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return False
        return (datetime.utcnow() - self.last_failure_time).seconds >= self.recovery_timeout

    @property
    def status(self) -> dict:
        """Get current circuit breaker status"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "service_name": self.service_name
        }
