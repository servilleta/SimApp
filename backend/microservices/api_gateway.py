"""
🌐 API GATEWAY - Microservices Architecture

Enterprise API Gateway providing:
- Request routing to appropriate microservices
- Authentication and authorization
- Rate limiting and quota enforcement
- Load balancing and circuit breaking
- Request/response transformation
- API versioning and monitoring

This gateway orchestrates all microservices and provides a unified API interface.
"""

import logging
import time
import httpx
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Request, Response, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import redis
import json

# Import from monolith (during transition)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from models import User
from auth.auth0_dependencies import get_current_active_auth0_user
from core.rate_limiter import limiter

logger = logging.getLogger(__name__)

# FastAPI app for API Gateway
app = FastAPI(
    title="Enterprise API Gateway",
    description="Unified API gateway for microservices architecture",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ===============================
# SERVICE DISCOVERY CONFIGURATION
# ===============================

class ServiceConfig:
    """Service discovery and configuration management."""
    
    def __init__(self):
        # Microservice endpoints (in production, use service discovery)
        self.services = {
            "user-service": {
                "host": "localhost",
                "port": 8001,
                "base_url": "http://localhost:8001",
                "health_endpoint": "/health",
                "timeout": 30.0
            },
            "file-service": {
                "host": "localhost", 
                "port": 8002,
                "base_url": "http://localhost:8002",
                "health_endpoint": "/health",
                "timeout": 60.0  # File operations may take longer
            },
            "simulation-service": {
                "host": "localhost",
                "port": 8003,
                "base_url": "http://localhost:8003",
                "health_endpoint": "/health",
                "timeout": 120.0  # Simulations may take longer
            }
        }
        
        # HTTP client for service communication
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Circuit breaker states
        self.circuit_breakers = {service: {"failures": 0, "last_failure": None, "is_open": False} 
                                for service in self.services}
        
        # Rate limiting (Redis-based)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ [API_GATEWAY] Redis connected for rate limiting")
        except Exception as e:
            logger.warning(f"⚠️ [API_GATEWAY] Redis not available: {e}")
            self.redis_client = None
    
    def get_service_url(self, service_name: str) -> str:
        """Get the full base URL for a service."""
        if service_name not in self.services:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        return self.services[service_name]["base_url"]
    
    def get_service_timeout(self, service_name: str) -> float:
        """Get timeout for a specific service."""
        return self.services.get(service_name, {}).get("timeout", 30.0)
    
    async def check_circuit_breaker(self, service_name: str) -> bool:
        """Check if circuit breaker allows requests to service."""
        breaker = self.circuit_breakers.get(service_name, {})
        
        # If circuit is open, check if we should try again
        if breaker.get("is_open"):
            if breaker["last_failure"] and (time.time() - breaker["last_failure"]) > 60:  # 1 minute timeout
                breaker["is_open"] = False
                breaker["failures"] = 0
                logger.info(f"🔄 [CIRCUIT_BREAKER] Reset for {service_name}")
            else:
                logger.warning(f"🚫 [CIRCUIT_BREAKER] {service_name} circuit is open")
                return False
        
        return True
    
    async def record_service_success(self, service_name: str):
        """Record successful service call."""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name]["failures"] = 0
            if self.circuit_breakers[service_name]["is_open"]:
                self.circuit_breakers[service_name]["is_open"] = False
                logger.info(f"✅ [CIRCUIT_BREAKER] {service_name} circuit closed")
    
    async def record_service_failure(self, service_name: str):
        """Record failed service call."""
        if service_name in self.circuit_breakers:
            breaker = self.circuit_breakers[service_name]
            breaker["failures"] += 1
            breaker["last_failure"] = time.time()
            
            # Open circuit if too many failures
            if breaker["failures"] >= 3:
                breaker["is_open"] = True
                logger.warning(f"🚫 [CIRCUIT_BREAKER] {service_name} circuit opened due to failures")

# Global service configuration
service_config = ServiceConfig()

# ===============================
# MIDDLEWARE AND HELPERS
# ===============================

class GatewayMiddleware:
    """Gateway-specific middleware for request processing."""
    
    @staticmethod
    async def check_rate_limit(user_id: int, endpoint: str) -> bool:
        """Check rate limit for user and endpoint."""
        if not service_config.redis_client:
            return True  # Allow if Redis not available
        
        try:
            # Rate limit: 100 requests per minute per user
            key = f"rate_limit:{user_id}:{endpoint}"
            current = service_config.redis_client.get(key)
            
            if current is None:
                service_config.redis_client.setex(key, 60, 1)
                return True
            elif int(current) < 100:
                service_config.redis_client.incr(key)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ [RATE_LIMIT] Check failed: {e}")
            return True  # Fail open
    
    @staticmethod
    async def forward_request(
        service_name: str, 
        method: str, 
        path: str, 
        headers: Dict[str, str] = None,
        json_data: Dict[str, Any] = None,
        files: Dict[str, Any] = None,
        params: Dict[str, str] = None
    ) -> httpx.Response:
        """Forward request to microservice with circuit breaker protection."""
        
        # Check circuit breaker
        if not await service_config.check_circuit_breaker(service_name):
            raise HTTPException(status_code=503, detail=f"Service {service_name} temporarily unavailable")
        
        try:
            # Build URL
            base_url = service_config.get_service_url(service_name)
            full_url = f"{base_url}{path}"
            
            # Prepare request
            request_kwargs = {
                "method": method,
                "url": full_url,
                "headers": headers or {},
                "timeout": service_config.get_service_timeout(service_name)
            }
            
            # Add data based on type
            if files:
                request_kwargs["files"] = files
            elif json_data:
                request_kwargs["json"] = json_data
            
            if params:
                request_kwargs["params"] = params
            
            # Make request
            response = await service_config.http_client.request(**request_kwargs)
            
            # Record success
            await service_config.record_service_success(service_name)
            
            return response
            
        except httpx.TimeoutException:
            await service_config.record_service_failure(service_name)
            logger.error(f"⏰ [API_GATEWAY] Timeout calling {service_name}{path}")
            raise HTTPException(status_code=504, detail=f"Service {service_name} timeout")
        
        except httpx.ConnectError:
            await service_config.record_service_failure(service_name)
            logger.error(f"🔌 [API_GATEWAY] Connection error to {service_name}")
            raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")
        
        except Exception as e:
            await service_config.record_service_failure(service_name)
            logger.error(f"❌ [API_GATEWAY] Error calling {service_name}{path}: {e}")
            raise HTTPException(status_code=500, detail=f"Gateway error: {str(e)}")

gateway_middleware = GatewayMiddleware()

# ===============================
# PYDANTIC MODELS
# ===============================

class GatewayHealthResponse(BaseModel):
    status: str
    gateway_version: str
    services: Dict[str, Dict[str, Any]]
    timestamp: str

class ServiceStatusResponse(BaseModel):
    service_name: str
    status: str
    response_time_ms: float
    circuit_breaker_open: bool

# ===============================
# USER SERVICE ROUTES
# ===============================

@app.get("/api/v2/users/profile")
async def get_user_profile(
    request: Request,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get user profile from User Service."""
    
    # Rate limiting
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "profile"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Forward to User Service
    response = await gateway_middleware.forward_request(
        service_name="user-service",
        method="GET",
        path="/profile",
        headers={"Authorization": request.headers.get("authorization")}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/api/v2/users/subscription")
async def get_user_subscription(
    request: Request,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get user subscription from User Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "subscription"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await gateway_middleware.forward_request(
        service_name="user-service",
        method="GET",
        path="/subscription",
        headers={"Authorization": request.headers.get("authorization")}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/api/v2/users/api-keys")
async def get_user_api_keys(
    request: Request,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get user API keys from User Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "api-keys"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await gateway_middleware.forward_request(
        service_name="user-service",
        method="GET",
        path="/api-keys",
        headers={"Authorization": request.headers.get("authorization")}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

# ===============================
# FILE SERVICE ROUTES
# ===============================

@app.post("/api/v2/files/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    category: str = "uploads",
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Upload file to File Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "upload"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Forward file upload to File Service
    files = {"file": (file.filename, await file.read(), file.content_type)}
    
    response = await gateway_middleware.forward_request(
        service_name="file-service",
        method="POST",
        path=f"/upload?category={category}",
        headers={"Authorization": request.headers.get("authorization")},
        files=files
    )
    
    if response.status_code in [200, 201]:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/api/v2/files/list")
async def list_files(
    request: Request,
    category: Optional[str] = None,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """List files from File Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "files"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    params = {"category": category} if category else {}
    
    response = await gateway_middleware.forward_request(
        service_name="file-service",
        method="GET",
        path="/list",
        headers={"Authorization": request.headers.get("authorization")},
        params=params
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/api/v2/files/{file_id}/download")
async def download_file(
    file_id: str,
    request: Request,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Download file from File Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "download"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await gateway_middleware.forward_request(
        service_name="file-service",
        method="GET",
        path=f"/{file_id}/download",
        headers={"Authorization": request.headers.get("authorization")}
    )
    
    if response.status_code == 200:
        return Response(
            content=response.content,
            media_type=response.headers.get("content-type", "application/octet-stream"),
            headers=dict(response.headers)
        )
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

# ===============================
# SIMULATION SERVICE ROUTES
# ===============================

@app.post("/api/v2/simulations")
async def create_simulation(
    request: Request,
    simulation_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Create simulation in Simulation Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "simulations"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await gateway_middleware.forward_request(
        service_name="simulation-service",
        method="POST",
        path="/simulations",
        headers={"Authorization": request.headers.get("authorization")},
        json_data=simulation_data
    )
    
    if response.status_code in [200, 201]:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/api/v2/simulations")
async def list_simulations(
    request: Request,
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """List simulations from Simulation Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "simulations"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    params = {"status_filter": status_filter} if status_filter else {}
    
    response = await gateway_middleware.forward_request(
        service_name="simulation-service",
        method="GET",
        path="/simulations",
        headers={"Authorization": request.headers.get("authorization")},
        params=params
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/api/v2/simulations/{simulation_id}/status")
async def get_simulation_status(
    simulation_id: str,
    request: Request,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get simulation status from Simulation Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "simulation-status"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await gateway_middleware.forward_request(
        service_name="simulation-service",
        method="GET",
        path=f"/simulations/{simulation_id}/status",
        headers={"Authorization": request.headers.get("authorization")}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/api/v2/simulations/{simulation_id}/results")
async def get_simulation_results(
    simulation_id: str,
    request: Request,
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get simulation results from Simulation Service."""
    
    if not await GatewayMiddleware.check_rate_limit(current_user.id, "simulation-results"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await gateway_middleware.forward_request(
        service_name="simulation-service",
        method="GET",
        path=f"/simulations/{simulation_id}/results",
        headers={"Authorization": request.headers.get("authorization")}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

# ===============================
# GATEWAY MANAGEMENT ENDPOINTS
# ===============================

@app.get("/health", response_model=GatewayHealthResponse)
async def gateway_health_check():
    """Comprehensive health check of gateway and all services."""
    services_status = {}
    
    for service_name, config in service_config.services.items():
        try:
            start_time = time.time()
            response = await service_config.http_client.get(
                f"{config['base_url']}{config['health_endpoint']}",
                timeout=5.0
            )
            response_time = (time.time() - start_time) * 1000
            
            services_status[service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": round(response_time, 2),
                "circuit_breaker_open": service_config.circuit_breakers[service_name]["is_open"],
                "last_checked": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            services_status[service_name] = {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": service_config.circuit_breakers[service_name]["is_open"],
                "last_checked": datetime.now(timezone.utc).isoformat()
            }
    
    # Overall gateway status
    all_healthy = all(service["status"] == "healthy" for service in services_status.values())
    gateway_status = "healthy" if all_healthy else "degraded"
    
    return GatewayHealthResponse(
        status=gateway_status,
        gateway_version="2.0.0",
        services=services_status,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/gateway/services")
async def list_services():
    """List all configured services."""
    services_info = {}
    
    for service_name, config in service_config.services.items():
        try:
            response = await service_config.http_client.get(
                f"{config['base_url']}/service-info",
                timeout=5.0
            )
            
            if response.status_code == 200:
                services_info[service_name] = response.json()
            else:
                services_info[service_name] = {"error": "Service info unavailable"}
                
        except Exception as e:
            services_info[service_name] = {"error": str(e)}
    
    return {
        "gateway_version": "2.0.0",
        "total_services": len(service_config.services),
        "services": services_info
    }

@app.get("/gateway/circuit-breakers")
async def get_circuit_breaker_status():
    """Get circuit breaker status for all services."""
    return {
        "circuit_breakers": service_config.circuit_breakers,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/gateway/circuit-breakers/{service_name}/reset")
async def reset_circuit_breaker(service_name: str):
    """Reset circuit breaker for a specific service."""
    if service_name not in service_config.circuit_breakers:
        raise HTTPException(status_code=404, detail="Service not found")
    
    service_config.circuit_breakers[service_name] = {
        "failures": 0,
        "last_failure": None,
        "is_open": False
    }
    
    logger.info(f"🔄 [CIRCUIT_BREAKER] Manual reset for {service_name}")
    return {"message": f"Circuit breaker reset for {service_name}"}

# ===============================
# LEGACY COMPATIBILITY ROUTES
# ===============================

@app.get("/api/v1/health")
async def legacy_health_check():
    """Legacy health check endpoint for backward compatibility."""
    return {"status": "healthy", "version": "2.0.0", "note": "Legacy endpoint - use /health"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
