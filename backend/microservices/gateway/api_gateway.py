"""
🌐 Enterprise API Gateway - Main Implementation
Phase 2 Week 5: Complete Microservices Decomposition
"""

import asyncio
import logging
import aiohttp
import json
from typing import Dict, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime

from service_registry import ServiceRegistry, ServiceInstance
from circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class EnterpriseAPIGateway:
    """Enterprise-grade API Gateway with service discovery, load balancing, and circuit breaking"""
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Service routing configuration
        self.service_routes = {
            "/api/users": "user-service",
            "/api/files": "file-service", 
            "/api/simulations": "simulation-service",
            "/api/results": "results-service",
            "/api/billing": "billing-service",
            "/api/notifications": "notification-service",
            # Enterprise routes
            "/enterprise": "simulation-service"  # Route enterprise endpoints to main service
        }
        
    async def initialize(self):
        """Initialize the API Gateway"""
        await self._register_services()
        
        # Start background health checking
        asyncio.create_task(self._health_check_loop())
        logger.info("✅ [API_GATEWAY] Enterprise API Gateway initialized")
        
    async def _register_services(self):
        """Register all microservices"""
        services = [
            # Main simulation service (already running on port 8000)
            ServiceInstance("simulation-service", "localhost", 8000, "/health"),
            
            # Future microservices (will be implemented in subsequent phases)
            ServiceInstance("user-service", "localhost", 8001, "/health"),
            ServiceInstance("file-service", "localhost", 8002, "/health"),
            ServiceInstance("results-service", "localhost", 8004, "/health"),
            ServiceInstance("billing-service", "localhost", 8005, "/health"),
            ServiceInstance("notification-service", "localhost", 8006, "/health"),
        ]
        
        for service in services:
            self.service_registry.register_service(service)
            self.circuit_breakers[service.service_name] = CircuitBreaker(service.service_name)
    
    async def _health_check_loop(self):
        """Background task for continuous health checking"""
        while True:
            try:
                await self.service_registry.health_check_all_services()
                await asyncio.sleep(30)  # Health check every 30 seconds
            except Exception as e:
                logger.error(f"❌ [HEALTH_CHECK] Error in health check loop: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def route_request(self, request: Request) -> JSONResponse:
        """Route incoming request to appropriate microservice"""
        path = request.url.path
        
        # Find matching service
        service_name = self._find_service_for_path(path)
        if not service_name:
            raise HTTPException(status_code=404, detail="Service not found")
        
        # Get healthy service instance
        instance = self.service_registry.get_healthy_instance(service_name)
        if not instance:
            raise HTTPException(status_code=503, detail=f"No healthy instances for {service_name}")
        
        # Get circuit breaker
        circuit_breaker = self.circuit_breakers[service_name]
        
        # Execute request with circuit breaker protection
        response = await circuit_breaker.call(
            self._forward_request, request, instance
        )
        
        return response
    
    def _find_service_for_path(self, path: str) -> Optional[str]:
        """Find which service should handle the given path"""
        for route_prefix, service_name in self.service_routes.items():
            if path.startswith(route_prefix):
                return service_name
        return None
    
    async def _forward_request(self, request: Request, instance: ServiceInstance) -> JSONResponse:
        """Forward request to the target service instance"""
        # Prepare request
        url = f"{instance.base_url}{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"
        
        headers = dict(request.headers)
        headers.pop('host', None)  # Remove host header
        
        # Get request body
        body = await request.body()
        
        # Forward request
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=url,
                headers=headers,
                data=body
            ) as response:
                content = await response.read()
                
                return JSONResponse(
                    content=json.loads(content) if content else {},
                    status_code=response.status,
                    headers=dict(response.headers)
                )

# ====================================================================
# FASTAPI APPLICATION
# ====================================================================

# Global API Gateway instance
api_gateway = EnterpriseAPIGateway()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 [API_GATEWAY] Starting Enterprise API Gateway...")
    await api_gateway.initialize()
    logger.info("✅ [API_GATEWAY] Enterprise API Gateway ready")
    
    yield
    
    # Shutdown
    logger.info("🔄 [API_GATEWAY] Shutting down...")
    logger.info("✅ [API_GATEWAY] Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Enterprise API Gateway",
    description="Central entry point for all microservices",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================================
# API GATEWAY ENDPOINTS
# ====================================================================

@app.get("/gateway/health")
async def gateway_health():
    """API Gateway health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gateway_version": "1.0.0",
        "services": {
            name: len([svc for svc in instances if svc.is_healthy])
            for name, instances in api_gateway.service_registry.services.items()
        }
    }

@app.get("/gateway/services")
async def list_services():
    """List all registered services and their status"""
    services = {}
    for name, instances in api_gateway.service_registry.services.items():
        services[name] = [
            {
                "instance_id": svc.instance_id,
                "host": svc.host,
                "port": svc.port,
                "status": svc.status.value,
                "failure_count": svc.failure_count,
                "base_url": svc.base_url
            }
            for svc in instances
        ]
    return {"services": services}

@app.get("/gateway/circuit-breakers")
async def circuit_breaker_status():
    """Get circuit breaker status for all services"""
    return {
        "circuit_breakers": {
            name: cb.status
            for name, cb in api_gateway.circuit_breakers.items()
        }
    }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def route_to_service(request: Request, path: str):
    """Route all requests to appropriate microservices"""
    return await api_gateway.route_request(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
