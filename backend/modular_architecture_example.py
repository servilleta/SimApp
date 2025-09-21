# Example of modular monolith structure ready for microservices

from abc import ABC, abstractmethod
from typing import Protocol

# Service interfaces (these become API contracts later)
class AuthServiceProtocol(Protocol):
    async def authenticate(self, email: str, password: str) -> dict:
        ...
    
    async def create_user(self, user_data: dict) -> dict:
        ...

class SimulationServiceProtocol(Protocol):
    async def run_simulation(self, config: dict) -> str:
        ...
    
    async def get_results(self, simulation_id: str) -> dict:
        ...

class BillingServiceProtocol(Protocol):
    async def create_subscription(self, user_id: int, plan: str) -> dict:
        ...
    
    async def check_usage(self, user_id: int) -> dict:
        ...

# Modular implementations
class AuthService:
    """Auth module - easily extractable to microservice"""
    def __init__(self, db, redis):
        self.db = db
        self.redis = redis
    
    async def authenticate(self, email: str, password: str) -> dict:
        # All auth logic contained here
        pass
    
    async def create_user(self, user_data: dict) -> dict:
        # User creation logic
        pass

class SimulationService:
    """Simulation module - the heaviest, first to extract"""
    def __init__(self, db, redis, engine_factory):
        self.db = db
        self.redis = redis
        self.engine_factory = engine_factory
    
    async def run_simulation(self, config: dict) -> str:
        # Queue simulation
        pass
    
    async def get_results(self, simulation_id: str) -> dict:
        # Get results
        pass

# Dependency injection container
class ServiceContainer:
    """Makes it easy to swap implementations later"""
    def __init__(self):
        # In monolith: direct instantiation
        self.auth_service = AuthService(db, redis)
        self.simulation_service = SimulationService(db, redis, engine_factory)
        self.billing_service = BillingService(stripe_client, db)
        
        # Future: these become HTTP/gRPC clients
        # self.auth_service = AuthServiceClient("http://auth-service")
        # self.simulation_service = SimulationServiceClient("http://sim-service")

# Router structure mirrors future API Gateway
from fastapi import APIRouter, Depends

def create_auth_router(auth_service: AuthServiceProtocol) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["auth"])
    
    @router.post("/login")
    async def login(credentials: dict):
        return await auth_service.authenticate(
            credentials["email"], 
            credentials["password"]
        )
    
    return router

# Main app assembly
def create_app():
    app = FastAPI()
    container = ServiceContainer()
    
    # Each router maps to a future microservice
    app.include_router(create_auth_router(container.auth_service))
    app.include_router(create_simulation_router(container.simulation_service))
    app.include_router(create_billing_router(container.billing_service))
    
    return app 