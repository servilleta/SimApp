"""
Service Container - Dependency Injection and Service Management

This container manages all services and their dependencies, making it easy
to swap implementations or extract services to microservices later.
"""

import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

from modules.base import ServiceRegistry
from .auth.service import AuthService
# from simulation.service import SimulationService  # TODO: Create SimulationService class
from .storage.service import StorageService
from .limits.service import LimitsService
from .billing.service import BillingService
from .excel_parser.service import ExcelParserService
from .security.service import SecurityService


logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Service container for dependency injection and service management
    
    This container:
    1. Initializes all services with their dependencies
    2. Manages service lifecycle (startup/shutdown)
    3. Provides dependency injection
    4. Handles service health checks
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ServiceRegistry()
        self._initialized = False
        self._db_session = None
        self._redis = None
    
    async def initialize(self, db_session: Session, redis_client=None):
        """Initialize all services with their dependencies"""
        if self._initialized:
            return
        
        self._db_session = db_session
        self._redis = redis_client
        
        try:
            # Initialize services in dependency order
            await self._initialize_core_services()
            await self._initialize_business_services()
            await self._initialize_integration_services()
            
            # Initialize the registry
            await self.registry.initialize_all()
            
            self._initialized = True
            logger.info("Service container initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service container: {e}")
            raise
    
    async def _initialize_core_services(self):
        """Initialize core services (no dependencies)"""
        
        # Storage service
        storage_service = StorageService(
            upload_dir=self.config.get("upload_dir", "uploads"),
            results_dir=self.config.get("results_dir", "results")
        )
        self.registry.register("storage", storage_service)
        
        # Limits service
        limits_service = LimitsService()
        if self._redis:
            limits_service.set_redis(self._redis)
        self.registry.register("limits", limits_service)
        
        # Auth service
        auth_service = AuthService(
            secret_key=self.config.get("secret_key", "your-secret-key-needs-to-be-changed"),
            algorithm=self.config.get("algorithm", "HS256"),
            access_token_expire_minutes=self.config.get("access_token_expire_minutes", 30)
        )
        auth_service.set_db_session(self._db_session)
        self.registry.register("auth", auth_service)
        
        # Security service (depends on auth service)
        security_service = SecurityService(
            auth_service=auth_service,
            redis_url=self.config.get("redis_url", "redis://localhost:6379")
        )
        self.registry.register("security", security_service)
        
        logger.info("Core services initialized")
    
    async def _initialize_business_services(self):
        """Initialize business services (depend on core services)"""
        
        # Get dependencies
        storage_service = self.registry.get("storage")
        limits_service = self.registry.get("limits")
        
        # Simulation service
        # TODO: Implement SimulationService class and uncomment below
        # simulation_service = SimulationService(
        #     storage_service=storage_service,
        #     limits_service=limits_service
        # )
        # if self._redis:
        #     simulation_service.set_redis(self._redis)
        # self.registry.register("simulation", simulation_service)
        
        # Excel parser service
        excel_parser_service = ExcelParserService(
            storage_service=storage_service
        )
        self.registry.register("excel_parser", excel_parser_service)
        
        logger.info("Business services initialized")
    
    async def _initialize_integration_services(self):
        """Initialize integration services (external dependencies)"""
        
        # Billing service (Stripe integration)
        billing_service = BillingService(
            stripe_key=self.config.get("stripe_secret_key"),
            webhook_secret=self.config.get("stripe_webhook_secret")
        )
        self.registry.register("billing", billing_service)
        
        logger.info("Integration services initialized")
    
    async def shutdown(self):
        """Shutdown all services"""
        if not self._initialized:
            return
        
        try:
            await self.registry.shutdown_all()
            self._initialized = False
            logger.info("Service container shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during service container shutdown: {e}")
    
    def get_service(self, name: str) -> Any:
        """Get a service by name"""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        return self.registry.get(name)
    
    def health_check(self) -> Dict[str, Any]:
        """Get health status of all services"""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "services": {}
            }
        
        return self.registry.health_check_all()
    
    @asynccontextmanager
    async def get_auth_service(self):
        """Context manager to get auth service with proper session management"""
        auth_service = self.get_service("auth")
        try:
            yield auth_service
        finally:
            # Any cleanup if needed
            pass
    
    @asynccontextmanager
    async def get_simulation_service(self):
        """Context manager to get simulation service"""
        simulation_service = self.get_service("simulation")
        try:
            yield simulation_service
        finally:
            # Any cleanup if needed
            pass
    
    def create_routers(self):
        """Create FastAPI routers with injected services"""
        # DISABLED FOR PRIVATE LAUNCH: The modular auth router has dependency issues
        # We'll use the existing auth system which works properly
        
        routers = []
        
        # Auth router - DISABLED temporarily due to get_current_active_user dependency issue
        # auth_service = self.get_service("auth")
        # from .auth.router import create_auth_router
        # auth_router = create_auth_router(auth_service)
        # routers.append(auth_router)
        
        # TODO: Fix the auth service protocol to include get_current_active_user method
        # For now, the main app will continue using the existing auth router
        
        return routers
    
    def get_dependencies(self):
        """Get dependency injection functions for FastAPI"""
        
        async def get_current_user(token: str):
            """Dependency to get current user"""
            auth_service = self.get_service("auth")
            user = await auth_service.verify_token(token)
            if not user:
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials"
                )
            return user
        
        async def get_auth_service():
            """Dependency to get auth service"""
            return self.get_service("auth")
        
        async def get_simulation_service():
            """Dependency to get simulation service"""
            return self.get_service("simulation")
        
        async def get_limits_service():
            """Dependency to get limits service"""
            return self.get_service("limits")
        
        return {
            "get_current_user": get_current_user,
            "get_auth_service": get_auth_service,
            "get_simulation_service": get_simulation_service,
            "get_limits_service": get_limits_service
        }


# Create placeholder services for modules we haven't implemented yet
class BillingService:
    """Placeholder billing service"""
    
    def __init__(self, stripe_key=None, webhook_secret=None):
        self.stripe_key = stripe_key
        self.webhook_secret = webhook_secret
    
    async def initialize(self):
        pass
    
    async def shutdown(self):
        pass
    
    def health_check(self):
        return {
            "service": "billing",
            "status": "placeholder",
            "timestamp": "2025-01-01T00:00:00"
        }


class ExcelParserService:
    """Placeholder Excel parser service"""
    
    def __init__(self, storage_service):
        self.storage_service = storage_service
    
    async def initialize(self):
        pass
    
    async def shutdown(self):
        pass
    
    def health_check(self):
        return {
            "service": "excel_parser",
            "status": "placeholder",
            "timestamp": "2025-01-01T00:00:00"
        }


# Global container instance
_global_container: Optional[ServiceContainer] = None

def get_service_container() -> ServiceContainer:
    """Get the global service container instance"""
    global _global_container
    if _global_container is None:
        # Create container with default config
        config = {
            "upload_dir": "uploads",
            "results_dir": "results",
            "secret_key": "your-secret-key-needs-to-be-changed",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30,
            "redis_url": "redis://localhost:6379",
            "stripe_secret_key": None,
            "stripe_webhook_secret": None
        }
        _global_container = ServiceContainer(config)
    return _global_container

def set_service_container(container: ServiceContainer) -> None:
    """Set the global service container instance"""
    global _global_container
    _global_container = container 