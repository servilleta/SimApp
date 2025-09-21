"""
Base service interfaces and protocols for modular monolith architecture.

These protocols define the contracts that each service module must implement,
making it easy to swap implementations or extract to microservices later.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, List, Optional, Any
from datetime import datetime


class AuthServiceProtocol(Protocol):
    """Authentication service interface"""
    
    async def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user with email/password"""
        ...
    
    async def create_access_token(self, data: dict, expires_delta: Optional[int] = None) -> str:
        """Create JWT access token"""
        ...
    
    async def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        ...
    
    async def create_user(self, user_data: dict) -> Dict:
        """Create new user account"""
        ...
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        ...


class SimulationServiceProtocol(Protocol):
    """Simulation service interface"""
    
    async def create_simulation(self, user_id: int, file_id: str, config: Dict) -> str:
        """Create new simulation job"""
        ...
    
    async def run_simulation(self, simulation_id: str) -> Dict:
        """Execute simulation"""
        ...
    
    async def get_simulation_status(self, simulation_id: str) -> Dict:
        """Get simulation status and progress"""
        ...
    
    async def get_simulation_results(self, simulation_id: str) -> Optional[Dict]:
        """Get simulation results"""
        ...
    
    async def list_user_simulations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """List simulations for a user"""
        ...


class ExcelParserServiceProtocol(Protocol):
    """Excel parser service interface"""
    
    async def parse_file(self, file_path: str) -> Dict:
        """Parse Excel file and extract formulas"""
        ...
    
    async def validate_file(self, file_path: str) -> Dict:
        """Validate Excel file for security and format"""
        ...
    
    async def get_file_info(self, file_id: str) -> Optional[Dict]:
        """Get parsed file information"""
        ...
    
    async def extract_dependencies(self, file_id: str) -> Dict:
        """Extract formula dependencies"""
        ...


class StorageServiceProtocol(Protocol):
    """Storage service interface"""
    
    async def save_file(self, file_content: bytes, filename: str, user_id: int) -> str:
        """Save uploaded file securely"""
        ...
    
    async def get_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve file content"""
        ...
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete file"""
        ...
    
    async def save_results(self, simulation_id: str, results: Dict) -> bool:
        """Save simulation results"""
        ...
    
    async def get_results(self, simulation_id: str) -> Optional[Dict]:
        """Get simulation results"""
        ...


class LimitsServiceProtocol(Protocol):
    """User limits and quota service interface"""
    
    async def check_simulation_allowed(self, user_id: int) -> tuple[bool, str]:
        """Check if user can run another simulation"""
        ...
    
    async def increment_usage(self, user_id: int, metric: str, amount: int = 1) -> None:
        """Increment usage counter"""
        ...
    
    async def get_usage(self, user_id: int) -> Dict:
        """Get user's current usage"""
        ...
    
    async def get_limits(self, user_id: int) -> Dict:
        """Get user's current limits"""
        ...
    
    async def reset_monthly_usage(self, user_id: int) -> None:
        """Reset monthly usage counters"""
        ...


class BillingServiceProtocol(Protocol):
    """Billing and subscription service interface"""
    
    async def create_subscription(self, user_id: int, plan: str) -> Dict:
        """Create new subscription"""
        ...
    
    async def cancel_subscription(self, user_id: int) -> bool:
        """Cancel subscription"""
        ...
    
    async def get_subscription(self, user_id: int) -> Optional[Dict]:
        """Get user's subscription details"""
        ...
    
    async def process_payment(self, user_id: int, amount: int) -> Dict:
        """Process payment"""
        ...


# Base service class for common functionality
class BaseService(ABC):
    """Base class for all services with common functionality"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize service (called during startup)"""
        self.initialized = True
    
    async def shutdown(self) -> None:
        """Cleanup service (called during shutdown)"""
        self.initialized = False
    
    def health_check(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service": self.name,
            "status": "healthy" if self.initialized else "not_initialized",
            "timestamp": datetime.utcnow().isoformat()
        }


# Service registry for dependency injection
class ServiceRegistry:
    """Registry to manage service dependencies"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._initialized = False
    
    def register(self, name: str, service: Any) -> None:
        """Register a service"""
        self._services[name] = service
    
    def get(self, name: str) -> Any:
        """Get a service by name"""
        if name not in self._services:
            raise ValueError(f"Service '{name}' not registered")
        return self._services[name]
    
    async def initialize_all(self) -> None:
        """Initialize all registered services"""
        for service in self._services.values():
            if hasattr(service, 'initialize'):
                await service.initialize()
        self._initialized = True
    
    async def shutdown_all(self) -> None:
        """Shutdown all services"""
        for service in self._services.values():
            if hasattr(service, 'shutdown'):
                await service.shutdown()
        self._initialized = False
    
    def health_check_all(self) -> Dict[str, Any]:
        """Get health status of all services"""
        health = {
            "registry_status": "healthy" if self._initialized else "not_initialized",
            "services": {}
        }
        
        for name, service in self._services.items():
            if hasattr(service, 'health_check'):
                health["services"][name] = service.health_check()
            else:
                health["services"][name] = {
                    "service": name,
                    "status": "unknown",
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return health


# Dependency injection decorators
def inject_service(service_name: str):
    """Decorator to inject service dependencies"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This will be implemented by the service container
            return func(*args, **kwargs)
        return wrapper
    return decorator 