# Modular Monolith Refactoring Plan

## Goal: Prepare for microservices WITHOUT delaying your 3-month launch

### Current Structure → Modular Structure (1 Week Sprint)

## Day 1-2: Create Module Structure

```bash
# Create new module structure alongside existing code
cd backend
mkdir -p modules/{auth,simulation,excel_parser,storage}/
touch modules/__init__.py
touch modules/{auth,simulation,excel_parser,storage}/__init__.py
```

### Move Auth Module
```python
# backend/modules/auth/service.py
from typing import Optional, Dict
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

class AuthService:
    """Centralized authentication service"""
    
    def __init__(self, db_session: Session, secret_key: str, algorithm: str = "HS256"):
        self.db = db_session
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    async def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user with email/password"""
        user = self.db.query(User).filter(User.email == email).first()
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        return {"id": user.id, "email": user.email, "tier": user.subscription_tier}
    
    async def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def hash_password(self, password: str) -> str:
        return self.pwd_context.hash(password)

# backend/modules/auth/router.py
from fastapi import APIRouter, Depends, HTTPException
from .service import AuthService
from .schemas import UserLogin, Token

def create_auth_router(auth_service: AuthService) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["auth"])
    
    @router.post("/token", response_model=Token)
    async def login(credentials: UserLogin):
        user = await auth_service.authenticate_user(credentials.email, credentials.password)
        if not user:
            raise HTTPException(401, "Invalid credentials")
        
        access_token = await auth_service.create_access_token(data={"sub": user["email"]})
        return {"access_token": access_token, "token_type": "bearer"}
    
    return router
```

### Move Simulation Module
```python
# backend/modules/simulation/service.py
from typing import Dict, List, Optional
import asyncio
from datetime import datetime

class SimulationService:
    """Centralized simulation orchestration"""
    
    def __init__(self, db, redis, storage_service, limits_service):
        self.db = db
        self.redis = redis
        self.storage = storage_service
        self.limits = limits_service
        
        # Existing engines remain unchanged
        from simulation.engine import PowerEngine
        from simulation.arrow_engine import ArrowEngine
        from simulation.enhanced_engine import EnhancedEngine
        
        self.engines = {
            "power": PowerEngine(),
            "arrow": ArrowEngine(), 
            "enhanced": EnhancedEngine()
        }
    
    async def create_simulation(self, user_id: int, file_id: str, config: Dict) -> str:
        """Create new simulation with all validations"""
        
        # Check user limits
        can_simulate, reason = await self.limits.check_simulation_allowed(user_id)
        if not can_simulate:
            raise ValueError(f"Simulation not allowed: {reason}")
        
        # Validate file
        file_data = await self.storage.get_file(file_id)
        if not file_data:
            raise ValueError("File not found")
        
        # Create simulation record
        sim_id = str(uuid.uuid4())
        await self.redis.set(f"sim:{sim_id}", json.dumps({
            "user_id": user_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "config": config
        }))
        
        # Queue for processing
        await self.redis.lpush("simulation_queue", sim_id)
        
        # Track usage
        await self.limits.increment_usage(user_id, "simulations")
        
        return sim_id
    
    async def run_simulation(self, sim_id: str) -> Dict:
        """Execute simulation - called by worker"""
        sim_data = await self.redis.get(f"sim:{sim_id}")
        if not sim_data:
            raise ValueError("Simulation not found")
        
        sim = json.loads(sim_data)
        engine = self.engines.get(sim["config"]["engine"], self.engines["power"])
        
        # Run simulation (existing logic)
        results = await engine.run(sim["config"])
        
        # Store results
        await self.storage.save_results(sim_id, results)
        
        # Update status
        sim["status"] = "completed"
        await self.redis.set(f"sim:{sim_id}", json.dumps(sim))
        
        return results
```

## Day 3-4: Create Service Container

```python
# backend/modules/container.py
from typing import Optional
from sqlalchemy.orm import Session
from .auth.service import AuthService
from .simulation.service import SimulationService
from .excel_parser.service import ExcelParserService
from .storage.service import StorageService

class ServiceContainer:
    """Dependency injection container for all services"""
    
    _instance: Optional['ServiceContainer'] = None
    
    def __init__(self):
        self._services = {}
    
    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(self, db: Session, redis, config: dict):
        """Initialize all services with dependencies"""
        
        # Core services
        self._services['auth'] = AuthService(
            db_session=db,
            secret_key=config['SECRET_KEY']
        )
        
        self._services['storage'] = StorageService(
            upload_path=config['UPLOAD_PATH'],
            max_file_size=config['MAX_FILE_SIZE']
        )
        
        self._services['limits'] = LimitsService(
            db=db,
            redis=redis,
            free_tier_limits=config['FREE_TIER_LIMITS']
        )
        
        self._services['simulation'] = SimulationService(
            db=db,
            redis=redis,
            storage_service=self._services['storage'],
            limits_service=self._services['limits']
        )
        
        self._services['excel_parser'] = ExcelParserService(
            storage_service=self._services['storage']
        )
    
    def get(self, service_name: str):
        """Get service instance"""
        return self._services.get(service_name)
    
    @property
    def auth(self) -> AuthService:
        return self._services['auth']
    
    @property
    def simulation(self) -> SimulationService:
        return self._services['simulation']
```

## Day 5: Update Main App

```python
# backend/main.py
from fastapi import FastAPI
from modules.container import ServiceContainer
from modules.auth.router import create_auth_router
from modules.simulation.router import create_simulation_router
from modules.excel_parser.router import create_excel_parser_router

def create_app() -> FastAPI:
    app = FastAPI(title="Monte Carlo Simulator API")
    
    # Initialize service container
    container = ServiceContainer.get_instance()
    
    @app.on_event("startup")
    async def startup_event():
        # Initialize services with dependencies
        container.initialize(
            db=next(get_db()),
            redis=get_redis_client(),
            config=get_config()
        )
    
    # Register routers (each maps to future microservice)
    app.include_router(create_auth_router(container.auth))
    app.include_router(create_simulation_router(container.simulation))
    app.include_router(create_excel_parser_router(container.excel_parser))
    
    # Health check shows service status
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "services": {
                "auth": "ready",
                "simulation": "ready",
                "excel_parser": "ready",
                "storage": "ready"
            }
        }
    
    return app

app = create_app()
```

## Benefits of This Approach

### 1. **Zero Launch Delay**
- Refactoring can be done in 1 week
- No functionality changes
- No new infrastructure needed

### 2. **Future-Proof**
Each module can be extracted by:
```python
# Future: Replace service with HTTP client
class SimulationServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def create_simulation(self, user_id: int, file_id: str, config: Dict) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/simulations",
                json={"user_id": user_id, "file_id": file_id, "config": config}
            )
            return response.json()["simulation_id"]

# In container.py, just swap implementation:
self._services['simulation'] = SimulationServiceClient("http://simulation-service:8001")
```

### 3. **Testing Benefits**
```python
# Easy to test modules in isolation
def test_simulation_service():
    mock_storage = Mock()
    mock_limits = Mock()
    
    service = SimulationService(
        db=mock_db,
        redis=mock_redis,
        storage_service=mock_storage,
        limits_service=mock_limits
    )
    
    # Test without dependencies
    result = await service.create_simulation(user_id=1, file_id="test", config={})
    assert result is not None
```

### 4. **Gradual Migration Path**

#### Phase 1 (Current): All in one process
```
┌──────────────────────────────┐
│      Monolithic App          │
│  ┌────────┐ ┌────────────┐  │
│  │  Auth  │ │ Simulation │  │
│  └────────┘ └────────────┘  │
│  ┌────────┐ ┌────────────┐  │
│  │ Parser │ │  Storage   │  │
│  └────────┘ └────────────┘  │
└──────────────────────────────┘
```

#### Phase 2 (Month 7): Extract simulation service
```
┌─────────────────┐     ┌──────────────────┐
│   Main App      │────▶│ Simulation       │
│  ┌──────┐       │ HTTP│  Microservice    │
│  │ Auth │       │     │  - GPU enabled   │
│  └──────┘       │     │  - Auto-scaling  │
│  ┌──────┐       │     └──────────────────┘
│  │Parser│       │
│  └──────┘       │
└─────────────────┘
```

## Summary

This modular approach gives you:
- ✅ Same 3-month launch timeline
- ✅ Clean separation for future extraction
- ✅ Better testing and maintenance
- ✅ Flexibility to stay monolithic if needed
- ✅ Clear path to microservices when ready

No over-engineering, just smart preparation! 🚀 