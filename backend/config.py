from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    # API settings
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # CORS settings
    CORS_ORIGINS: str = "*"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./montecarlo_app.db"
    
    # Admin user settings (for demo/initial setup)
    ADMIN_EMAIL: str = "admin@montecarlo.local"
    ADMIN_PASSWORD: str = Field(default="ChangeMeInProduction", env="ADMIN_PASSWORD")
    ADMIN_USERNAME: str = "admin"
    
    # File upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB (increased from 5MB)
    MAX_EXCEL_CELLS: int = 1_000_000  # 1M cells limit
    CHUNK_SIZE_CELLS: int = 100_000   # Process in chunks
    STREAMING_THRESHOLD: int = 50_000  # Stream large files
    
    # File cleanup settings
    FILE_CLEANUP_ENABLED: bool = True
    FILE_RETENTION_DAYS: int = 7  # Keep temporary files for 7 days
    SIMULATION_FILE_RETENTION_DAYS: int = 365  # Keep simulation files for 1 year
    CLEANUP_INTERVAL_HOURS: int = 6  # Run cleanup every 6 hours
    SAVED_SIMULATIONS_DIR: str = "saved_simulations_files"
    PERSISTENT_EXCEL_DIR: str = "persistent_excel_files"  # Long-term Excel file storage
    
    # Simulation settings - SUPERFAST OPTIMIZED
    DEFAULT_ITERATIONS: int = 1000
    MAX_ITERATIONS: int = 10_000_000  # SUPERFAST: 10x increase for massive simulations
    
    # GPU settings - SUPERFAST OPTIMIZED
    USE_GPU: bool = True # Master switch for GPU usage
    GPU_MEMORY_FRACTION: float = 0.8  # SUPERFAST: Increased from 0.6 to 0.8 for better performance
    MAX_CPU_FALLBACK_TASKS: int = 4 # Max concurrent tasks if GPU init fails
    GPU_MEMORY_PER_TASK_MB: int = 2048 # Estimated VRAM per GPU task
    
    # Forecasting readiness settings
    FORECASTING_ENABLED: bool = False  # Future feature flag
    DEEP_LEARNING_BACKEND: str = "pytorch"  # pytorch, tensorflow, jax
    MAX_TIME_SERIES_LENGTH: int = 100_000  # Max time series points
    FORECASTING_MEMORY_FRACTION: float = 0.3  # Reserved for forecasting models
    ENABLE_MIXED_PRECISION: bool = True  # FP16 training for efficiency

    # Authentication settings
    SECRET_KEY: str = Field(default="your-secret-key-needs-to-be-changed-in-env", env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # Increased from 30 to 8 hours for better UX
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # Added refresh token support
    
    # Auth0 settings (for token validation from frontend)
    AUTH0_DOMAIN: str = "dev-jw6k27f0v5tcgl56.eu.auth0.com"
    AUTH0_CLIENT_ID: str = "UDXorRodTlUmgkigfaWW81Rr40gKpeAJ"  # Frontend SPA Client ID
    AUTH0_CLIENT_SECRET: str = ""  # SPAs don't use client secrets
    AUTH0_AUDIENCE: str = "https://simapp.ai/api"  # SIMAPP API identifier
    AUTH0_ALGORITHMS: list[str] = ["RS256"]

    # Auth0 Management API settings (M2M Application for backend operations)
    AUTH0_MANAGEMENT_CLIENT_ID: str = "RTtzMRmBkOeygIb49xAZMzBRQnLO5QP0"  # M2M Client ID
    AUTH0_MANAGEMENT_CLIENT_SECRET: str = Field(default="", env="AUTH0_MANAGEMENT_CLIENT_SECRET")  # M2M Client Secret
    AUTH0_MANAGEMENT_AUDIENCE: str = "https://dev-jw6k27f0v5tcgl56.eu.auth0.com/api/v2/"
    
    # Auth0 Frontend settings (support both domain and IP)
    AUTH0_CALLBACK_URL: str = "https://simapp.ai/callback,http://localhost:9090/callback,http://64.71.146.187:9090/callback"
    AUTH0_LOGOUT_URL: str = "https://simapp.ai,http://localhost:9090,http://64.71.146.187:9090"
    
    # Stripe Payment settings
    STRIPE_PUBLISHABLE_KEY: str = Field(default="", env="STRIPE_PUBLISHABLE_KEY")
    STRIPE_SECRET_KEY: str = Field(default="", env="STRIPE_SECRET_KEY")
    STRIPE_WEBHOOK_SECRET: str = Field(default="", env="STRIPE_WEBHOOK_SECRET")
    STRIPE_WEBHOOK_ENDPOINT_SECRET: str = Field(default="", env="STRIPE_WEBHOOK_ENDPOINT_SECRET")  # For webhook signature verification
    
    # Simulation Webhook settings
    WEBHOOK_DEFAULT_SECRET: str = Field(default="simulation_webhook_secret_2024", env="WEBHOOK_DEFAULT_SECRET")
    WEBHOOK_TIMEOUT_SECONDS: int = 30
    WEBHOOK_MAX_RETRIES: int = 3
    WEBHOOK_RETRY_DELAYS: List[int] = [30, 300, 1800]  # 30s, 5m, 30m
        
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Progress system configuration - 🚀 ULTRA-FAST OPTIMIZED
    PROGRESS_REDIS_TIMEOUT: int = 3  # INCREASED: More reliable timeout for Redis operations (seconds)
    PROGRESS_REDIS_CONNECT_TIMEOUT: int = 2  # INCREASED: More reliable connection timeout (seconds)
    PROGRESS_CIRCUIT_BREAKER_THRESHOLD: int = 5  # FIXED: Open circuit after 5 failures instead of 1 for stability
    PROGRESS_CIRCUIT_BREAKER_DURATION: int = 5  # INCREASED: 5s circuit breaker for proper recovery
    PROGRESS_HEALTH_CHECK_INTERVAL: int = 30  # Redis health check interval (seconds)
    PROGRESS_CACHE_TTL: int = 2  # Progress response cache TTL (seconds)
    PROGRESS_POLLING_FREQUENCY: int = 2000  # Default frontend polling frequency (ms)
    PROGRESS_ENDPOINT_TIMEOUT: int = 10  # Progress endpoint timeout (seconds)
    PROGRESS_FALLBACK_ENABLED: bool = True  # Enable memory fallback when Redis fails
    PROGRESS_PERFORMANCE_LOGGING: bool = True  # Enable progress endpoint performance logging
    
    # Deployment settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    
    # Environment-specific progress settings
    @property
    def progress_redis_timeout(self) -> int:
        """Get Redis timeout based on environment"""
        if self.ENVIRONMENT == "production":
            return 2  # Slightly higher timeout in production
        return self.PROGRESS_REDIS_TIMEOUT
    
    @property
    def progress_polling_frequency(self) -> int:
        """Get polling frequency based on environment"""
        if self.ENVIRONMENT == "development":
            return 1000  # Faster polling in development
        return self.PROGRESS_POLLING_FREQUENCY

    # File Upload settings (Added)
    ALLOWED_EXCEL_MIME_TYPES: list[str] = [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # .xlsx
        "application/vnd.ms-excel" # .xls
    ]
        
    # BIG engine settings
    BIG_MAX_NODES: int = 1_000_000  # Max nodes for dependency graph
    BIG_TIMEOUT_SEC: int = 1200     # Timeout for big engine (20 min)
    BIG_BATCH_SIZE: int = 10_000    # Batch size for chunked graph traversal
    
    # Excel parsing timeout settings
    EXCEL_PARSE_TIMEOUT_SEC: int = 300  # 5 minutes timeout for Excel parsing
    EXCEL_PARSE_CHUNK_ROWS: int = 1000  # Process Excel in chunks of 1000 rows
    EXCEL_PARSE_PROGRESS_INTERVAL: int = 100  # Report progress every 100 rows
        
    class Config:
        env_file = ".env"
        extra = "ignore" # Allow extra fields in .env not defined in Settings

settings = Settings()

# Print a warning if the default SECRET_KEY is used in a non-debug environment
if not settings.DEBUG and settings.SECRET_KEY == "your-secret-key-needs-to-be-changed-in-env":
    print("="*80)
    print("WARNING: SECURITY RISK! YOU ARE USING THE DEFAULT SECRET_KEY.")
    print("PLEASE SET A STRONG, UNIQUE SECRET_KEY IN YOUR .env FILE OR ENVIRONMENT VARIABLES.")
    print("Example: openssl rand -hex 32")
    print("="*80) 