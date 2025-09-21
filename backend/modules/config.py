"""
Configuration management for modular monolith

Centralizes all configuration and provides environment-based settings.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field


class ModularConfig(BaseSettings):
    """Configuration for the modular monolith"""
    
    # Database
    database_url: str = Field(default="sqlite:///./montecarlo_app.db", env="DATABASE_URL")
    
    # Security
    secret_key: str = Field(default="your-secret-key-needs-to-be-changed", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Storage
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    results_dir: str = Field(default="results", env="RESULTS_DIR")
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    
    # Redis
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Stripe (for billing)
    stripe_secret_key: Optional[str] = Field(default=None, env="STRIPE_SECRET_KEY")
    stripe_webhook_secret: Optional[str] = Field(default=None, env="STRIPE_WEBHOOK_SECRET")
    
    # Feature flags
    enable_gpu: bool = Field(default=True, env="ENABLE_GPU")
    enable_super_engine: bool = Field(default=True, env="ENABLE_SUPER_ENGINE")
    enable_billing: bool = Field(default=False, env="ENABLE_BILLING")
    
    # Limits
    free_tier_simulations: int = Field(default=100, env="FREE_TIER_SIMULATIONS")
    free_tier_file_size_mb: int = Field(default=10, env="FREE_TIER_FILE_SIZE_MB")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    # Development
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_config() -> ModularConfig:
    """Get configuration instance"""
    return ModularConfig()


def get_service_config() -> Dict[str, Any]:
    """Get configuration dict for service container"""
    config = get_config()
    
    return {
        # Database
        "database_url": config.database_url,
        
        # Security
        "secret_key": config.secret_key,
        "algorithm": config.algorithm,
        "access_token_expire_minutes": config.access_token_expire_minutes,
        
        # Storage
        "upload_dir": config.upload_dir,
        "results_dir": config.results_dir,
        "max_file_size_mb": config.max_file_size_mb,
        
        # External services
        "redis_url": config.redis_url,
        "stripe_secret_key": config.stripe_secret_key,
        "stripe_webhook_secret": config.stripe_webhook_secret,
        
        # Feature flags
        "enable_gpu": config.enable_gpu,
        "enable_super_engine": config.enable_super_engine,
        "enable_billing": config.enable_billing,
        
        # Limits
        "free_tier_simulations": config.free_tier_simulations,
        "free_tier_file_size_mb": config.free_tier_file_size_mb,
        
        # Monitoring
        "sentry_dsn": config.sentry_dsn,
        "enable_metrics": config.enable_metrics,
        
        # Development
        "debug": config.debug,
        "reload": config.reload
    }


# Environment-specific configurations
class DevelopmentConfig(ModularConfig):
    """Development configuration"""
    debug: bool = True
    reload: bool = True
    enable_billing: bool = False


class ProductionConfig(ModularConfig):
    """Production configuration"""
    debug: bool = False
    reload: bool = False
    enable_billing: bool = True
    
    # Override defaults for production
    secret_key: str = Field(env="SECRET_KEY")  # Required in production
    database_url: str = Field(env="DATABASE_URL")  # Required in production


class TestingConfig(ModularConfig):
    """Testing configuration"""
    debug: bool = True
    database_url: str = "sqlite:///./test.db"
    enable_billing: bool = False
    redis_url: Optional[str] = None


def get_config_for_environment(env: str = None) -> ModularConfig:
    """Get configuration for specific environment"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig() 