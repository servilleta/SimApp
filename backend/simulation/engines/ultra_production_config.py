"""
ULTRA ENGINE PRODUCTION CONFIGURATION
Phase 7: Production deployment configuration management
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class ResourceProfile(Enum):
    """Resource allocation profiles"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    PERFORMANCE = "performance"
    ENTERPRISE = "enterprise"

@dataclass
class UltraProductionConfig:
    """Production configuration for Ultra engine"""
    
    # Environment Configuration
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    resource_profile: ResourceProfile = ResourceProfile.STANDARD
    
    # Performance Configuration
    max_concurrent_simulations: int = 5
    max_iterations_per_simulation: int = 1000000
    gpu_memory_limit_gb: float = 8.0
    cpu_cores_limit: int = 4
    memory_limit_gb: float = 16.0
    
    # GPU Configuration
    enable_gpu_acceleration: bool = True
    gpu_block_size: int = 256
    cuda_device_id: int = 0
    enable_unified_memory: bool = True
    
    # Database Configuration
    database_url: str = "sqlite:///ultra_production.db"
    database_pool_size: int = 10
    enable_database_encryption: bool = True
    
    # Security Configuration
    enable_ssl: bool = True
    api_key_required: bool = True
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 100
    
    # Monitoring Configuration
    enable_prometheus_metrics: bool = True
    prometheus_port: int = 8001
    enable_health_checks: bool = True
    health_check_interval: int = 30
    log_level: str = "INFO"
    
    # Backup Configuration
    enable_automated_backups: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    backup_storage_path: str = "/var/backups/ultra"
    
    # Feature Flags
    enable_phase_3_excel_parsing: bool = True
    enable_phase_4_formula_optimization: bool = True
    enable_phase_5_async_processing: bool = True
    
    def __post_init__(self):
        """Adjust configurations based on resource profile"""
        if self.resource_profile == ResourceProfile.MINIMAL:
            self.max_concurrent_simulations = 1
            self.enable_gpu_acceleration = False
            self.cpu_cores_limit = 1
            self.memory_limit_gb = 2.0
            
        elif self.resource_profile == ResourceProfile.STANDARD:
            self.max_concurrent_simulations = 3
            self.cpu_cores_limit = 2
            self.memory_limit_gb = 4.0
            
        elif self.resource_profile == ResourceProfile.PERFORMANCE:
            self.max_concurrent_simulations = 5
            self.cpu_cores_limit = 4
            self.memory_limit_gb = 8.0
            
        elif self.resource_profile == ResourceProfile.ENTERPRISE:
            self.max_concurrent_simulations = 10
            self.cpu_cores_limit = 8
            self.memory_limit_gb = 16.0
            
        # Environment-specific adjustments
        if self.environment == DeploymentEnvironment.DEVELOPMENT:
            self.enable_ssl = False
            self.api_key_required = False
            self.rate_limiting_enabled = False
            self.log_level = "DEBUG"

class UltraProductionConfigManager:
    """Production configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "ultra_config.json"
        self.config = self._load_configuration()
        
    def _load_configuration(self) -> UltraProductionConfig:
        """Load configuration from file and environment variables"""
        try:
            # Load from file if exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                logger.info(f"✅ [ULTRA] Loaded configuration from {self.config_path}")
            else:
                file_config = {}
                logger.info(f"ℹ️ [ULTRA] No configuration file found, using defaults")
            
            # Load environment variable overrides
            env_overrides = self._load_environment_overrides()
            
            # Merge configurations
            merged_config = {**file_config, **env_overrides}
            
            # Create configuration object
            config = UltraProductionConfig(**merged_config)
            
            logger.info(f"🔧 [ULTRA] Configuration loaded for {config.environment.value} environment")
            return config
            
        except Exception as e:
            logger.error(f"❌ [ULTRA] Failed to load configuration: {e}")
            return UltraProductionConfig()
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables"""
        overrides = {}
        
        env_mappings = {
            "ULTRA_ENVIRONMENT": ("environment", str),
            "ULTRA_RESOURCE_PROFILE": ("resource_profile", str),
            "ULTRA_MAX_CONCURRENT": ("max_concurrent_simulations", int),
            "ULTRA_GPU_ENABLED": ("enable_gpu_acceleration", bool),
            "ULTRA_DATABASE_URL": ("database_url", str),
            "ULTRA_LOG_LEVEL": ("log_level", str),
            "ULTRA_MEMORY_LIMIT": ("memory_limit_gb", float),
            "ULTRA_CPU_LIMIT": ("cpu_cores_limit", int),
        }
        
        for env_var, (config_key, type_func) in env_mappings.items():
            if env_var in os.environ:
                try:
                    if type_func == bool:
                        overrides[config_key] = os.environ[env_var].lower() in ('true', '1', 'yes')
                    else:
                        overrides[config_key] = type_func(os.environ[env_var])
                except ValueError as e:
                    logger.warning(f"⚠️ [ULTRA] Invalid environment variable {env_var}: {e}")
        
        return overrides
    
    def get_config(self) -> UltraProductionConfig:
        """Get current configuration"""
        return self.config
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        if self.config.memory_limit_gb < 1.0:
            issues.append("Memory limit too low (minimum 1GB)")
        
        if self.config.cpu_cores_limit < 1:
            issues.append("CPU cores limit too low (minimum 1)")
        
        if self.config.max_concurrent_simulations < 1:
            issues.append("Max concurrent simulations too low (minimum 1)")
        
        if self.config.enable_gpu_acceleration and self.config.gpu_memory_limit_gb < 1.0:
            issues.append("GPU memory limit too low (minimum 1GB)")
        
        if not self.config.database_url:
            issues.append("Database URL not configured")
        
        return issues

# Global configuration manager
_config_manager: Optional[UltraProductionConfigManager] = None

def get_production_config() -> UltraProductionConfig:
    """Get global production configuration"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = UltraProductionConfigManager()
    
    return _config_manager.get_config()

def initialize_production_config(config_path: Optional[str] = None) -> UltraProductionConfigManager:
    """Initialize global production configuration"""
    global _config_manager
    
    _config_manager = UltraProductionConfigManager(config_path)
    
    # Validate configuration
    issues = _config_manager.validate_configuration()
    if issues:
        logger.warning(f"⚠️ [ULTRA] Configuration validation issues: {issues}")
    else:
        logger.info("✅ [ULTRA] Configuration validation passed")
    
    return _config_manager 