"""
PHASE 7: PRODUCTION CONFIGURATION SYSTEM
Implementation of comprehensive production configuration for Ultra engine deployment.

This module provides production-ready configuration management for:
- Environment-specific configurations
- Performance tuning parameters
- Security configurations
- Monitoring and logging settings
- Resource allocation optimization
- Deployment automation support
"""

import os
import json
import logging
import platform
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class UltraResourceProfile(Enum):
    """Resource allocation profiles for different deployment scenarios"""
    MINIMAL = "minimal"          # 1 CPU, 2GB RAM, no GPU
    STANDARD = "standard"        # 2 CPU, 4GB RAM, optional GPU
    PERFORMANCE = "performance"  # 4 CPU, 8GB RAM, dedicated GPU
    ENTERPRISE = "enterprise"    # 8+ CPU, 16GB+ RAM, multiple GPUs

@dataclass
class UltraProductionConfig:
    """
    Comprehensive production configuration for Ultra engine
    Based on research and production deployment best practices
    """
    
    # Environment Configuration
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    resource_profile: UltraResourceProfile = UltraResourceProfile.STANDARD
    
    # Performance Configuration
    max_concurrent_simulations: int = 5
    max_iterations_per_simulation: int = 1000000
    gpu_memory_limit_gb: float = 8.0
    cpu_cores_limit: int = 4
    memory_limit_gb: float = 16.0
    
    # GPU Configuration
    enable_gpu_acceleration: bool = True
    gpu_block_size: int = 256
    gpu_memory_pool_size_gb: float = 4.0
    cuda_device_id: int = 0
    enable_unified_memory: bool = True
    
    # Database Configuration
    database_url: str = "sqlite:///ultra_production.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_connection_timeout: int = 30
    enable_database_encryption: bool = True
    
    # Security Configuration
    enable_ssl: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    api_key_required: bool = True
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 100
    
    # Monitoring Configuration
    enable_prometheus_metrics: bool = True
    prometheus_port: int = 8001
    enable_health_checks: bool = True
    health_check_interval: int = 30
    log_level: str = "INFO"
    enable_performance_profiling: bool = True
    
    # Caching Configuration
    enable_redis_cache: bool = True
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    enable_result_caching: bool = True
    
    # Backup Configuration
    enable_automated_backups: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    backup_storage_path: str = "/var/backups/ultra"
    
    # Deployment Configuration
    deployment_mode: str = "container"  # container, systemd, kubernetes
    container_registry: str = "docker.io"
    image_tag: str = "latest"
    replicas: int = 3
    
    # Feature Flags
    enable_phase_3_excel_parsing: bool = True
    enable_phase_4_formula_optimization: bool = True
    enable_phase_5_async_processing: bool = True
    enable_experimental_features: bool = False
    
    # Alerting Configuration
    enable_alerting: bool = True
    alert_webhook_url: Optional[str] = None
    alert_email_recipients: List[str] = None
    critical_memory_threshold: float = 0.85
    critical_cpu_threshold: float = 0.90
    critical_gpu_threshold: float = 0.95

    def __post_init__(self):
        """Post-initialization validation and adjustments"""
        # Adjust configurations based on resource profile
        if self.resource_profile == UltraResourceProfile.MINIMAL:
            self.max_concurrent_simulations = 1
            self.enable_gpu_acceleration = False
            self.cpu_cores_limit = 1
            self.memory_limit_gb = 2.0
            
        elif self.resource_profile == UltraResourceProfile.STANDARD:
            self.max_concurrent_simulations = 3
            self.cpu_cores_limit = 2
            self.memory_limit_gb = 4.0
            
        elif self.resource_profile == UltraResourceProfile.PERFORMANCE:
            self.max_concurrent_simulations = 5
            self.cpu_cores_limit = 4
            self.memory_limit_gb = 8.0
            
        elif self.resource_profile == UltraResourceProfile.ENTERPRISE:
            self.max_concurrent_simulations = 10
            self.cpu_cores_limit = 8
            self.memory_limit_gb = 16.0
            
        # Environment-specific adjustments
        if self.environment == DeploymentEnvironment.DEVELOPMENT:
            self.enable_ssl = False
            self.api_key_required = False
            self.rate_limiting_enabled = False
            self.log_level = "DEBUG"
            
        elif self.environment == DeploymentEnvironment.PRODUCTION:
            self.enable_ssl = True
            self.api_key_required = True
            self.rate_limiting_enabled = True
            self.log_level = "INFO"
            
        # Initialize alert recipients if not provided
        if self.alert_email_recipients is None:
            self.alert_email_recipients = []

class UltraProductionConfigManager:
    """
    Production configuration management system
    Handles loading, validation, and environment-specific configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._default_config_path()
        self.config: Optional[UltraProductionConfig] = None
        self.environment_overrides: Dict[str, Any] = {}
        
        # Load configuration
        self._load_configuration()
        
    def _default_config_path(self) -> str:
        """Get default configuration path"""
        # Check environment variable first
        if "ULTRA_CONFIG_PATH" in os.environ:
            return os.environ["ULTRA_CONFIG_PATH"]
            
        # Default paths based on environment
        if os.path.exists("/etc/ultra/config.yaml"):
            return "/etc/ultra/config.yaml"
        elif os.path.exists("config/ultra_production.yaml"):
            return "config/ultra_production.yaml"
        else:
            return "ultra_config.yaml"
    
    def _load_configuration(self):
        """Load configuration from file and environment variables"""
        try:
            # Load from file if exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                        
                logger.info(f"✅ [ULTRA] Loaded configuration from {self.config_path}")
            else:
                file_config = {}
                logger.info(f"ℹ️ [ULTRA] No configuration file found at {self.config_path}, using defaults")
            
            # Load environment variable overrides
            env_overrides = self._load_environment_overrides()
            
            # Merge configurations (env overrides take precedence)
            merged_config = {**file_config, **env_overrides}
            
            # Create configuration object
            self.config = UltraProductionConfig(**merged_config)
            
            logger.info(f"🔧 [ULTRA] Configuration loaded for {self.config.environment.value} environment")
            logger.info(f"🔧 [ULTRA] Resource profile: {self.config.resource_profile.value}")
            logger.info(f"🔧 [ULTRA] Max concurrent simulations: {self.config.max_concurrent_simulations}")
            logger.info(f"🔧 [ULTRA] GPU acceleration: {self.config.enable_gpu_acceleration}")
            
        except Exception as e:
            logger.error(f"❌ [ULTRA] Failed to load configuration: {e}")
            # Fallback to default configuration
            self.config = UltraProductionConfig()
            logger.warning("⚠️ [ULTRA] Using default configuration due to load failure")
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables"""
        overrides = {}
        
        # Define environment variable mappings
        env_mappings = {
            "ULTRA_ENVIRONMENT": ("environment", str),
            "ULTRA_RESOURCE_PROFILE": ("resource_profile", str),
            "ULTRA_MAX_CONCURRENT": ("max_concurrent_simulations", int),
            "ULTRA_GPU_ENABLED": ("enable_gpu_acceleration", bool),
            "ULTRA_DATABASE_URL": ("database_url", str),
            "ULTRA_LOG_LEVEL": ("log_level", str),
            "ULTRA_ENABLE_SSL": ("enable_ssl", bool),
            "ULTRA_API_KEY_REQUIRED": ("api_key_required", bool),
            "ULTRA_PROMETHEUS_PORT": ("prometheus_port", int),
            "ULTRA_REDIS_URL": ("redis_url", str),
            "ULTRA_BACKUP_PATH": ("backup_storage_path", str),
            "ULTRA_ALERT_WEBHOOK": ("alert_webhook_url", str),
            "ULTRA_MEMORY_LIMIT": ("memory_limit_gb", float),
            "ULTRA_CPU_LIMIT": ("cpu_cores_limit", int),
            "ULTRA_GPU_MEMORY_LIMIT": ("gpu_memory_limit_gb", float),
        }
        
        for env_var, (config_key, type_func) in env_mappings.items():
            if env_var in os.environ:
                try:
                    if type_func == bool:
                        overrides[config_key] = os.environ[env_var].lower() in ('true', '1', 'yes', 'on')
                    else:
                        overrides[config_key] = type_func(os.environ[env_var])
                    logger.debug(f"🔧 [ULTRA] Environment override: {config_key} = {overrides[config_key]}")
                except ValueError as e:
                    logger.warning(f"⚠️ [ULTRA] Invalid environment variable {env_var}: {e}")
        
        return overrides
    
    def get_config(self) -> UltraProductionConfig:
        """Get current configuration"""
        return self.config
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        if not self.config:
            issues.append("Configuration not loaded")
            return issues
        
        # Validate resource limits
        if self.config.memory_limit_gb < 1.0:
            issues.append("Memory limit too low (minimum 1GB)")
        
        if self.config.cpu_cores_limit < 1:
            issues.append("CPU cores limit too low (minimum 1)")
        
        if self.config.max_concurrent_simulations < 1:
            issues.append("Max concurrent simulations too low (minimum 1)")
        
        # Validate GPU configuration
        if self.config.enable_gpu_acceleration and self.config.gpu_memory_limit_gb < 1.0:
            issues.append("GPU memory limit too low (minimum 1GB)")
        
        # Validate database configuration
        if not self.config.database_url:
            issues.append("Database URL not configured")
        
        # Validate SSL configuration
        if self.config.enable_ssl and not self.config.ssl_cert_path:
            issues.append("SSL enabled but certificate path not configured")
        
        # Validate backup configuration
        if self.config.enable_automated_backups and not self.config.backup_storage_path:
            issues.append("Automated backups enabled but storage path not configured")
        
        # Validate alerting configuration
        if self.config.enable_alerting and not self.config.alert_webhook_url and not self.config.alert_email_recipients:
            issues.append("Alerting enabled but no notification targets configured")
        
        return issues
    
    def generate_deployment_config(self) -> Dict[str, Any]:
        """Generate deployment configuration for various platforms"""
        config = self.config
        
        deployment_config = {
            "docker": {
                "image": f"{config.container_registry}/ultra-engine:{config.image_tag}",
                "ports": [
                    {"containerPort": 8000, "name": "api"},
                    {"containerPort": config.prometheus_port, "name": "metrics"}
                ],
                "environment": {
                    "ULTRA_ENVIRONMENT": config.environment.value,
                    "ULTRA_RESOURCE_PROFILE": config.resource_profile.value,
                    "ULTRA_MAX_CONCURRENT": str(config.max_concurrent_simulations),
                    "ULTRA_GPU_ENABLED": str(config.enable_gpu_acceleration),
                    "ULTRA_DATABASE_URL": config.database_url,
                    "ULTRA_LOG_LEVEL": config.log_level,
                    "ULTRA_PROMETHEUS_PORT": str(config.prometheus_port),
                    "ULTRA_REDIS_URL": config.redis_url,
                },
                "resources": {
                    "requests": {
                        "cpu": f"{config.cpu_cores_limit}",
                        "memory": f"{config.memory_limit_gb}Gi"
                    },
                    "limits": {
                        "cpu": f"{config.cpu_cores_limit}",
                        "memory": f"{config.memory_limit_gb}Gi"
                    }
                },
                "health_check": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            },
            "kubernetes": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "ultra-engine",
                    "labels": {"app": "ultra-engine"}
                },
                "spec": {
                    "replicas": config.replicas,
                    "selector": {"matchLabels": {"app": "ultra-engine"}},
                    "template": {
                        "metadata": {"labels": {"app": "ultra-engine"}},
                        "spec": {
                            "containers": [{
                                "name": "ultra-engine",
                                "image": f"{config.container_registry}/ultra-engine:{config.image_tag}",
                                "ports": [
                                    {"containerPort": 8000, "name": "api"},
                                    {"containerPort": config.prometheus_port, "name": "metrics"}
                                ],
                                "env": [
                                    {"name": "ULTRA_ENVIRONMENT", "value": config.environment.value},
                                    {"name": "ULTRA_RESOURCE_PROFILE", "value": config.resource_profile.value},
                                    {"name": "ULTRA_MAX_CONCURRENT", "value": str(config.max_concurrent_simulations)},
                                    {"name": "ULTRA_GPU_ENABLED", "value": str(config.enable_gpu_acceleration)},
                                    {"name": "ULTRA_DATABASE_URL", "value": config.database_url},
                                    {"name": "ULTRA_LOG_LEVEL", "value": config.log_level},
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": f"{config.cpu_cores_limit}",
                                        "memory": f"{config.memory_limit_gb}Gi"
                                    },
                                    "limits": {
                                        "cpu": f"{config.cpu_cores_limit}",
                                        "memory": f"{config.memory_limit_gb}Gi"
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 30
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/ready", "port": 8000},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }]
                        }
                    }
                }
            },
            "systemd": {
                "unit_file": f"""
[Unit]
Description=Ultra Monte Carlo Engine
After=network.target

[Service]
Type=simple
User=ultra
Group=ultra
WorkingDirectory=/opt/ultra
ExecStart=/opt/ultra/venv/bin/python -m ultra_engine
Environment=ULTRA_ENVIRONMENT={config.environment.value}
Environment=ULTRA_RESOURCE_PROFILE={config.resource_profile.value}
Environment=ULTRA_MAX_CONCURRENT={config.max_concurrent_simulations}
Environment=ULTRA_GPU_ENABLED={config.enable_gpu_acceleration}
Environment=ULTRA_DATABASE_URL={config.database_url}
Environment=ULTRA_LOG_LEVEL={config.log_level}
Restart=always
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
""".strip()
            }
        }
        
        return deployment_config
    
    def save_configuration(self, path: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        try:
            save_path = path or self.config_path
            
            config_dict = asdict(self.config)
            
            # Convert enums to strings for serialization
            config_dict['environment'] = config_dict['environment'].value
            config_dict['resource_profile'] = config_dict['resource_profile'].value
            
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                with open(save_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                with open(save_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"✅ [ULTRA] Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [ULTRA] Failed to save configuration: {e}")
            return False

# Global configuration manager instance
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

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for deployment planning"""
    try:
        import psutil
        
        # Get CPU information
        cpu_info = {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        }
        
        # Get memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
        }
        
        # Get disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            "total_gb": disk.total / (1024**3),
            "free_gb": disk.free / (1024**3),
            "used_gb": disk.used / (1024**3),
            "percent": (disk.used / disk.total) * 100,
        }
        
    except ImportError:
        cpu_info = {"cpu_count": os.cpu_count()}
        memory_info = {"total_gb": "unknown"}
        disk_info = {"total_gb": "unknown"}
    
    # Get GPU information if available
    gpu_info = {}
    try:
        import cupy as cp
        gpu_info = {
            "cuda_available": True,
            "device_count": cp.cuda.runtime.getDeviceCount(),
            "current_device": cp.cuda.runtime.getDevice(),
            "memory_info": cp.cuda.runtime.memGetInfo(),
        }
    except ImportError:
        gpu_info = {"cuda_available": False}
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture(),
        "cpu": cpu_info,
        "memory": memory_info,
        "disk": disk_info,
        "gpu": gpu_info,
        "hostname": platform.node(),
        "timestamp": os.times(),
    }

# Factory function
def create_production_config(
    environment: str = "production",
    resource_profile: str = "standard",
    **kwargs
) -> UltraProductionConfig:
    """Create production configuration with specified parameters"""
    
    config_kwargs = {
        "environment": DeploymentEnvironment(environment),
        "resource_profile": UltraResourceProfile(resource_profile),
        **kwargs
    }
    
    return UltraProductionConfig(**config_kwargs) 