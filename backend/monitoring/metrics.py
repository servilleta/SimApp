"""
Production Metrics and Monitoring System
Phase 5: Production Deployment
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from fastapi import Request, Response
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Prometheus metrics registry
REGISTRY = CollectorRegistry()

# Application metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

SIMULATION_COUNT = Counter(
    'simulations_total',
    'Total simulations run',
    ['engine', 'status'],
    registry=REGISTRY
)

SIMULATION_DURATION = Histogram(
    'simulation_duration_seconds',
    'Simulation duration in seconds',
    ['engine'],
    registry=REGISTRY
)

ACTIVE_USERS = Gauge(
    'active_users_total',
    'Number of active users',
    registry=REGISTRY
)

ACTIVE_SIMULATIONS = Gauge(
    'active_simulations_total',
    'Number of active simulations',
    registry=REGISTRY
)

FILE_UPLOADS = Counter(
    'file_uploads_total',
    'Total file uploads',
    ['status'],
    registry=REGISTRY
)

ERROR_COUNT = Counter(
    'errors_total',
    'Total errors',
    ['type', 'severity'],
    registry=REGISTRY
)

# System metrics
CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    registry=REGISTRY
)

DISK_USAGE = Gauge(
    'disk_usage_bytes',
    'Disk usage in bytes',
    ['mount_point'],
    registry=REGISTRY
)

GPU_USAGE = Gauge(
    'gpu_usage_percent',
    'GPU usage percentage',
    ['gpu_id'],
    registry=REGISTRY
)

GPU_MEMORY = Gauge(
    'gpu_memory_bytes',
    'GPU memory usage in bytes',
    ['gpu_id', 'type'],
    registry=REGISTRY
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections',
    registry=REGISTRY
)

REDIS_CONNECTIONS = Gauge(
    'redis_connections_active',
    'Active Redis connections',
    registry=REGISTRY
)

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_metrics_task: Optional[asyncio.Task] = None
        
    async def start_collection(self):
        """Start background metrics collection"""
        self.system_metrics_task = asyncio.create_task(self._collect_system_metrics())
        logger.info("Metrics collection started")
        
    async def stop_collection(self):
        """Stop background metrics collection"""
        if self.system_metrics_task:
            self.system_metrics_task.cancel()
            try:
                await self.system_metrics_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collection stopped")
        
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                CPU_USAGE.set(cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                MEMORY_USAGE.set(memory.used)
                
                # Disk metrics
                for disk in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(disk.mountpoint)
                        DISK_USAGE.labels(mount_point=disk.mountpoint).set(usage.used)
                    except (PermissionError, FileNotFoundError):
                        continue
                
                # GPU metrics (if available)
                await self._collect_gpu_metrics()
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _collect_gpu_metrics(self):
        """Collect GPU metrics using nvidia-ml-py"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                GPU_USAGE.labels(gpu_id=str(i)).set(util.gpu)
                
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                GPU_MEMORY.labels(gpu_id=str(i), type='used').set(mem_info.used)
                GPU_MEMORY.labels(gpu_id=str(i), type='total').set(mem_info.total)
                
        except ImportError:
            # nvidia-ml-py not available
            pass
        except Exception as e:
            logger.warning(f"Error collecting GPU metrics: {e}")

class MetricsMiddleware:
    """FastAPI middleware for request metrics"""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        
        # Normalize path for metrics (remove dynamic parts)
        normalized_path = self._normalize_path(path)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                duration = time.time() - start_time
                
                # Record metrics
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=normalized_path,
                    status=str(status_code)
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=normalized_path
                ).observe(duration)
                
            await send(message)
            
        await self.app(scope, receive, send_wrapper)
        
    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics to avoid high cardinality"""
        # Replace UUIDs and IDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            path
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
        
    def register_check(self, name: str, check_func):
        """Register a health check"""
        self.checks[name] = check_func
        
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - MetricsCollector().start_time,
            "checks": {}
        }
        
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                health_status["checks"][name] = {
                    "status": "healthy" if result else "unhealthy",
                    "details": result if isinstance(result, dict) else {}
                }
                if not result:
                    overall_healthy = False
            except Exception as e:
                health_status["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_healthy = False
                
        health_status["status"] = "healthy" if overall_healthy else "unhealthy"
        return health_status

# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()

def record_simulation_metrics(engine: str, status: str, duration: float):
    """Record simulation metrics"""
    SIMULATION_COUNT.labels(engine=engine, status=status).inc()
    SIMULATION_DURATION.labels(engine=engine).observe(duration)

def record_file_upload(status: str):
    """Record file upload metrics"""
    FILE_UPLOADS.labels(status=status).inc()

def record_error(error_type: str, severity: str = "error"):
    """Record error metrics"""
    ERROR_COUNT.labels(type=error_type, severity=severity).inc()

def update_active_users(count: int):
    """Update active users count"""
    ACTIVE_USERS.set(count)

def update_active_simulations(count: int):
    """Update active simulations count"""
    ACTIVE_SIMULATIONS.set(count)

def get_metrics() -> str:
    """Get metrics in Prometheus format"""
    return generate_latest(REGISTRY).decode('utf-8')

# Health check functions
async def check_database_health():
    """Check database connectivity"""
    try:
        from database import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def check_redis_health():
    """Check Redis connectivity"""
    try:
        import redis
        import os
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url)
        r.ping()
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def check_gpu_health():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory
                devices.append({"id": i, "name": name, "memory": memory})
            return {"status": "available", "devices": devices}
        else:
            return {"status": "unavailable", "reason": "CUDA not available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Register default health checks
health_checker.register_check("database", check_database_health)
health_checker.register_check("redis", check_redis_health)
health_checker.register_check("gpu", check_gpu_health)

@asynccontextmanager
async def metrics_lifespan():
    """Context manager for metrics lifecycle"""
    await metrics_collector.start_collection()
    try:
        yield
    finally:
        await metrics_collector.stop_collection() 