"""
🔍 Service Discovery and Registration
Enterprise API Gateway Component
"""

import asyncio
import logging
import aiohttp
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

class ServiceInstance:
    def __init__(self, service_name: str, host: str, port: int, health_check_url: str):
        self.service_name = service_name
        self.host = host
        self.port = port
        self.health_check_url = health_check_url
        self.status = ServiceStatus.HEALTHY
        self.last_health_check = datetime.utcnow()
        self.failure_count = 0
        self.instance_id = str(uuid.uuid4())
        
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY

class ServiceRegistry:
    """Service Discovery and Registration with Health Checking"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.load_balancer_index: Dict[str, int] = {}
        
    def register_service(self, service: ServiceInstance):
        """Register a new service instance"""
        if service.service_name not in self.services:
            self.services[service.service_name] = []
        
        self.services[service.service_name].append(service)
        logger.info(f"✅ [SERVICE_REGISTRY] Registered {service.service_name} at {service.base_url}")
    
    def get_healthy_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get a healthy service instance using round-robin load balancing"""
        if service_name not in self.services:
            return None
        
        instances = [svc for svc in self.services[service_name] if svc.is_healthy]
        if not instances:
            logger.warning(f"⚠️ [SERVICE_REGISTRY] No healthy instances for {service_name}")
            return None
        
        # Round-robin load balancing
        current_index = self.load_balancer_index.get(service_name, 0)
        instance = instances[current_index % len(instances)]
        self.load_balancer_index[service_name] = (current_index + 1) % len(instances)
        
        return instance
    
    async def health_check_all_services(self):
        """Perform health checks on all registered services"""
        for service_name, instances in self.services.items():
            for instance in instances:
                await self._health_check_instance(instance)
    
    async def _health_check_instance(self, instance: ServiceInstance):
        """Perform health check on a single service instance"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{instance.base_url}{instance.health_check_url}") as response:
                    if response.status == 200:
                        instance.status = ServiceStatus.HEALTHY
                        instance.failure_count = 0
                        logger.debug(f"✅ [HEALTH_CHECK] {instance.service_name} is healthy")
                    else:
                        await self._mark_instance_unhealthy(instance, f"Health check returned {response.status}")
        except Exception as e:
            await self._mark_instance_unhealthy(instance, str(e))
    
    async def _mark_instance_unhealthy(self, instance: ServiceInstance, reason: str):
        """Mark service instance as unhealthy"""
        instance.failure_count += 1
        
        if instance.failure_count >= 3:
            instance.status = ServiceStatus.UNHEALTHY
            logger.error(f"❌ [HEALTH_CHECK] {instance.service_name} marked as unhealthy: {reason}")
        else:
            instance.status = ServiceStatus.DEGRADED
            logger.warning(f"⚠️ [HEALTH_CHECK] {instance.service_name} degraded ({instance.failure_count}/3): {reason}")
