"""
ENTERPRISE LOAD BALANCER & AUTO-SCALING SERVICE
Phase 3 Week 9-10: Load Balancing & Auto-Scaling Implementation

This module implements:
- Intelligent load balancing for simulation requests
- Auto-scaling based on system metrics
- Health monitoring and failover
- Session affinity for WebSocket connections (progress bar)

CRITICAL: This preserves Ultra engine and progress bar functionality
while adding enterprise load balancing capabilities.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import aiohttp

logger = logging.getLogger(__name__)

class InstanceStatus(Enum):
    """Status of simulation service instances"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    OVERLOADED = "overloaded"

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"

@dataclass
class ServiceInstance:
    """Represents a simulation service instance"""
    id: str
    host: str
    port: int
    status: InstanceStatus = InstanceStatus.HEALTHY
    active_simulations: int = 0
    max_simulations: int = 10
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    response_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    weight: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0
    websocket_connections: int = 0
    
    @property
    def is_healthy(self) -> bool:
        """Check if instance is healthy"""
        return (
            self.status == InstanceStatus.HEALTHY and
            self.active_simulations < self.max_simulations and
            self.cpu_usage < 90.0 and
            self.memory_usage < 90.0 and
            self.gpu_usage < 95.0
        )
    
    @property
    def load_score(self) -> float:
        """Calculate load score for resource-based balancing"""
        # Lower score = better choice
        cpu_score = self.cpu_usage / 100.0
        memory_score = self.memory_usage / 100.0
        gpu_score = self.gpu_usage / 100.0
        simulation_score = self.active_simulations / self.max_simulations
        
        return (cpu_score + memory_score + gpu_score + simulation_score) / 4.0
    
    @property
    def availability_score(self) -> float:
        """Calculate availability score"""
        if self.total_requests == 0:
            return 1.0
        
        success_rate = 1.0 - (self.failed_requests / self.total_requests)
        return success_rate

class EnterpriseLoadBalancer:
    """
    Enterprise-grade load balancer for Monte Carlo simulation services
    
    Features:
    - Multiple load balancing algorithms
    - Health monitoring and automatic failover
    - Session affinity for WebSocket connections (progress bar)
    - Auto-scaling based on metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".EnterpriseLoadBalancer")
        
        # Service instances
        self.instances: Dict[str, ServiceInstance] = {}
        self.algorithm = LoadBalancingAlgorithm.RESOURCE_BASED
        
        # Session affinity for WebSocket connections (progress bar support)
        self.session_affinity: Dict[str, str] = {}  # user_id -> instance_id
        
        # Round-robin counter
        self.round_robin_counter = 0
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = True
        self.min_instances = 3
        self.max_instances = 20
        self.target_cpu_utilization = 70.0
        self.target_gpu_utilization = 75.0
        self.scale_up_threshold = 80.0
        self.scale_down_threshold = 30.0
        
        # Metrics collection
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'active_instances': 0,
            'scaling_events': 0
        }
        
        # Initialize with default instances (Kubernetes pods)
        self._initialize_default_instances()
        
        # TEMPORARILY DISABLED: Background tasks causing performance issues
        # These will be re-enabled when we have actual multiple instances
        # asyncio.create_task(self._health_monitor_loop())
        # asyncio.create_task(self._auto_scaling_loop())
        
        self.logger.warning("⚠️ [LOAD_BALANCER] Background health checks disabled to preserve progress bar performance")
    
    def _initialize_default_instances(self):
        """Initialize default service instances"""
        # In Kubernetes, these would be discovered via service discovery
        default_instances = [
            ServiceInstance(
                id="simulation-service-0",
                host="simulation-service-0.simulation-service-lb",
                port=8000,
                max_simulations=10,
                weight=1.0
            ),
            ServiceInstance(
                id="simulation-service-1", 
                host="simulation-service-1.simulation-service-lb",
                port=8000,
                max_simulations=10,
                weight=1.0
            ),
            ServiceInstance(
                id="simulation-service-2",
                host="simulation-service-2.simulation-service-lb",
                port=8000,
                max_simulations=10,
                weight=1.0
            )
        ]
        
        for instance in default_instances:
            self.instances[instance.id] = instance
            
        self.logger.info(f"✅ [LOAD_BALANCER] Initialized with {len(default_instances)} default instances")
    
    async def select_instance(self, user_id: Optional[int] = None, session_id: Optional[str] = None, 
                            requires_websocket: bool = False) -> Optional[ServiceInstance]:
        """
        Select best instance for request based on load balancing algorithm
        
        CRITICAL: For WebSocket connections (progress bar), this maintains
        session affinity to preserve real-time progress updates.
        """
        try:
            # Filter healthy instances
            healthy_instances = [
                instance for instance in self.instances.values()
                if instance.is_healthy
            ]
            
            if not healthy_instances:
                self.logger.error("❌ [LOAD_BALANCER] No healthy instances available")
                return None
            
            # Session affinity for WebSocket connections (progress bar)
            if requires_websocket and user_id:
                session_key = f"user_{user_id}"
                if session_key in self.session_affinity:
                    instance_id = self.session_affinity[session_key]
                    if instance_id in self.instances and self.instances[instance_id].is_healthy:
                        selected = self.instances[instance_id]
                        self.logger.debug(f"🔗 [SESSION_AFFINITY] User {user_id} → {selected.id} (WebSocket)")
                        return selected
                    else:
                        # Remove stale affinity
                        del self.session_affinity[session_key]
            
            # Select instance based on algorithm
            if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                selected = self._select_round_robin(healthy_instances)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                selected = self._select_least_connections(healthy_instances)
            elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                selected = self._select_weighted_round_robin(healthy_instances)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                selected = self._select_least_response_time(healthy_instances)
            elif self.algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
                selected = self._select_resource_based(healthy_instances)
            else:
                selected = healthy_instances[0]  # Fallback
            
            # Establish session affinity for WebSocket connections
            if requires_websocket and user_id and selected:
                session_key = f"user_{user_id}"
                self.session_affinity[session_key] = selected.id
                selected.websocket_connections += 1
                self.logger.debug(f"🔗 [SESSION_AFFINITY] Established {session_key} → {selected.id}")
            
            if selected:
                selected.total_requests += 1
                self.metrics['total_requests'] += 1
                
                self.logger.debug(f"🎯 [LOAD_BALANCER] Selected {selected.id} (algorithm: {self.algorithm.value})")
            
            return selected
            
        except Exception as e:
            self.logger.error(f"❌ [LOAD_BALANCER] Instance selection failed: {e}")
            return None
    
    def _select_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection"""
        selected = instances[self.round_robin_counter % len(instances)]
        self.round_robin_counter += 1
        return selected
    
    def _select_least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least active simulations"""
        return min(instances, key=lambda x: x.active_simulations)
    
    def _select_weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round-robin based on instance weights"""
        # Simple weighted selection
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return instances[0]
        
        # Normalize weights and select
        weights = [instance.weight / total_weight for instance in instances]
        import random
        return random.choices(instances, weights=weights)[0]
    
    def _select_least_response_time(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with lowest response time"""
        return min(instances, key=lambda x: x.response_time_ms)
    
    def _select_resource_based(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with best resource availability"""
        return min(instances, key=lambda x: x.load_score)
    
    async def report_instance_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """Update instance metrics from health checks"""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        
        # Update metrics
        instance.cpu_usage = metrics.get('cpu_usage', 0.0)
        instance.memory_usage = metrics.get('memory_usage', 0.0)
        instance.gpu_usage = metrics.get('gpu_usage', 0.0)
        instance.active_simulations = metrics.get('active_simulations', 0)
        instance.response_time_ms = metrics.get('response_time_ms', 0.0)
        instance.last_health_check = datetime.utcnow()
        
        # Update status based on metrics
        if (instance.cpu_usage > 95 or instance.memory_usage > 95 or 
            instance.active_simulations >= instance.max_simulations):
            instance.status = InstanceStatus.OVERLOADED
        elif (instance.cpu_usage < 90 and instance.memory_usage < 90 and 
              instance.active_simulations < instance.max_simulations):
            instance.status = InstanceStatus.HEALTHY
    
    async def report_request_result(self, instance_id: str, success: bool, response_time_ms: float):
        """Report request result for metrics"""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            instance.failed_requests += 1
            self.metrics['failed_requests'] += 1
        
        # Update average response time
        instance.response_time_ms = response_time_ms
        
        # Update global average response time
        total_requests = self.metrics['successful_requests'] + self.metrics['failed_requests']
        if total_requests > 0:
            self.metrics['average_response_time'] = (
                (self.metrics['average_response_time'] * (total_requests - 1) + response_time_ms) 
                / total_requests
            )
    
    async def _health_monitor_loop(self):
        """Background task to monitor instance health"""
        while True:
            try:
                await self._check_all_instances_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"❌ [HEALTH_MONITOR] Health check loop failed: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _check_all_instances_health(self):
        """Check health of all instances"""
        tasks = []
        for instance_id in self.instances:
            tasks.append(self._check_instance_health(instance_id))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_instance_health(self, instance_id: str):
        """Check health of specific instance"""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        
        try:
            start_time = time.time()
            
            # Health check HTTP request
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"http://{instance.host}:{instance.port}/health"
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Update instance metrics
                        await self.report_instance_metrics(instance_id, {
                            'cpu_usage': health_data.get('cpu_usage', 0.0),
                            'memory_usage': health_data.get('memory_usage', 0.0),
                            'gpu_usage': health_data.get('gpu_usage', 0.0),
                            'active_simulations': health_data.get('active_simulations', 0),
                            'response_time_ms': response_time
                        })
                        
                        instance.status = InstanceStatus.HEALTHY
                        
                    else:
                        instance.status = InstanceStatus.UNHEALTHY
                        
        except Exception as e:
            self.logger.warning(f"⚠️ [HEALTH_CHECK] Instance {instance_id} health check failed: {e}")
            instance.status = InstanceStatus.UNHEALTHY
            instance.last_health_check = datetime.utcnow()
    
    async def _auto_scaling_loop(self):
        """Background task for auto-scaling decisions"""
        while True:
            try:
                if self.auto_scaling_enabled:
                    await self._evaluate_scaling()
                await asyncio.sleep(60)  # Evaluate every minute
            except Exception as e:
                self.logger.error(f"❌ [AUTO_SCALING] Scaling loop failed: {e}")
                await asyncio.sleep(120)  # Back off on error
    
    async def _evaluate_scaling(self):
        """Evaluate if scaling up or down is needed"""
        healthy_instances = [
            instance for instance in self.instances.values()
            if instance.status == InstanceStatus.HEALTHY
        ]
        
        if not healthy_instances:
            self.logger.warning("⚠️ [AUTO_SCALING] No healthy instances for scaling evaluation")
            return
        
        # Calculate average metrics
        avg_cpu = sum(instance.cpu_usage for instance in healthy_instances) / len(healthy_instances)
        avg_gpu = sum(instance.gpu_usage for instance in healthy_instances) / len(healthy_instances)
        avg_simulations = sum(instance.active_simulations for instance in healthy_instances) / len(healthy_instances)
        
        current_instances = len(healthy_instances)
        
        # Scale up conditions
        should_scale_up = (
            (avg_cpu > self.scale_up_threshold or avg_gpu > self.scale_up_threshold) and
            current_instances < self.max_instances
        )
        
        # Scale down conditions
        should_scale_down = (
            avg_cpu < self.scale_down_threshold and 
            avg_gpu < self.scale_down_threshold and
            avg_simulations < 2 and  # Low simulation load
            current_instances > self.min_instances
        )
        
        if should_scale_up:
            await self._scale_up()
        elif should_scale_down:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up by adding new instance"""
        try:
            # In Kubernetes, this would trigger HPA or VPA
            # For now, we simulate adding a new instance
            
            new_instance_id = f"simulation-service-{len(self.instances)}"
            new_instance = ServiceInstance(
                id=new_instance_id,
                host=f"{new_instance_id}.simulation-service-lb",
                port=8000,
                status=InstanceStatus.STARTING,
                max_simulations=10
            )
            
            self.instances[new_instance_id] = new_instance
            self.metrics['scaling_events'] += 1
            
            self.logger.info(f"📈 [AUTO_SCALING] Scaled UP - Added instance {new_instance_id}")
            
            # Simulate startup time
            await asyncio.sleep(30)
            new_instance.status = InstanceStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"❌ [AUTO_SCALING] Scale up failed: {e}")
    
    async def _scale_down(self):
        """Scale down by removing instance"""
        try:
            # Find instance with least load to remove
            healthy_instances = [
                instance for instance in self.instances.values()
                if instance.status == InstanceStatus.HEALTHY
            ]
            
            if len(healthy_instances) <= self.min_instances:
                return
            
            # Select instance with lowest load
            instance_to_remove = min(healthy_instances, key=lambda x: x.active_simulations)
            
            # Mark as stopping
            instance_to_remove.status = InstanceStatus.STOPPING
            
            # Wait for active simulations to complete
            timeout = 300  # 5 minutes
            start_time = time.time()
            
            while instance_to_remove.active_simulations > 0 and (time.time() - start_time) < timeout:
                await asyncio.sleep(10)
            
            # Remove instance
            del self.instances[instance_to_remove.id]
            self.metrics['scaling_events'] += 1
            
            self.logger.info(f"📉 [AUTO_SCALING] Scaled DOWN - Removed instance {instance_to_remove.id}")
            
        except Exception as e:
            self.logger.error(f"❌ [AUTO_SCALING] Scale down failed: {e}")
    
    async def get_load_balancer_stats(self) -> dict:
        """Get comprehensive load balancer statistics"""
        try:
            healthy_instances = [
                instance for instance in self.instances.values()
                if instance.is_healthy
            ]
            
            total_capacity = sum(instance.max_simulations for instance in healthy_instances)
            active_simulations = sum(instance.active_simulations for instance in healthy_instances)
            
            return {
                "instances": {
                    "total": len(self.instances),
                    "healthy": len(healthy_instances),
                    "unhealthy": len(self.instances) - len(healthy_instances),
                    "details": [
                        {
                            "id": instance.id,
                            "status": instance.status.value,
                            "active_simulations": instance.active_simulations,
                            "max_simulations": instance.max_simulations,
                            "cpu_usage": instance.cpu_usage,
                            "memory_usage": instance.memory_usage,
                            "gpu_usage": instance.gpu_usage,
                            "response_time_ms": instance.response_time_ms,
                            "websocket_connections": instance.websocket_connections,
                            "load_score": round(instance.load_score, 3)
                        }
                        for instance in self.instances.values()
                    ]
                },
                "capacity": {
                    "total_simulation_slots": total_capacity,
                    "active_simulations": active_simulations,
                    "utilization_percent": round(
                        (active_simulations / total_capacity * 100) if total_capacity > 0 else 0, 2
                    )
                },
                "performance": {
                    "algorithm": self.algorithm.value,
                    "total_requests": self.metrics['total_requests'],
                    "successful_requests": self.metrics['successful_requests'],
                    "failed_requests": self.metrics['failed_requests'],
                    "success_rate_percent": round(
                        (self.metrics['successful_requests'] / self.metrics['total_requests'] * 100)
                        if self.metrics['total_requests'] > 0 else 0, 2
                    ),
                    "average_response_time_ms": round(self.metrics['average_response_time'], 2)
                },
                "auto_scaling": {
                    "enabled": self.auto_scaling_enabled,
                    "min_instances": self.min_instances,
                    "max_instances": self.max_instances,
                    "target_cpu_utilization": self.target_cpu_utilization,
                    "target_gpu_utilization": self.target_gpu_utilization,
                    "scaling_events": self.metrics['scaling_events']
                },
                "session_affinity": {
                    "active_sessions": len(self.session_affinity),
                    "websocket_preservation": "enabled"  # Critical for progress bar
                },
                "ultra_engine_compatibility": {
                    "preserved": True,
                    "progress_bar_support": True,
                    "websocket_affinity": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ [LOAD_BALANCER_STATS] Failed to get statistics: {e}")
            return {"error": str(e)}

# Global load balancer instance
enterprise_load_balancer = EnterpriseLoadBalancer()

# Convenience functions
async def select_simulation_instance(user_id: int, requires_websocket: bool = False) -> Optional[ServiceInstance]:
    """Select instance for simulation (preserves Ultra engine functionality)"""
    return await enterprise_load_balancer.select_instance(
        user_id=user_id, 
        requires_websocket=requires_websocket
    )

async def report_simulation_metrics(instance_id: str, metrics: Dict[str, Any]):
    """Report simulation metrics for load balancing"""
    await enterprise_load_balancer.report_instance_metrics(instance_id, metrics)
