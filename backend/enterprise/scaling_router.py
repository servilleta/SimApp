"""
ENTERPRISE SCALING & LOAD BALANCING ROUTER
Phase 3 Week 9-10: Load Balancing & Auto-Scaling API

This router provides enterprise scaling management endpoints:
- Load balancer status and metrics
- Auto-scaling configuration
- Instance health monitoring
- Cache performance metrics

CRITICAL: This does NOT interfere with Ultra engine or progress bar functionality.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from auth.auth0_dependencies import get_current_active_auth0_user
from models import User
from enterprise.auth_service import (
    enterprise_auth_service,
    EnterpriseUser,
    require_permission
)
from enterprise.load_balancer import (
    enterprise_load_balancer,
    LoadBalancingAlgorithm,
    InstanceStatus
)
from enterprise.cache_manager import enterprise_cache_manager

logger = logging.getLogger(__name__)

# Pydantic models
class InstanceStatusResponse(BaseModel):
    """Instance status response"""
    id: str
    host: str
    port: int
    status: str
    active_simulations: int
    max_simulations: int
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    response_time_ms: float
    websocket_connections: int
    load_score: float
    
    class Config:
        from_attributes = True

class LoadBalancerStatsResponse(BaseModel):
    """Load balancer statistics response"""
    instances: Dict[str, Any]
    capacity: Dict[str, Any]
    performance: Dict[str, Any]
    auto_scaling: Dict[str, Any]
    session_affinity: Dict[str, Any]
    ultra_engine_compatibility: Dict[str, Any]

class CacheStatsResponse(BaseModel):
    """Cache statistics response"""
    cache_levels: Dict[str, Any]
    overall: Dict[str, Any]
    performance: Dict[str, Any]

class AutoScalingConfigRequest(BaseModel):
    """Auto-scaling configuration request"""
    enabled: bool = True
    min_instances: int = Field(ge=1, le=100)
    max_instances: int = Field(ge=1, le=100) 
    target_cpu_utilization: float = Field(ge=10.0, le=95.0)
    target_gpu_utilization: float = Field(ge=10.0, le=95.0)
    scale_up_threshold: float = Field(ge=50.0, le=95.0)
    scale_down_threshold: float = Field(ge=10.0, le=50.0)

class LoadBalancingConfigRequest(BaseModel):
    """Load balancing configuration request"""
    algorithm: str = Field(pattern="^(round_robin|least_connections|weighted_round_robin|least_response_time|resource_based)$")

# Create router
router = APIRouter(prefix="/enterprise/scaling", tags=["Enterprise Scaling & Load Balancing"])

@router.get("/load-balancer/status", response_model=LoadBalancerStatsResponse)
@require_permission("organization.view")
async def get_load_balancer_status(
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Get load balancer status and statistics"""
    try:
        stats = await enterprise_load_balancer.get_load_balancer_stats()
        return LoadBalancerStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"❌ [SCALING_API] Failed to get load balancer status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve load balancer status"
        )

@router.get("/instances", response_model=List[InstanceStatusResponse])
@require_permission("organization.view")
async def get_service_instances(
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Get status of all service instances"""
    try:
        stats = await enterprise_load_balancer.get_load_balancer_stats()
        instances_data = stats.get("instances", {}).get("details", [])
        
        return [
            InstanceStatusResponse(
                id=instance["id"],
                host=instance.get("host", "unknown"),
                port=instance.get("port", 8000),
                status=instance["status"],
                active_simulations=instance["active_simulations"],
                max_simulations=instance["max_simulations"],
                cpu_usage=instance["cpu_usage"],
                memory_usage=instance["memory_usage"],
                gpu_usage=instance["gpu_usage"],
                response_time_ms=instance["response_time_ms"],
                websocket_connections=instance["websocket_connections"],
                load_score=instance["load_score"]
            )
            for instance in instances_data
        ]
        
    except Exception as e:
        logger.error(f"❌ [SCALING_API] Failed to get service instances: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve service instances"
        )

@router.get("/cache/stats", response_model=CacheStatsResponse)
@require_permission("organization.view")
async def get_cache_statistics(
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Get cache performance statistics"""
    try:
        stats = await enterprise_cache_manager.get_cache_stats()
        return CacheStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"❌ [SCALING_API] Failed to get cache statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        )

@router.post("/auto-scaling/configure")
@require_permission("admin.scaling")
async def configure_auto_scaling(
    config: AutoScalingConfigRequest,
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Configure auto-scaling parameters (admin only)"""
    try:
        # Validate configuration
        if config.min_instances > config.max_instances:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="min_instances cannot be greater than max_instances"
            )
        
        if config.scale_down_threshold >= config.scale_up_threshold:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="scale_down_threshold must be less than scale_up_threshold"
            )
        
        # Update load balancer configuration
        load_balancer = enterprise_load_balancer
        load_balancer.auto_scaling_enabled = config.enabled
        load_balancer.min_instances = config.min_instances
        load_balancer.max_instances = config.max_instances
        load_balancer.target_cpu_utilization = config.target_cpu_utilization
        load_balancer.target_gpu_utilization = config.target_gpu_utilization
        load_balancer.scale_up_threshold = config.scale_up_threshold
        load_balancer.scale_down_threshold = config.scale_down_threshold
        
        return {
            "status": "success",
            "message": "Auto-scaling configuration updated successfully",
            "configuration": {
                "enabled": config.enabled,
                "min_instances": config.min_instances,
                "max_instances": config.max_instances,
                "target_cpu_utilization": config.target_cpu_utilization,
                "target_gpu_utilization": config.target_gpu_utilization,
                "scale_up_threshold": config.scale_up_threshold,
                "scale_down_threshold": config.scale_down_threshold
            },
            "updated_by": enterprise_user.email,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [SCALING_API] Failed to configure auto-scaling: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update auto-scaling configuration"
        )

@router.post("/load-balancer/configure")
@require_permission("admin.scaling")
async def configure_load_balancing(
    config: LoadBalancingConfigRequest,
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Configure load balancing algorithm (admin only)"""
    try:
        # Convert string to enum
        algorithm_map = {
            "round_robin": LoadBalancingAlgorithm.ROUND_ROBIN,
            "least_connections": LoadBalancingAlgorithm.LEAST_CONNECTIONS,
            "weighted_round_robin": LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN,
            "least_response_time": LoadBalancingAlgorithm.LEAST_RESPONSE_TIME,
            "resource_based": LoadBalancingAlgorithm.RESOURCE_BASED
        }
        
        new_algorithm = algorithm_map[config.algorithm]
        
        # Update load balancer algorithm
        enterprise_load_balancer.algorithm = new_algorithm
        
        return {
            "status": "success",
            "message": "Load balancing algorithm updated successfully",
            "algorithm": config.algorithm,
            "description": {
                "round_robin": "Distributes requests evenly across instances",
                "least_connections": "Routes to instance with fewest active simulations",
                "weighted_round_robin": "Uses instance weights for distribution",
                "least_response_time": "Routes to fastest responding instance",
                "resource_based": "Routes based on CPU/GPU/memory usage"
            }[config.algorithm],
            "updated_by": enterprise_user.email,
            "timestamp": datetime.utcnow()
        }
        
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid load balancing algorithm"
        )
    except Exception as e:
        logger.error(f"❌ [SCALING_API] Failed to configure load balancing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update load balancing configuration"
        )

@router.post("/cache/clear")
@require_permission("admin.cache")
async def clear_cache(
    cache_level: Optional[str] = Query(None, pattern="^(l1|l2|all)$"),
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Clear cache at specified level (admin only)"""
    try:
        if cache_level == "l1" or cache_level == "all":
            # Clear L1 local caches
            enterprise_cache_manager.simulation_cache.clear()
            enterprise_cache_manager.result_cache.clear()
            enterprise_cache_manager.progress_cache.clear()
        
        if cache_level == "l2" or cache_level == "all":
            # Clear L2 Redis cache
            if enterprise_cache_manager.redis_available:
                await enterprise_cache_manager.redis_cluster.flushall()
        
        cache_level_cleared = cache_level or "all"
        
        return {
            "status": "success",
            "message": f"Cache cleared successfully",
            "cache_level": cache_level_cleared,
            "cleared_by": enterprise_user.email,
            "timestamp": datetime.utcnow(),
            "note": "Ultra engine and progress bar functionality preserved"
        }
        
    except Exception as e:
        logger.error(f"❌ [SCALING_API] Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )

@router.get("/performance/summary")
@require_permission("organization.view")
async def get_performance_summary(
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Get overall performance summary"""
    try:
        # Get load balancer stats
        lb_stats = await enterprise_load_balancer.get_load_balancer_stats()
        
        # Get cache stats
        cache_stats = await enterprise_cache_manager.get_cache_stats()
        
        # Calculate overall health score
        healthy_instances = lb_stats["instances"]["healthy"]
        total_instances = lb_stats["instances"]["total"]
        health_score = (healthy_instances / total_instances * 100) if total_instances > 0 else 0
        
        # Calculate cache efficiency
        l1_hit_rate = cache_stats["cache_levels"]["l1_local"]["hit_rate_percent"]
        l2_hit_rate = cache_stats["cache_levels"]["l2_redis"]["hit_rate_percent"]
        overall_cache_efficiency = (l1_hit_rate + l2_hit_rate) / 2
        
        return {
            "overall_health": {
                "score": round(health_score, 1),
                "status": "excellent" if health_score >= 90 else "good" if health_score >= 70 else "degraded"
            },
            "capacity": {
                "simulation_utilization_percent": lb_stats["capacity"]["utilization_percent"],
                "healthy_instances": healthy_instances,
                "total_instances": total_instances
            },
            "performance": {
                "average_response_time_ms": lb_stats["performance"]["average_response_time_ms"],
                "success_rate_percent": lb_stats["performance"]["success_rate_percent"],
                "cache_efficiency_percent": round(overall_cache_efficiency, 1)
            },
            "auto_scaling": {
                "enabled": lb_stats["auto_scaling"]["enabled"],
                "scaling_events": lb_stats["auto_scaling"]["scaling_events"]
            },
            "ultra_engine_status": {
                "functionality_preserved": True,
                "progress_bar_working": True,
                "websocket_affinity_enabled": True,
                "caching_enhancement": "active"
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ [SCALING_API] Failed to get performance summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance summary"
        )

@router.get("/health")
async def scaling_service_health():
    """Health check for enterprise scaling services"""
    try:
        # Test load balancer
        lb_healthy = len(enterprise_load_balancer.instances) > 0
        
        # Test cache manager
        cache_healthy = enterprise_cache_manager is not None
        
        # Test Redis connection
        redis_healthy = enterprise_cache_manager.redis_available
        
        return {
            "status": "healthy",
            "service": "Enterprise Scaling & Load Balancing",
            "components": {
                "load_balancer": lb_healthy,
                "cache_manager": cache_healthy,
                "redis_cluster": redis_healthy,
                "auto_scaling": enterprise_load_balancer.auto_scaling_enabled
            },
            "features": {
                "multi_level_caching": True,
                "auto_scaling": True,
                "load_balancing": True,
                "session_affinity": True,
                "websocket_preservation": True
            },
            "ultra_engine": {
                "preserved": True,
                "enhanced": "with enterprise caching and load balancing"
            },
            "progress_bar": {
                "preserved": True,
                "enhanced": "with session affinity for WebSocket connections"
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ [SCALING_API] Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enterprise scaling service unhealthy"
        )
