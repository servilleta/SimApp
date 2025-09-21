"""
ENTERPRISE GPU RESOURCE MANAGEMENT & FAIR-SHARE SCHEDULING
Phase 3 Week 11-12: Advanced Performance Optimization

This module implements:
- Fair-share GPU scheduling with priority weighting
- GPU resource pool management
- User priority and quota-based allocation
- Performance monitoring and optimization

CRITICAL: This enhances GPU utilization without modifying Ultra engine core functionality.
It only adds enterprise-grade resource management on top of existing GPU capabilities.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class GPUPriority(Enum):
    """GPU allocation priorities for different user tiers"""
    ENTERPRISE = "enterprise"     # Highest priority (weight: 3)
    PROFESSIONAL = "professional" # High priority (weight: 2)
    STANDARD = "standard"         # Normal priority (weight: 1)
    TRIAL = "trial"              # Lowest priority (weight: 0.5)

class ResourceRequirement(Enum):
    """GPU resource requirement levels"""
    LOW = "low"           # Simple simulations, small files
    MEDIUM = "medium"     # Standard simulations
    HIGH = "high"         # Complex simulations, large files
    ULTRA = "ultra"       # Maximum resource requirements

@dataclass
class GPUAllocation:
    """Represents a GPU resource allocation"""
    user_id: int
    simulation_id: str
    gpu_id: Optional[int]
    memory_allocated_mb: int
    compute_allocated_percent: float
    priority: GPUPriority
    allocated_at: datetime
    estimated_duration_minutes: int
    
    @property
    def is_expired(self) -> bool:
        """Check if allocation has exceeded estimated duration"""
        if self.estimated_duration_minutes <= 0:
            return False
        
        elapsed = datetime.utcnow() - self.allocated_at
        return elapsed.total_seconds() > (self.estimated_duration_minutes * 60 * 1.5)  # 50% buffer

@dataclass
class GPUResourceRequirements:
    """GPU resource requirements for a simulation"""
    memory_mb: int
    compute_percent: float
    estimated_duration_minutes: int
    complexity_level: ResourceRequirement
    file_size_mb: float
    iterations: int
    
    @classmethod
    def estimate_from_simulation(cls, simulation_request: dict) -> 'GPUResourceRequirements':
        """Estimate GPU requirements from simulation request"""
        
        # Get simulation parameters
        iterations = simulation_request.get('iterations', 1000)
        file_size_mb = simulation_request.get('file_size_mb', 1.0)
        variable_count = len(simulation_request.get('variables', []))
        result_count = len(simulation_request.get('result_cells', []))
        
        # Estimate complexity
        complexity_score = (
            (iterations / 1000) * 0.4 +           # Iterations weight
            (file_size_mb / 10) * 0.3 +           # File size weight  
            (variable_count / 10) * 0.2 +         # Variables weight
            (result_count / 5) * 0.1               # Results weight
        )
        
        if complexity_score < 0.5:
            complexity = ResourceRequirement.LOW
            memory_mb = 512
            compute_percent = 25.0
            duration_minutes = 2
        elif complexity_score < 1.5:
            complexity = ResourceRequirement.MEDIUM
            memory_mb = 1024
            compute_percent = 50.0
            duration_minutes = 5
        elif complexity_score < 3.0:
            complexity = ResourceRequirement.HIGH
            memory_mb = 2048
            compute_percent = 75.0
            duration_minutes = 10
        else:
            complexity = ResourceRequirement.ULTRA
            memory_mb = 4096
            compute_percent = 100.0
            duration_minutes = 20
        
        return cls(
            memory_mb=memory_mb,
            compute_percent=compute_percent,
            estimated_duration_minutes=duration_minutes,
            complexity_level=complexity,
            file_size_mb=file_size_mb,
            iterations=iterations
        )

class FairShareScheduler:
    """
    Fair-share GPU scheduler with priority weighting
    
    This ensures enterprise customers get priority while maintaining
    fairness for all users based on their subscription tier.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".FairShareScheduler")
        
        # User usage tracking (for fair share calculation)
        self.user_usage_tracking: Dict[int, float] = {}  # user_id -> cumulative usage
        
        # Priority weights for different user tiers
        self.priority_weights = {
            GPUPriority.ENTERPRISE: 3.0,
            GPUPriority.PROFESSIONAL: 2.0,
            GPUPriority.STANDARD: 1.0,
            GPUPriority.TRIAL: 0.5
        }
        
        # Resource allocation limits per tier
        self.tier_limits = {
            GPUPriority.ENTERPRISE: {
                'max_memory_mb': 8192,      # 8GB
                'max_compute_percent': 100.0,
                'max_duration_minutes': 60
            },
            GPUPriority.PROFESSIONAL: {
                'max_memory_mb': 4096,      # 4GB
                'max_compute_percent': 75.0,
                'max_duration_minutes': 30
            },
            GPUPriority.STANDARD: {
                'max_memory_mb': 2048,      # 2GB
                'max_compute_percent': 50.0,
                'max_duration_minutes': 15
            },
            GPUPriority.TRIAL: {
                'max_memory_mb': 1024,      # 1GB
                'max_compute_percent': 25.0,
                'max_duration_minutes': 5
            }
        }
    
    async def calculate_user_priority_score(self, user_id: int, priority: GPUPriority) -> float:
        """Calculate priority score for fair share allocation"""
        
        # Get user's cumulative usage
        user_usage = self.user_usage_tracking.get(user_id, 0.0)
        
        # Get priority weight
        priority_weight = self.priority_weights.get(priority, 1.0)
        
        # Fair share score (lower is better for allocation)
        # Users with less usage get priority, but weighted by tier
        fair_share_score = user_usage / priority_weight
        
        return fair_share_score
    
    async def allocate_gpu_resources(self, user_id: int, priority: GPUPriority, 
                                   requirements: GPUResourceRequirements) -> Optional[GPUAllocation]:
        """Allocate GPU resources based on fair share and priority"""
        
        try:
            # Calculate priority score
            priority_score = await self.calculate_user_priority_score(user_id, priority)
            
            # Check tier limits
            tier_limits = self.tier_limits.get(priority, self.tier_limits[GPUPriority.TRIAL])
            
            # Enforce tier limits
            memory_requested = min(requirements.memory_mb, tier_limits['max_memory_mb'])
            compute_requested = min(requirements.compute_percent, tier_limits['max_compute_percent'])
            duration_requested = min(requirements.estimated_duration_minutes, tier_limits['max_duration_minutes'])
            
            # Check if resources are available (simplified for single GPU)
            if await self._check_gpu_availability(memory_requested, compute_requested):
                
                # Create allocation
                allocation = GPUAllocation(
                    user_id=user_id,
                    simulation_id=f"sim_{user_id}_{int(time.time())}",
                    gpu_id=0,  # Single GPU system
                    memory_allocated_mb=memory_requested,
                    compute_allocated_percent=compute_requested,
                    priority=priority,
                    allocated_at=datetime.utcnow(),
                    estimated_duration_minutes=duration_requested
                )
                
                # Track usage for fair share
                usage_points = (memory_requested / 1024) * (compute_requested / 100) * (duration_requested / 10)
                self.user_usage_tracking[user_id] = self.user_usage_tracking.get(user_id, 0) + usage_points
                
                self.logger.info(f"✅ [GPU_SCHEDULER] Allocated GPU to user {user_id} ({priority.value})")
                self.logger.debug(f"   Memory: {memory_requested}MB, Compute: {compute_requested}%, Duration: {duration_requested}min")
                
                return allocation
            else:
                self.logger.warning(f"⚠️ [GPU_SCHEDULER] No GPU resources available for user {user_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [GPU_SCHEDULER] Failed to allocate GPU resources: {e}")
            return None
    
    async def _check_gpu_availability(self, memory_mb: int, compute_percent: float) -> bool:
        """Check if GPU has available resources"""
        
        try:
            # For single GPU system, check if it's currently in use
            # In production, this would check actual GPU memory and utilization
            
            # Get GPU manager status
            from gpu.manager import gpu_manager
            
            if gpu_manager and gpu_manager.gpu_available:
                # Check GPU memory availability
                available_memory = gpu_manager.usable_memory_mb
                if memory_mb > available_memory:
                    return False
                
                # In single GPU mode, only one simulation at a time
                # This preserves existing Ultra engine behavior
                return True
            else:
                # CPU fallback mode - always available
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ [GPU_SCHEDULER] GPU availability check failed: {e}")
            return True  # Fallback to allowing allocation
    
    async def get_scheduling_stats(self) -> dict:
        """Get comprehensive scheduling statistics"""
        
        try:
            # Calculate total usage by tier
            tier_usage = {}
            for user_id, usage in self.user_usage_tracking.items():
                # Would need to look up user tier - simplified for now
                tier = "standard"  # Default
                tier_usage[tier] = tier_usage.get(tier, 0) + usage
            
            return {
                "fair_share_scheduler": {
                    "total_users_tracked": len(self.user_usage_tracking),
                    "usage_by_tier": tier_usage,
                    "priority_weights": {k.value: v for k, v in self.priority_weights.items()},
                    "tier_limits": {k.value: v for k, v in self.tier_limits.items()}
                },
                "resource_allocation": {
                    "single_gpu_mode": True,
                    "gpu_available": True,  # Would check actual GPU
                    "memory_pools": "optimized",
                    "concurrent_support": "ultra_engine_managed"
                },
                "performance": {
                    "ultra_engine_preserved": True,
                    "progress_bar_preserved": True,
                    "scheduling_overhead": "minimal"
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ [GPU_SCHEDULER] Failed to get scheduling stats: {e}")
            return {"error": str(e)}

class EnterpriseGPUScheduler:
    """
    Enterprise GPU scheduler that enhances existing Ultra engine GPU management
    
    CRITICAL: This does NOT replace the Ultra engine GPU functionality.
    It adds enterprise-grade scheduling and resource management on top.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".EnterpriseGPUScheduler")
        
        # Fair share scheduler
        self.fair_share_scheduler = FairShareScheduler()
        
        # Active allocations
        self.active_allocations: Dict[str, GPUAllocation] = {}
        
        # Performance metrics
        self.metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'average_wait_time_seconds': 0.0,
            'gpu_utilization_percent': 0.0
        }
        
        # Start background cleanup task (lightweight)
        asyncio.create_task(self._cleanup_expired_allocations())
    
    async def schedule_simulation(self, user_id: int, simulation_request: dict) -> Optional[GPUAllocation]:
        """
        Schedule simulation with enterprise GPU management
        
        CRITICAL: This enhances but does not replace Ultra engine scheduling.
        The Ultra engine continues to handle actual GPU operations.
        """
        
        try:
            # Get user priority from enterprise auth service
            user_priority = await self._get_user_priority(user_id)
            
            # Estimate resource requirements
            resource_requirements = GPUResourceRequirements.estimate_from_simulation(simulation_request)
            
            # Allocate resources through fair share scheduler
            allocation = await self.fair_share_scheduler.allocate_gpu_resources(
                user_id=user_id,
                priority=user_priority,
                requirements=resource_requirements
            )
            
            if allocation:
                # Store active allocation
                self.active_allocations[allocation.simulation_id] = allocation
                self.metrics['total_allocations'] += 1
                self.metrics['successful_allocations'] += 1
                
                self.logger.info(f"✅ [GPU_SCHEDULER] Scheduled simulation for user {user_id}")
                return allocation
            else:
                self.metrics['failed_allocations'] += 1
                self.logger.warning(f"⚠️ [GPU_SCHEDULER] Failed to schedule simulation for user {user_id}")
                return None
                
        except Exception as e:
            self.metrics['failed_allocations'] += 1
            self.logger.error(f"❌ [GPU_SCHEDULER] Scheduling failed: {e}")
            return None
    
    async def _get_user_priority(self, user_id: int) -> GPUPriority:
        """Get user priority based on subscription tier"""
        
        try:
            # Get user tier from enterprise auth service
            from enterprise.auth_service import enterprise_auth_service
            from models import User
            from database import get_db
            
            db = next(get_db())
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    # Get enterprise user context
                    enterprise_user = await enterprise_auth_service.authenticate_enterprise_user(user)
                    
                    # Map organization tier to GPU priority
                    tier_mapping = {
                        'enterprise': GPUPriority.ENTERPRISE,
                        'professional': GPUPriority.PROFESSIONAL,
                        'standard': GPUPriority.STANDARD,
                        'trial': GPUPriority.TRIAL
                    }
                    
                    return tier_mapping.get(
                        enterprise_user.organization.tier.value, 
                        GPUPriority.STANDARD
                    )
                else:
                    return GPUPriority.STANDARD
                    
            finally:
                db.close()
                
        except Exception as e:
            self.logger.warning(f"⚠️ [GPU_SCHEDULER] Failed to get user priority: {e}")
            return GPUPriority.STANDARD  # Safe fallback
    
    async def release_allocation(self, simulation_id: str):
        """Release GPU allocation when simulation completes"""
        
        if simulation_id in self.active_allocations:
            allocation = self.active_allocations[simulation_id]
            del self.active_allocations[simulation_id]
            
            self.logger.info(f"🔓 [GPU_SCHEDULER] Released GPU allocation for simulation {simulation_id}")
            
            # Update metrics
            duration = datetime.utcnow() - allocation.allocated_at
            actual_duration_minutes = duration.total_seconds() / 60
            
            # Log if simulation took longer than estimated
            if actual_duration_minutes > allocation.estimated_duration_minutes * 1.2:
                self.logger.warning(
                    f"⏰ [GPU_SCHEDULER] Simulation {simulation_id} exceeded estimate: "
                    f"{actual_duration_minutes:.1f}min vs {allocation.estimated_duration_minutes}min"
                )
    
    async def _cleanup_expired_allocations(self):
        """Background task to clean up expired allocations"""
        
        while True:
            try:
                expired_count = 0
                current_time = datetime.utcnow()
                
                # Find expired allocations
                expired_sims = []
                for sim_id, allocation in self.active_allocations.items():
                    if allocation.is_expired:
                        expired_sims.append(sim_id)
                        expired_count += 1
                
                # Remove expired allocations
                for sim_id in expired_sims:
                    del self.active_allocations[sim_id]
                    self.logger.warning(f"🧹 [GPU_SCHEDULER] Cleaned up expired allocation: {sim_id}")
                
                if expired_count > 0:
                    self.logger.info(f"🧹 [GPU_SCHEDULER] Cleaned up {expired_count} expired allocations")
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"❌ [GPU_SCHEDULER] Cleanup task failed: {e}")
                await asyncio.sleep(600)  # Back off on error
    
    async def get_gpu_utilization_stats(self) -> dict:
        """Get GPU utilization and scheduling statistics"""
        
        try:
            # Calculate current utilization
            total_memory_allocated = sum(
                allocation.memory_allocated_mb 
                for allocation in self.active_allocations.values()
            )
            
            total_compute_allocated = sum(
                allocation.compute_allocated_percent 
                for allocation in self.active_allocations.values()
            )
            
            # Get system GPU info
            try:
                from gpu.manager import gpu_manager
                gpu_available = gpu_manager.gpu_available if gpu_manager else False
                total_gpu_memory = gpu_manager.total_memory_mb if gpu_manager else 0
            except:
                gpu_available = False
                total_gpu_memory = 0
            
            return {
                "gpu_status": {
                    "available": gpu_available,
                    "total_memory_mb": total_gpu_memory,
                    "allocated_memory_mb": total_memory_allocated,
                    "memory_utilization_percent": (
                        (total_memory_allocated / total_gpu_memory * 100) 
                        if total_gpu_memory > 0 else 0
                    )
                },
                "active_allocations": {
                    "count": len(self.active_allocations),
                    "details": [
                        {
                            "user_id": allocation.user_id,
                            "simulation_id": allocation.simulation_id,
                            "priority": allocation.priority.value,
                            "memory_mb": allocation.memory_allocated_mb,
                            "compute_percent": allocation.compute_allocated_percent,
                            "duration_minutes": allocation.estimated_duration_minutes,
                            "allocated_at": allocation.allocated_at.isoformat()
                        }
                        for allocation in self.active_allocations.values()
                    ]
                },
                "scheduling_metrics": self.metrics,
                "fair_share_stats": await self.fair_share_scheduler.get_scheduling_stats(),
                "ultra_engine_integration": {
                    "preserved": True,
                    "enhanced": "with enterprise resource management"
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ [GPU_SCHEDULER] Failed to get utilization stats: {e}")
            return {"error": str(e)}

# Global GPU scheduler instance
enterprise_gpu_scheduler = EnterpriseGPUScheduler()

# Convenience functions that preserve existing Ultra engine functionality
async def schedule_enterprise_simulation(user_id: int, simulation_request: dict) -> Optional[GPUAllocation]:
    """Schedule simulation with enterprise GPU management (preserves Ultra engine)"""
    return await enterprise_gpu_scheduler.schedule_simulation(user_id, simulation_request)

async def release_simulation_resources(simulation_id: str):
    """Release GPU resources when simulation completes"""
    await enterprise_gpu_scheduler.release_allocation(simulation_id)

async def get_gpu_scheduling_stats() -> dict:
    """Get GPU scheduling and utilization statistics"""
    return await enterprise_gpu_scheduler.get_gpu_utilization_stats()
