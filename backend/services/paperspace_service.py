"""
Paperspace Service for SimApp Backend
====================================

Integrates Paperspace API management into the SimApp backend for:
- Automatic scaling based on simulation load
- Cost optimization through intelligent server management
- Real-time cluster status monitoring
- Blue-Green deployment support

Author: SimApp DevOps Team
Date: September 21, 2025
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
sys.path.append('/home/paperspace/SimApp')

from paperspace_api_manager import PaperspaceAPIManager, MachineState
from backend.database.database import get_db_session
from backend.models.models import SimulationRequest
from sqlalchemy import func, and_

logger = logging.getLogger(__name__)

class PaperspaceScalingService:
    """
    Service for managing Paperspace servers based on SimApp workload
    """
    
    def __init__(self):
        """Initialize the Paperspace scaling service"""
        self.api_manager = PaperspaceAPIManager()
        self.scaling_cooldown = 300  # 5 minutes between scaling operations
        self.last_scale_action = None
        self.scale_up_threshold = 5  # Scale up if >5 pending simulations
        self.scale_down_threshold = 2  # Scale down if <2 pending simulations
        self.secondary_server_cost_per_hour = 0.40  # Estimated cost
        
        logger.info("🚀 Paperspace Scaling Service initialized")
    
    async def get_simulation_load(self) -> Dict[str, int]:
        """
        Get current simulation workload statistics
        
        Returns:
            Dictionary with workload metrics
        """
        try:
            async with get_db_session() as session:
                # Count simulations by status
                pending_count = await session.execute(
                    "SELECT COUNT(*) FROM simulation_requests WHERE status = 'pending'"
                ).scalar()
                
                running_count = await session.execute(
                    "SELECT COUNT(*) FROM simulation_requests WHERE status = 'running'"
                ).scalar()
                
                # Count simulations in last hour
                last_hour = datetime.utcnow() - timedelta(hours=1)
                recent_count = await session.execute(
                    "SELECT COUNT(*) FROM simulation_requests WHERE created_at > :last_hour",
                    {"last_hour": last_hour}
                ).scalar()
                
                return {
                    "pending_simulations": pending_count or 0,
                    "running_simulations": running_count or 0,
                    "recent_simulations_1h": recent_count or 0,
                    "total_load": (pending_count or 0) + (running_count or 0)
                }
        
        except Exception as e:
            logger.error(f"❌ Failed to get simulation load: {e}")
            return {
                "pending_simulations": 0,
                "running_simulations": 0,
                "recent_simulations_1h": 0,
                "total_load": 0
            }
    
    async def should_scale_up(self) -> bool:
        """
        Determine if we should scale up (start secondary server)
        
        Returns:
            True if scaling up is recommended
        """
        load = await self.get_simulation_load()
        
        # Scale up conditions:
        # 1. High pending simulation count
        # 2. High total load
        # 3. Recent activity surge
        should_scale = (
            load["pending_simulations"] >= self.scale_up_threshold or
            load["total_load"] >= 8 or
            load["recent_simulations_1h"] >= 10
        )
        
        if should_scale:
            logger.info(f"📈 Scale up recommended: {load}")
        
        return should_scale
    
    async def should_scale_down(self) -> bool:
        """
        Determine if we should scale down (stop secondary server)
        
        Returns:
            True if scaling down is recommended
        """
        load = await self.get_simulation_load()
        
        # Scale down conditions:
        # 1. Low pending simulation count
        # 2. Low total load
        # 3. No recent activity
        should_scale = (
            load["pending_simulations"] <= self.scale_down_threshold and
            load["total_load"] <= 3 and
            load["recent_simulations_1h"] <= 2
        )
        
        if should_scale:
            logger.info(f"📉 Scale down recommended: {load}")
        
        return should_scale
    
    def can_perform_scaling_action(self) -> bool:
        """
        Check if enough time has passed since last scaling action
        
        Returns:
            True if scaling action can be performed
        """
        if not self.last_scale_action:
            return True
        
        time_since_last = datetime.utcnow() - self.last_scale_action
        return time_since_last.total_seconds() >= self.scaling_cooldown
    
    async def auto_scale_cluster(self) -> Dict[str, Any]:
        """
        Perform automatic scaling based on current workload
        
        Returns:
            Dictionary with scaling action results
        """
        logger.info("🤖 Running automatic cluster scaling check...")
        
        if not self.can_perform_scaling_action():
            return {
                "action": "none",
                "reason": "cooling_down",
                "next_check_in": self.scaling_cooldown - 
                    (datetime.utcnow() - self.last_scale_action).total_seconds()
            }
        
        # Get cluster status
        cluster_status = self.api_manager.get_cluster_status()
        secondary_running = cluster_status["secondary_server"]["status"] == "running"
        
        load = await self.get_simulation_load()
        
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "cluster_status": cluster_status,
            "simulation_load": load,
            "action": "none",
            "reason": "no_action_needed",
            "success": False
        }
        
        # Decide on scaling action
        if not secondary_running and await self.should_scale_up():
            logger.info("🚀 AUTO-SCALING: Starting secondary server...")
            success = self.api_manager.scale_up_secondary_server()
            result.update({
                "action": "scale_up",
                "reason": "high_load",
                "success": success
            })
            if success:
                self.last_scale_action = datetime.utcnow()
        
        elif secondary_running and await self.should_scale_down():
            logger.info("🛑 AUTO-SCALING: Stopping secondary server...")
            success = self.api_manager.scale_down_secondary_server()
            result.update({
                "action": "scale_down", 
                "reason": "low_load",
                "success": success
            })
            if success:
                self.last_scale_action = datetime.utcnow()
        
        return result
    
    async def get_cost_optimization_report(self) -> Dict[str, Any]:
        """
        Generate cost optimization report
        
        Returns:
            Dictionary with cost analysis and recommendations
        """
        cluster_status = self.api_manager.get_cluster_status()
        load = await self.get_simulation_load()
        
        # Calculate estimated costs
        primary_cost = 24 * 0.25  # Estimated $0.25/hour for primary
        secondary_running = cluster_status["secondary_server"]["status"] == "running"
        secondary_cost = 24 * self.secondary_server_cost_per_hour if secondary_running else 0
        
        total_daily_cost = primary_cost + secondary_cost
        
        # Optimization recommendations
        recommendations = []
        
        if secondary_running and load["total_load"] < 3:
            potential_savings = 24 * self.secondary_server_cost_per_hour
            recommendations.append({
                "type": "cost_saving",
                "action": "stop_secondary_server",
                "potential_daily_savings": potential_savings,
                "reason": "Low simulation load detected"
            })
        
        if not secondary_running and load["pending_simulations"] > 5:
            recommendations.append({
                "type": "performance",
                "action": "start_secondary_server", 
                "estimated_daily_cost": 24 * self.secondary_server_cost_per_hour,
                "reason": "High pending simulation backlog"
            })
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "current_costs": {
                "primary_server_daily": primary_cost,
                "secondary_server_daily": secondary_cost,
                "total_daily": total_daily_cost,
                "estimated_monthly": total_daily_cost * 30
            },
            "cluster_utilization": {
                "secondary_server_running": secondary_running,
                "simulation_load": load,
                "efficiency_score": min(100, (load["total_load"] / 10) * 100)
            },
            "recommendations": recommendations
        }

# Global service instance
paperspace_service = PaperspaceScalingService()

async def get_paperspace_service() -> PaperspaceScalingService:
    """
    Dependency injection for FastAPI endpoints
    
    Returns:
        PaperspaceScalingService instance
    """
    return paperspace_service
