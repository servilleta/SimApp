"""
🔍 ENTERPRISE METRICS COLLECTOR - Phase 5 Week 17-18
Advanced monitoring and metrics collection for Monte Carlo Enterprise Platform

This module implements the custom business metrics described in enterprise.txt:
- Simulation performance metrics
- User satisfaction tracking
- Revenue per user calculations
- Ultra Engine optimization metrics
- Success rate monitoring

Integrates with Prometheus for time-series data collection.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import logging
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import psutil
import GPUtil

logger = logging.getLogger(__name__)


@dataclass
class SimulationMetrics:
    """Simulation performance metrics"""
    simulation_id: str
    user_id: int
    duration: float
    success: bool
    complexity: str
    engine: str
    gpu_utilization: float
    memory_usage: float


@dataclass
class UserMetrics:
    """User satisfaction and usage metrics"""
    user_id: int
    organization_id: int
    tier: str
    satisfaction_score: float
    monthly_usage: int
    monthly_revenue: float


class EnterpriseMetricsCollector:
    """
    Enterprise-grade metrics collector for Monte Carlo platform
    
    Implements the custom business metrics from enterprise.txt:
    - simulation_duration_seconds: Time taken to complete simulations
    - user_satisfaction_score: User satisfaction score based on NPS
    - monthly_revenue_per_user: Monthly revenue per user in USD
    - simulation_success_rate: Percentage of successful simulations
    """
    
    def __init__(self):
        # Prometheus metrics as defined in enterprise.txt
        self.simulation_duration = Histogram(
            'simulation_duration_seconds',
            'Time taken to complete simulations',
            ['user_tier', 'complexity', 'engine']
        )
        
        self.user_satisfaction = Gauge(
            'user_satisfaction_score',
            'User satisfaction score based on NPS',
            ['organization', 'user_tier']
        )
        
        self.revenue_per_user = Gauge(
            'monthly_revenue_per_user',
            'Monthly revenue per user in USD',
            ['organization', 'user_tier']
        )
        
        self.simulation_success_rate = Gauge(
            'simulation_success_rate',
            'Percentage of successful simulations',
            ['engine', 'complexity']
        )
        
        # Additional Ultra Engine metrics
        self.ultra_engine_response_time = Histogram(
            'ultra_engine_response_time_seconds',
            'Ultra Engine API response times',
            ['endpoint', 'method']
        )
        
        self.progress_bar_latency = Histogram(
            'progress_bar_latency_milliseconds',
            'Progress bar update latency (critical metric)',
            ['user_tier']
        )
        
        self.active_simulations = Gauge(
            'active_simulations_total',
            'Number of currently active simulations'
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['component']
        )
        
        # Business metrics
        self.total_users = Gauge('total_users', 'Total number of users')
        self.total_organizations = Gauge('total_organizations', 'Total number of organizations')
        self.monthly_revenue = Gauge('monthly_revenue_total', 'Total monthly revenue')
        
        # Performance tracking
        self.metrics_cache: Dict[str, Any] = {}
        self.last_collection = datetime.utcnow()
        
    async def record_simulation_completion(
        self, 
        user_id: int, 
        simulation_id: str, 
        duration: float, 
        success: bool,
        user_tier: str = "standard",
        complexity: str = "medium",
        engine: str = "ultra"
    ):
        """
        Record simulation completion metrics
        
        This implements the simulation_duration_seconds metric from enterprise.txt
        """
        try:
            # Record duration histogram
            self.simulation_duration.labels(
                user_tier=user_tier,
                complexity=complexity,
                engine=engine
            ).observe(duration)
            
            # Update success rate
            await self._update_success_rate(engine, complexity, success)
            
            # Log for debugging
            logger.info(
                f"Recorded simulation metrics: {simulation_id}, "
                f"duration={duration}s, success={success}, tier={user_tier}"
            )
            
        except Exception as e:
            logger.error(f"Error recording simulation metrics: {e}")
    
    async def record_progress_bar_latency(self, latency_ms: float, user_tier: str = "standard"):
        """
        Record progress bar latency - CRITICAL for Ultra Engine performance
        
        This is the key metric that ensures we maintain the 51ms response time
        mentioned in enterprise.txt
        """
        try:
            self.progress_bar_latency.labels(user_tier=user_tier).observe(latency_ms)
            
            # Alert if latency exceeds threshold
            if latency_ms > 100:  # 100ms threshold
                logger.warning(
                    f"Progress bar latency HIGH: {latency_ms}ms (threshold: 100ms)"
                )
            
        except Exception as e:
            logger.error(f"Error recording progress bar latency: {e}")
    
    async def update_user_satisfaction(
        self, 
        user_id: int, 
        organization: str, 
        user_tier: str, 
        nps_score: float
    ):
        """
        Update user satisfaction metrics based on NPS score
        
        Implements user_satisfaction_score metric from enterprise.txt
        """
        try:
            self.user_satisfaction.labels(
                organization=organization,
                user_tier=user_tier
            ).set(nps_score)
            
            logger.info(f"Updated user satisfaction: {organization}/{user_tier} = {nps_score}")
            
        except Exception as e:
            logger.error(f"Error updating user satisfaction: {e}")
    
    async def update_revenue_metrics(
        self, 
        user_id: int, 
        organization: str, 
        user_tier: str, 
        monthly_revenue: float
    ):
        """
        Update revenue per user metrics
        
        Implements monthly_revenue_per_user metric from enterprise.txt
        """
        try:
            self.revenue_per_user.labels(
                organization=organization,
                user_tier=user_tier
            ).set(monthly_revenue)
            
            logger.info(f"Updated revenue metrics: {organization}/{user_tier} = ${monthly_revenue}")
            
        except Exception as e:
            logger.error(f"Error updating revenue metrics: {e}")
    
    async def collect_system_metrics(self):
        """
        Collect system-level metrics for monitoring
        """
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.memory_usage.labels(component="system").set(memory.used)
            
            # GPU metrics (if available)
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.gpu_utilization.labels(gpu_id=str(i)).set(gpu.load * 100)
            except Exception:
                # GPU monitoring not available
                pass
            
            logger.debug(f"Collected system metrics: CPU={cpu_percent}%, Memory={memory.percent}%")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _update_success_rate(self, engine: str, complexity: str, success: bool):
        """Update simulation success rate metrics"""
        try:
            # This is a simplified implementation
            # In production, you'd calculate this from a sliding window of results
            cache_key = f"success_rate_{engine}_{complexity}"
            
            if cache_key not in self.metrics_cache:
                self.metrics_cache[cache_key] = {"total": 0, "successful": 0}
            
            self.metrics_cache[cache_key]["total"] += 1
            if success:
                self.metrics_cache[cache_key]["successful"] += 1
            
            success_rate = (
                self.metrics_cache[cache_key]["successful"] / 
                self.metrics_cache[cache_key]["total"] * 100
            )
            
            self.simulation_success_rate.labels(
                engine=engine,
                complexity=complexity
            ).set(success_rate)
            
        except Exception as e:
            logger.error(f"Error updating success rate: {e}")
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get current metrics summary for health checks and debugging
        """
        try:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "collection_status": "healthy",
                "metrics_cache_size": len(self.metrics_cache),
                "last_collection": self.last_collection.isoformat(),
                "ultra_engine": {
                    "monitoring": "active",
                    "progress_bar_tracking": "enabled",
                    "performance_impact": "zero"
                }
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_prometheus_metrics(self) -> str:
        """
        Get Prometheus-formatted metrics
        
        This endpoint is scraped by Prometheus as configured in prometheus.yml
        """
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Error generating Prometheus metrics: {e}")
            return ""


# Global metrics collector instance
metrics_collector = EnterpriseMetricsCollector()


async def start_metrics_collection():
    """
    Start background metrics collection
    
    This runs continuously to collect system metrics
    """
    logger.info("Starting enterprise metrics collection...")
    
    while True:
        try:
            await metrics_collector.collect_system_metrics()
            await asyncio.sleep(15)  # Collect every 15 seconds
            
        except Exception as e:
            logger.error(f"Error in metrics collection loop: {e}")
            await asyncio.sleep(60)  # Wait longer on error


# Convenience functions for easy integration
async def record_simulation_metrics(
    user_id: int,
    simulation_id: str, 
    duration: float,
    success: bool,
    **kwargs
):
    """Convenience function to record simulation metrics"""
    await metrics_collector.record_simulation_completion(
        user_id, simulation_id, duration, success, **kwargs
    )


async def record_progress_bar_performance(latency_ms: float, user_tier: str = "standard"):
    """Convenience function to record progress bar performance"""
    await metrics_collector.record_progress_bar_latency(latency_ms, user_tier)


async def update_business_metrics(user_metrics: UserMetrics):
    """Convenience function to update business metrics"""
    await metrics_collector.update_user_satisfaction(
        user_metrics.user_id,
        f"org_{user_metrics.organization_id}",
        user_metrics.tier,
        user_metrics.satisfaction_score
    )
    
    await metrics_collector.update_revenue_metrics(
        user_metrics.user_id,
        f"org_{user_metrics.organization_id}",
        user_metrics.tier,
        user_metrics.monthly_revenue
    )
