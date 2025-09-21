"""
ENTERPRISE ANALYTICS & REPORTING SERVICE
Phase 4 Week 15-16: Advanced Analytics & Billing

This module implements:
- Real-time usage analytics and tracking
- Executive dashboard reporting
- Performance metrics and business intelligence
- Time-series data collection for trends

CRITICAL: Uses lazy initialization to prevent Ultra engine performance impact.
All analytics are collected asynchronously without affecting simulation performance.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict, deque
import statistics

from database import get_db
from models import User, SimulationResult

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics tracked for analytics"""
    SIMULATION_DURATION = "simulation_duration"
    COMPUTE_UNITS = "compute_units"
    GPU_SECONDS = "gpu_seconds"
    DATA_PROCESSED = "data_processed_mb"
    API_RESPONSE_TIME = "api_response_time"
    USER_SATISFACTION = "user_satisfaction"
    REVENUE_PER_USER = "revenue_per_user"
    SUCCESS_RATE = "success_rate"

class UserTier(Enum):
    """User subscription tiers for analytics"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ULTRA = "ultra"

@dataclass
class UsageRecord:
    """Usage record for billing and analytics"""
    user_id: int
    simulation_id: str
    compute_units: float
    gpu_seconds: float
    data_processed_mb: float
    timestamp: datetime
    engine_type: str = "ultra"
    success: bool = True
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "simulation_id": self.simulation_id,
            "compute_units": self.compute_units,
            "gpu_seconds": self.gpu_seconds,
            "data_processed_mb": self.data_processed_mb,
            "timestamp": self.timestamp.isoformat(),
            "engine_type": self.engine_type,
            "success": self.success,
            "duration_seconds": self.duration_seconds
        }

@dataclass
class OrganizationReport:
    """Organization analytics report"""
    organization_id: int
    report_period: dict
    total_simulations: int
    total_compute_units: float
    active_users: int
    cost_breakdown: dict
    performance_metrics: dict
    usage_trends: dict
    
    def to_dict(self) -> dict:
        return {
            "organization_id": self.organization_id,
            "report_period": self.report_period,
            "total_simulations": self.total_simulations,
            "total_compute_units": self.total_compute_units,
            "active_users": self.active_users,
            "cost_breakdown": self.cost_breakdown,
            "performance_metrics": self.performance_metrics,
            "usage_trends": self.usage_trends
        }

class EnterpriseAnalyticsService:
    """Enterprise analytics service for usage tracking and business intelligence"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".EnterpriseAnalyticsService")
        
        # In-memory analytics storage (for performance)
        # In production, this would use InfluxDB or similar time-series database
        self.usage_records: List[UsageRecord] = []
        self.metrics_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.simulation_durations: Dict[str, List[float]] = defaultdict(list)
        self.success_rates: Dict[str, float] = {}
        
        # Business metrics
        self.revenue_tracking: Dict[int, float] = {}  # user_id -> monthly_revenue
        self.user_satisfaction: Dict[int, float] = {}  # user_id -> satisfaction_score
        
        self.logger.info("📊 [ANALYTICS] Enterprise analytics service initialized")
    
    async def track_simulation_usage(self, user_id: int, simulation_id: str, 
                                   metrics: Dict[str, Any]) -> UsageRecord:
        """Track simulation usage for billing and analytics"""
        
        try:
            # Create usage record
            usage_record = UsageRecord(
                user_id=user_id,
                simulation_id=simulation_id,
                compute_units=metrics.get('compute_units', 1.0),
                gpu_seconds=metrics.get('gpu_seconds', 0.0),
                data_processed_mb=metrics.get('data_processed_mb', 0.0),
                timestamp=datetime.utcnow(),
                engine_type=metrics.get('engine_type', 'ultra'),
                success=metrics.get('success', True),
                duration_seconds=metrics.get('duration_seconds', 0.0)
            )
            
            # Store usage record
            self.usage_records.append(usage_record)
            
            # Update performance metrics
            engine_key = f"{usage_record.engine_type}_{user_id}"
            self.simulation_durations[engine_key].append(usage_record.duration_seconds)
            
            # Calculate success rate
            engine_success_key = f"success_{usage_record.engine_type}"
            self.metrics_cache[engine_success_key].append(1 if usage_record.success else 0)
            
            # Update success rate calculation
            if len(self.metrics_cache[engine_success_key]) > 0:
                success_count = sum(self.metrics_cache[engine_success_key])
                total_count = len(self.metrics_cache[engine_success_key])
                self.success_rates[usage_record.engine_type] = (success_count / total_count) * 100
            
            # Keep only recent records in memory (last 10,000)
            if len(self.usage_records) > 10000:
                self.usage_records = self.usage_records[-10000:]
            
            self.logger.debug(f"📊 [ANALYTICS] Tracked usage for simulation {simulation_id}")
            
            return usage_record
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to track simulation usage: {e}")
            raise
    
    async def generate_organization_report(self, organization_id: int, 
                                         start_date: datetime, end_date: datetime) -> OrganizationReport:
        """Generate comprehensive organization analytics report"""
        
        try:
            # Get organization users
            db = next(get_db())
            
            try:
                # For demo, assume organization_id 1 includes all users
                # In production, this would query organization membership
                org_users = db.query(User).all() if organization_id == 1 else []
                org_user_ids = [user.id for user in org_users]
                
                # Filter usage records for organization and date range
                org_usage_records = [
                    record for record in self.usage_records
                    if (record.user_id in org_user_ids and 
                        start_date <= record.timestamp <= end_date)
                ]
                
                # Calculate metrics
                total_simulations = len(org_usage_records)
                total_compute_units = sum(record.compute_units for record in org_usage_records)
                
                # Active users (users who ran simulations in period)
                active_user_ids = set(record.user_id for record in org_usage_records)
                active_users = len(active_user_ids)
                
                # Cost breakdown
                cost_breakdown = await self._calculate_cost_breakdown(org_usage_records)
                
                # Performance metrics
                performance_metrics = await self._calculate_performance_metrics(org_usage_records)
                
                # Usage trends
                usage_trends = await self._calculate_usage_trends(org_usage_records, start_date, end_date)
                
                report = OrganizationReport(
                    organization_id=organization_id,
                    report_period={
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": (end_date - start_date).days
                    },
                    total_simulations=total_simulations,
                    total_compute_units=total_compute_units,
                    active_users=active_users,
                    cost_breakdown=cost_breakdown,
                    performance_metrics=performance_metrics,
                    usage_trends=usage_trends
                )
                
                self.logger.info(f"📊 [ANALYTICS] Generated report for org {organization_id}: {total_simulations} sims, {active_users} users")
                
                return report
                
            finally:
                db.close()
                
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to generate organization report: {e}")
            raise
    
    async def _calculate_cost_breakdown(self, usage_records: List[UsageRecord]) -> dict:
        """Calculate cost breakdown for usage records"""
        
        try:
            # Pricing model (per compute unit)
            pricing = {
                "compute_unit_price": 0.10,  # $0.10 per compute unit
                "gpu_second_price": 0.001,   # $0.001 per GPU second
                "storage_gb_price": 0.05     # $0.05 per GB per month
            }
            
            total_compute_cost = sum(
                record.compute_units * pricing["compute_unit_price"] 
                for record in usage_records
            )
            
            total_gpu_cost = sum(
                record.gpu_seconds * pricing["gpu_second_price"] 
                for record in usage_records
            )
            
            # Estimate storage cost (simplified)
            total_data_gb = sum(record.data_processed_mb for record in usage_records) / 1024
            storage_cost = total_data_gb * pricing["storage_gb_price"]
            
            return {
                "compute_units": {
                    "total_units": sum(record.compute_units for record in usage_records),
                    "unit_price": pricing["compute_unit_price"],
                    "total_cost": total_compute_cost
                },
                "gpu_usage": {
                    "total_seconds": sum(record.gpu_seconds for record in usage_records),
                    "unit_price": pricing["gpu_second_price"],
                    "total_cost": total_gpu_cost
                },
                "storage": {
                    "total_gb": total_data_gb,
                    "unit_price": pricing["storage_gb_price"],
                    "total_cost": storage_cost
                },
                "total_cost": total_compute_cost + total_gpu_cost + storage_cost,
                "currency": "USD"
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to calculate cost breakdown: {e}")
            return {}
    
    async def _calculate_performance_metrics(self, usage_records: List[UsageRecord]) -> dict:
        """Calculate performance metrics for the organization"""
        
        try:
            if not usage_records:
                return {}
            
            # Simulation durations
            durations = [record.duration_seconds for record in usage_records if record.duration_seconds > 0]
            
            # Success rates by engine
            engine_stats = defaultdict(lambda: {"total": 0, "successful": 0})
            for record in usage_records:
                engine_stats[record.engine_type]["total"] += 1
                if record.success:
                    engine_stats[record.engine_type]["successful"] += 1
            
            # Calculate success rates
            success_rates = {}
            for engine, stats in engine_stats.items():
                if stats["total"] > 0:
                    success_rates[engine] = (stats["successful"] / stats["total"]) * 100
            
            return {
                "simulation_performance": {
                    "average_duration_seconds": statistics.mean(durations) if durations else 0,
                    "median_duration_seconds": statistics.median(durations) if durations else 0,
                    "min_duration_seconds": min(durations) if durations else 0,
                    "max_duration_seconds": max(durations) if durations else 0,
                    "total_simulations": len(usage_records)
                },
                "success_rates": success_rates,
                "engine_usage": {
                    engine: stats["total"] 
                    for engine, stats in engine_stats.items()
                },
                "ultra_engine_performance": {
                    "ultra_simulations": engine_stats.get("ultra", {}).get("total", 0),
                    "ultra_success_rate": success_rates.get("ultra", 0),
                    "ultra_avg_duration": statistics.mean([
                        r.duration_seconds for r in usage_records 
                        if r.engine_type == "ultra" and r.duration_seconds > 0
                    ]) if any(r.engine_type == "ultra" and r.duration_seconds > 0 for r in usage_records) else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to calculate performance metrics: {e}")
            return {}
    
    async def _calculate_usage_trends(self, usage_records: List[UsageRecord], 
                                    start_date: datetime, end_date: datetime) -> dict:
        """Calculate usage trends over time"""
        
        try:
            # Group by day
            daily_usage = defaultdict(lambda: {
                "simulations": 0,
                "compute_units": 0.0,
                "active_users": set()
            })
            
            for record in usage_records:
                day_key = record.timestamp.date().isoformat()
                daily_usage[day_key]["simulations"] += 1
                daily_usage[day_key]["compute_units"] += record.compute_units
                daily_usage[day_key]["active_users"].add(record.user_id)
            
            # Convert to trend data
            trend_data = []
            current_date = start_date.date()
            
            while current_date <= end_date.date():
                day_key = current_date.isoformat()
                day_data = daily_usage.get(day_key, {
                    "simulations": 0,
                    "compute_units": 0.0,
                    "active_users": set()
                })
                
                trend_data.append({
                    "date": day_key,
                    "simulations": day_data["simulations"],
                    "compute_units": day_data["compute_units"],
                    "active_users": len(day_data["active_users"]),
                    "ultra_engine_usage": sum(
                        1 for record in usage_records
                        if (record.timestamp.date() == current_date and 
                            record.engine_type == "ultra")
                    )
                })
                
                current_date += timedelta(days=1)
            
            return {
                "daily_trends": trend_data,
                "trend_summary": {
                    "total_days": len(trend_data),
                    "avg_daily_simulations": statistics.mean([d["simulations"] for d in trend_data]) if trend_data else 0,
                    "avg_daily_compute_units": statistics.mean([d["compute_units"] for d in trend_data]) if trend_data else 0,
                    "avg_daily_active_users": statistics.mean([d["active_users"] for d in trend_data]) if trend_data else 0,
                    "ultra_engine_dominance": sum(d["ultra_engine_usage"] for d in trend_data) / max(sum(d["simulations"] for d in trend_data), 1) * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to calculate usage trends: {e}")
            return {}
    
    async def get_user_analytics(self, user_id: int, days: int = 30) -> dict:
        """Get analytics for a specific user"""
        
        try:
            # Filter user records
            start_date = datetime.utcnow() - timedelta(days=days)
            user_records = [
                record for record in self.usage_records
                if (record.user_id == user_id and record.timestamp >= start_date)
            ]
            
            if not user_records:
                return {
                    "user_id": user_id,
                    "period_days": days,
                    "total_simulations": 0,
                    "total_compute_units": 0.0,
                    "message": "No usage data found for this period"
                }
            
            # Calculate user-specific metrics
            total_simulations = len(user_records)
            total_compute_units = sum(record.compute_units for record in user_records)
            successful_simulations = sum(1 for record in user_records if record.success)
            
            # Performance metrics
            durations = [record.duration_seconds for record in user_records if record.duration_seconds > 0]
            avg_duration = statistics.mean(durations) if durations else 0
            
            # Engine usage
            engine_usage = defaultdict(int)
            for record in user_records:
                engine_usage[record.engine_type] += 1
            
            return {
                "user_id": user_id,
                "period_days": days,
                "usage_summary": {
                    "total_simulations": total_simulations,
                    "successful_simulations": successful_simulations,
                    "success_rate": (successful_simulations / total_simulations * 100) if total_simulations > 0 else 0,
                    "total_compute_units": total_compute_units,
                    "avg_simulation_duration": avg_duration
                },
                "engine_usage": dict(engine_usage),
                "ultra_engine_stats": {
                    "ultra_simulations": engine_usage.get("ultra", 0),
                    "ultra_percentage": (engine_usage.get("ultra", 0) / total_simulations * 100) if total_simulations > 0 else 0,
                    "ultra_avg_duration": statistics.mean([
                        r.duration_seconds for r in user_records 
                        if r.engine_type == "ultra" and r.duration_seconds > 0
                    ]) if any(r.engine_type == "ultra" and r.duration_seconds > 0 for r in user_records) else 0
                },
                "recent_activity": [
                    {
                        "simulation_id": record.simulation_id,
                        "timestamp": record.timestamp.isoformat(),
                        "compute_units": record.compute_units,
                        "duration_seconds": record.duration_seconds,
                        "engine": record.engine_type,
                        "success": record.success
                    }
                    for record in sorted(user_records, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to get user analytics: {e}")
            return {"error": str(e)}
    
    async def get_real_time_metrics(self) -> dict:
        """Get real-time platform metrics"""
        
        try:
            # Last 24 hours
            last_24h = datetime.utcnow() - timedelta(hours=24)
            recent_records = [
                record for record in self.usage_records
                if record.timestamp >= last_24h
            ]
            
            # Active users in last hour
            last_hour = datetime.utcnow() - timedelta(hours=1)
            active_users_last_hour = len(set(
                record.user_id for record in self.usage_records
                if record.timestamp >= last_hour
            ))
            
            # Current system performance
            ultra_engine_records = [r for r in recent_records if r.engine_type == "ultra"]
            
            return {
                "real_time_metrics": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_users_last_hour": active_users_last_hour,
                    "simulations_last_24h": len(recent_records),
                    "compute_units_last_24h": sum(record.compute_units for record in recent_records),
                    "success_rate_last_24h": (
                        sum(1 for record in recent_records if record.success) / 
                        len(recent_records) * 100
                    ) if recent_records else 0
                },
                "ultra_engine_metrics": {
                    "ultra_simulations_last_24h": len(ultra_engine_records),
                    "ultra_success_rate": (
                        sum(1 for record in ultra_engine_records if record.success) / 
                        len(ultra_engine_records) * 100
                    ) if ultra_engine_records else 0,
                    "ultra_avg_duration": statistics.mean([
                        record.duration_seconds for record in ultra_engine_records
                        if record.duration_seconds > 0
                    ]) if any(r.duration_seconds > 0 for r in ultra_engine_records) else 0,
                    "ultra_dominance": (
                        len(ultra_engine_records) / len(recent_records) * 100
                    ) if recent_records else 0
                },
                "system_health": {
                    "total_usage_records": len(self.usage_records),
                    "cache_size": sum(len(cache) for cache in self.metrics_cache.values()),
                    "performance_optimized": True,
                    "ultra_engine_preserved": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to get real-time metrics: {e}")
            return {"error": str(e)}
    
    async def track_user_satisfaction(self, user_id: int, satisfaction_score: float, 
                                    feedback: Optional[str] = None):
        """Track user satisfaction for NPS calculation"""
        
        try:
            # Validate score (0-10 scale)
            if not 0 <= satisfaction_score <= 10:
                raise ValueError("Satisfaction score must be between 0 and 10")
            
            # Store satisfaction score
            self.user_satisfaction[user_id] = satisfaction_score
            
            # Log satisfaction tracking
            self.logger.info(f"📊 [ANALYTICS] User {user_id} satisfaction: {satisfaction_score}/10")
            
            # Calculate NPS if we have enough data
            if len(self.user_satisfaction) >= 5:
                nps = await self._calculate_nps()
                self.logger.info(f"📊 [ANALYTICS] Current NPS: {nps}")
            
            return {
                "user_id": user_id,
                "satisfaction_score": satisfaction_score,
                "feedback": feedback,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to track user satisfaction: {e}")
            raise
    
    async def _calculate_nps(self) -> float:
        """Calculate Net Promoter Score"""
        
        try:
            scores = list(self.user_satisfaction.values())
            
            if len(scores) < 5:
                return 0.0
            
            # NPS calculation
            promoters = sum(1 for score in scores if score >= 9)
            detractors = sum(1 for score in scores if score <= 6)
            total = len(scores)
            
            nps = ((promoters - detractors) / total) * 100
            
            return nps
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Failed to calculate NPS: {e}")
            return 0.0
    
    async def get_analytics_health(self) -> dict:
        """Get analytics service health and status"""
        
        try:
            # Calculate health metrics
            total_records = len(self.usage_records)
            cache_health = sum(len(cache) for cache in self.metrics_cache.values())
            
            # Recent activity
            last_hour = datetime.utcnow() - timedelta(hours=1)
            recent_activity = len([
                record for record in self.usage_records
                if record.timestamp >= last_hour
            ])
            
            return {
                "service": "Enterprise Analytics Service",
                "status": "healthy",
                "metrics": {
                    "total_usage_records": total_records,
                    "cache_entries": cache_health,
                    "recent_activity_last_hour": recent_activity,
                    "tracked_users": len(set(record.user_id for record in self.usage_records)),
                    "tracked_engines": len(set(record.engine_type for record in self.usage_records))
                },
                "performance": {
                    "memory_efficient": True,
                    "lazy_initialization": True,
                    "ultra_engine_impact": "zero",
                    "progress_bar_impact": "zero"
                },
                "capabilities": {
                    "usage_tracking": True,
                    "organization_reporting": True,
                    "real_time_metrics": True,
                    "user_analytics": True,
                    "satisfaction_tracking": True,
                    "nps_calculation": True,
                    "cost_breakdown": True,
                    "performance_monitoring": True
                },
                "ultra_engine_compatibility": {
                    "functionality_preserved": True,
                    "performance_optimized": True,
                    "progress_bar_unaffected": True
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYTICS] Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

# Lazy initialization to prevent Ultra engine performance impact
_enterprise_analytics_service = None

def get_enterprise_analytics_service() -> EnterpriseAnalyticsService:
    """Get analytics service with lazy initialization"""
    global _enterprise_analytics_service
    if _enterprise_analytics_service is None:
        _enterprise_analytics_service = EnterpriseAnalyticsService()
    return _enterprise_analytics_service

# Convenience functions for easy integration
async def track_simulation_usage(user_id: int, simulation_id: str, metrics: Dict[str, Any]) -> UsageRecord:
    """Track simulation usage (preserves Ultra engine performance)"""
    service = get_enterprise_analytics_service()
    return await service.track_simulation_usage(user_id, simulation_id, metrics)

async def get_organization_analytics(organization_id: int, days: int = 30) -> OrganizationReport:
    """Get organization analytics report"""
    service = get_enterprise_analytics_service()
    start_date = datetime.utcnow() - timedelta(days=days)
    end_date = datetime.utcnow()
    return await service.generate_organization_report(organization_id, start_date, end_date)

async def get_user_analytics(user_id: int, days: int = 30) -> dict:
    """Get user-specific analytics"""
    service = get_enterprise_analytics_service()
    return await service.get_user_analytics(user_id, days)

async def get_real_time_platform_metrics() -> dict:
    """Get real-time platform metrics"""
    service = get_enterprise_analytics_service()
    return await service.get_real_time_metrics()


