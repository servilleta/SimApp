"""
System Monitoring for Admin Panel
=================================

Provides real-time system monitoring and Paperspace scaling insights
for the admin dashboard, including Server 1 stress levels and
auto-scaling trigger visualization.

Author: SimApp DevOps Team
Date: September 21, 2025
"""

import os
import sys
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import asyncio
import requests

# Add path for paperspace integration
sys.path.append('/home/paperspace/SimApp')
sys.path.append('/app')

logger = logging.getLogger(__name__)

# Response models
class SystemMetrics(BaseModel):
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_total_gb: float
    memory_used_gb: float
    disk_usage: float
    disk_total_gb: float
    disk_free_gb: float
    load_average: List[float]
    active_processes: int
    network_io: Dict[str, int]

class PaperspaceStatus(BaseModel):
    server1_running: bool
    server2_running: bool
    server1_ip: Optional[str]
    server2_ip: Optional[str]
    last_scale_event: Optional[datetime]
    scaling_triggers_active: List[str]

class AutoScalingMetrics(BaseModel):
    cpu_threshold: float
    memory_threshold: float
    response_time_threshold: float
    active_simulations_threshold: int
    current_cpu: float
    current_memory: float
    current_response_time: float
    current_simulations: int
    scaling_recommendation: str
    cost_impact: Dict[str, float]

class SystemMonitoringResponse(BaseModel):
    system_metrics: SystemMetrics
    paperspace_status: PaperspaceStatus
    scaling_metrics: AutoScalingMetrics
    alerts: List[Dict[str, Any]]
    recommendations: List[str]

router = APIRouter()

class SystemMonitor:
    """System monitoring service for admin panel"""
    
    def __init__(self):
        self.paperspace_api = None
        self.load_paperspace_api()
        
        # Scaling thresholds (matching development_scaling.py)
        self.stress_thresholds = {
            'cpu_usage': 70.0,
            'memory_usage': 75.0,
            'api_response_time': 3000,
            'active_simulations': 3
        }
    
    def load_paperspace_api(self):
        """Load Paperspace API manager if available"""
        try:
            # Import and create API manager with proper environment setup
            import os
            api_key = os.getenv('PAPERSPACE_API_KEY')
            if not api_key:
                logger.warning("⚠️ PAPERSPACE_API_KEY not found in environment")
                self.paperspace_api = None
                return
                
            from paperspace_api_manager import PaperspaceAPIManager
            self.paperspace_api = PaperspaceAPIManager(api_key=api_key)
            logger.info("✅ Paperspace API integration loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Paperspace API not available: {e}")
            self.paperspace_api = None
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_total_gb = memory.total / (1024**3)
            memory_used_gb = memory.used / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            disk_total_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            
            # Load average
            try:
                load_avg = os.getloadavg()
            except:
                load_avg = [0.0, 0.0, 0.0]
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                network_io = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            except:
                network_io = {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_total_gb=round(memory_total_gb, 2),
                memory_used_gb=round(memory_used_gb, 2),
                disk_usage=round(disk_usage, 1),
                disk_total_gb=round(disk_total_gb, 2),
                disk_free_gb=round(disk_free_gb, 2),
                load_average=[round(x, 2) for x in load_avg],
                active_processes=active_processes,
                network_io=network_io
            )
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0, memory_usage=0, memory_total_gb=0, memory_used_gb=0,
                disk_usage=0, disk_total_gb=0, disk_free_gb=0,
                load_average=[0, 0, 0], active_processes=0, network_io={}
            )
    
    def get_paperspace_status(self) -> PaperspaceStatus:
        """Get Paperspace servers status"""
        server1_running = True  # We know Server 1 is running (we're on it!)
        server2_running = False  # Default to off for cost savings
        server1_ip = "64.71.146.187"  # Current server IP
        server2_ip = "72.52.107.230"  # Server 2 IP
        last_scale_event = None
        scaling_triggers = []
        
        # Try to get real Paperspace status if API is available
        if self.paperspace_api:
            try:
                machines = self.paperspace_api.list_machines()
                for machine in machines:
                    machine_data = machine if isinstance(machine, dict) else machine.__dict__
                    
                    if machine_data.get('id') == 'psotdtcda5ap':  # Server 1
                        server1_running = machine_data.get('state') == 'ready'
                        server1_ip = machine_data.get('public_ip', server1_ip)
                    elif machine_data.get('id') == 'pso1zne8qfxx':  # Server 2
                        server2_running = machine_data.get('state') == 'ready'
                        server2_ip = machine_data.get('public_ip', server2_ip)
                logger.info(f"✅ Real Paperspace data: Server1={server1_running}, Server2={server2_running}")
            except Exception as e:
                logger.warning(f"⚠️ Using fallback Paperspace status (API unavailable): {e}")
        else:
            # Use direct API call as fallback
            try:
                import requests
                api_key = os.getenv('PAPERSPACE_API_KEY', 'fe6ccc742bb4e07d4951c11ebc360b')
                headers = {'X-API-Key': api_key, 'Content-Type': 'application/json'}
                
                response = requests.get('https://api.paperspace.io/machines/getMachines', 
                                     headers=headers, timeout=10)
                if response.status_code == 200:
                    machines = response.json()
                    for machine in machines:
                        if machine.get('id') == 'psotdtcda5ap':  # Server 1
                            server1_running = machine.get('state') == 'ready'
                            server1_ip = machine.get('publicIpAddress', server1_ip)
                        elif machine.get('id') == 'pso1zne8qfxx':  # Server 2
                            server2_running = machine.get('state') == 'ready'
                            server2_ip = machine.get('publicIpAddress', server2_ip)
                    logger.info(f"✅ Direct API call successful: Server1={server1_running}, Server2={server2_running}")
                else:
                    logger.warning(f"⚠️ Paperspace API returned {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ Direct Paperspace API call failed: {e}")
        
        # Check for active scaling triggers
        metrics = self.get_system_metrics()
        if metrics.cpu_usage > self.stress_thresholds['cpu_usage']:
            scaling_triggers.append(f"High CPU: {metrics.cpu_usage:.1f}%")
        if metrics.memory_usage > self.stress_thresholds['memory_usage']:
            scaling_triggers.append(f"High Memory: {metrics.memory_usage:.1f}%")
        
        return PaperspaceStatus(
            server1_running=server1_running,
            server2_running=server2_running,
            server1_ip=server1_ip,
            server2_ip=server2_ip,
            last_scale_event=last_scale_event,
            scaling_triggers_active=scaling_triggers
        )
    
    def get_api_response_time(self) -> float:
        """Check API response time"""
        try:
            start_time = time.time()
            # From inside Docker container, call backend directly
            response = requests.get('http://localhost:8000/api/health', timeout=3)
            response_time = (time.time() - start_time) * 1000
            logger.info(f"✅ API health check: {response_time:.2f}ms (status: {response.status_code})")
            return response_time if response.status_code == 200 else 5000
        except Exception as e:
            logger.warning(f"⚠️ API health check failed: {e}")
            return 5000  # Return more reasonable fallback
    
    def estimate_active_simulations(self, cpu_usage: float) -> int:
        """Estimate active simulations based on system load"""
        if cpu_usage < 20:
            return 0
        elif cpu_usage < 40:
            return 1
        elif cpu_usage < 60:
            return 2
        else:
            return max(2, int(cpu_usage / 25))
    
    def get_scaling_metrics(self) -> AutoScalingMetrics:
        """Get auto-scaling metrics and recommendations"""
        system_metrics = self.get_system_metrics()
        api_response_time = self.get_api_response_time()
        active_simulations = self.estimate_active_simulations(system_metrics.cpu_usage)
        
        # Determine scaling recommendation
        stress_score = 0
        if system_metrics.cpu_usage > self.stress_thresholds['cpu_usage']:
            stress_score += 2
        if system_metrics.memory_usage > self.stress_thresholds['memory_usage']:
            stress_score += 2
        if api_response_time > self.stress_thresholds['api_response_time']:
            stress_score += 1
        if active_simulations > self.stress_thresholds['active_simulations']:
            stress_score += 1
        
        if stress_score >= 3:
            recommendation = "SCALE_UP_RECOMMENDED"
        elif stress_score >= 2:
            recommendation = "SCALE_UP_CONSIDER"
        elif stress_score >= 1:
            recommendation = "MONITOR_CLOSELY"
        else:
            recommendation = "OPTIMAL"
        
        # Cost impact calculation
        cost_impact = {
            'current_hourly': 0.45,  # Server 1 only
            'scaled_hourly': 1.21,   # Both servers
            'daily_if_scaled': 29.04,
            'monthly_if_always_scaled': 871.0
        }
        
        return AutoScalingMetrics(
            cpu_threshold=self.stress_thresholds['cpu_usage'],
            memory_threshold=self.stress_thresholds['memory_usage'],
            response_time_threshold=self.stress_thresholds['api_response_time'],
            active_simulations_threshold=self.stress_thresholds['active_simulations'],
            current_cpu=system_metrics.cpu_usage,
            current_memory=system_metrics.memory_usage,
            current_response_time=api_response_time,
            current_simulations=active_simulations,
            scaling_recommendation=recommendation,
            cost_impact=cost_impact
        )
    
    def get_alerts(self, system_metrics: SystemMetrics, paperspace_status: PaperspaceStatus, scaling_metrics: AutoScalingMetrics) -> List[Dict[str, Any]]:
        """Generate system alerts"""
        alerts = []
        
        # High resource usage alerts
        if system_metrics.cpu_usage > 90:
            alerts.append({
                'level': 'critical',
                'message': f'Critical CPU usage: {system_metrics.cpu_usage:.1f}%',
                'action': 'Consider immediate scaling or load reduction'
            })
        elif system_metrics.cpu_usage > 75:
            alerts.append({
                'level': 'warning',
                'message': f'High CPU usage: {system_metrics.cpu_usage:.1f}%',
                'action': 'Monitor for auto-scaling trigger'
            })
        
        if system_metrics.memory_usage > 90:
            alerts.append({
                'level': 'critical',
                'message': f'Critical memory usage: {system_metrics.memory_usage:.1f}%',
                'action': 'Scale up immediately or restart services'
            })
        elif system_metrics.memory_usage > 80:
            alerts.append({
                'level': 'warning',
                'message': f'High memory usage: {system_metrics.memory_usage:.1f}%',
                'action': 'Consider scaling up'
            })
        
        # Disk space alerts
        if system_metrics.disk_usage > 90:
            alerts.append({
                'level': 'critical',
                'message': f'Critical disk usage: {system_metrics.disk_usage:.1f}%',
                'action': 'Clean up files immediately'
            })
        elif system_metrics.disk_usage > 80:
            alerts.append({
                'level': 'warning',
                'message': f'High disk usage: {system_metrics.disk_usage:.1f}%',
                'action': 'Schedule cleanup'
            })
        
        # Paperspace scaling alerts
        if scaling_metrics.scaling_recommendation == "SCALE_UP_RECOMMENDED":
            if not paperspace_status.server2_running:
                alerts.append({
                    'level': 'info',
                    'message': 'Auto-scaling recommended - Server 2 should start soon',
                    'action': 'Monitor for automatic Server 2 startup'
                })
        
        # Cost alerts
        if paperspace_status.server2_running:
            alerts.append({
                'level': 'info',
                'message': 'High-performance mode active (Server 2 running)',
                'action': f'Additional cost: ${scaling_metrics.cost_impact["scaled_hourly"] - scaling_metrics.cost_impact["current_hourly"]:.2f}/hour'
            })
        
        return alerts
    
    def get_recommendations(self, system_metrics: SystemMetrics, scaling_metrics: AutoScalingMetrics) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if scaling_metrics.scaling_recommendation == "SCALE_UP_RECOMMENDED":
            recommendations.append("🚀 High load detected - Server 2 auto-scaling recommended")
        elif scaling_metrics.scaling_recommendation == "OPTIMAL":
            recommendations.append("✅ System performance optimal - single server sufficient")
        
        if system_metrics.memory_usage > 70:
            recommendations.append("🧠 Consider optimizing memory usage or scaling up")
        
        if scaling_metrics.current_response_time > 2000:
            recommendations.append("⚡ API response time elevated - check for bottlenecks")
        
        if system_metrics.disk_usage > 70:
            recommendations.append("💾 Disk usage high - schedule cleanup or expand storage")
        
        if len(recommendations) == 0:
            recommendations.append("🎯 System running optimally")
        
        return recommendations

# Global monitor instance
system_monitor = SystemMonitor()

@router.get("/system/monitoring", response_model=SystemMonitoringResponse)
async def get_system_monitoring():
    """
    Get comprehensive system monitoring data for admin dashboard.
    
    Returns real-time metrics including:
    - System resource usage (CPU, memory, disk)
    - Paperspace servers status
    - Auto-scaling metrics and triggers
    - Alerts and recommendations
    """
    try:
        system_metrics = system_monitor.get_system_metrics()
        paperspace_status = system_monitor.get_paperspace_status()
        scaling_metrics = system_monitor.get_scaling_metrics()
        alerts = system_monitor.get_alerts(system_metrics, paperspace_status, scaling_metrics)
        recommendations = system_monitor.get_recommendations(system_metrics, scaling_metrics)
        
        return SystemMonitoringResponse(
            system_metrics=system_metrics,
            paperspace_status=paperspace_status,
            scaling_metrics=scaling_metrics,
            alerts=alerts,
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error getting system monitoring data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring data: {str(e)}")

@router.get("/system/stress-history")
async def get_stress_history(hours: int = 24):
    """
    Get historical stress data for the specified time period.
    This would typically come from a time-series database.
    For now, returns mock data structure.
    """
    try:
        # In a real implementation, this would query a time-series database
        # For now, return current metrics as a sample
        current_metrics = system_monitor.get_system_metrics()
        
        # Mock historical data structure
        stress_history = {
            'time_range_hours': hours,
            'data_points': [
                {
                    'timestamp': datetime.now() - timedelta(minutes=i*5),
                    'cpu_usage': max(0, current_metrics.cpu_usage + (i % 10 - 5) * 2),
                    'memory_usage': max(0, current_metrics.memory_usage + (i % 8 - 4) * 1.5),
                    'scaling_events': []
                }
                for i in range(min(hours * 12, 288))  # 5-minute intervals, max 24h
            ]
        }
        
        return stress_history
    except Exception as e:
        logger.error(f"Error getting stress history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stress history: {str(e)}")

@router.get("/system/paperspace-cost")
async def get_paperspace_cost_analysis():
    """
    Get detailed Paperspace cost analysis and optimization suggestions.
    """
    try:
        paperspace_status = system_monitor.get_paperspace_status()
        
        cost_analysis = {
            'current_status': {
                'server1_running': paperspace_status.server1_running,
                'server2_running': paperspace_status.server2_running,
                'current_hourly_cost': 0.45 if paperspace_status.server1_running else 0,
                'current_hourly_cost_with_server2': 1.21 if paperspace_status.server2_running else 0.45
            },
            'cost_scenarios': {
                'development_mode': {
                    'description': 'Smart auto-scaling (current setup)',
                    'estimated_monthly_cost': 150,
                    'hours_server1': 160,  # ~5 hours/day
                    'hours_server2': 40,   # ~1.3 hours/day
                    'savings_vs_always_on': 721
                },
                'always_on': {
                    'description': 'Both servers always running',
                    'estimated_monthly_cost': 871,
                    'hours_server1': 720,
                    'hours_server2': 720,
                    'savings_vs_always_on': 0
                },
                'server1_only': {
                    'description': 'Only Server 1, no auto-scaling',
                    'estimated_monthly_cost': 324,
                    'hours_server1': 720,
                    'hours_server2': 0,
                    'savings_vs_always_on': 547
                }
            },
            'optimization_tips': [
                "✅ Current setup saves ~$721/month vs always-on",
                "🎯 Auto-scaling activates only when Server 1 is stressed",
                "💰 Server 2 auto-stops after idle periods to save costs",
                "📊 Monitor scaling triggers to optimize thresholds",
                "⏰ Consider scheduled scaling for predictable workloads"
            ]
        }
        
        return cost_analysis
    except Exception as e:
        logger.error(f"Error getting cost analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost analysis: {str(e)}")
