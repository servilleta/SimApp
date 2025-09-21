#!/usr/bin/env python3
"""
🚀 Enhanced Dynamic Scaling System
Multi-Core Server 1 → Full Server 2 Auto-Scaling
"""

import asyncio
import time
import logging
import psutil
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import requests

from multicore_worker_config import multicore_config
from paperspace_api_manager import PaperspaceAPIManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedScalingManager:
    """
    Enhanced scaling system that:
    1. First maximizes Server 1 cores (8 cores)
    2. Then scales to Server 2 when Server 1 is fully utilized
    3. Monitors both servers and scales intelligently
    """
    
    def __init__(self):
        self.paperspace_api = self._init_paperspace_api()
        self.server_config = {
            'server_1': {
                'machine_id': 'psotdtcda5ap',
                'max_cores': 8,
                'cost_per_hour': 0.45,
                'ip_address': '64.71.146.187'
            },
            'server_2': {
                'machine_id': 'pso1zne8qfxx', 
                'max_cores': 8,
                'cost_per_hour': 0.76,
                'ip_address': '72.52.107.230'
            }
        }
        
        # Enhanced scaling thresholds
        self.scaling_thresholds = {
            # Server 1 Multi-Core Thresholds
            'server1_cpu_high': 70.0,      # Start considering Server 2
            'server1_cpu_critical': 85.0,  # Definitely start Server 2
            'server1_memory_high': 75.0,   # Memory pressure
            'server1_cores_utilized': 6,   # 6+ cores actively used
            
            # System-wide thresholds
            'api_response_time': 3000,     # 3 seconds max response time
            'queue_depth': 8,              # More than 8 simulations queued
            'load_average_threshold': 6.0, # Load average > 6 (out of 8 cores)
            
            # Scale-down thresholds
            'idle_time_minutes': 10,       # Server 2 idle for 10 minutes
            'low_utilization': 20.0,       # Combined utilization < 20%
        }
        
        self.scaling_state = {
            'server_2_running': False,
            'last_scale_up': None,
            'last_scale_down': None,
            'scale_up_count': 0,
            'scale_down_count': 0,
            'total_cost_saved': 0.0
        }
        
        logger.info("🚀 Enhanced Multi-Core + Multi-Server Scaling System Initialized")
        logger.info(f"💰 Server 1: ${self.server_config['server_1']['cost_per_hour']}/hr")
        logger.info(f"💰 Server 2: ${self.server_config['server_2']['cost_per_hour']}/hr")
    
    def _init_paperspace_api(self) -> Optional[PaperspaceAPIManager]:
        """Initialize Paperspace API if available"""
        try:
            api_key = os.getenv('PAPERSPACE_API_KEY')
            if not api_key:
                logger.warning("⚠️ PAPERSPACE_API_KEY not found - simulated scaling only")
                return None
            return PaperspaceAPIManager(api_key=api_key)
        except Exception as e:
            logger.warning(f"⚠️ Paperspace API unavailable: {e}")
            return None
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system metrics for scaling decisions
        """
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        load_avg = os.getloadavg()
        
        # Per-core metrics
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        cores_above_50 = sum(1 for core in cpu_per_core if core > 50.0)
        cores_above_80 = sum(1 for core in cpu_per_core if core > 80.0)
        
        # API performance
        api_response_time = self._check_api_performance()
        
        # Simulation queue (if available)
        queue_depth = self._estimate_queue_depth()
        
        # Paperspace server status
        server_2_status = self._get_server_2_status()
        
        return {
            'timestamp': datetime.now(timezone.utc),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'load_average_1min': load_avg[0],
                'load_average_5min': load_avg[1],
                'cores_above_50_percent': cores_above_50,
                'cores_above_80_percent': cores_above_80,
                'total_cores': len(cpu_per_core)
            },
            'performance': {
                'api_response_time_ms': api_response_time,
                'estimated_queue_depth': queue_depth,
            },
            'servers': {
                'server_1_running': True,  # We're on Server 1
                'server_2_running': server_2_status['running'],
                'server_2_last_activity': server_2_status.get('last_activity')
            },
            'scaling_state': self.scaling_state.copy()
        }
    
    def _check_api_performance(self) -> float:
        """Check API response time"""
        try:
            start_time = time.time()
            response = requests.get('http://localhost:8000/api/health', timeout=5)
            response_time = (time.time() - start_time) * 1000
            return response_time if response.status_code == 200 else 5000
        except Exception:
            return 5000  # Assume slow if unreachable
    
    def _estimate_queue_depth(self) -> int:
        """Estimate current simulation queue depth"""
        try:
            # This would integrate with your actual queue system
            # For now, estimate based on CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.5)
            if cpu_percent > 90:
                return 10  # High load = likely queue
            elif cpu_percent > 70:
                return 5
            elif cpu_percent > 50:
                return 2
            else:
                return 0
        except Exception:
            return 0
    
    def _get_server_2_status(self) -> Dict[str, Any]:
        """Get Server 2 status from Paperspace API"""
        if not self.paperspace_api:
            return {'running': False, 'last_activity': None}
        
        try:
            status = self.paperspace_api.get_machine_status(
                self.server_config['server_2']['machine_id']
            )
            return {
                'running': status.get('state') == 'ready',
                'last_activity': status.get('last_activity')
            }
        except Exception as e:
            logger.warning(f"⚠️ Could not get Server 2 status: {e}")
            return {'running': False, 'last_activity': None}
    
    def should_scale_up(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if Server 2 should be started
        
        Enhanced logic:
        1. Server 1 must be highly utilized (multiple cores busy)
        2. System performance is degraded
        3. Cost vs benefit analysis
        """
        reasons = []
        scale_needed = False
        confidence = 0.0
        
        system = metrics['system']
        performance = metrics['performance']
        
        # Check CPU utilization (multi-core awareness)
        if system['cpu_percent'] > self.scaling_thresholds['server1_cpu_critical']:
            reasons.append(f"Critical CPU usage: {system['cpu_percent']:.1f}%")
            scale_needed = True
            confidence += 0.4
        elif system['cpu_percent'] > self.scaling_thresholds['server1_cpu_high']:
            reasons.append(f"High CPU usage: {system['cpu_percent']:.1f}%")
            scale_needed = True
            confidence += 0.2
        
        # Check core utilization
        if system['cores_above_80_percent'] >= 4:
            reasons.append(f"Many cores highly utilized: {system['cores_above_80_percent']}/8 cores >80%")
            scale_needed = True
            confidence += 0.3
        elif system['cores_above_50_percent'] >= 6:
            reasons.append(f"Most cores active: {system['cores_above_50_percent']}/8 cores >50%")
            confidence += 0.2
        
        # Check memory pressure
        if system['memory_percent'] > self.scaling_thresholds['server1_memory_high']:
            reasons.append(f"High memory usage: {system['memory_percent']:.1f}%")
            scale_needed = True
            confidence += 0.2
        
        # Check load average
        if system['load_average_1min'] > self.scaling_thresholds['load_average_threshold']:
            reasons.append(f"High load average: {system['load_average_1min']:.2f}")
            scale_needed = True
            confidence += 0.3
        
        # Check API performance
        if performance['api_response_time_ms'] > self.scaling_thresholds['api_response_time']:
            reasons.append(f"Slow API response: {performance['api_response_time_ms']:.0f}ms")
            scale_needed = True
            confidence += 0.3
        
        # Check queue depth
        if performance['estimated_queue_depth'] > self.scaling_thresholds['queue_depth']:
            reasons.append(f"High queue depth: {performance['estimated_queue_depth']} simulations")
            scale_needed = True
            confidence += 0.4
        
        # Cost-benefit analysis
        current_cost = self.server_config['server_1']['cost_per_hour']
        scaled_cost = current_cost + self.server_config['server_2']['cost_per_hour']
        cost_increase = (scaled_cost - current_cost) / current_cost * 100
        
        # Estimate performance benefit
        performance_improvement = min(confidence * 100, 80)  # Cap at 80% improvement
        
        return {
            'scale_needed': scale_needed and confidence > 0.3,  # Require moderate confidence
            'confidence': confidence,
            'reasons': reasons,
            'cost_analysis': {
                'current_hourly': current_cost,
                'scaled_hourly': scaled_cost,
                'cost_increase_percent': cost_increase,
                'estimated_performance_gain': f"{performance_improvement:.1f}%"
            },
            'recommendation': self._get_scaling_recommendation(scale_needed, confidence, reasons)
        }
    
    def should_scale_down(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if Server 2 should be stopped
        """
        if not metrics['servers']['server_2_running']:
            return {'scale_down_needed': False, 'reason': 'Server 2 not running'}
        
        system = metrics['system']
        performance = metrics['performance']
        
        # Check for sustained low utilization
        low_cpu = system['cpu_percent'] < self.scaling_thresholds['low_utilization']
        low_cores = system['cores_above_50_percent'] < 2
        fast_api = performance['api_response_time_ms'] < 1000
        empty_queue = performance['estimated_queue_depth'] == 0
        
        if low_cpu and low_cores and fast_api and empty_queue:
            return {
                'scale_down_needed': True,
                'reason': 'Low utilization across all metrics',
                'cost_savings_per_hour': self.server_config['server_2']['cost_per_hour']
            }
        
        return {'scale_down_needed': False, 'reason': 'System still under load'}
    
    def _get_scaling_recommendation(self, scale_needed: bool, confidence: float, reasons: list) -> str:
        """Generate human-readable scaling recommendation"""
        if not scale_needed:
            return "OPTIMAL - Server 1 handling load efficiently"
        elif confidence > 0.7:
            return "SCALE_NOW - Strong indicators for Server 2 startup"
        elif confidence > 0.5:
            return "SCALE_RECOMMENDED - Multiple indicators suggest Server 2 needed"
        elif confidence > 0.3:
            return "SCALE_CONSIDER - Some indicators suggest Server 2 may help"
        else:
            return "MONITOR - Watch for sustained high usage"
    
    async def execute_scale_up(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Server 2 startup
        """
        if not self.paperspace_api:
            logger.info("🚧 SIMULATED: Would start Server 2 (Paperspace API not available)")
            return {'success': True, 'simulated': True}
        
        try:
            logger.info("🚀 Starting Server 2 for additional capacity...")
            
            result = self.paperspace_api.start_machine(
                self.server_config['server_2']['machine_id']
            )
            
            if result.get('success'):
                self.scaling_state['server_2_running'] = True
                self.scaling_state['last_scale_up'] = datetime.now(timezone.utc)
                self.scaling_state['scale_up_count'] += 1
                
                logger.info("✅ Server 2 startup initiated successfully")
                return {'success': True, 'message': 'Server 2 scaling up'}
            else:
                logger.error(f"❌ Failed to start Server 2: {result}")
                return {'success': False, 'error': result}
                
        except Exception as e:
            logger.error(f"❌ Error starting Server 2: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_scale_down(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Server 2 shutdown
        """
        if not self.paperspace_api:
            logger.info("🚧 SIMULATED: Would stop Server 2 (Paperspace API not available)")
            return {'success': True, 'simulated': True}
        
        try:
            logger.info("💤 Stopping Server 2 to save costs...")
            
            result = self.paperspace_api.stop_machine(
                self.server_config['server_2']['machine_id']
            )
            
            if result.get('success'):
                self.scaling_state['server_2_running'] = False
                self.scaling_state['last_scale_down'] = datetime.now(timezone.utc)
                self.scaling_state['scale_down_count'] += 1
                
                # Calculate cost savings
                if self.scaling_state['last_scale_up']:
                    uptime_hours = (datetime.now(timezone.utc) - self.scaling_state['last_scale_up']).total_seconds() / 3600
                    cost_for_session = uptime_hours * self.server_config['server_2']['cost_per_hour']
                    logger.info(f"💰 Server 2 session cost: ${cost_for_session:.2f}")
                
                logger.info("✅ Server 2 shutdown initiated successfully")
                return {'success': True, 'message': 'Server 2 scaling down'}
            else:
                logger.error(f"❌ Failed to stop Server 2: {result}")
                return {'success': False, 'error': result}
                
        except Exception as e:
            logger.error(f"❌ Error stopping Server 2: {e}")
            return {'success': False, 'error': str(e)}
    
    async def run_scaling_loop(self, check_interval_seconds: int = 60):
        """
        Main scaling loop
        """
        logger.info(f"🔄 Starting enhanced scaling loop (checking every {check_interval_seconds}s)")
        
        while True:
            try:
                # Get comprehensive metrics
                metrics = self.get_comprehensive_metrics()
                
                # Log current status
                system = metrics['system']
                logger.info(f"📊 Server 1: CPU {system['cpu_percent']:.1f}%, "
                          f"Memory {system['memory_percent']:.1f}%, "
                          f"Active cores: {system['cores_above_50_percent']}/8")
                
                # Check for scale up
                scale_up_analysis = self.should_scale_up(metrics)
                if scale_up_analysis['scale_needed'] and not metrics['servers']['server_2_running']:
                    logger.warning(f"🚨 Scale up triggered: {scale_up_analysis['reasons']}")
                    logger.info(f"💡 Confidence: {scale_up_analysis['confidence']:.2f}")
                    
                    await self.execute_scale_up(metrics)
                
                # Check for scale down
                scale_down_analysis = self.should_scale_down(metrics)
                if scale_down_analysis['scale_down_needed']:
                    logger.info(f"💤 Scale down triggered: {scale_down_analysis['reason']}")
                    
                    await self.execute_scale_down(metrics)
                
                # Log scaling recommendation
                if not scale_up_analysis['scale_needed']:
                    logger.info(f"✅ {scale_up_analysis['recommendation']}")
                
                await asyncio.sleep(check_interval_seconds)
                
            except Exception as e:
                logger.error(f"❌ Error in scaling loop: {e}")
                await asyncio.sleep(60)  # Longer sleep on error

async def main():
    """
    Run the enhanced scaling system
    """
    scaling_manager = EnhancedScalingManager()
    
    print("🚀 Enhanced Multi-Core + Multi-Server Scaling System")
    print("=" * 60)
    print("Features:")
    print("• Server 1: Multi-core utilization (up to 7 cores)")
    print("• Server 2: Auto-scaling when Server 1 is fully utilized")
    print("• Cost optimization: Scale down when idle")
    print("• Real-time performance monitoring")
    print()
    
    # Show current metrics
    metrics = scaling_manager.get_comprehensive_metrics()
    print("📊 Current System Status:")
    print(f"CPU Usage: {metrics['system']['cpu_percent']:.1f}%")
    print(f"Memory Usage: {metrics['system']['memory_percent']:.1f}%")
    print(f"Active Cores: {metrics['system']['cores_above_50_percent']}/8")
    print(f"Server 2 Running: {metrics['servers']['server_2_running']}")
    print()
    
    # Run scaling analysis
    scale_analysis = scaling_manager.should_scale_up(metrics)
    print("🎯 Scaling Analysis:")
    print(f"Recommendation: {scale_analysis['recommendation']}")
    print(f"Scale Needed: {scale_analysis['scale_needed']}")
    print(f"Confidence: {scale_analysis['confidence']:.2f}")
    if scale_analysis['reasons']:
        print(f"Reasons: {', '.join(scale_analysis['reasons'])}")
    print()
    
    # Start the scaling loop
    print("🔄 Starting continuous monitoring...")
    await scaling_manager.run_scaling_loop(check_interval_seconds=30)

if __name__ == "__main__":
    asyncio.run(main())
