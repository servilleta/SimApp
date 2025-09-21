#!/usr/bin/env python3
"""
Intelligent Scaling Monitor for SimApp
=====================================

Automatically manages Server 2 based on workload demands.
Monitors simulation queue, user load, and system resources.

Author: SimApp DevOps Team
Date: September 21, 2025
"""

import os
import sys
import time
import json
import logging
import requests
import psutil
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

# Add the app root to the path
sys.path.append('/home/paperspace/SimApp')
from paperspace_api_manager import PaperspaceAPIManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScalingMetrics:
    """System metrics for scaling decisions"""
    cpu_usage: float
    memory_usage: float
    active_simulations: int
    queued_simulations: int
    concurrent_users: int
    response_time_ms: float

class IntelligentScalingManager:
    """
    Manages automatic scaling of Server 2 based on intelligent metrics
    """
    
    def __init__(self):
        self.api = PaperspaceAPIManager()
        self.server2_id = "pso1zne8qfxx"
        self.server2_running = False
        self.last_scale_action = None
        self.cooldown_minutes = 15  # Prevent rapid scaling
        
        # Scaling thresholds
        self.scale_up_thresholds = {
            'cpu_usage': 80.0,           # CPU > 80%
            'memory_usage': 85.0,        # Memory > 85%
            'active_simulations': 5,     # More than 5 concurrent simulations
            'queued_simulations': 3,     # More than 3 queued
            'concurrent_users': 10,      # More than 10 users
            'response_time_ms': 5000     # Response time > 5 seconds
        }
        
        self.scale_down_thresholds = {
            'cpu_usage': 30.0,           # CPU < 30%
            'memory_usage': 40.0,        # Memory < 40%
            'active_simulations': 2,     # Less than 2 simulations
            'queued_simulations': 0,     # No queue
            'concurrent_users': 3,       # Less than 3 users
            'response_time_ms': 2000     # Response time < 2 seconds
        }
    
    def get_system_metrics(self) -> ScalingMetrics:
        """Gather current system metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # SimApp metrics (mock for now - integrate with your actual API)
            simapp_metrics = self.get_simapp_metrics()
            
            return ScalingMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                active_simulations=simapp_metrics.get('active_simulations', 0),
                queued_simulations=simapp_metrics.get('queued_simulations', 0),
                concurrent_users=simapp_metrics.get('concurrent_users', 0),
                response_time_ms=simapp_metrics.get('response_time_ms', 1000)
            )
        except Exception as e:
            logger.error(f"Error gathering metrics: {e}")
            return ScalingMetrics(0, 0, 0, 0, 0, 1000)
    
    def get_simapp_metrics(self) -> Dict:
        """Get metrics from SimApp API"""
        try:
            # Try to get metrics from local SimApp API
            response = requests.get('http://localhost:8000/api/metrics', timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        
        # Fallback: estimate based on system load
        return {
            'active_simulations': max(0, int((psutil.cpu_percent() - 20) / 15)),
            'queued_simulations': 0,
            'concurrent_users': max(1, int(psutil.cpu_percent() / 20)),
            'response_time_ms': 1000 + (psutil.cpu_percent() * 50)
        }
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale up to Server 2"""
        if self.server2_running:
            return False
        
        if self.in_cooldown():
            return False
        
        # Check if any threshold is exceeded
        conditions = [
            metrics.cpu_usage > self.scale_up_thresholds['cpu_usage'],
            metrics.memory_usage > self.scale_up_thresholds['memory_usage'],
            metrics.active_simulations > self.scale_up_thresholds['active_simulations'],
            metrics.queued_simulations > self.scale_up_thresholds['queued_simulations'],
            metrics.concurrent_users > self.scale_up_thresholds['concurrent_users'],
            metrics.response_time_ms > self.scale_up_thresholds['response_time_ms']
        ]
        
        # Scale up if at least 2 conditions are met
        return sum(conditions) >= 2
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale down Server 2"""
        if not self.server2_running:
            return False
        
        if self.in_cooldown():
            return False
        
        # Check if all thresholds are below scale-down limits
        conditions = [
            metrics.cpu_usage < self.scale_down_thresholds['cpu_usage'],
            metrics.memory_usage < self.scale_down_thresholds['memory_usage'],
            metrics.active_simulations < self.scale_down_thresholds['active_simulations'],
            metrics.queued_simulations <= self.scale_down_thresholds['queued_simulations'],
            metrics.concurrent_users < self.scale_down_thresholds['concurrent_users'],
            metrics.response_time_ms < self.scale_down_thresholds['response_time_ms']
        ]
        
        # Scale down only if ALL conditions are met
        return all(conditions)
    
    def in_cooldown(self) -> bool:
        """Check if we're in cooldown period"""
        if not self.last_scale_action:
            return False
        
        cooldown_end = self.last_scale_action + timedelta(minutes=self.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def scale_up(self) -> bool:
        """Scale up by starting Server 2"""
        try:
            logger.info("🚀 SCALING UP: Starting Server 2 (A4000 GPU)")
            result = self.api.start_machine(self.server2_id)
            
            if result:
                self.server2_running = True
                self.last_scale_action = datetime.now()
                logger.info("✅ Server 2 started successfully")
                return True
            else:
                logger.error("❌ Failed to start Server 2")
                return False
        except Exception as e:
            logger.error(f"❌ Error scaling up: {e}")
            return False
    
    def scale_down(self) -> bool:
        """Scale down by stopping Server 2"""
        try:
            logger.info("⬇️ SCALING DOWN: Stopping Server 2")
            result = self.api.stop_machine(self.server2_id)
            
            if result:
                self.server2_running = False
                self.last_scale_action = datetime.now()
                logger.info("✅ Server 2 stopped successfully")
                return True
            else:
                logger.error("❌ Failed to stop Server 2")
                return False
        except Exception as e:
            logger.error(f"❌ Error scaling down: {e}")
            return False
    
    def update_server2_status(self):
        """Update current Server 2 status"""
        try:
            machines = self.api.list_machines()
            for machine in machines:
                if machine.get('id') == self.server2_id:
                    self.server2_running = machine.get('state') == 'ready'
                    break
        except Exception as e:
            logger.error(f"Error updating Server 2 status: {e}")
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        # Update current status
        self.update_server2_status()
        
        # Get current metrics
        metrics = self.get_system_metrics()
        
        # Log current status
        logger.info(f"📊 Metrics: CPU:{metrics.cpu_usage:.1f}% Memory:{metrics.memory_usage:.1f}% "
                   f"Sims:{metrics.active_simulations} Queue:{metrics.queued_simulations} "
                   f"Users:{metrics.concurrent_users} Response:{metrics.response_time_ms:.0f}ms")
        
        logger.info(f"🖥️ Server 2 Status: {'🟢 RUNNING' if self.server2_running else '🔴 STOPPED'}")
        
        # Make scaling decisions
        if self.should_scale_up(metrics):
            logger.info("🚀 TRIGGER: Scaling up conditions met")
            self.scale_up()
        elif self.should_scale_down(metrics):
            logger.info("⬇️ TRIGGER: Scaling down conditions met")
            self.scale_down()
        else:
            logger.info("✅ No scaling action needed")
    
    def run_continuous_monitoring(self, interval_seconds: int = 60):
        """Run continuous monitoring"""
        logger.info(f"🔄 Starting intelligent scaling monitor (checking every {interval_seconds}s)")
        
        try:
            while True:
                self.run_monitoring_cycle()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("⏹️ Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SimApp Intelligent Scaling Manager')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Monitoring interval in seconds (default: 60)')
    parser.add_argument('--once', action='store_true', 
                       help='Run once instead of continuous monitoring')
    
    args = parser.parse_args()
    
    manager = IntelligentScalingManager()
    
    if args.once:
        manager.run_monitoring_cycle()
    else:
        manager.run_continuous_monitoring(args.interval)

if __name__ == "__main__":
    main()
