#!/usr/bin/env python3
"""
Development Mode Scaling Manager for SimApp
==========================================

Perfect for development mode where:
- Both servers start OFF (save money during development)
- When you start Server 1 for work, Server 2 auto-scales based on load
- Automatic shutdown when both servers are idle

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
    format='%(asctime)s - 🔧 DEV-SCALING - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DevScalingMetrics:
    """Development-focused scaling metrics"""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    active_simulations: int
    api_response_time: float
    server1_running: bool
    server2_running: bool

class DevelopmentScalingManager:
    """
    Development-optimized scaling manager
    - Monitors Server 1 stress levels
    - Auto-starts Server 2 when Server 1 is overwhelmed
    - Auto-stops both servers during idle periods
    """
    
    def __init__(self):
        self.api = PaperspaceAPIManager()
        self.server1_id = "psotdtcda5ap"  # P4000 Primary
        self.server2_id = "pso1zne8qfxx"  # A4000 Performance
        
        self.server1_running = False
        self.server2_running = False
        self.last_scale_action = None
        
        # Development-specific settings
        self.cooldown_minutes = 5          # Faster scaling in dev
        self.idle_shutdown_minutes = 30    # Auto-shutdown after 30min idle
        self.check_interval = 30           # Check every 30 seconds
        
        # Development scaling thresholds (more aggressive)
        self.stress_thresholds = {
            'cpu_usage': 70.0,          # CPU > 70% (lower for dev)
            'memory_usage': 75.0,       # Memory > 75%
            'disk_io': 80.0,           # High disk I/O
            'api_response_time': 3000,  # Response > 3 seconds
            'active_simulations': 3     # More than 3 concurrent sims
        }
        
        self.idle_thresholds = {
            'cpu_usage': 20.0,          # CPU < 20%
            'memory_usage': 30.0,       # Memory < 30%
            'api_response_time': 1000,  # Response < 1 second
            'active_simulations': 0     # No active simulations
        }
        
        self.last_activity_time = datetime.now()
    
    def get_server_status(self):
        """Check current status of both servers"""
        try:
            machines = self.api.list_machines()
            for machine in machines:
                if machine.get('id') == self.server1_id:
                    self.server1_running = machine.get('state') == 'ready'
                elif machine.get('id') == self.server2_id:
                    self.server2_running = machine.get('state') == 'ready'
        except Exception as e:
            logger.error(f"Error checking server status: {e}")
    
    def get_dev_metrics(self) -> DevScalingMetrics:
        """Get development-focused metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # Disk I/O metrics
            disk_io = 0
            try:
                disk_stats = psutil.disk_io_counters()
                if disk_stats:
                    # Simplified disk usage metric
                    disk_io = (disk_stats.read_bytes + disk_stats.write_bytes) / (1024**3)  # GB
            except:
                pass
            
            # SimApp-specific metrics
            api_response_time = self.check_api_response_time()
            active_simulations = self.estimate_active_simulations(cpu_usage)
            
            return DevScalingMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_io=disk_io,
                active_simulations=active_simulations,
                api_response_time=api_response_time,
                server1_running=self.server1_running,
                server2_running=self.server2_running
            )
        except Exception as e:
            logger.error(f"Error gathering dev metrics: {e}")
            return DevScalingMetrics(0, 0, 0, 0, 1000, False, False)
    
    def check_api_response_time(self) -> float:
        """Check SimApp API response time"""
        try:
            start_time = time.time()
            response = requests.get('http://localhost:8000/api/health', timeout=10)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                return response_time
            else:
                return 5000  # High response time for errors
        except:
            return 10000  # Very high for API unavailable
    
    def estimate_active_simulations(self, cpu_usage: float) -> int:
        """Estimate active simulations based on CPU usage"""
        if cpu_usage < 20:
            return 0
        elif cpu_usage < 40:
            return 1
        elif cpu_usage < 60:
            return 2
        elif cpu_usage < 80:
            return 3
        else:
            return max(3, int(cpu_usage / 20))
    
    def is_server1_stressed(self, metrics: DevScalingMetrics) -> bool:
        """Determine if Server 1 is under stress"""
        if not metrics.server1_running:
            return False
        
        stress_conditions = [
            metrics.cpu_usage > self.stress_thresholds['cpu_usage'],
            metrics.memory_usage > self.stress_thresholds['memory_usage'],
            metrics.api_response_time > self.stress_thresholds['api_response_time'],
            metrics.active_simulations > self.stress_thresholds['active_simulations']
        ]
        
        # Server 1 is stressed if 2+ conditions are met
        stressed = sum(stress_conditions) >= 2
        
        if stressed:
            logger.warning(f"🚨 SERVER 1 STRESSED: CPU:{metrics.cpu_usage:.1f}% "
                         f"Memory:{metrics.memory_usage:.1f}% "
                         f"Response:{metrics.api_response_time:.0f}ms "
                         f"Sims:{metrics.active_simulations}")
        
        return stressed
    
    def is_system_idle(self, metrics: DevScalingMetrics) -> bool:
        """Determine if the entire system is idle"""
        idle_conditions = [
            metrics.cpu_usage < self.idle_thresholds['cpu_usage'],
            metrics.memory_usage < self.idle_thresholds['memory_usage'],
            metrics.api_response_time < self.idle_thresholds['api_response_time'],
            metrics.active_simulations <= self.idle_thresholds['active_simulations']
        ]
        
        return all(idle_conditions)
    
    def in_cooldown(self) -> bool:
        """Check if we're in cooldown period"""
        if not self.last_scale_action:
            return False
        
        cooldown_end = self.last_scale_action + timedelta(minutes=self.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def scale_up_server2(self) -> bool:
        """Start Server 2 to help with load"""
        if self.server2_running or self.in_cooldown():
            return False
        
        try:
            logger.info("🚀 DEV SCALING UP: Starting Server 2 (A4000) to help Server 1")
            result = self.api.start_machine(self.server2_id)
            
            if result:
                self.server2_running = True
                self.last_scale_action = datetime.now()
                self.last_activity_time = datetime.now()
                logger.info("✅ Server 2 started - High performance mode active")
                return True
            else:
                logger.error("❌ Failed to start Server 2")
                return False
        except Exception as e:
            logger.error(f"❌ Error starting Server 2: {e}")
            return False
    
    def scale_down_server2(self) -> bool:
        """Stop Server 2 when no longer needed"""
        if not self.server2_running or self.in_cooldown():
            return False
        
        try:
            logger.info("⬇️ DEV SCALING DOWN: Stopping Server 2 (load reduced)")
            result = self.api.stop_machine(self.server2_id)
            
            if result:
                self.server2_running = False
                self.last_scale_action = datetime.now()
                logger.info("✅ Server 2 stopped - Back to single server mode")
                return True
            else:
                logger.error("❌ Failed to stop Server 2")
                return False
        except Exception as e:
            logger.error(f"❌ Error stopping Server 2: {e}")
            return False
    
    def auto_shutdown_idle_system(self, metrics: DevScalingMetrics):
        """Auto-shutdown both servers if idle for too long"""
        if self.is_system_idle(metrics):
            time_since_activity = datetime.now() - self.last_activity_time
            
            if time_since_activity.total_seconds() > (self.idle_shutdown_minutes * 60):
                logger.info(f"💤 SYSTEM IDLE for {self.idle_shutdown_minutes} minutes - Auto-shutdown initiated")
                
                # Stop Server 2 first
                if self.server2_running:
                    logger.info("⬇️ Stopping Server 2 (idle)")
                    self.api.stop_machine(self.server2_id)
                
                # Note: In development, you might want to stop Server 1 too
                # Uncomment the next lines if you want full auto-shutdown
                # if self.server1_running:
                #     logger.info("⬇️ Stopping Server 1 (idle)")
                #     self.api.stop_machine(self.server1_id)
                
                logger.info("💰 Development session complete - Servers optimized for cost savings")
        else:
            # Reset activity timer if system is active
            self.last_activity_time = datetime.now()
    
    def run_dev_cycle(self):
        """Run one development monitoring cycle"""
        # Update server status
        self.get_server_status()
        
        # Get current metrics
        metrics = self.get_dev_metrics()
        
        # Log status
        status_emoji = "🟢" if metrics.server1_running else "🔴"
        server2_emoji = "🔥" if metrics.server2_running else "💤"
        
        logger.info(f"📊 DEV STATUS: Server1:{status_emoji} Server2:{server2_emoji} "
                   f"CPU:{metrics.cpu_usage:.1f}% Memory:{metrics.memory_usage:.1f}% "
                   f"API:{metrics.api_response_time:.0f}ms Sims:{metrics.active_simulations}")
        
        # Development scaling decisions
        if metrics.server1_running and not metrics.server2_running:
            # Server 1 is running, check if it needs help
            if self.is_server1_stressed(metrics):
                logger.info("🚨 Server 1 is stressed - Scaling up Server 2")
                self.scale_up_server2()
        
        elif metrics.server1_running and metrics.server2_running:
            # Both running, check if we can scale down
            if not self.is_server1_stressed(metrics):
                logger.info("✅ Load reduced - Considering Server 2 scale-down")
                time.sleep(60)  # Wait 1 minute to confirm load reduction
                
                # Re-check after waiting
                new_metrics = self.get_dev_metrics()
                if not self.is_server1_stressed(new_metrics):
                    self.scale_down_server2()
        
        # Auto-shutdown logic for development
        self.auto_shutdown_idle_system(metrics)
    
    def run_development_monitoring(self):
        """Run continuous development monitoring"""
        logger.info("🔧 Starting DEVELOPMENT scaling monitor")
        logger.info("📋 Development Mode Features:")
        logger.info("   • Server 2 auto-starts when Server 1 is stressed")
        logger.info("   • Server 2 auto-stops when load reduces")
        logger.info("   • System auto-optimizes for cost during idle periods")
        logger.info("   • Faster scaling (5min cooldown vs 15min production)")
        logger.info(f"   • Checking every {self.check_interval} seconds")
        
        try:
            while True:
                self.run_dev_cycle()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("⏹️ Development monitoring stopped by user")
            
            # Optionally stop Server 2 on exit to save costs
            if self.server2_running:
                logger.info("💰 Stopping Server 2 to save development costs...")
                self.api.stop_machine(self.server2_id)
        except Exception as e:
            logger.error(f"❌ Development monitoring error: {e}")

def main():
    """Main function for development scaling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SimApp Development Scaling Manager')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Monitoring interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true', 
                       help='Run once instead of continuous monitoring')
    parser.add_argument('--idle-shutdown', type=int, default=30,
                       help='Auto-shutdown after N minutes idle (default: 30)')
    
    args = parser.parse_args()
    
    manager = DevelopmentScalingManager()
    manager.idle_shutdown_minutes = args.idle_shutdown
    manager.check_interval = args.interval
    
    if args.once:
        manager.run_dev_cycle()
    else:
        manager.run_development_monitoring()

if __name__ == "__main__":
    main()
