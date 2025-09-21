#!/usr/bin/env python3
"""
🚀 Multi-Core Worker Configuration for SimApp Backend
Dynamic CPU utilization and intelligent scaling system.
"""

import os
import psutil
import multiprocessing
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MultiCoreConfig:
    """
    Dynamic configuration for optimal CPU utilization
    """
    
    def __init__(self):
        self.total_cores = multiprocessing.cpu_count()
        self.recommended_workers = self._calculate_optimal_workers()
        self.scaling_thresholds = {
            'cpu_high': 70.0,      # When to consider Server 2
            'cpu_critical': 85.0,   # When to definitely scale
            'memory_high': 75.0,    # Memory threshold
            'queue_depth': 10       # Simulation queue depth threshold
        }
    
    def _calculate_optimal_workers(self) -> int:
        """
        Calculate optimal number of worker processes
        
        Rules:
        - For CPU-intensive tasks: cores
        - Leave 1 core for system/other processes
        - Maximum 8 workers (diminishing returns beyond this)
        """
        # Reserve 1 core for system
        optimal = max(1, self.total_cores - 1)
        
        # Cap at 8 for optimal performance
        optimal = min(optimal, 8)
        
        logger.info(f"💻 Detected {self.total_cores} cores, recommending {optimal} workers")
        return optimal
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """
        Get uvicorn configuration for optimal multi-core usage
        """
        return {
            'workers': self.recommended_workers,
            'worker_class': 'uvicorn.workers.UvicornWorker',
            'worker_connections': 1000,
            'max_requests': 1000,  # Restart workers after 1000 requests (memory cleanup)
            'max_requests_jitter': 100,
            'preload_app': True,   # Share code across workers
            'timeout': 300,        # 5 minutes for long simulations
            'keepalive': 2,
            'bind': '0.0.0.0:8000'
        }
    
    def get_current_system_load(self) -> Dict[str, float]:
        """
        Get current system resource utilization
        """
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'load_average_1min': os.getloadavg()[0],
            'load_average_5min': os.getloadavg()[1],
            'active_cores': min(os.getloadavg()[0], self.total_cores)
        }
    
    def should_scale_to_server2(self, current_load: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Determine if Server 2 should be started based on current load
        """
        if current_load is None:
            current_load = self.get_current_system_load()
        
        reasons = []
        scale_needed = False
        
        # CPU threshold check
        if current_load['cpu_percent'] > self.scaling_thresholds['cpu_critical']:
            reasons.append(f"Critical CPU usage: {current_load['cpu_percent']:.1f}%")
            scale_needed = True
        elif current_load['cpu_percent'] > self.scaling_thresholds['cpu_high']:
            reasons.append(f"High CPU usage: {current_load['cpu_percent']:.1f}%")
            scale_needed = True
        
        # Memory threshold check
        if current_load['memory_percent'] > self.scaling_thresholds['memory_high']:
            reasons.append(f"High memory usage: {current_load['memory_percent']:.1f}%")
            scale_needed = True
        
        # Load average check (cores fully utilized)
        if current_load['load_average_1min'] > (self.total_cores * 0.8):
            reasons.append(f"High load average: {current_load['load_average_1min']:.2f}")
            scale_needed = True
        
        return {
            'scale_needed': scale_needed,
            'reasons': reasons,
            'current_load': current_load,
            'estimated_benefit': self._estimate_scaling_benefit(current_load)
        }
    
    def _estimate_scaling_benefit(self, load: Dict[str, float]) -> Dict[str, str]:
        """
        Estimate performance benefit of scaling to Server 2
        """
        current_efficiency = min(load['cpu_percent'] / 100, 1.0)
        
        # With Server 2: distribute load across 16 cores instead of 8
        potential_speedup = min(2.0, 1 / (current_efficiency * 0.5))
        
        return {
            'current_efficiency': f"{current_efficiency * 100:.1f}%",
            'potential_speedup': f"{potential_speedup:.1f}x",
            'estimated_time_savings': f"{((potential_speedup - 1) / potential_speedup) * 100:.1f}%"
        }

# Global configuration instance
multicore_config = MultiCoreConfig()

def get_optimal_worker_count() -> int:
    """
    Get the optimal number of workers for current system
    """
    return multicore_config.recommended_workers

def get_scaling_recommendation() -> Dict[str, Any]:
    """
    Get current scaling recommendation
    """
    return multicore_config.should_scale_to_server2()

if __name__ == "__main__":
    # Test the configuration
    config = MultiCoreConfig()
    print("🚀 Multi-Core Configuration Analysis")
    print("=" * 50)
    print(f"Total CPU cores: {config.total_cores}")
    print(f"Recommended workers: {config.recommended_workers}")
    print(f"Uvicorn config: {config.get_uvicorn_config()}")
    print()
    
    load = config.get_current_system_load()
    print("📊 Current System Load:")
    for key, value in load.items():
        print(f"  {key}: {value}")
    print()
    
    scaling = config.should_scale_to_server2(load)
    print("🎯 Scaling Recommendation:")
    print(f"  Scale needed: {scaling['scale_needed']}")
    print(f"  Reasons: {scaling['reasons']}")
    print(f"  Benefits: {scaling['estimated_benefit']}")
