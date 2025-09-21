#!/usr/bin/env python3
"""
Paperspace Auto-Scaler for Monte Carlo Platform
Automatically starts/stops secondary instances based on user load
"""

import asyncio
import aiohttp
import os
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PaperspaceInstance:
    machine_id: str
    name: str
    state: str  # 'ready', 'off', 'starting', 'stopping'
    hourly_cost: float
    max_users: int

@dataclass
class ScalingMetrics:
    current_users: int
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    response_time_ms: float
    queue_length: int

class PaperspaceAutoScaler:
    def __init__(self):
        self.api_key = os.getenv('PAPERSPACE_API_KEY')
        self.api_base = "https://api.paperspace.io"
        
        # Instance configuration
        self.primary_instance = PaperspaceInstance(
            machine_id=os.getenv('PRIMARY_MACHINE_ID'),
            name="monte-carlo-primary",
            state="ready",
            hourly_cost=0.51,  # P4000
            max_users=8
        )
        
        self.secondary_instance = PaperspaceInstance(
            machine_id=os.getenv('SECONDARY_MACHINE_ID'),
            name="monte-carlo-secondary", 
            state="off",
            hourly_cost=0.51,  # P4000
            max_users=8
        )
        
        # Scaling thresholds
        self.scale_up_threshold = 6    # Start secondary when primary has 6+ users
        self.scale_down_threshold = 4  # Stop secondary when total users < 4
        self.scale_up_delay = 300      # 5 minutes before scaling up
        self.scale_down_delay = 600    # 10 minutes before scaling down
        
        # State tracking
        self.last_scale_action = datetime.now()
        self.pending_scale_action = None
        
    async def get_current_metrics(self) -> ScalingMetrics:
        """Get current system metrics from your Monte Carlo platform"""
        try:
            async with aiohttp.ClientSession() as session:
                # Call your platform's metrics API
                async with session.get('http://localhost:8000/api/metrics') as response:
                    if response.status == 200:
                        data = await response.json()
                        return ScalingMetrics(
                            current_users=data.get('active_users', 0),
                            cpu_usage=data.get('cpu_usage', 0),
                            memory_usage=data.get('memory_usage', 0),
                            gpu_usage=data.get('gpu_usage', 0),
                            response_time_ms=data.get('avg_response_time', 0),
                            queue_length=data.get('simulation_queue_length', 0)
                        )
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            
        # Fallback metrics
        return ScalingMetrics(
            current_users=0,
            cpu_usage=0,
            memory_usage=0, 
            gpu_usage=0,
            response_time_ms=0,
            queue_length=0
        )
    
    async def get_machine_state(self, machine_id: str) -> str:
        """Get current state of a Paperspace machine"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'X-Api-Key': self.api_key}
                url = f"{self.api_base}/machines/{machine_id}"
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('state', 'unknown')
        except Exception as e:
            logger.error(f"Failed to get machine state: {e}")
            
        return 'unknown'
    
    async def start_machine(self, machine_id: str) -> bool:
        """Start a Paperspace machine"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'X-Api-Key': self.api_key}
                url = f"{self.api_base}/machines/{machine_id}/start"
                
                async with session.post(url, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Successfully started machine {machine_id}")
                        return True
                    else:
                        logger.error(f"Failed to start machine: {response.status}")
        except Exception as e:
            logger.error(f"Error starting machine: {e}")
            
        return False
    
    async def stop_machine(self, machine_id: str) -> bool:
        """Stop a Paperspace machine"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'X-Api-Key': self.api_key}
                url = f"{self.api_base}/machines/{machine_id}/stop"
                
                async with session.post(url, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Successfully stopped machine {machine_id}")
                        return True
                    else:
                        logger.error(f"Failed to stop machine: {response.status}")
        except Exception as e:
            logger.error(f"Error stopping machine: {e}")
            
        return False
    
    async def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale up"""
        conditions = [
            metrics.current_users >= self.scale_up_threshold,
            metrics.cpu_usage > 70,  # High CPU usage
            metrics.gpu_usage > 80,  # High GPU usage  
            metrics.queue_length > 5,  # Long simulation queue
            metrics.response_time_ms > 3000  # Slow response times
        ]
        
        # Scale up if any 2 conditions are met
        return sum(conditions) >= 2
    
    async def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale down"""
        conditions = [
            metrics.current_users <= self.scale_down_threshold,
            metrics.cpu_usage < 30,  # Low CPU usage
            metrics.gpu_usage < 40,  # Low GPU usage
            metrics.queue_length == 0,  # No queue
            metrics.response_time_ms < 1000  # Fast response times
        ]
        
        # Scale down only if ALL conditions are met (conservative)
        return all(conditions)
    
    async def execute_scaling_decision(self, action: str, metrics: ScalingMetrics):
        """Execute scaling up or down"""
        now = datetime.now()
        
        # Prevent rapid scaling (cooldown period)
        if (now - self.last_scale_action).seconds < 300:  # 5 minute cooldown
            logger.info(f"Scaling action '{action}' delayed due to cooldown")
            return
        
        if action == "scale_up":
            if self.secondary_instance.state == "off":
                logger.info(f"Scaling UP: Starting secondary instance (Users: {metrics.current_users})")
                success = await self.start_machine(self.secondary_instance.machine_id)
                if success:
                    self.secondary_instance.state = "starting"
                    self.last_scale_action = now
                    
                    # Update load balancer to include secondary instance
                    await self.update_load_balancer("add_instance")
        
        elif action == "scale_down":
            if self.secondary_instance.state == "ready":
                logger.info(f"Scaling DOWN: Stopping secondary instance (Users: {metrics.current_users})")
                
                # Remove from load balancer first
                await self.update_load_balancer("remove_instance")
                
                # Wait for current simulations to finish
                await asyncio.sleep(60)
                
                success = await self.stop_machine(self.secondary_instance.machine_id)
                if success:
                    self.secondary_instance.state = "stopping"
                    self.last_scale_action = now
    
    async def update_load_balancer(self, action: str):
        """Update load balancer configuration"""
        try:
            # This would update your nginx/haproxy config
            # or update service discovery
            logger.info(f"Load balancer action: {action}")
            
            if action == "add_instance":
                # Add secondary instance to load balancer
                # Implementation depends on your load balancer
                pass
            elif action == "remove_instance":
                # Remove secondary instance from load balancer
                pass
                
        except Exception as e:
            logger.error(f"Failed to update load balancer: {e}")
    
    async def monitor_and_scale(self):
        """Main monitoring loop"""
        logger.info("Starting Paperspace Auto-Scaler...")
        
        while True:
            try:
                # Get current metrics
                metrics = await self.get_current_metrics()
                
                # Update instance states
                self.secondary_instance.state = await self.get_machine_state(
                    self.secondary_instance.machine_id
                )
                
                logger.info(f"Metrics - Users: {metrics.current_users}, "
                          f"CPU: {metrics.cpu_usage}%, GPU: {metrics.gpu_usage}%, "
                          f"Queue: {metrics.queue_length}, "
                          f"Secondary: {self.secondary_instance.state}")
                
                # Make scaling decisions
                if await self.should_scale_up(metrics) and self.secondary_instance.state == "off":
                    await self.execute_scaling_decision("scale_up", metrics)
                    
                elif await self.should_scale_down(metrics) and self.secondary_instance.state == "ready":
                    await self.execute_scaling_decision("scale_down", metrics)
                
                # Calculate current hourly cost
                cost_per_hour = self.primary_instance.hourly_cost
                if self.secondary_instance.state == "ready":
                    cost_per_hour += self.secondary_instance.hourly_cost
                
                logger.info(f"Current hourly cost: ${cost_per_hour:.2f}")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Check every 2 minutes
            await asyncio.sleep(120)

# Configuration for environment variables
def create_env_template():
    """Create environment variables template"""
    env_content = """
# Paperspace Auto-Scaler Configuration
PAPERSPACE_API_KEY=your_paperspace_api_key_here
PRIMARY_MACHINE_ID=your_primary_machine_id
SECONDARY_MACHINE_ID=your_secondary_machine_id

# Optional: Custom thresholds
SCALE_UP_THRESHOLD=6
SCALE_DOWN_THRESHOLD=4
SCALE_UP_DELAY=300
SCALE_DOWN_DELAY=600
"""
    
    with open('.env.autoscaler', 'w') as f:
        f.write(env_content.strip())
    
    print("Created .env.autoscaler template")
    print("Please fill in your Paperspace API key and machine IDs")

if __name__ == "__main__":
    # Create environment template if it doesn't exist
    if not os.path.exists('.env.autoscaler'):
        create_env_template()
    
    # Load environment variables
    try:
        with open('.env.autoscaler', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        print("Please create .env.autoscaler with your configuration")
        exit(1)
    
    # Start the auto-scaler
    scaler = PaperspaceAutoScaler()
    asyncio.run(scaler.monitor_and_scale())







