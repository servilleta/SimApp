#!/usr/bin/env python3
"""
🚀 Enterprise Microservices Startup Script
Phase 2 Week 5: Complete Microservices Decomposition

This script starts all microservices for the enterprise platform.
"""

import asyncio
import subprocess
import sys
import time
import logging
import signal
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class MicroserviceManager:
    """Manage multiple microservices"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = True
        
        # Define microservices configuration
        self.services = {
            "api-gateway": {
                "module": "microservices.gateway.api_gateway",
                "port": 8080,
                "description": "Enterprise API Gateway"
            },
            "main-service": {
                "module": "main:app",
                "port": 8000,
                "description": "Main Simulation Service"
            }
            # Additional services will be added as they're implemented
        }
    
    async def start_all_services(self):
        """Start all microservices"""
        logger.info("🚀 [MICROSERVICES] Starting Enterprise Microservices Platform...")
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start each service
        for service_name, config in self.services.items():
            await self._start_service(service_name, config)
            time.sleep(2)  # Small delay between service starts
        
        logger.info("✅ [MICROSERVICES] All services started successfully")
        
        # Monitor services
        await self._monitor_services()
    
    async def _start_service(self, service_name: str, config: Dict):
        """Start a single microservice"""
        try:
            cmd = [
                sys.executable, "-m", "uvicorn",
                config["module"],
                "--host", "0.0.0.0",
                "--port", str(config["port"]),
                "--reload",
                "--log-level", "info"
            ]
            
            logger.info(f"🚀 [MICROSERVICES] Starting {service_name}: {config['description']}")
            logger.info(f"🔧 [MICROSERVICES] Command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[service_name] = process
            logger.info(f"✅ [MICROSERVICES] {service_name} started on port {config['port']}")
            
        except Exception as e:
            logger.error(f"❌ [MICROSERVICES] Failed to start {service_name}: {e}")
    
    async def _monitor_services(self):
        """Monitor all running services"""
        logger.info("👁️ [MICROSERVICES] Monitoring services (Ctrl+C to stop)...")
        
        while self.running:
            try:
                # Check if any service has died
                for service_name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.error(f"❌ [MICROSERVICES] {service_name} has stopped unexpectedly")
                        # Could implement auto-restart here
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ [MICROSERVICES] Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 [MICROSERVICES] Received signal {signum}, shutting down...")
        self.running = False
        asyncio.create_task(self._shutdown_all_services())
    
    async def _shutdown_all_services(self):
        """Gracefully shutdown all services"""
        logger.info("🔄 [MICROSERVICES] Shutting down all services...")
        
        for service_name, process in self.processes.items():
            try:
                logger.info(f"🔄 [MICROSERVICES] Stopping {service_name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    logger.info(f"✅ [MICROSERVICES] {service_name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ [MICROSERVICES] Force killing {service_name}...")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                logger.error(f"❌ [MICROSERVICES] Error stopping {service_name}: {e}")
        
        logger.info("✅ [MICROSERVICES] All services stopped")
    
    def print_service_status(self):
        """Print status of all services"""
        print("\n" + "="*60)
        print("🏢 ENTERPRISE MICROSERVICES STATUS")
        print("="*60)
        
        for service_name, config in self.services.items():
            process = self.processes.get(service_name)
            if process and process.poll() is None:
                status = "🟢 RUNNING"
            else:
                status = "🔴 STOPPED"
            
            print(f"{status} {service_name:<15} | Port {config['port']:<4} | {config['description']}")
        
        print("="*60)
        print("📊 Access services:")
        for service_name, config in self.services.items():
            print(f"   {service_name}: http://localhost:{config['port']}")
        print("="*60 + "\n")

async def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Change to backend directory
    backend_dir = Path(__file__).parent.parent
    import os
    os.chdir(backend_dir)
    
    manager = MicroserviceManager()
    
    try:
        # Print initial status
        manager.print_service_status()
        
        # Start all services
        await manager.start_all_services()
        
    except KeyboardInterrupt:
        logger.info("👋 [MICROSERVICES] Shutting down...")
    except Exception as e:
        logger.error(f"❌ [MICROSERVICES] Fatal error: {e}")
    finally:
        await manager._shutdown_all_services()

if __name__ == "__main__":
    asyncio.run(main())