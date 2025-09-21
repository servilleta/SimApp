"""
🚀 PROCESS SIMULATION: Ultra-robust solution using multiprocessing
This completely eliminates event loop conflicts by running simulations in separate processes
"""

import multiprocessing
import asyncio
import logging
import os
import sys
from typing import Dict, Any

logger = logging.getLogger(__name__)

def run_simulation_in_process(simulation_data: Dict[str, Any]):
    """
    Run simulation in a completely separate process
    This eliminates ALL event loop and Redis connection issues
    """
    try:
        # Import inside process to avoid import issues
        import asyncio
        from simulation.service import run_multi_target_simulation_task
        from shared.progress_store import reset_redis_connection_async
        
        # Create fresh event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def process_simulation():
            try:
                # Reset Redis connections for this process
                await reset_redis_connection_async()
                
                # Run the simulation
                result = await run_multi_target_simulation_task(simulation_data)
                
                logger.info(f"✅ [PROCESS] Simulation completed: {simulation_data.get('simulation_id')}")
                return result
                
            except Exception as e:
                logger.error(f"❌ [PROCESS] Simulation failed: {e}")
                raise
        
        # Run simulation in process event loop
        return loop.run_until_complete(process_simulation())
        
    except Exception as e:
        logger.error(f"🚨 [PROCESS] Process simulation failed: {e}")
        return None
    finally:
        try:
            loop.close()
        except:
            pass


class ProcessSimulationManager:
    """Manages simulation processes"""
    
    def __init__(self):
        self.active_processes = {}
    
    def start_simulation_process(self, simulation_data: Dict[str, Any]) -> str:
        """
        Start simulation in a separate process
        Returns immediately with simulation ID
        """
        simulation_id = simulation_data.get('simulation_id')
        
        try:
            # Start process
            process = multiprocessing.Process(
                target=run_simulation_in_process,
                args=(simulation_data,),
                name=f"sim_{simulation_id}"
            )
            
            process.start()
            self.active_processes[simulation_id] = process
            
            logger.info(f"🚀 [PROCESS_MANAGER] Started simulation process: {simulation_id}")
            return simulation_id
            
        except Exception as e:
            logger.error(f"❌ [PROCESS_MANAGER] Failed to start process: {e}")
            raise
    
    def cleanup_completed_processes(self):
        """Clean up completed processes"""
        completed = []
        for sim_id, process in self.active_processes.items():
            if not process.is_alive():
                process.join(timeout=1)
                completed.append(sim_id)
        
        for sim_id in completed:
            del self.active_processes[sim_id]
            logger.info(f"🧹 [PROCESS_MANAGER] Cleaned up completed process: {sim_id}")

# Global manager
_process_manager = ProcessSimulationManager()

def start_simulation_process(simulation_data: Dict[str, Any]) -> str:
    """Start simulation in separate process"""
    return _process_manager.start_simulation_process(simulation_data)
