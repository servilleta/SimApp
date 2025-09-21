"""
🎛️ PROGRESS SMOOTHING UTILITY
Provides smoother progress updates by interpolating between values and preventing large jumps.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ProgressSmoother:
    """Smooths progress updates to avoid jarring jumps from 0% to 100%"""
    
    def __init__(self, max_jump: float = 5.0, min_interval: float = 0.1):
        """
        Args:
            max_jump: Maximum allowed progress jump in percentage points
            min_interval: Minimum time between progress updates in seconds
        """
        self.max_jump = max_jump
        self.min_interval = min_interval
        self.last_progress = {}  # simulation_id -> (progress, timestamp)
        
    def smooth_progress(self, simulation_id: str, new_progress: float, 
                       progress_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Smooth a progress update to avoid large jumps.
        
        Returns:
            Dict with smoothed progress data, or None if update should be skipped
        """
        current_time = time.time()
        
        # Get last progress for this simulation
        if simulation_id in self.last_progress:
            last_progress, last_time = self.last_progress[simulation_id]
            
            # Check if enough time has passed
            if current_time - last_time < self.min_interval:
                return None  # Skip this update
            
            # Check for large jumps
            progress_jump = new_progress - last_progress
            if progress_jump > self.max_jump:
                # Limit the jump to max_jump
                smoothed_progress = last_progress + self.max_jump
                logger.debug(f"[ProgressSmoother] Large jump detected for {simulation_id}: "
                           f"{last_progress:.1f}% -> {new_progress:.1f}%, limiting to {smoothed_progress:.1f}%")
            else:
                smoothed_progress = new_progress
        else:
            # First update for this simulation
            smoothed_progress = new_progress
        
        # Store the smoothed progress
        self.last_progress[simulation_id] = (smoothed_progress, current_time)
        
        # Update progress data with smoothed value
        smoothed_data = progress_data.copy()
        smoothed_data["progress_percentage"] = smoothed_progress
        
        return smoothed_data
    
    def force_progress(self, simulation_id: str, progress: float) -> None:
        """Force set progress without smoothing (for completion, etc.)"""
        self.last_progress[simulation_id] = (progress, time.time())
    
    def reset_simulation(self, simulation_id: str) -> None:
        """Reset progress tracking for a simulation"""
        if simulation_id in self.last_progress:
            del self.last_progress[simulation_id]
    
    def cleanup_old_simulations(self, max_age: float = 3600) -> None:
        """Remove old simulation progress data"""
        current_time = time.time()
        old_sims = []
        
        for sim_id, (progress, timestamp) in self.last_progress.items():
            if current_time - timestamp > max_age:
                old_sims.append(sim_id)
        
        for sim_id in old_sims:
            del self.last_progress[sim_id]
        
        if old_sims:
            logger.debug(f"[ProgressSmoother] Cleaned up {len(old_sims)} old simulations")


# Global progress smoother instance - FIXED: More permissive settings for Ultra engine
_progress_smoother = ProgressSmoother(max_jump=25.0, min_interval=0.05)

def smooth_simulation_progress(simulation_id: str, progress_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Smooth progress update for a simulation.
    
    Returns smoothed progress data or None if update should be skipped.
    """
    if "progress_percentage" not in progress_data:
        return progress_data
    
    # 🎯 CRITICAL FIX: Be more permissive for simulation stage - allow 1-second updates
    stage = progress_data.get("stage", "")
    if stage == "simulation":
        # Use a more permissive smoother for simulation stage
        simulation_smoother = ProgressSmoother(max_jump=30.0, min_interval=1.0)  # 1 second for simulation
        return simulation_smoother.smooth_progress(
            simulation_id, 
            progress_data["progress_percentage"],
            progress_data
        )
    
    return _progress_smoother.smooth_progress(
        simulation_id, 
        progress_data["progress_percentage"],
        progress_data
    )

def force_simulation_progress(simulation_id: str, progress: float) -> None:
    """Force set progress without smoothing"""
    _progress_smoother.force_progress(simulation_id, progress)

def reset_simulation_progress(simulation_id: str) -> None:
    """Reset progress tracking for a simulation"""
    _progress_smoother.reset_simulation(simulation_id) 