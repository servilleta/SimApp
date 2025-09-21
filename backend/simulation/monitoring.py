import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class ProgressMonitor:
    """Monitors and validates simulation progress updates"""
    
    def __init__(self):
        self.progress_history = defaultdict(list)  # simulation_id -> List[Dict]
        self.phase_timings = defaultdict(dict)     # simulation_id -> Dict[phase, timing]
        self.performance_metrics = defaultdict(dict)  # simulation_id -> Dict[metric, value]
        self.alerts = defaultdict(list)            # simulation_id -> List[str]
        
        # Thresholds for monitoring
        self.thresholds = {
            'max_progress_jump': 10.0,        # Maximum allowed progress jump (%)
            'min_update_interval': 0.1,       # Minimum time between updates (seconds)
            'max_update_interval': 5.0,       # Maximum time between updates (seconds)
            'max_phase_duration': 300.0,      # Maximum duration for any phase (seconds)
            'max_total_duration': 1800.0,     # Maximum total simulation duration (seconds)
            'min_progress_rate': 0.1,         # Minimum progress rate (%/second)
        }
    
    def record_progress(self, simulation_id: str, progress_data: Dict[str, Any]):
        """Record a progress update and perform validation"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in progress_data:
                progress_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Get previous progress
            prev_progress = self.progress_history[simulation_id][-1] if self.progress_history[simulation_id] else None
            
            # Record the update
            self.progress_history[simulation_id].append(progress_data)
            
            # Update phase timing
            current_phase = progress_data.get('phase')
            if current_phase:
                if current_phase not in self.phase_timings[simulation_id]:
                    self.phase_timings[simulation_id][current_phase] = {
                        'start_time': time.time(),
                        'updates': 0
                    }
                self.phase_timings[simulation_id][current_phase]['updates'] += 1
            
            # Validate the update
            self._validate_progress_update(simulation_id, progress_data, prev_progress)
            
            # Update performance metrics
            self._update_performance_metrics(simulation_id, progress_data, prev_progress)
            
        except Exception as e:
            logger.error(f"Error recording progress for {simulation_id}: {e}")
            self.alerts[simulation_id].append(f"Progress recording error: {e}")
    
    def _validate_progress_update(self, simulation_id: str, current: Dict[str, Any], previous: Optional[Dict[str, Any]]):
        """Validate a progress update against thresholds"""
        try:
            if not previous:
                return
            
            # Check progress jump
            curr_progress = current.get('progress_percentage', 0)
            prev_progress = previous.get('progress_percentage', 0)
            progress_jump = curr_progress - prev_progress
            
            if progress_jump > self.thresholds['max_progress_jump']:
                self.alerts[simulation_id].append(
                    f"Large progress jump detected: {progress_jump:.1f}% "
                    f"(from {prev_progress:.1f}% to {curr_progress:.1f}%)"
                )
            
            # Check update interval
            curr_time = datetime.fromisoformat(current['timestamp'].replace('Z', '+00:00'))
            prev_time = datetime.fromisoformat(previous['timestamp'].replace('Z', '+00:00'))
            update_interval = (curr_time - prev_time).total_seconds()
            
            if update_interval < self.thresholds['min_update_interval']:
                self.alerts[simulation_id].append(
                    f"Update interval too short: {update_interval:.2f}s "
                    f"(minimum: {self.thresholds['min_update_interval']}s)"
                )
            elif update_interval > self.thresholds['max_update_interval']:
                self.alerts[simulation_id].append(
                    f"Update interval too long: {update_interval:.2f}s "
                    f"(maximum: {self.thresholds['max_update_interval']}s)"
                )
            
            # Check phase duration
            current_phase = current.get('phase')
            if current_phase and current_phase in self.phase_timings[simulation_id]:
                phase_start = self.phase_timings[simulation_id][current_phase]['start_time']
                phase_duration = time.time() - phase_start
                
                if phase_duration > self.thresholds['max_phase_duration']:
                    self.alerts[simulation_id].append(
                        f"Phase {current_phase} duration exceeded: {phase_duration:.1f}s "
                        f"(maximum: {self.thresholds['max_phase_duration']}s)"
                    )
            
            # Check total duration
            if len(self.progress_history[simulation_id]) > 1:
                first_update = self.progress_history[simulation_id][0]
                first_time = datetime.fromisoformat(first_update['timestamp'].replace('Z', '+00:00'))
                total_duration = (curr_time - first_time).total_seconds()
                
                if total_duration > self.thresholds['max_total_duration']:
                    self.alerts[simulation_id].append(
                        f"Total duration exceeded: {total_duration:.1f}s "
                        f"(maximum: {self.thresholds['max_total_duration']}s)"
                    )
            
            # Check progress rate
            if update_interval > 0:
                progress_rate = progress_jump / update_interval
                if progress_rate < self.thresholds['min_progress_rate'] and curr_progress < 100:
                    self.alerts[simulation_id].append(
                        f"Progress rate too low: {progress_rate:.2f}%/s "
                        f"(minimum: {self.thresholds['min_progress_rate']}%/s)"
                    )
            
        except Exception as e:
            logger.error(f"Error validating progress for {simulation_id}: {e}")
            self.alerts[simulation_id].append(f"Progress validation error: {e}")
    
    def _update_performance_metrics(self, simulation_id: str, current: Dict[str, Any], previous: Optional[Dict[str, Any]]):
        """Update performance metrics based on progress updates"""
        try:
            metrics = self.performance_metrics[simulation_id]
            
            # Update basic metrics
            metrics['total_updates'] = len(self.progress_history[simulation_id])
            metrics['current_phase'] = current.get('phase', 'unknown')
            metrics['current_progress'] = current.get('progress_percentage', 0)
            
            if previous:
                # Calculate update rate
                curr_time = datetime.fromisoformat(current['timestamp'].replace('Z', '+00:00'))
                prev_time = datetime.fromisoformat(previous['timestamp'].replace('Z', '+00:00'))
                update_interval = (curr_time - prev_time).total_seconds()
                
                if 'avg_update_interval' not in metrics:
                    metrics['avg_update_interval'] = update_interval
                else:
                    metrics['avg_update_interval'] = (
                        metrics['avg_update_interval'] * 0.9 + update_interval * 0.1
                    )
                
                # Calculate progress rate
                progress_change = current.get('progress_percentage', 0) - previous.get('progress_percentage', 0)
                if update_interval > 0:
                    progress_rate = progress_change / update_interval
                    if 'avg_progress_rate' not in metrics:
                        metrics['avg_progress_rate'] = progress_rate
                    else:
                        metrics['avg_progress_rate'] = (
                            metrics['avg_progress_rate'] * 0.9 + progress_rate * 0.1
                        )
            
            # Update phase metrics
            current_phase = current.get('phase')
            if current_phase:
                phase_timing = self.phase_timings[simulation_id][current_phase]
                phase_duration = time.time() - phase_timing['start_time']
                metrics[f'{current_phase}_duration'] = phase_duration
                metrics[f'{current_phase}_updates'] = phase_timing['updates']
            
        except Exception as e:
            logger.error(f"Error updating metrics for {simulation_id}: {e}")
            self.alerts[simulation_id].append(f"Metrics update error: {e}")
    
    def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get current status and metrics for a simulation"""
        try:
            if simulation_id not in self.progress_history:
                return {"error": "Simulation not found"}
            
            history = self.progress_history[simulation_id]
            if not history:
                return {"error": "No progress updates recorded"}
            
            current = history[-1]
            metrics = self.performance_metrics[simulation_id]
            alerts = self.alerts[simulation_id]
            
            return {
                "current_status": current,
                "performance_metrics": metrics,
                "alerts": alerts,
                "total_updates": len(history),
                "phase_timings": self.phase_timings[simulation_id]
            }
            
        except Exception as e:
            logger.error(f"Error getting status for {simulation_id}: {e}")
            return {"error": str(e)}
    
    def cleanup_simulation(self, simulation_id: str):
        """Clean up monitoring data for a completed simulation"""
        try:
            self.progress_history.pop(simulation_id, None)
            self.phase_timings.pop(simulation_id, None)
            self.performance_metrics.pop(simulation_id, None)
            self.alerts.pop(simulation_id, None)
        except Exception as e:
            logger.error(f"Error cleaning up monitoring data for {simulation_id}: {e}")

# Create a global progress monitor instance
progress_monitor = ProgressMonitor() 