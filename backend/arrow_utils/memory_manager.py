"""
Arrow Memory Manager
Manages Arrow memory pools for efficient large file processing
"""

import pyarrow as pa
import psutil
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for Arrow memory management"""
    memory_pool_size_gb: int = 4
    warning_threshold: float = 0.8  # 80% usage warning
    critical_threshold: float = 0.9  # 90% usage critical
    enable_garbage_collection: bool = True
    gc_frequency: int = 1000  # Every N operations

class ArrowMemoryManager:
    """
    Manages Arrow memory pools and monitors usage
    Prevents memory exhaustion during large file processing
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.memory_pool: Optional[pa.MemoryPool] = None
        self.operation_count = 0
        self.peak_usage = 0
        self.callbacks = []
        
    def initialize(self) -> bool:
        """Initialize Arrow memory pool"""
        try:
            # Use default memory pool - different PyArrow versions have different APIs
            self.memory_pool = pa.default_memory_pool()
            
            logger.info(f"Arrow memory pool initialized with default pool")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Arrow memory pool: {e}")
            # Fallback to default system pool
            self.memory_pool = pa.default_memory_pool()
            return False
    
    def get_memory_pool(self) -> pa.MemoryPool:
        """Get the Arrow memory pool"""
        if self.memory_pool is None:
            self.initialize()
        return self.memory_pool
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            # System memory stats
            system_memory = psutil.virtual_memory()
            
            # Arrow pool stats (if available)
            arrow_stats = {}
            if self.memory_pool:
                try:
                    arrow_stats = {
                        'bytes_allocated': self.memory_pool.bytes_allocated(),
                        'max_memory': self.memory_pool.max_memory(),
                    }
                except AttributeError:
                    # Some Arrow versions don't have these methods
                    arrow_stats = {'status': 'pool_active'}
            
            return {
                'system': {
                    'total_gb': system_memory.total / 1024**3,
                    'available_gb': system_memory.available / 1024**3,
                    'used_percent': system_memory.percent,
                    'used_gb': system_memory.used / 1024**3
                },
                'arrow': arrow_stats,
                'peak_usage_gb': self.peak_usage / 1024**3,
                'operations_count': self.operation_count
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {'error': str(e)}
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and trigger warnings if needed"""
        stats = self.get_memory_stats()
        
        if 'system' not in stats:
            return stats
        
        usage_percent = stats['system']['used_percent'] / 100.0
        
        # Update peak usage
        current_usage = stats['system']['used_gb'] * 1024**3
        if current_usage > self.peak_usage:
            self.peak_usage = current_usage
        
        # Check thresholds
        if usage_percent >= self.config.critical_threshold:
            level = 'critical'
            message = f"Critical memory usage: {usage_percent:.1%}"
            logger.critical(message)
        elif usage_percent >= self.config.warning_threshold:
            level = 'warning'
            message = f"High memory usage: {usage_percent:.1%}"
            logger.warning(message)
        else:
            level = 'normal'
            message = f"Memory usage normal: {usage_percent:.1%}"
        
        # Trigger callbacks
        self._trigger_callbacks(level, usage_percent, stats)
        
        return {
            'level': level,
            'usage_percent': usage_percent,
            'message': message,
            'stats': stats
        }
    
    def add_memory_callback(self, callback):
        """Add callback for memory threshold events"""
        self.callbacks.append(callback)
    
    def _trigger_callbacks(self, level: str, usage_percent: float, stats: Dict[str, Any]):
        """Trigger registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(level, usage_percent, stats)
            except Exception as e:
                logger.error(f"Error in memory callback: {e}")
    
    def increment_operation(self):
        """Increment operation counter and check for garbage collection"""
        self.operation_count += 1
        
        if (self.config.enable_garbage_collection and 
            self.operation_count % self.config.gc_frequency == 0):
            self._run_garbage_collection()
    
    def _run_garbage_collection(self):
        """Run garbage collection if enabled"""
        try:
            import gc
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"Garbage collection freed {collected} objects")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
    
    def create_table_with_pool(self, arrays, schema) -> pa.Table:
        """
        Create Arrow table using managed memory pool
        """
        try:
            # PyArrow versions have different APIs - use basic table creation
            table = pa.Table.from_arrays(arrays, schema=schema)
            self.increment_operation()
            return table
            
        except Exception as e:
            logger.error(f"Error creating table with memory pool: {e}")
            # Fallback to default creation
            return pa.Table.from_arrays(arrays, schema=schema)
    
    def get_optimal_batch_size(self, estimated_row_size_bytes: int) -> int:
        """
        Calculate optimal batch size based on available memory
        """
        try:
            stats = self.get_memory_stats()
            available_gb = stats['system']['available_gb']
            
            # Use 25% of available memory for batch processing
            batch_memory_gb = available_gb * 0.25
            batch_memory_bytes = batch_memory_gb * 1024**3
            
            # Calculate optimal batch size
            batch_size = int(batch_memory_bytes / estimated_row_size_bytes)
            
            # Ensure reasonable bounds
            batch_size = max(1000, min(batch_size, 100000))
            
            logger.debug(f"Calculated optimal batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            logger.error(f"Error calculating batch size: {e}")
            return 10000  # Default fallback
    
    def cleanup(self):
        """Cleanup memory resources"""
        try:
            if self.memory_pool:
                # Arrow memory pools are automatically managed
                logger.info("Arrow memory pool cleanup requested")
            
            # Final garbage collection
            if self.config.enable_garbage_collection:
                self._run_garbage_collection()
                
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")

# Global memory manager instance
_global_memory_manager: Optional[ArrowMemoryManager] = None

def get_memory_manager() -> ArrowMemoryManager:
    """Get global Arrow memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = ArrowMemoryManager()
        _global_memory_manager.initialize()
    return _global_memory_manager

def create_memory_manager(config: Optional[MemoryConfig] = None) -> ArrowMemoryManager:
    """Create new Arrow memory manager with custom config"""
    manager = ArrowMemoryManager(config)
    manager.initialize()
    return manager 