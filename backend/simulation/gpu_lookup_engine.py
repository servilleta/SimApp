"""
🚀 GPU LOOKUP ENGINE
===================
High-performance GPU-accelerated lookup operations for Excel functions
like VLOOKUP, INDEX, MATCH with vectorized processing.

Features:
- Vectorized VLOOKUP operations
- Binary search algorithms on GPU
- Memory-efficient batch processing
- Parallel lookup execution

Author: World-Class AI Assistant
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from gpu.manager import GPUManager
from super_engine.gpu_kernels import gpu_vlookup_exact, is_gpu_available

logger = logging.getLogger(__name__)

class GPULookupEngine:
    """
    🔍 GPU-ACCELERATED LOOKUP ENGINE
    
    Provides high-performance lookup operations using GPU vectorization.
    """
    
    def __init__(self):
        self.gpu_manager = GPUManager()
        logger.info("🔍 GPU Lookup Engine initialized")
    
    async def vectorized_vlookup(self, lookup_values: np.ndarray, 
                               lookup_table: np.ndarray, 
                               col_index: int,
                               approximate_match: bool = False) -> np.ndarray:
        """
        Execute vectorized VLOOKUP operation on GPU.
        
        Args:
            lookup_values: Values to look up
            lookup_table: Table to search in (2D array)
            col_index: Column index to return values from
            approximate_match: Whether to use approximate matching
            
        Returns:
            Array of looked up values
        """
        try:
            if self.gpu_manager.is_available() and is_gpu_available():
                return await self._gpu_vlookup(lookup_values, lookup_table, col_index, approximate_match)
            else:
                return self._cpu_vlookup(lookup_values, lookup_table, col_index, approximate_match)
        except Exception as e:
            logger.warning(f"GPU VLOOKUP failed, falling back to CPU: {e}")
            return self._cpu_vlookup(lookup_values, lookup_table, col_index, approximate_match)
    
    async def _gpu_vlookup(self, lookup_values: np.ndarray, 
                          lookup_table: np.ndarray, 
                          col_index: int,
                          approximate_match: bool) -> np.ndarray:
        """GPU-accelerated VLOOKUP implementation."""
        # Fallback to CPU if string/object dtype is detected
        if lookup_values.dtype.char == 'O' or lookup_table.dtype.char == 'O':
            logger.warning("_gpu_vlookup: String/object dtype detected, falling back to CPU for VLOOKUP.")
            return self._cpu_vlookup(lookup_values, lookup_table, col_index, approximate_match)
        if approximate_match:
            logger.warning("GPU VLOOKUP approximate match not implemented, falling back to CPU.")
            return self._cpu_vlookup(lookup_values, lookup_table, col_index, True)
        logger.info(f"🚀 Performing exact VLOOKUP on GPU for {len(lookup_values)} values.")
        import cupy as cp
        # Move data to GPU
        lookup_values_gpu = cp.asarray(lookup_values)
        lookup_table_gpu = cp.asarray(lookup_table)
        # Call the SuperEngine kernel
        result_gpu = gpu_vlookup_exact(lookup_values_gpu, lookup_table_gpu, col_index)
        # Move result back to CPU
        return cp.asnumpy(result_gpu)
    
    def _cpu_vlookup(self, lookup_values: np.ndarray, 
                     lookup_table: np.ndarray, 
                     col_index: int,
                     approximate_match: bool) -> np.ndarray:
        """Optimized CPU VLOOKUP implementation."""
        results = []
        
        for value in lookup_values:
            try:
                if approximate_match:
                    # Find closest match
                    differences = np.abs(lookup_table[:, 0] - value)
                    best_idx = np.argmin(differences)
                    result = lookup_table[best_idx, col_index]
                else:
                    # Exact match
                    matches = np.where(lookup_table[:, 0] == value)[0]
                    if len(matches) > 0:
                        result = lookup_table[matches[0], col_index]
                    else:
                        result = np.nan
                
                results.append(result)
            except Exception as e:
                logger.warning(f"VLOOKUP error for value {value}: {e}")
                results.append(np.nan)
        
        return np.array(results)
    
    async def batch_lookup(self, operations: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Execute multiple lookup operations in batch.
        
        Args:
            operations: List of lookup operation definitions
            
        Returns:
            Dictionary of operation results
        """
        results = {}
        
        for i, operation in enumerate(operations):
            op_type = operation.get('type', 'vlookup')
            op_id = operation.get('id', f'lookup_{i}')
            
            try:
                if op_type == 'vlookup':
                    result = await self.vectorized_vlookup(
                        operation['lookup_values'],
                        operation['lookup_table'],
                        operation['col_index'],
                        operation.get('approximate_match', False)
                    )
                    results[op_id] = result
                else:
                    logger.warning(f"Unsupported lookup operation: {op_type}")
                    results[op_id] = np.array([])
            except Exception as e:
                logger.error(f"Batch lookup failed for operation {op_id}: {e}")
                results[op_id] = np.array([])
        
        return results
