"""
GPU Kernels for Power Monte Carlo Engine

Provides optimized GPU kernels for common Excel operations.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CuPy not available - GPU kernels disabled")


class PowerEngineGPUKernels:
    """GPU kernel manager for Power engine"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.compiled_kernels = {}
        
        if self.gpu_available:
            self._compile_standard_kernels()
    
    def _compile_standard_kernels(self):
        """Compile standard GPU kernels"""
        try:
            # Sum reduction kernel
            self.sum_kernel = cp.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a + b',
                'y = a',
                '0',
                'power_sum'
            )
            
            # Element-wise operations kernel
            self.elementwise_add = cp.ElementwiseKernel(
                'float64 x, float64 y',
                'float64 z',
                'z = x + y',
                'power_add'
            )
            
            self.elementwise_mul = cp.ElementwiseKernel(
                'float64 x, float64 y',
                'float64 z',
                'z = x * y',
                'power_mul'
            )
            
            logger.info("✅ Power engine GPU kernels compiled successfully")
            
        except Exception as e:
            logger.error(f"Failed to compile GPU kernels: {e}")
            self.gpu_available = False
    
    def sum_range(self, values: np.ndarray) -> float:
        """GPU-accelerated sum of values"""
        if not self.gpu_available:
            return np.sum(values)
        
        try:
            gpu_values = cp.asarray(values)
            result = self.sum_kernel(gpu_values)
            return float(result)
        except Exception as e:
            logger.warning(f"GPU sum failed, falling back to CPU: {e}")
            return np.sum(values)
    
    def batch_arithmetic(self, a: np.ndarray, b: np.ndarray, operation: str) -> np.ndarray:
        """GPU-accelerated batch arithmetic operations"""
        if not self.gpu_available:
            if operation == 'add':
                return a + b
            elif operation == 'mul':
                return a * b
            else:
                return a
        
        try:
            gpu_a = cp.asarray(a)
            gpu_b = cp.asarray(b)
            
            if operation == 'add':
                gpu_result = self.elementwise_add(gpu_a, gpu_b)
            elif operation == 'mul':
                gpu_result = self.elementwise_mul(gpu_a, gpu_b)
            else:
                gpu_result = gpu_a
            
            return cp.asnumpy(gpu_result)
            
        except Exception as e:
            logger.warning(f"GPU batch arithmetic failed, falling back to CPU: {e}")
            if operation == 'add':
                return a + b
            elif operation == 'mul':
                return a * b
            else:
                return a 