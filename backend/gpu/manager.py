import asyncio
import pynvml
from typing import Optional, Callable, Any, Dict, Tuple # Added Callable, Any, Dict, Tuple
from config import settings # Changed to absolute import
import logging # Added
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__) # Added

@dataclass
class MemoryPlan:
    """Memory allocation plan for GPU operations"""
    total_memory_mb: int
    allocated_memory_mb: int
    available_memory_mb: int
    pools: Dict[str, int]  # pool_name -> size_mb
    estimated_usage: Dict[str, int]  # operation -> memory_mb

@dataclass
class ExcelFileStats:
    """Statistics about Excel file for memory estimation"""
    num_cells: int
    num_formulas: int
    num_sheets: int
    file_size_mb: float
    max_range_size: int
    formula_complexity: float

class WorkloadType(Enum):
    """Types of GPU workloads"""
    MONTE_CARLO = "monte_carlo"
    FORECASTING = "forecasting"
    MIXED = "mixed"

class GPUManager:
    _instance: Optional["GPUManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, memory_fraction: float = settings.GPU_MEMORY_FRACTION):
        # Ensure __init__ is idempotent for the singleton pattern
        if not hasattr(self, 'initialized'): # Check if already initialized
            self.memory_fraction = memory_fraction
            self.initialized = False
            self.gpu_available = False # Tracks if a GPU is actually usable
            self.semaphore: Optional[asyncio.Semaphore] = None
            self.max_concurrent_tasks = 1 # Default to 1 (CPU-like behavior if no GPU)
            self.device_count = 0
            
            # Enhanced memory management
            self.memory_pools = {}  # Pre-allocated memory pools
            self.tensor_cache = {}  # Cached GPU tensors
            self.total_memory_mb = 0
            self.available_memory_mb = 0
            
            # Forecasting readiness
            self.dl_frameworks = {}  # PyTorch, TensorFlow devices
            self.model_cache = {}    # Cached trained models
            
            print("GPUManager instance created.")

    async def initialize(self):
        """Initialize NVIDIA Management Library and set up resources. Idempotent."""
        if self.initialized:
            # print("GPUManager already initialized.")
            return
        
        print("Initializing GPUManager...")
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            
            if self.device_count == 0:
                logger.info("No GPU devices found. Running in CPU-only mode.") # Changed print to logger.info
                self.gpu_available = False
                self.max_concurrent_tasks = settings.MAX_CPU_FALLBACK_TASKS if hasattr(settings, 'MAX_CPU_FALLBACK_TASKS') else 1 # Configurable
            else:
                print(f"Found {self.device_count} GPU device(s).")
                # Use the first GPU for simplicity, as in the original guide
                # In a multi-GPU setup, more sophisticated allocation would be needed.
                handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming at least one GPU
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                self.total_memory_mb = mem_info.total / (1024 * 1024)
                # Using configured fraction of *total* memory, not free. This is a common strategy.
                allocatable_memory_mb = self.total_memory_mb * self.memory_fraction
                self.available_memory_mb = allocatable_memory_mb
                
                # Estimate memory required per simulation task (adjust based on your workload)
                memory_per_task_mb = settings.GPU_MEMORY_PER_TASK_MB
                
                if memory_per_task_mb <= 0:
                    print("Warning: memory_per_task_mb is not configured properly. Defaulting to 1 task.")
                    self.max_concurrent_tasks = 1
                else:
                    self.max_concurrent_tasks = max(1, int(allocatable_memory_mb / memory_per_task_mb))
                
                # Initialize memory pools
                self._create_memory_pools()
                
                self.gpu_available = True
                logger.info(f"GPU initialized. Total Memory: {self.total_memory_mb:.2f}MB, Usable: {allocatable_memory_mb:.2f}MB, Tasks: {self.max_concurrent_tasks}") # Changed print to logger.info
            
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            self.initialized = True
            logger.info(f"GPUManager initialization complete. GPU Available: {self.gpu_available}, Max Concurrent Tasks: {self.max_concurrent_tasks}")
            
        except pynvml.NVMLError_LibraryNotFound:
            logger.warning("NVIDIA NVML library not found. GPU support disabled. Running in CPU-only mode.") # Changed print to logger.warning
            self.gpu_available = False
            self.max_concurrent_tasks = settings.MAX_CPU_FALLBACK_TASKS if hasattr(settings, 'MAX_CPU_FALLBACK_TASKS') else 1
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks) # Still need semaphore for CPU fallback task limiting
            self.initialized = True # Considered initialized, but with no GPU
        except Exception as e:
            logger.error(f"Failed to initialize GPU: {str(e)}. Running in CPU-only mode.", exc_info=True) # Changed print to logger.error and added exc_info
            self.gpu_available = False
            self.max_concurrent_tasks = settings.MAX_CPU_FALLBACK_TASKS if hasattr(settings, 'MAX_CPU_FALLBACK_TASKS') else 1
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            self.initialized = True
    
    def _create_memory_pools(self):
        """Create specialized memory pools for different data types - SUPERFAST OPTIMIZED"""
        try:
            import cupy as cp
            
            # Calculate pool sizes based on available memory
            pool_memory = self.available_memory_mb * 0.8  # Reserve 20% for overhead
            
            # SUPERFAST: Dynamic pool sizing based on workload optimization
            dynamic_pool_sizes = {
                'variables': pool_memory * 0.4,     # 40% for random variables (main workload)
                'constants': pool_memory * 0.1,     # 10% for constants (small but frequent)
                'results': pool_memory * 0.3,       # 30% for results (large arrays)
                'lookup_tables': pool_memory * 0.15, # 15% for VLOOKUP (medium usage)
                'forecasting': pool_memory * 0.05   # 5% reserved for future features
            }
            
            self.memory_pools = {}
            
            # Create pools with optimized sizes
            for pool_name, size_mb in dynamic_pool_sizes.items():
                pool = cp.cuda.MemoryPool()
                pool.set_limit(size=int(size_mb * 1024 * 1024))  # Convert to bytes
                self.memory_pools[pool_name] = pool
                logger.info(f"🚀 Created optimized memory pool '{pool_name}': {size_mb:.1f}MB ({size_mb/pool_memory*100:.1f}%)")
                
            logger.info(f"✅ SUPERFAST GPU Memory Pools initialized: {pool_memory:.1f}MB total across {len(self.memory_pools)} specialized pools")
            
        except Exception as e:
            logger.warning(f"Failed to create memory pools: {e}")
            self.memory_pools = {}
    
    async def allocate_simulation_memory(self, 
                                       iterations: int, 
                                       num_variables: int,
                                       formula_complexity: int) -> MemoryPlan:
        """Pre-calculate and reserve GPU memory for simulation"""
        
        # Estimate memory requirements
        variable_memory_mb = (iterations * num_variables * 4) / (1024 * 1024)  # 4 bytes per float32
        formula_memory_mb = formula_complexity * 100  # Rough estimate
        result_memory_mb = (iterations * 4) / (1024 * 1024)  # Results array
        
        total_estimated_mb = variable_memory_mb + formula_memory_mb + result_memory_mb
        
        if total_estimated_mb > self.available_memory_mb:
            raise RuntimeError(f"Insufficient GPU memory: need {total_estimated_mb:.1f}MB, have {self.available_memory_mb:.1f}MB")
        
        return MemoryPlan(
            total_memory_mb=int(self.total_memory_mb),
            allocated_memory_mb=int(total_estimated_mb),
            available_memory_mb=int(self.available_memory_mb - total_estimated_mb),
            pools={
                'variables': int(variable_memory_mb),
                'formulas': int(formula_memory_mb),
                'results': int(result_memory_mb)
            },
            estimated_usage={
                'random_generation': int(variable_memory_mb),
                'formula_evaluation': int(formula_memory_mb),
                'result_storage': int(result_memory_mb)
            }
        )
        
    def estimate_memory_requirements(self, excel_stats: ExcelFileStats) -> int:
        """Estimate VRAM needed based on file size and complexity"""
        
        # Base memory for cell data
        cell_memory_mb = (excel_stats.num_cells * 16) / (1024 * 1024)  # 16 bytes per cell estimate
        
        # Formula processing memory
        formula_memory_mb = excel_stats.num_formulas * 0.1  # 100KB per formula estimate
        
        # Lookup table memory
        lookup_memory_mb = excel_stats.max_range_size * 0.01  # 10KB per lookup cell
        
        # Complexity multiplier
        complexity_multiplier = 1.0 + (excel_stats.formula_complexity * 0.5)
        
        total_mb = (cell_memory_mb + formula_memory_mb + lookup_memory_mb) * complexity_multiplier
        
        return int(total_mb)
        
    def create_memory_pool(self, pool_name: str, size_mb: int):
        """Create dedicated memory pool for specific operations"""
        try:
            import cupy as cp
            
            if pool_name not in self.memory_pools:
                pool = cp.cuda.MemoryPool()
                pool.set_limit(size=size_mb * 1024 * 1024)
                self.memory_pools[pool_name] = pool
                logger.info(f"Created memory pool '{pool_name}' with {size_mb}MB limit")
            else:
                logger.warning(f"Memory pool '{pool_name}' already exists")
                
        except Exception as e:
            logger.error(f"Failed to create memory pool '{pool_name}': {e}")

    def is_gpu_available(self) -> bool:
        """Check if a GPU is initialized and available for use."""
        return self.initialized and self.gpu_available

    async def run_task(self, task_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Run an async task with GPU resource management (semaphore).
        Ensures initialization before running.
        """
        if not self.initialized:
            await self.initialize() # Ensure initialized
        
        if not self.semaphore: # Should not happen if initialize was called
            print("Error: GPUManager semaphore not initialized. Task cannot run.")
            raise RuntimeError("GPUManager semaphore not initialized.")

        # print(f"Task {task_func.__name__} waiting for semaphore ({self.semaphore._value}/{self.max_concurrent_tasks})...")
        async with self.semaphore:
            # print(f"Task {task_func.__name__} acquired semaphore. Running...")
            try:
                result = await task_func(*args, **kwargs)
                # print(f"Task {task_func.__name__} completed.")
                return result
            finally:
                # print(f"Task {task_func.__name__} released semaphore.")
                pass # Semaphore is released automatically by async with
    
    def shutdown(self):
        """Clean up GPU resources. Should be called on application shutdown."""
        if self.initialized and self.device_count > 0: # Only shutdown nvml if it was used with devices
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown successful.") # Added logger info
            except pynvml.NVMLError as e:
                logger.error(f"Error shutting down NVML: {str(e)}", exc_info=True) # Added logger error with exc_info
            except Exception as e:
                logger.error(f"An unexpected error occurred during NVML shutdown: {str(e)}", exc_info=True) # Added logger error with exc_info
        self.initialized = False
        self.gpu_available = False
        logger.info("GPUManager shutdown complete.") # Added logger info

# Create a singleton instance of the GPUManager
# The application should call await gpu_manager.initialize() at startup 
# and gpu_manager.shutdown() at shutdown if GPU is intended to be used.

gpu_manager = GPUManager()

# Example of how to add shutdown to FastAPI app (in main.py):
# @app.on_event("startup")
# async def startup_event():
#     if settings.USE_GPU:
#         await gpu_manager.initialize()

# @app.on_event("shutdown")
# def shutdown_event():
#     if settings.USE_GPU: # Or just always call it, it's safe
#         gpu_manager.shutdown() 