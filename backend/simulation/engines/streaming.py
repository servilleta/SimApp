"""
Streaming Simulation Engine for Large Excel Files
Handles simulations too large for memory with GPU-optimized batch processing
"""

import numpy as np
import cupy as cp
from typing import Dict, Tuple, List, Iterator, AsyncIterator, Optional, Any
import asyncio
import math
from dataclasses import dataclass
from enum import Enum

from config import settings
from gpu.manager import gpu_manager, ExcelFileStats, MemoryPlan
from simulation.random_engine import get_multi_stream_generator, RNGType
from simulation.schemas import VariableConfig

@dataclass
class ChunkData:
    """Data chunk for streaming processing"""
    chunk_id: int
    start_iteration: int
    end_iteration: int
    size: int
    variable_data: Dict[str, np.ndarray]
    intermediate_results: Optional[np.ndarray] = None

@dataclass
class BatchResult:
    """Result from a batch of iterations"""
    batch_id: int
    iterations_processed: int
    results: np.ndarray
    statistics: Dict[str, Any]
    errors: List[str]
    memory_used_mb: float

@dataclass
class LargeExcelData:
    """Large Excel dataset for streaming processing"""
    file_id: str
    total_cells: int
    total_formulas: int
    sheets: List[str]
    complexity_score: float
    estimated_memory_mb: float

class StreamingMode(Enum):
    """Streaming processing modes"""
    MEMORY_OPTIMIZED = "memory_optimized"
    SPEED_OPTIMIZED = "speed_optimized"
    BALANCED = "balanced"

class StreamingSimulationEngine:
    """Handle simulations too large for memory"""
    
    def __init__(self, gpu_manager_instance=None):
        self.gpu_manager = gpu_manager_instance or gpu_manager
        self.batch_size = None
        self.streaming_mode = StreamingMode.BALANCED
        self.memory_limit_mb = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the streaming engine"""
        if self._initialized:
            return
            
        await self.gpu_manager.initialize()
        self.batch_size = self._calculate_optimal_batch_size()
        self.memory_limit_mb = self._get_memory_limit()
        self._initialized = True
        
        print(f"✅ Streaming engine initialized: batch_size={self.batch_size}, memory_limit={self.memory_limit_mb}MB")
        
    def _calculate_optimal_batch_size(self) -> int:
        """Determine batch size based on available GPU memory"""
        if not self.gpu_manager.is_gpu_available():
            # CPU-only mode - use smaller batches
            return min(settings.DEFAULT_ITERATIONS // 4, 1000)
            
        # GPU mode - calculate based on available memory
        available_mb = self.gpu_manager.available_memory_mb
        
        # Estimate memory per iteration (rough calculation)
        memory_per_iteration_kb = 10  # 10KB per iteration estimate
        max_iterations = int((available_mb * 1024) / memory_per_iteration_kb)
        
        # Use 50% of available memory for batch processing
        optimal_batch = max_iterations // 2
        
        # Clamp to reasonable bounds
        return max(100, min(optimal_batch, 50000))
        
    def _get_memory_limit(self) -> int:
        """Get memory limit for streaming operations"""
        if self.gpu_manager.is_gpu_available():
            return int(self.gpu_manager.available_memory_mb * 0.8)
        else:
            # CPU mode - use system memory estimate
            return 2048  # 2GB default limit
            
    async def run_streaming_simulation(self, 
                                     excel_data: LargeExcelData,
                                     mc_input_configs: List[VariableConfig],
                                     ordered_calc_steps: List[Tuple[str, str, str]],
                                     target_sheet_name: str,
                                     target_cell_coordinate: str,
                                     constant_values: Dict[Tuple[str, str], Any],
                                     total_iterations: int) -> AsyncIterator[BatchResult]:
        """Process simulation in GPU-sized batches"""
        
        if not self._initialized:
            await self.initialize()
            
        print(f"🚀 Starting streaming simulation: {total_iterations} iterations, batch_size={self.batch_size}")
        
        # Calculate number of batches needed
        num_batches = math.ceil(total_iterations / self.batch_size)
        
        # Pre-calculate memory requirements
        try:
            memory_plan = await self._estimate_streaming_memory(
                total_iterations, len(mc_input_configs), excel_data.complexity_score
            )
            print(f"📊 Memory plan: {memory_plan.total_memory_mb}MB total, {memory_plan.allocated_memory_mb}MB allocated")
        except Exception as e:
            print(f"⚠️ Memory estimation failed: {e}")
            memory_plan = None
            
        # Process batches
        for batch_id in range(num_batches):
            start_iter = batch_id * self.batch_size
            end_iter = min(start_iter + self.batch_size, total_iterations)
            actual_batch_size = end_iter - start_iter
            
            print(f"🔄 Processing batch {batch_id + 1}/{num_batches}: iterations {start_iter}-{end_iter}")
            
            try:
                batch_result = await self._process_batch(
                    batch_id=batch_id,
                    start_iteration=start_iter,
                    batch_size=actual_batch_size,
                    mc_input_configs=mc_input_configs,
                    ordered_calc_steps=ordered_calc_steps,
                    target_sheet_name=target_sheet_name,
                    target_cell_coordinate=target_cell_coordinate,
                    constant_values=constant_values
                )
                
                yield batch_result
                
            except Exception as e:
                print(f"❌ Batch {batch_id} failed: {e}")
                # Yield error result
                yield BatchResult(
                    batch_id=batch_id,
                    iterations_processed=0,
                    results=np.array([]),
                    statistics={},
                    errors=[f"Batch processing failed: {str(e)}"],
                    memory_used_mb=0.0
                )
                
    async def _process_batch(self,
                           batch_id: int,
                           start_iteration: int,
                           batch_size: int,
                           mc_input_configs: List[VariableConfig],
                           ordered_calc_steps: List[Tuple[str, str, str]],
                           target_sheet_name: str,
                           target_cell_coordinate: str,
                           constant_values: Dict[Tuple[str, str], Any]) -> BatchResult:
        """Process a single batch of iterations"""
        
        batch_start_time = asyncio.get_event_loop().time()
        
        # Create batch-specific random generator
        multi_stream_gen = get_multi_stream_generator(num_streams=4)
        
        # Generate random variables for this batch
        variable_configs = []
        mc_input_params_map = {}
        
        for var_config in mc_input_configs:
            sheet_cell_key = (var_config.sheet_name, var_config.cell_coordinate)
            mc_input_params_map[sheet_cell_key] = (
                var_config.min_value,
                var_config.most_likely,  # mode
                var_config.max_value
            )
            variable_configs.append({
                'name': f"{var_config.sheet_name}!{var_config.cell_coordinate}",
                'min_value': var_config.min_value,
                'mode_value': var_config.most_likely,
                'max_value': var_config.max_value
            })
            
        # Generate random values for this batch
        try:
            if self.gpu_manager.is_gpu_available() and settings.USE_GPU:
                batch_random_values = await self._generate_batch_random_gpu(
                    variable_configs, batch_size
                )
            else:
                batch_random_values = await self._generate_batch_random_cpu(
                    variable_configs, batch_size
                )
        except Exception as e:
            raise RuntimeError(f"Random generation failed for batch {batch_id}: {e}")
            
        # Process iterations in this batch
        batch_results = np.full(batch_size, np.nan)
        batch_errors = []
        
        # Convert GPU arrays to CPU for formula evaluation
        mc_input_iter_values = {}
        for var_name, gpu_array in batch_random_values.items():
            # Extract sheet and cell from variable name
            sheet_name, cell_coord = var_name.split('!')
            mc_input_iter_values[(sheet_name, cell_coord)] = cp.asnumpy(gpu_array) if hasattr(gpu_array, 'get') else gpu_array
            
        # Evaluate formulas for each iteration in the batch
        for i in range(batch_size):
            current_iter_cell_values = constant_values.copy()
            
            # Populate with this iteration's MC input values
            for (sheet, cell), all_vals_for_input in mc_input_iter_values.items():
                current_iter_cell_values[(sheet, cell)] = all_vals_for_input[i]
                
            try:
                # Import the safe eval function
from simulation.engine import _safe_excel_eval, SAFE_EVAL_NAMESPACE
                
                # Evaluate ordered calculation steps
                for calc_sheet, calc_cell, calc_formula_str in ordered_calc_steps:
                    eval_result = _safe_excel_eval(
                        calc_formula_str,
                        calc_sheet,
                        current_iter_cell_values,
                        SAFE_EVAL_NAMESPACE,
                        current_calc_cell_coord=f"{calc_sheet}!{calc_cell}",
                        constant_values=constant_values
                    )
                    current_iter_cell_values[(calc_sheet, calc_cell)] = eval_result
                    
                # Get the final target cell value
                final_value_key = (target_sheet_name, target_cell_coordinate)
                if final_value_key in current_iter_cell_values:
                    iter_final_result = current_iter_cell_values[final_value_key]
                    try:
                        batch_results[i] = float(iter_final_result)
                    except (ValueError, TypeError):
                        batch_errors.append(f"Batch {batch_id}, Iteration {i}: Non-numeric result '{iter_final_result}'")
                else:
                    batch_errors.append(f"Batch {batch_id}, Iteration {i}: Target cell not found")
                    
            except Exception as e:
                batch_errors.append(f"Batch {batch_id}, Iteration {i}: {str(e)}")
                
        # Calculate batch statistics
        batch_stats = self._calculate_batch_statistics(batch_results)
        
        # Estimate memory usage
        memory_used_mb = self._estimate_batch_memory_usage(batch_size, len(variable_configs))
        
        batch_end_time = asyncio.get_event_loop().time()
        processing_time = batch_end_time - batch_start_time
        
        print(f"✅ Batch {batch_id} completed: {batch_size} iterations, {processing_time:.2f}s, {memory_used_mb:.1f}MB")
        
        return BatchResult(
            batch_id=batch_id,
            iterations_processed=batch_size,
            results=batch_results,
            statistics=batch_stats,
            errors=batch_errors,
            memory_used_mb=memory_used_mb
        )
        
    async def _generate_batch_random_gpu(self, variable_configs: List[Dict], batch_size: int) -> Dict[str, cp.ndarray]:
        """Generate random values for batch on GPU"""
        multi_stream_gen = get_multi_stream_generator(num_streams=min(4, len(variable_configs)))
        return multi_stream_gen.generate_all_variables_batch(variable_configs, batch_size)
        
    async def _generate_batch_random_cpu(self, variable_configs: List[Dict], batch_size: int) -> Dict[str, np.ndarray]:
        """Generate random values for batch on CPU"""
        results = {}
        for var_config in variable_configs:
            var_name = var_config['name']
            left = var_config['min_value']
            mode = var_config['mode_value']
            right = var_config['max_value']
            
            results[var_name] = np.random.triangular(left, mode, right, size=batch_size)
            
        return results
        
    def _calculate_batch_statistics(self, batch_results: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for a batch of results"""
        valid_results = batch_results[~np.isnan(batch_results)]
        
        if len(valid_results) == 0:
            return {
                "mean": float('nan'),
                "std_dev": float('nan'),
                "min_value": float('nan'),
                "max_value": float('nan'),
                "successful_iterations": 0
            }
            
        return {
            "mean": float(np.mean(valid_results)),
            "std_dev": float(np.std(valid_results)),
            "min_value": float(np.min(valid_results)),
            "max_value": float(np.max(valid_results)),
            "successful_iterations": len(valid_results)
        }
        
    def _estimate_batch_memory_usage(self, batch_size: int, num_variables: int) -> float:
        """Estimate memory usage for a batch"""
        # Rough estimate in MB
        variable_memory = (batch_size * num_variables * 4) / (1024 * 1024)  # 4 bytes per float32
        result_memory = (batch_size * 4) / (1024 * 1024)
        overhead_memory = 10  # 10MB overhead estimate
        
        return variable_memory + result_memory + overhead_memory
        
    async def _estimate_streaming_memory(self, iterations: int, num_variables: int, complexity: float) -> MemoryPlan:
        """Estimate memory requirements for streaming simulation"""
        return await self.gpu_manager.allocate_simulation_memory(
            iterations=self.batch_size,  # Memory for one batch
            num_variables=num_variables,
            formula_complexity=int(complexity)
        )

# Global streaming engine instance
streaming_engine = StreamingSimulationEngine()

def get_streaming_engine() -> StreamingSimulationEngine:
    """Get the global streaming engine instance"""
    return streaming_engine 