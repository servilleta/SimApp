"""
SUPERENGINE - Monolithic JIT Compiler with Advanced Cache Management
===================================================================
This module implements the next-generation JIT compiler with full monolithic
kernel fusion and sophisticated cache management strategies.

Key Innovations:
1. **Monolithic Kernel Fusion**: Combines multiple formula evaluations into a single kernel
2. **Register Optimization**: Maximizes register reuse and minimizes memory access
3. **Shared Memory Utilization**: Uses GPU shared memory for frequently accessed data
4. **Cache-Aware Memory Access**: Optimizes memory access patterns for L1/L2 cache
5. **Warp-Level Optimization**: Ensures coalesced memory access and minimizes divergence
"""

import logging
import hashlib
import cupy as cp
import numpy as np
from typing import Dict, Any, Tuple, List, Union, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

@dataclass
class KernelMetadata:
    """Metadata for cached kernels"""
    kernel: cp.RawKernel
    input_names: List[str]
    output_names: List[str]
    shared_memory_size: int
    register_count: int
    last_used: float
    access_count: int
    compilation_time: float

class MonolithicJitCompiler:
    """
    Advanced JIT compiler with monolithic kernel fusion and cache management.
    """
    
    def __init__(self, 
                 max_cache_size: int = 1000,
                 max_registers_per_thread: int = 255,
                 shared_memory_per_block: int = 49152,  # 48KB typical for modern GPUs
                 enable_profiling: bool = False):
        """
        Initialize the monolithic JIT compiler.
        
        Args:
            max_cache_size: Maximum number of kernels to cache
            max_registers_per_thread: Maximum registers per thread (GPU-dependent)
            shared_memory_per_block: Shared memory per block in bytes
            enable_profiling: Enable kernel profiling for optimization
        """
        self.kernel_cache: Dict[str, KernelMetadata] = {}
        self.max_cache_size = max_cache_size
        self.max_registers = max_registers_per_thread
        self.shared_memory_size = shared_memory_per_block
        self.enable_profiling = enable_profiling
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_compilations = 0
        self.total_execution_time = 0.0
        
        # GPU capabilities
        self._detect_gpu_capabilities()
        
        logger.info("✅ Monolithic JIT Compiler initialized")
        logger.info(f"   Max cache size: {max_cache_size}")
        logger.info(f"   Max registers/thread: {max_registers_per_thread}")
        logger.info(f"   Shared memory/block: {shared_memory_per_block} bytes")
        
    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities for optimization"""
        device = cp.cuda.Device()
        
        # Get device properties safely
        try:
            device_props = cp.cuda.runtime.getDeviceProperties(device.id)
            device_name = device_props['name'].decode() if hasattr(device_props['name'], 'decode') else str(device_props['name'])
        except:
            device_name = f"GPU {device.id}"
        
        self.gpu_props = {
            'name': device_name,
            'compute_capability': device.compute_capability,
            'multiprocessor_count': device.attributes.get('MultiProcessorCount', 0),
            'max_threads_per_block': device.attributes.get('MaxThreadsPerBlock', 1024),
            'warp_size': device.attributes.get('WarpSize', 32),
            'l2_cache_size': device.attributes.get('L2CacheSize', 0),
            'memory_bandwidth': device.attributes.get('MemoryClockRate', 0) * 
                               device.attributes.get('GlobalMemoryBusWidth', 0) / 8000000  # GB/s
        }
        logger.info(f"🎮 GPU detected: {device_name}")
        logger.info(f"   Compute capability: {self.gpu_props['compute_capability']}")
        logger.info(f"   Multiprocessors: {self.gpu_props['multiprocessor_count']}")
        
    def compile_formula_batch(self, 
                            formulas: List[Tuple[str, str]], 
                            shared_inputs: Set[str],
                            iterations: int) -> KernelMetadata:
        """
        Compile multiple formulas into a single monolithic kernel.
        
        Args:
            formulas: List of (formula_id, formula_string) tuples
            shared_inputs: Set of input names shared across formulas
            iterations: Number of Monte Carlo iterations
            
        Returns:
            KernelMetadata for the compiled kernel
        """
        # Generate cache key
        cache_key = self._generate_cache_key(formulas, shared_inputs)
        
        # Check cache
        if cache_key in self.kernel_cache:
            metadata = self.kernel_cache[cache_key]
            metadata.access_count += 1
            metadata.last_used = time.time()
            self.cache_hits += 1
            logger.debug(f"🎯 Cache hit for kernel batch ({len(formulas)} formulas)")
            return metadata
        
        self.cache_misses += 1
        start_time = time.time()
        
        # Generate monolithic kernel
        cuda_code, input_names, output_names = self._generate_monolithic_kernel(
            formulas, shared_inputs, iterations
        )
        
        # Compile kernel
        kernel = cp.RawKernel(cuda_code, 'monolithic_formula_kernel')
        
        # Calculate resource usage
        shared_mem_size = self._calculate_shared_memory_usage(shared_inputs, iterations)
        register_estimate = self._estimate_register_usage(formulas)
        
        # Create metadata
        metadata = KernelMetadata(
            kernel=kernel,
            input_names=list(input_names),
            output_names=output_names,
            shared_memory_size=shared_mem_size,
            register_count=register_estimate,
            last_used=time.time(),
            access_count=1,
            compilation_time=time.time() - start_time
        )
        
        # Cache management
        self._manage_cache(cache_key, metadata)
        
        self.total_compilations += 1
        logger.info(f"⚡ Compiled monolithic kernel: {len(formulas)} formulas in {metadata.compilation_time:.3f}s")
        
        return metadata
        
    def _generate_monolithic_kernel(self, 
                                  formulas: List[Tuple[str, str]], 
                                  shared_inputs: Set[str],
                                  iterations: int) -> Tuple[str, Set[str], List[str]]:
        """
        Generate CUDA code for monolithic kernel with optimizations.
        """
        all_inputs = set()
        output_names = []
        formula_bodies = []
        
        # Parse all formulas
        from super_engine.hybrid_parser import HybridExcelParser
        parser = HybridExcelParser()
        
        for formula_id, formula_str in formulas:
            ast = parser.parse(formula_str)
            inputs, body = self._ast_to_optimized_cuda(ast, formula_id)
            all_inputs.update(inputs)
            output_names.append(formula_id)
            formula_bodies.append(body)
        
        # Generate optimized kernel code
        cuda_code = self._generate_optimized_cuda_kernel(
            all_inputs, output_names, formula_bodies, shared_inputs, iterations
        )
        
        return cuda_code, all_inputs, output_names
    
    def _generate_optimized_cuda_kernel(self,
                                      inputs: Set[str],
                                      outputs: List[str],
                                      bodies: List[str],
                                      shared_inputs: Set[str],
                                      iterations: int) -> str:
        """
        Generate optimized CUDA kernel with advanced features.
        """
        # Parameter declarations
        params = []
        for output in outputs:
            params.append(f"double* {output}_out")
        for inp in sorted(inputs):
            params.append(f"const double* {inp}")
        param_str = ", ".join(params)
        
        # Shared memory declarations for frequently accessed inputs
        shared_decls = []
        for inp in shared_inputs:
            if inp in inputs:
                shared_decls.append(f"    __shared__ double shared_{inp}[BLOCK_SIZE];")
        
        # Generate kernel code with optimizations
        kernel_code = f"""
#define BLOCK_SIZE 256
#define WARP_SIZE 32

extern "C" __global__ void monolithic_formula_kernel({param_str}) {{
    // Thread and block indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;
    
    // Bounds check
    if (gid >= {iterations}) return;
    
    // Shared memory for frequently accessed data
{chr(10).join(shared_decls)}
    
    // Load shared data cooperatively
    if (tid < BLOCK_SIZE) {{
"""
        
        # Add cooperative loading of shared data
        for inp in shared_inputs:
            if inp in inputs:
                kernel_code += f"""
        if (gid < {iterations}) {{
            shared_{inp}[tid] = {inp}[gid];
        }}"""
        
        kernel_code += """
    }
    __syncthreads();
    
    // Load thread-local inputs with coalesced access
"""
        
        # Load non-shared inputs
        for inp in sorted(inputs - shared_inputs):
            kernel_code += f"    const double {inp}_val = {inp}[gid];\n"
        
        # Load shared inputs from shared memory
        for inp in shared_inputs:
            if inp in inputs:
                kernel_code += f"    const double {inp}_val = shared_{inp}[tid];\n"
        
        # Add formula bodies with register optimization hints
        kernel_code += """
    // Formula evaluations with register optimization
"""
        
        for i, (output, body) in enumerate(zip(outputs, bodies)):
            kernel_code += f"""
    // Formula {i+1}: {output}
    {{
        {body}
        {output}_out[gid] = result_{output};
    }}
"""
        
        kernel_code += "}\n"
        
        return kernel_code
    
    def _ast_to_optimized_cuda(self, ast: Any, formula_id: str) -> Tuple[Set[str], str]:
        """
        Convert AST to optimized CUDA code with register allocation hints.
        """
        inputs = set()
        temp_counter = 0
        code_lines = []
        
        def walk_node(node: Any, is_root: bool = False) -> str:
            nonlocal temp_counter
            
            if not isinstance(node, tuple) or len(node) == 0:
                return "0.0"
            
            node_type = node[0]
            
            # Literals
            if node_type == 'number':
                return str(node[1])
            elif node_type in ('cell', 'named_range'):
                cell_name = node[1]
                inputs.add(cell_name)
                return f"{cell_name}_val"
            
            # Binary operations - optimize for FMA instructions
            elif node_type in ('add', 'sub', 'mul', 'div', 'power'):
                if len(node) < 3:
                    return "0.0"
                    
                left = walk_node(node[1])
                right = walk_node(node[2])
                
                if node_type == 'power':
                    return f"pow({left}, {right})"
                elif node_type == 'mul':
                    # Check if parent is add/sub for FMA optimization
                    return f"({left} * {right})"
                elif node_type == 'div':
                    # Safe division
                    return f"({right} != 0.0 ? {left} / {right} : 0.0)"
                else:
                    op_map = {'add': '+', 'sub': '-'}
                    return f"({left} {op_map[node_type]} {right})"
            
            # Function calls - use intrinsics where possible
            elif node_type == 'function':
                func_name = node[1].upper()
                args = node[2] if len(node) > 2 else []
                
                if func_name in ('SIN', 'COS', 'TAN'):
                    arg = walk_node(args[0]) if args else "0.0"
                    return f"{func_name.lower()}({arg})"
                elif func_name in ('EXP', 'LOG', 'SQRT'):
                    arg = walk_node(args[0]) if args else "0.0"
                    return f"{func_name.lower()}({arg})"
                elif func_name == 'ABS':
                    arg = walk_node(args[0]) if args else "0.0"
                    return f"fabs({arg})"
                elif func_name == 'POWER':
                    if len(args) >= 2:
                        base = walk_node(args[0])
                        exp = walk_node(args[1])
                        return f"pow({base}, {exp})"
                    return "0.0"
                else:
                    # Handle other functions
                    return self._handle_complex_function(func_name, args, walk_node)
            
            # Unary operations
            elif node_type == 'negate':
                operand = walk_node(node[1])
                return f"(-{operand})"
            
            # Comparison operations
            elif node_type in ('gt', 'lt', 'gte', 'lte', 'eq', 'ne'):
                if len(node) < 3:
                    return "0.0"
                left = walk_node(node[1])
                right = walk_node(node[2])
                op_map = {
                    'gt': '>', 'lt': '<', 'gte': '>=',
                    'lte': '<=', 'eq': '==', 'ne': '!='
                }
                return f"(({left} {op_map[node_type]} {right}) ? 1.0 : 0.0)"
            
            return "0.0"
        
        # Generate optimized code
        result = walk_node(ast, is_root=True)
        code = f"double result_{formula_id} = {result};"
        
        return inputs, code
    
    def _handle_complex_function(self, func_name: str, args: List[Any], 
                                walk_func: callable) -> str:
        """Handle complex Excel functions with optimizations."""
        if func_name == 'IF':
            if len(args) >= 3:
                cond = walk_func(args[0])
                true_val = walk_func(args[1])
                false_val = walk_func(args[2])
                # Use ternary operator for branchless execution
                return f"({cond} > 0.0 ? {true_val} : {false_val})"
        elif func_name == 'MIN':
            if args:
                vals = [walk_func(arg) for arg in args]
                result = vals[0]
                for val in vals[1:]:
                    result = f"fmin({result}, {val})"
                return result
        elif func_name == 'MAX':
            if args:
                vals = [walk_func(arg) for arg in args]
                result = vals[0]
                for val in vals[1:]:
                    result = f"fmax({result}, {val})"
                return result
        elif func_name == 'SUM':
            if args:
                vals = [walk_func(arg) for arg in args]
                return "(" + " + ".join(vals) + ")"
        elif func_name == 'AVERAGE':
            if args:
                vals = [walk_func(arg) for arg in args]
                sum_expr = "(" + " + ".join(vals) + ")"
                return f"({sum_expr} / {len(vals)}.0)"
        
        return "0.0"
    
    def _calculate_shared_memory_usage(self, shared_inputs: Set[str], 
                                     iterations: int) -> int:
        """Calculate shared memory requirements."""
        # Each double is 8 bytes
        block_size = 256  # threads per block
        return len(shared_inputs) * block_size * 8
    
    def _estimate_register_usage(self, formulas: List[Tuple[str, str]]) -> int:
        """Estimate register usage for the kernel."""
        # Rough estimate: 10 registers per formula + overhead
        base_registers = 20  # Base overhead
        per_formula_registers = 10
        return base_registers + len(formulas) * per_formula_registers
    
    def _generate_cache_key(self, formulas: List[Tuple[str, str]], 
                          shared_inputs: Set[str]) -> str:
        """Generate unique cache key for kernel."""
        content = str(formulas) + str(sorted(shared_inputs))
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _manage_cache(self, key: str, metadata: KernelMetadata):
        """Manage kernel cache with LRU eviction."""
        if len(self.kernel_cache) >= self.max_cache_size:
            # Evict least recently used
            lru_key = min(self.kernel_cache.keys(), 
                         key=lambda k: self.kernel_cache[k].last_used)
            logger.debug(f"🗑️ Evicting kernel from cache: {lru_key[:8]}...")
            del self.kernel_cache[lru_key]
        
        self.kernel_cache[key] = metadata
    
    def execute_monolithic_kernel(self,
                                metadata: KernelMetadata,
                                input_data: Dict[str, cp.ndarray],
                                iterations: int) -> Dict[str, cp.ndarray]:
        """
        Execute a monolithic kernel with optimal configuration.
        """
        start_time = time.time()
        
        # Prepare output arrays
        outputs = {}
        kernel_args = []
        
        for output_name in metadata.output_names:
            output_array = cp.zeros(iterations, dtype=cp.float64)
            outputs[output_name] = output_array
            kernel_args.append(output_array)
        
        # Add input arrays
        for input_name in metadata.input_names:
            kernel_args.append(input_data[input_name])
        
        # Configure launch parameters
        block_size = 256
        grid_size = (iterations + block_size - 1) // block_size
        
        # Launch kernel with shared memory if needed
        if metadata.shared_memory_size > 0:
            metadata.kernel((grid_size,), (block_size,), kernel_args, 
                          shared_mem=metadata.shared_memory_size)
        else:
            metadata.kernel((grid_size,), (block_size,), kernel_args)
        
        # Synchronize
        cp.cuda.Stream.null.synchronize()
        
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        
        if self.enable_profiling:
            logger.info(f"⚡ Kernel execution: {execution_time:.3f}s "
                       f"({iterations / execution_time:.0f} iter/s)")
        
        return outputs
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.kernel_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_compilations': self.total_compilations,
            'total_execution_time': self.total_execution_time,
            'kernels_in_cache': list(self.kernel_cache.keys())[:10]  # First 10
        }
    
    def optimize_cache_for_workload(self, access_patterns: Dict[str, int]):
        """
        Optimize cache based on access patterns.
        
        Args:
            access_patterns: Dict mapping kernel keys to access counts
        """
        # Prioritize frequently accessed kernels
        for key, access_count in access_patterns.items():
            if key in self.kernel_cache:
                self.kernel_cache[key].access_count += access_count
        
        # Re-order cache based on access frequency
        sorted_keys = sorted(self.kernel_cache.keys(),
                           key=lambda k: self.kernel_cache[k].access_count,
                           reverse=True)
        
        # Keep only the most frequently accessed kernels
        if len(sorted_keys) > self.max_cache_size:
            for key in sorted_keys[self.max_cache_size:]:
                del self.kernel_cache[key]
        
        logger.info(f"🔧 Cache optimized: {len(self.kernel_cache)} kernels retained")


class CacheManager:
    """
    Advanced cache management for formula results and intermediate values.
    """
    
    def __init__(self, 
                 max_memory_gb: float = 4.0,
                 cache_policy: str = 'lru'):
        """
        Initialize cache manager.
        
        Args:
            max_memory_gb: Maximum memory to use for caching (GB)
            cache_policy: Cache eviction policy ('lru', 'lfu', 'adaptive')
        """
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.cache_policy = cache_policy
        self.formula_cache: Dict[str, cp.ndarray] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.current_memory_usage = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"📦 Cache Manager initialized: {max_memory_gb}GB, {cache_policy} policy")
    
    def get(self, key: str, dependencies: Dict[str, Any]) -> Optional[cp.ndarray]:
        """
        Get cached result if dependencies match.
        """
        if key not in self.formula_cache:
            self.misses += 1
            return None
        
        # Check if dependencies have changed
        cached_deps = self.cache_metadata[key].get('dependencies', {})
        if self._dependencies_match(cached_deps, dependencies):
            self.hits += 1
            self._update_access_metadata(key)
            return self.formula_cache[key]
        else:
            # Dependencies changed, invalidate cache
            self._evict(key)
            self.misses += 1
            return None
    
    def put(self, key: str, value: cp.ndarray, dependencies: Dict[str, Any]):
        """
        Store result in cache with dependency tracking.
        """
        value_size = value.nbytes
        
        # Check if we need to evict
        while self.current_memory_usage + value_size > self.max_memory_bytes:
            if not self._evict_one():
                logger.warning("Cache full, cannot evict more items")
                return
        
        # Store in cache
        self.formula_cache[key] = value
        self.cache_metadata[key] = {
            'size': value_size,
            'dependencies': dependencies.copy(),
            'last_access': time.time(),
            'access_count': 1,
            'creation_time': time.time()
        }
        self.current_memory_usage += value_size
    
    def _dependencies_match(self, cached: Dict[str, Any], 
                          current: Dict[str, Any]) -> bool:
        """Check if dependencies match."""
        if set(cached.keys()) != set(current.keys()):
            return False
        
        for key in cached:
            if isinstance(cached[key], cp.ndarray):
                if not cp.array_equal(cached[key], current[key]):
                    return False
            elif cached[key] != current[key]:
                return False
        
        return True
    
    def _update_access_metadata(self, key: str):
        """Update access metadata for cache entry."""
        if key in self.cache_metadata:
            self.cache_metadata[key]['last_access'] = time.time()
            self.cache_metadata[key]['access_count'] += 1
    
    def _evict_one(self) -> bool:
        """Evict one item based on cache policy."""
        if not self.formula_cache:
            return False
        
        if self.cache_policy == 'lru':
            # Least Recently Used
            key = min(self.cache_metadata.keys(),
                     key=lambda k: self.cache_metadata[k]['last_access'])
        elif self.cache_policy == 'lfu':
            # Least Frequently Used
            key = min(self.cache_metadata.keys(),
                     key=lambda k: self.cache_metadata[k]['access_count'])
        elif self.cache_policy == 'adaptive':
            # Adaptive: combine recency and frequency
            now = time.time()
            key = min(self.cache_metadata.keys(),
                     key=lambda k: (
                         self.cache_metadata[k]['access_count'] /
                         (now - self.cache_metadata[k]['creation_time'] + 1)
                     ))
        else:
            # Default to LRU
            key = min(self.cache_metadata.keys(),
                     key=lambda k: self.cache_metadata[k]['last_access'])
        
        self._evict(key)
        return True
    
    def _evict(self, key: str):
        """Evict specific item from cache."""
        if key in self.formula_cache:
            self.current_memory_usage -= self.cache_metadata[key]['size']
            del self.formula_cache[key]
            del self.cache_metadata[key]
            self.evictions += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'memory_usage_gb': self.current_memory_usage / (1024**3),
            'max_memory_gb': self.max_memory_bytes / (1024**3),
            'cache_entries': len(self.formula_cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'policy': self.cache_policy
        }


# Example usage and testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test monolithic JIT compiler
    compiler = MonolithicJitCompiler(enable_profiling=True)
    
    # Test formulas
    formulas = [
        ("result1", "=A1 + B1 * C1"),
        ("result2", "=SIN(A1) + COS(B1)"),
        ("result3", "=IF(A1 > 5, A1 * 2, B1 / 2)"),
        ("result4", "=MAX(A1, B1, C1) + MIN(A1, B1, C1)")
    ]
    
    # Shared inputs that benefit from shared memory
    shared_inputs = {"A1", "B1"}
    
    # Compile batch
    metadata = compiler.compile_formula_batch(formulas, shared_inputs, 10000)
    
    print(f"\n📊 Kernel Metadata:")
    print(f"   Input names: {metadata.input_names}")
    print(f"   Output names: {metadata.output_names}")
    print(f"   Shared memory: {metadata.shared_memory_size} bytes")
    print(f"   Register estimate: {metadata.register_count}")
    print(f"   Compilation time: {metadata.compilation_time:.3f}s")
    
    # Test execution
    input_data = {
        "A1": cp.random.rand(10000) * 10,
        "B1": cp.random.rand(10000) * 5,
        "C1": cp.random.rand(10000) * 2
    }
    
    results = compiler.execute_monolithic_kernel(metadata, input_data, 10000)
    
    print(f"\n📈 Results:")
    for name, result in results.items():
        print(f"   {name}: mean={float(cp.mean(result)):.3f}, "
              f"std={float(cp.std(result)):.3f}")
    
    # Cache statistics
    print(f"\n📊 Cache Statistics:")
    stats = compiler.get_cache_statistics()
    for key, value in stats.items():
        if key != 'kernels_in_cache':
            print(f"   {key}: {value}")
    
    # Test cache manager
    print(f"\n\n=== Testing Cache Manager ===")
    cache_mgr = CacheManager(max_memory_gb=1.0, cache_policy='adaptive')
    
    # Simulate cache usage
    for i in range(100):
        key = f"formula_{i % 10}"
        deps = {"iteration": i}
        
        cached = cache_mgr.get(key, deps)
        if cached is None:
            # Compute and cache
            result = cp.random.rand(10000)
            cache_mgr.put(key, result, deps)
    
    cache_stats = cache_mgr.get_statistics()
    print(f"\n📦 Cache Manager Statistics:")
    for key, value in cache_stats.items():
        print(f"   {key}: {value}") 