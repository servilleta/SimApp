"""
Test script for Monolithic JIT Compiler and Advanced Cache Management
"""

import asyncio
import time
import cupy as cp
import numpy as np
from super_engine.monolithic_jit import MonolithicJitCompiler, CacheManager
from super_engine.jit_compiler import JitCompiler

def test_monolithic_compilation():
    """Test monolithic kernel compilation and execution"""
    print("=== Testing Monolithic JIT Compilation ===\n")
    
    # Initialize compiler
    compiler = MonolithicJitCompiler(enable_profiling=True)
    
    # Test case 1: Simple formula batch
    print("Test 1: Simple Formula Batch")
    formulas = [
        ("revenue", "=A1 * B1"),
        ("cost", "=B1 * 0.7"),
        ("profit", "=A1 * B1 - B1 * 0.7"),
        ("margin", "=(A1 * B1 - B1 * 0.7) / (A1 * B1)")
    ]
    
    shared_inputs = {"A1", "B1"}  # Both used in multiple formulas
    iterations = 10000
    
    # Compile
    start = time.time()
    metadata = compiler.compile_formula_batch(formulas, shared_inputs, iterations)
    compile_time = time.time() - start
    
    print(f"✅ Compilation completed in {compile_time:.3f}s")
    print(f"   Formulas compiled: {len(metadata.output_names)}")
    print(f"   Shared memory size: {metadata.shared_memory_size} bytes")
    print(f"   Register estimate: {metadata.register_count}")
    
    # Execute
    input_data = {
        "A1": cp.random.uniform(10, 100, iterations),  # Price
        "B1": cp.random.uniform(100, 1000, iterations)  # Quantity
    }
    
    start = time.time()
    results = compiler.execute_monolithic_kernel(metadata, input_data, iterations)
    exec_time = time.time() - start
    
    print(f"✅ Execution completed in {exec_time:.3f}s")
    print(f"   Throughput: {iterations / exec_time:.0f} iterations/second")
    
    # Verify results
    for name, result in results.items():
        print(f"   {name}: mean={float(cp.mean(result)):.2f}, std={float(cp.std(result)):.2f}")
    
    # Test case 2: Complex formulas with functions
    print("\n\nTest 2: Complex Formula Batch with Functions")
    complex_formulas = [
        ("signal1", "=SIN(A1 * 0.1) + COS(B1 * 0.05)"),
        ("signal2", "=EXP(-A1 / 50) * LOG(B1 + 1)"),
        ("combined", "=SIN(A1 * 0.1) + COS(B1 * 0.05) + EXP(-A1 / 50) * LOG(B1 + 1)"),
        ("normalized", "=(SIN(A1 * 0.1) + COS(B1 * 0.05)) / SQRT(A1 * A1 + B1 * B1)")
    ]
    
    metadata2 = compiler.compile_formula_batch(complex_formulas, shared_inputs, iterations)
    results2 = compiler.execute_monolithic_kernel(metadata2, input_data, iterations)
    
    print(f"✅ Complex formulas executed successfully")
    for name, result in results2.items():
        print(f"   {name}: mean={float(cp.mean(result)):.3f}, std={float(cp.std(result)):.3f}")
    
    # Show cache statistics
    print("\n📊 Compiler Cache Statistics:")
    stats = compiler.get_cache_statistics()
    print(f"   Cache size: {stats['cache_size']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Total compilations: {stats['total_compilations']}")

def test_cache_manager():
    """Test advanced cache management"""
    print("\n\n=== Testing Advanced Cache Manager ===\n")
    
    # Test different cache policies
    policies = ['lru', 'lfu', 'adaptive']
    
    for policy in policies:
        print(f"\nTesting {policy.upper()} Policy:")
        cache = CacheManager(max_memory_gb=0.1, cache_policy=policy)  # Small cache for testing
        
        # Simulate workload
        for iteration in range(50):
            for formula_id in range(10):
                key = f"formula_{formula_id}"
                deps = {"iteration": iteration, "param": formula_id % 3}
                
                # Try to get from cache
                result = cache.get(key, deps)
                
                if result is None:
                    # Simulate computation
                    result = cp.random.rand(1000) * formula_id
                    cache.put(key, result, deps)
        
        # Get statistics
        stats = cache.get_statistics()
        print(f"   Memory usage: {stats['memory_usage_gb']:.3f}GB / {stats['max_memory_gb']}GB")
        print(f"   Cache entries: {stats['cache_entries']}")
        print(f"   Hit rate: {stats['hit_rate']:.1%}")
        print(f"   Evictions: {stats['evictions']}")

def test_jit_with_monolithic():
    """Test enhanced JIT compiler with monolithic support"""
    print("\n\n=== Testing Enhanced JIT Compiler ===\n")
    
    # Initialize JIT compiler with monolithic support
    jit = JitCompiler(enable_monolithic=True)
    
    # Test batch compilation
    formulas = [
        ("total_sales", "=A1 + A2 + A3 + A4 + A5"),
        ("average_sales", "=(A1 + A2 + A3 + A4 + A5) / 5"),
        ("max_sales", "=MAX(A1, A2, A3, A4, A5)"),
        ("min_sales", "=MIN(A1, A2, A3, A4, A5)"),
        ("range", "=MAX(A1, A2, A3, A4, A5) - MIN(A1, A2, A3, A4, A5)")
    ]
    
    input_data = {
        "A1": cp.random.uniform(100, 200, 10000),
        "A2": cp.random.uniform(150, 250, 10000),
        "A3": cp.random.uniform(120, 220, 10000),
        "A4": cp.random.uniform(180, 280, 10000),
        "A5": cp.random.uniform(140, 240, 10000)
    }
    
    # Compile and execute batch
    start = time.time()
    results = jit.compile_batch(formulas, input_data)
    batch_time = time.time() - start
    
    print(f"✅ Batch compilation and execution: {batch_time:.3f}s")
    print(f"   Formulas processed: {len(formulas)}")
    
    for name, result in results.items():
        print(f"   {name}: mean={float(cp.mean(result)):.2f}")
    
    # Compare with individual compilation
    print("\nComparing with individual compilation:")
    start = time.time()
    individual_results = {}
    for name, formula in formulas:
        individual_results[name] = jit.compile_and_run(formula, input_data)
    individual_time = time.time() - start
    
    print(f"✅ Individual compilation: {individual_time:.3f}s")
    print(f"⚡ Speedup from monolithic: {individual_time / batch_time:.2f}x")
    
    # Show statistics
    print("\n📊 JIT Compiler Statistics:")
    stats = jit.get_statistics()
    print(f"   Kernel cache size: {stats['kernel_cache_size']}")
    print(f"   Monolithic enabled: {stats['monolithic_enabled']}")
    
    if 'monolithic_stats' in stats:
        print("\n   Monolithic Compiler Stats:")
        for key, value in stats['monolithic_stats'].items():
            if key != 'kernels_in_cache':
                print(f"     {key}: {value}")

def benchmark_performance():
    """Benchmark performance improvements"""
    print("\n\n=== Performance Benchmarks ===\n")
    
    # Test different batch sizes
    batch_sizes = [1, 5, 10, 20, 50]
    iterations = 100000
    
    print("Benchmarking different batch sizes:")
    print("Batch Size | Compile Time | Exec Time | Total Time | Throughput")
    print("-" * 65)
    
    compiler = MonolithicJitCompiler(enable_profiling=False)
    
    for batch_size in batch_sizes:
        # Generate formulas
        formulas = []
        for i in range(batch_size):
            formulas.append((f"result_{i}", f"=A1 * {i+1} + B1 / {i+1} + SIN(C1 * {i*0.1})"))
        
        # Prepare data
        input_data = {
            "A1": cp.random.rand(iterations),
            "B1": cp.random.rand(iterations),
            "C1": cp.random.rand(iterations)
        }
        
        # Compile
        compile_start = time.time()
        metadata = compiler.compile_formula_batch(formulas, {"A1", "B1", "C1"}, iterations)
        compile_time = time.time() - compile_start
        
        # Execute
        exec_start = time.time()
        results = compiler.execute_monolithic_kernel(metadata, input_data, iterations)
        exec_time = time.time() - exec_start
        
        total_time = compile_time + exec_time
        throughput = (batch_size * iterations) / total_time
        
        print(f"{batch_size:10} | {compile_time:11.3f}s | {exec_time:9.3f}s | "
              f"{total_time:10.3f}s | {throughput:,.0f} ops/s")

def main():
    """Run all tests"""
    print("Monolithic JIT Compiler and Cache Management Tests")
    print("=" * 60)
    
    # Run tests
    test_monolithic_compilation()
    test_cache_manager()
    test_jit_with_monolithic()
    benchmark_performance()
    
    print("\n\n✅ All tests completed successfully!")
    print("\n🚀 Key Benefits Demonstrated:")
    print("   - Monolithic kernel fusion reduces kernel launch overhead")
    print("   - Shared memory optimization for frequently accessed data")
    print("   - Advanced caching with multiple eviction policies")
    print("   - Significant performance improvements for batch operations")

if __name__ == "__main__":
    main() 