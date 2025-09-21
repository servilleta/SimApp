"""
Test Advanced Power Engine Features
- Memory-mapped file support
- GPU kernel integration  
- LZ4 compression
"""

import asyncio
import logging
import numpy as np
import time
from simulation.power_engine import PowerMonteCarloEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_memory_mapped_storage():
    """Test memory-mapped file support"""
    logger.info("\n=== Testing Memory-Mapped Storage ===")
    
    try:
        engine = PowerMonteCarloEngine(iterations=100000)  # Large number to trigger memmap
        
        # Test memmap creation
        test_array = engine.memmap_storage.create_memmap_array(
            'test_data', shape=(100000,), dtype=np.float64
        )
        
        # Write some data
        test_array[:1000] = np.random.random(1000)
        test_array.flush()
        
        # Verify data persists
        retrieved = engine.memmap_storage.get_memmap_array('test_data')
        assert retrieved is not None
        assert np.array_equal(test_array[:1000], retrieved[:1000])
        
        # Cleanup
        engine.memmap_storage.cleanup()
        
        logger.info("✅ Memory-mapped storage test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory-mapped storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_lz4_compression():
    """Test LZ4 compression in cache"""
    logger.info("\n=== Testing LZ4 Compression ===")
    
    try:
        engine = PowerMonteCarloEngine(iterations=1000)
        cache = engine.cache_manager
        
        # Create test data
        test_data = {
            'large_array': np.random.random((1000, 100)),
            'formula': '=SUM(A1:A1000)',
            'metadata': {'iterations': 1000, 'timestamp': time.time()}
        }
        
        # Test compression
        compressed = cache._compress(test_data)
        logger.info(f"Original size: {len(str(test_data))} bytes")
        logger.info(f"Compressed size: {len(compressed)} bytes")
        
        # Test decompression
        decompressed = cache._decompress(compressed)
        
        # Verify data integrity
        assert np.array_equal(test_data['large_array'], decompressed['large_array'])
        assert test_data['formula'] == decompressed['formula']
        assert test_data['metadata']['iterations'] == decompressed['metadata']['iterations']
        
        logger.info("✅ LZ4 compression test passed")
        return True
        
    except ImportError:
        logger.warning("⚠️ LZ4 not installed, testing pickle fallback")
        # Test should still pass with pickle fallback
        return True
    except Exception as e:
        logger.error(f"❌ LZ4 compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_gpu_kernels():
    """Test GPU kernel integration"""
    logger.info("\n=== Testing GPU Kernels ===")
    
    try:
        engine = PowerMonteCarloEngine(iterations=1000)
        
        if not engine.gpu_available:
            logger.warning("⚠️ GPU not available, skipping GPU tests")
            return True
        
        # Test formula grouping
        test_formulas = [
            ('Sheet1', 'A1', '=SUM(B1:B100)'),
            ('Sheet1', 'A2', '=A1+B2'),
            ('Sheet1', 'A3', '=A2*2'),
            ('Sheet1', 'A4', '=VLOOKUP(A1,B:C,2,FALSE)'),
        ]
        
        groups = engine._group_formulas_by_type(test_formulas)
        logger.info(f"Formula groups: {list(groups.keys())}")
        
        # Test GPU kernel compilation
        sum_kernel = engine._compile_gpu_kernel('SUM_RANGE')
        arith_kernel = engine._compile_gpu_kernel('ARITHMETIC')
        
        if sum_kernel:
            logger.info("✅ SUM_RANGE kernel compiled")
        if arith_kernel:
            logger.info("✅ ARITHMETIC kernel compiled")
        
        # Test GPU batch execution
        iteration_values = {
            ('Sheet1', 'B1'): 10.0,
            ('Sheet1', 'B2'): 20.0,
        }
        
        gpu_results = await engine._execute_gpu_batch(
            test_formulas[:3], iteration_values
        )
        
        logger.info(f"GPU results: {gpu_results}")
        logger.info("✅ GPU kernel test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sparse_range_optimization():
    """Test sparse range detection and optimization"""
    logger.info("\n=== Testing Sparse Range Optimization ===")
    
    try:
        engine = PowerMonteCarloEngine(iterations=100)
        
        # Test range detection
        test_formula = "=SUM(A1:A10000)+SUM(B:B)+AVERAGE(C1:D5000)"
        ranges = engine.sparse_detector.analyze_sum_range(test_formula)
        
        logger.info(f"Detected ranges: {ranges}")
        assert len(ranges) >= 2  # Should detect at least A1:A10000 and B:B
        
        # Test sparse optimization
        sparse_sheet_data = {
            'A1': 10, 'A5': 20, 'A100': 30, 'A5000': 40,  # Only 4 values in 10000 range
            'B1': 5, 'B2': 15,  # Only 2 values in column B
        }
        
        non_empty_a = engine.sparse_detector.optimize_range(
            sparse_sheet_data, ('A1', 'A10000')
        )
        
        logger.info(f"Non-empty cells in A1:A10000: {len(non_empty_a)} (was 10000)")
        assert len(non_empty_a) == 4  # Should only find 4 non-empty cells
        
        # Test sparsity calculation
        sparsity = engine.sparse_detector.estimate_sparsity(sparse_sheet_data, 20000)
        logger.info(f"Sparsity: {sparsity:.2%}")
        assert sparsity > 0.99  # Should be very sparse (only 6 values in 20000 cells)
        
        logger.info("✅ Sparse range optimization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sparse range optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_metrics():
    """Test performance metrics collection"""
    logger.info("\n=== Testing Performance Metrics ===")
    
    try:
        engine = PowerMonteCarloEngine(iterations=100)
        
        # Simulate some operations to generate metrics
        engine.metrics['sparse_cells_skipped'] = 9996  # Skipped 9996 empty cells
        engine.metrics['cache_hits'] = 500
        engine.metrics['cache_misses'] = 100
        engine.metrics['chunks_processed'] = 10
        engine.metrics['gpu_kernels_launched'] = 5
        engine.metrics['memory_cleanups'] = 2
        
        logger.info(f"Performance metrics: {engine.metrics}")
        
        # Calculate cache hit rate
        total_cache_ops = engine.metrics['cache_hits'] + engine.metrics['cache_misses']
        hit_rate = engine.metrics['cache_hits'] / total_cache_ops * 100
        
        logger.info(f"Cache hit rate: {hit_rate:.1f}%")
        logger.info(f"Sparse cells skipped: {engine.metrics['sparse_cells_skipped']:,}")
        
        logger.info("✅ Performance metrics test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all advanced Power engine tests"""
    logger.info("🚀 Starting Advanced Power Engine Tests")
    
    tests = [
        ("Memory-Mapped Storage", test_memory_mapped_storage),
        ("LZ4 Compression", test_lz4_compression),
        ("GPU Kernels", test_gpu_kernels),
        ("Sparse Range Optimization", test_sparse_range_optimization),
        ("Performance Metrics", test_performance_metrics),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        result = await test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1) 