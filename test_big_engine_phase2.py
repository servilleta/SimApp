#!/usr/bin/env python3
"""
Test script for BIG engine Phase 2 implementation
Tests: Monte Carlo simulation logic, batch processing, result generation
"""

import sys
import os
import time
import numpy as np

# Add backend to path
sys.path.insert(0, '/app')

def test_monte_carlo_simulation():
    """Test Monte Carlo simulation logic"""
    print("🧪 Testing Monte Carlo simulation logic...")
    
    try:
        from simulation.big_engine import BigMonteCarloEngine
        
        # Create test data
        dependency_graph = {1: {2, 3}, 2: {4}, 3: {4}, 4: set()}
        formulas = {('Sheet1', 'A1'): {'formula': '=B1+C1', 'precedents': [('Sheet1', 'B1'), ('Sheet1', 'C1')]}}
        
        # Test inputs
        mc_inputs = [
            {
                'sheet': 'Sheet1',
                'cell': 'B1',
                'distribution': 'normal',
                'mean': 10.0,
                'std': 2.0
            },
            {
                'sheet': 'Sheet1',
                'cell': 'C1',
                'distribution': 'uniform',
                'min': 5.0,
                'max': 15.0
            }
        ]
        
        constants = [
            {
                'sheet': 'Sheet1',
                'cell': 'D1',
                'value': 100.0
            }
        ]
        
        # Run simulation
        target_node = BigMonteCarloEngine.encode_node('Sheet1', 'A1')
        iterations = 100
        batch_size = 20
        
        print(f"Running simulation with {iterations} iterations, batch size {batch_size}")
        
        all_results = []
        batch_count = 0
        
        for batch_data in BigMonteCarloEngine.run_monte_carlo_simulation(
            dependency_graph=dependency_graph,
            formulas=formulas,
            target_node=target_node,
            mc_inputs=mc_inputs,
            constants=constants,
            iterations=iterations,
            batch_size=batch_size,
            timeout_seconds=60
        ):
            batch_count += 1
            all_results.extend(batch_data['batch_results'])
            print(f"  Batch {batch_count}: {batch_data['total_processed']}/{iterations} iterations")
        
        print(f"✅ Simulation completed: {len(all_results)} results")
        print(f"   Mean: {np.mean(all_results):.2f}")
        print(f"   Std: {np.std(all_results):.2f}")
        print(f"   Min: {np.min(all_results):.2f}")
        print(f"   Max: {np.max(all_results):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monte Carlo simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_histogram_generation():
    """Test histogram data generation"""
    print("\n🧪 Testing histogram generation...")
    
    try:
        from simulation.big_engine import BigMonteCarloEngine
        
        # Generate test results
        test_results = np.random.normal(100, 20, 1000).tolist()
        
        # Create histogram
        histogram_data = BigMonteCarloEngine.create_histogram_data(test_results, bins=20)
        
        print(f"✅ Histogram created:")
        print(f"   Bins: {len(histogram_data['histogram'])}")
        print(f"   Bin edges: {len(histogram_data['bin_edges'])}")
        print(f"   Bin centers: {len(histogram_data['bin_centers'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Histogram generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sensitivity_analysis():
    """Test sensitivity analysis"""
    print("\n🧪 Testing sensitivity analysis...")
    
    try:
        from simulation.big_engine import BigMonteCarloEngine
        
        # Generate test results
        test_results = np.random.normal(100, 20, 1000).tolist()
        
        # Test input configs
        input_configs = [
            {'sheet': 'Sheet1', 'cell': 'B1'},
            {'sheet': 'Sheet1', 'cell': 'C1'},
            {'sheet': 'Sheet1', 'cell': 'D1'}
        ]
        
        # Calculate sensitivity
        sensitivity_data = BigMonteCarloEngine.calculate_sensitivity_analysis(test_results, input_configs)
        
        print(f"✅ Sensitivity analysis completed:")
        for cell, data in sensitivity_data.items():
            print(f"   {cell}: impact={data['impact']:.3f}, correlation={data['correlation']:.3f}, rank={data['rank']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sensitivity analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_node_encoding_performance():
    """Test node encoding performance with large datasets"""
    print("\n🧪 Testing node encoding performance...")
    
    try:
        from simulation.big_engine import BigMonteCarloEngine
        import time
        
        # Test encoding many nodes
        start_time = time.time()
        encoded_nodes = []
        
        for sheet_num in range(10):
            for row in range(1000):
                for col in range(26):  # A-Z
                    sheet = f"Sheet{sheet_num}"
                    cell = f"{chr(65+col)}{row+1}"
                    encoded = BigMonteCarloEngine.encode_node(sheet, cell)
                    encoded_nodes.append(encoded)
        
        encoding_time = time.time() - start_time
        print(f"✅ Encoded {len(encoded_nodes)} nodes in {encoding_time:.3f} seconds")
        print(f"   Rate: {len(encoded_nodes)/encoding_time:.0f} nodes/second")
        
        # Test decoding performance
        start_time = time.time()
        for encoded in encoded_nodes[:1000]:  # Test first 1000
            decoded = BigMonteCarloEngine.decode_node(encoded)
        
        decoding_time = time.time() - start_time
        print(f"✅ Decoded 1000 nodes in {decoding_time:.3f} seconds")
        print(f"   Rate: {1000/decoding_time:.0f} nodes/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Node encoding performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 2 tests"""
    print("🚀 Testing BIG Engine Phase 2 Implementation")
    print("=" * 60)
    
    tests = [
        test_monte_carlo_simulation,
        test_histogram_generation,
        test_sensitivity_analysis,
        test_node_encoding_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"✅ BIG Engine Phase 2 testing complete: {passed}/{total} tests passed!")
    
    if passed == total:
        print("🎉 All Phase 2 tests passed! BIG engine is ready for production.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main() 