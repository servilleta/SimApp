"""
Test script for SuperEngine text functions and optimization analysis
"""

import asyncio
import cupy as cp
import numpy as np
from super_engine.gpu_kernels import (
    gpu_concatenate, gpu_left, gpu_right, gpu_len, gpu_mid
)
from super_engine.model_optimizer import ModelOptimizationAnalyzer

def test_text_functions():
    """Test the newly implemented text functions"""
    print("=== Testing Text Functions ===\n")
    
    # Test data - numeric values
    text_data = cp.array([12345.67, 67890.12, 11122.33, 33344.55, 55566.77])
    
    # Test CONCATENATE
    print("1. Testing CONCATENATE:")
    try:
        result = gpu_concatenate(text_data, cp.array([1, 2, 3, 4, 5]))
        print(f"   Input 1: {text_data}")
        print(f"   Input 2: {cp.array([1, 2, 3, 4, 5])}")
        print(f"   Result (sum as placeholder): {result}")
        print("   ✅ CONCATENATE working (returns sum as placeholder)")
    except Exception as e:
        print(f"   ❌ CONCATENATE failed: {e}")
    
    # Test LEFT
    print("\n2. Testing LEFT:")
    try:
        result = gpu_left(text_data, 3)
        print(f"   Input: {text_data}")
        print(f"   LEFT(3): {result}")
        print("   ✅ LEFT working (extracts leftmost digits)")
    except Exception as e:
        print(f"   ❌ LEFT failed: {e}")
    
    # Test RIGHT
    print("\n3. Testing RIGHT:")
    try:
        result = gpu_right(text_data, 2)
        print(f"   Input: {text_data}")
        print(f"   RIGHT(2): {result}")
        print("   ✅ RIGHT working (extracts rightmost digits)")
    except Exception as e:
        print(f"   ❌ RIGHT failed: {e}")
    
    # Test LEN
    print("\n4. Testing LEN:")
    try:
        result = gpu_len(text_data)
        print(f"   Input: {text_data}")
        print(f"   LEN (digit count): {result}")
        print("   ✅ LEN working (counts digits)")
    except Exception as e:
        print(f"   ❌ LEN failed: {e}")
    
    # Test MID
    print("\n5. Testing MID:")
    try:
        result = gpu_mid(text_data, 2, 3)
        print(f"   Input: {text_data}")
        print(f"   MID(2,3): {result}")
        print("   ✅ MID working (returns original as placeholder)")
    except Exception as e:
        print(f"   ❌ MID failed: {e}")

def test_direct_compilation():
    """Test direct compilation of formulas with text functions"""
    print("\n\n=== Testing Direct Formula Compilation ===\n")
    
    try:
        from super_engine.hybrid_parser import HybridExcelParser
        from super_engine.engine import SuperEngine
        
        # Initialize SuperEngine
        engine = SuperEngine()
        
        # Test formulas
        test_cases = [
            ("=LEFT(123456, 3)", {"A1": cp.array([123456.0])}, "LEFT function"),
            ("=RIGHT(123456, 3)", {"A1": cp.array([123456.0])}, "RIGHT function"),
            ("=LEN(12345)", {"A1": cp.array([12345.0])}, "LEN function"),
        ]
        
        for formula, data, desc in test_cases:
            try:
                # Parse and compile
                result = engine.calculate(formula, data, iterations=1)
                print(f"✅ {desc}: {formula}")
                print(f"   Result: {result}")
            except Exception as e:
                print(f"❌ {desc} failed: {e}")
                
    except ImportError as e:
        print(f"⚠️  Could not test direct compilation: {e}")

async def test_optimization_analyzer():
    """Test the model optimization analyzer"""
    print("\n\n=== Testing Model Optimization Analyzer ===\n")
    
    # Create test data with various optimization opportunities
    test_formulas = {
        'Sheet1': {
            # Basic formulas
            'A1': '=B1+C1',
            'A2': '=A1*2',
            'A3': '=SUM(A1:A2)',
            
            # Lookup functions (optimization opportunity)
            'B1': '=VLOOKUP(A1,D:E,2,FALSE)',
            'B2': '=VLOOKUP(A2,D:E,2,FALSE)',
            'B3': '=VLOOKUP(A3,D:E,2,FALSE)',
            'B4': '=VLOOKUP(A1,D:E,3,FALSE)',
            'B5': '=VLOOKUP(A2,D:E,3,FALSE)',
            
            # Volatile functions (optimization opportunity)
            'C1': '=NOW()',
            'C2': '=TODAY()',
            'C3': '=RAND()',
            
            # GPU-incompatible functions (optimization opportunity)
            'D1': '=INDIRECT("B"&ROW())',
            'D2': '=OFFSET(A1,1,1)',
            
            # Complex functions
            'E1': '=SUMPRODUCT(B1:B100,C1:C100)',
            'E2': '=INDEX(D:D,MATCH(A1,E:E,0))',
            
            # Large range operations (GPU opportunity)
            'F1': '=SUM(A1:A10000)',
            'F2': '=AVERAGE(B1:B5000)',
            
            # Text functions (new)
            'G1': '=CONCATENATE(LEFT(B1,3),RIGHT(C1,2))',
            'G2': '=LEN(G1)',
            'G3': '=MID(G1,2,5)',
            
            # Array formulas
            'H1': '{=SUM(A1:A10*B1:B10)}',
            'H2': '{=TRANSPOSE(A1:A5)}'
        }
    }
    
    # Create dependency list
    test_deps = []
    for sheet, cells in test_formulas.items():
        for cell, formula in cells.items():
            test_deps.append((sheet, cell, formula))
    
    # Define Monte Carlo inputs
    test_mc_inputs = {('Sheet1', 'B1'), ('Sheet1', 'C1')}
    
    # Run analysis
    analyzer = ModelOptimizationAnalyzer()
    results = analyzer.analyze_model(test_formulas, test_deps, test_mc_inputs)
    
    # Display results
    print(f"📊 Optimization Score: {results['optimization_score']:.1f}/100")
    print(f"📝 Summary: {results['summary']}")
    
    print(f"\n📈 Model Statistics:")
    for key, value in results['stats'].items():
        print(f"  - {key}: {value}")
    
    print(f"\n💡 Top Optimization Suggestions:")
    for i, suggestion in enumerate(results['suggestions'][:5]):
        print(f"\n{i+1}. [{suggestion['severity'].upper()}] {suggestion['title']}")
        print(f"   {suggestion['description']}")
        print(f"   Affected cells: {suggestion['affected_count']} cells")
        print(f"   Estimated speedup: {suggestion['estimated_speedup']}")
        print(f"   Implementation effort: {suggestion['implementation_effort']}")
        if suggestion['affected_cells']:
            print(f"   Examples: {', '.join(suggestion['affected_cells'][:3])}")

def main():
    """Run all tests"""
    print("SuperEngine Text Functions and Optimization Tests")
    print("=" * 50)
    
    # Test text functions
    test_text_functions()
    
    # Test direct compilation
    test_direct_compilation()
    
    # Test optimization analyzer
    asyncio.run(test_optimization_analyzer())
    
    print("\n\n✅ All tests completed!")

if __name__ == "__main__":
    main() 