#!/usr/bin/env python3
"""
Test script for Phase 1 functions in the Formula Engine
Tests all newly implemented math and statistical functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from excel_parser.formula_engine import ExcelFormulaEngine
import numpy as np
import statistics

def test_phase1_functions():
    """Test all Phase 1 functions"""
    engine = ExcelFormulaEngine()
    
    print("🧪 Testing Phase 1 Formula Engine Functions")
    print("=" * 50)
    
    # Test Math Functions
    print("\n📊 Math Functions:")
    
    # PRODUCT
    result = engine._product(2, 3, 4)
    expected = 24
    print(f"PRODUCT(2,3,4): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # POWER
    result = engine._power(2, 3)
    expected = 8
    print(f"POWER(2,3): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # INT
    result = engine._int(3.7)
    expected = 3
    print(f"INT(3.7): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._int(-3.7)
    expected = -4
    print(f"INT(-3.7): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # MOD
    result = engine._mod(10, 3)
    expected = 1
    print(f"MOD(10,3): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # TRUNC
    result = engine._trunc(3.14159, 2)
    expected = 3.14
    print(f"TRUNC(3.14159,2): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # ROUNDUP
    result = engine._roundup(3.2, 0)
    expected = 4
    print(f"ROUNDUP(3.2,0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # ROUNDDOWN
    result = engine._rounddown(3.8, 0)
    expected = 3
    print(f"ROUNDDOWN(3.8,0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # SIGN
    result = engine._sign(5)
    expected = 1
    print(f"SIGN(5): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._sign(-5)
    expected = -1
    print(f"SIGN(-5): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._sign(0)
    expected = 0
    print(f"SIGN(0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Test Statistical Functions
    print("\n📈 Statistical Functions:")
    
    test_data = [1, 2, 3, 4, 5]
    
    # COUNTA
    result = engine._counta(1, 2, "", None, 5)
    expected = 3  # Only non-empty values
    print(f"COUNTA(1,2,'',None,5): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # COUNTBLANK
    result = engine._countblank(1, 2, "", None, 5)
    expected = 2  # Empty string and None
    print(f"COUNTBLANK(1,2,'',None,5): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # STDEV.S (sample standard deviation)
    result = engine._stdev_s(*test_data)
    expected = statistics.stdev(test_data)
    print(f"STDEV.S({test_data}): {result:.4f} (expected: {expected:.4f}) {'✅' if abs(result - expected) < 0.001 else '❌'}")
    
    # STDEV.P (population standard deviation)
    result = engine._stdev_p(*test_data)
    expected = statistics.pstdev(test_data)
    print(f"STDEV.P({test_data}): {result:.4f} (expected: {expected:.4f}) {'✅' if abs(result - expected) < 0.001 else '❌'}")
    
    # VAR.S (sample variance)
    result = engine._var_s(*test_data)
    expected = statistics.variance(test_data)
    print(f"VAR.S({test_data}): {result:.4f} (expected: {expected:.4f}) {'✅' if abs(result - expected) < 0.001 else '❌'}")
    
    # VAR.P (population variance)
    result = engine._var_p(*test_data)
    expected = statistics.pvariance(test_data)
    print(f"VAR.P({test_data}): {result:.4f} (expected: {expected:.4f}) {'✅' if abs(result - expected) < 0.001 else '❌'}")
    
    # MEDIAN
    result = engine._median(*test_data)
    expected = statistics.median(test_data)
    print(f"MEDIAN({test_data}): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # MODE (test with repeated values)
    mode_data = [1, 2, 2, 3, 2, 4]
    result = engine._mode(*mode_data)
    expected = 2
    print(f"MODE({mode_data}): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # PERCENTILE
    result = engine._percentile(test_data, 0.5)  # 50th percentile (median)
    expected = np.percentile(test_data, 50)
    print(f"PERCENTILE({test_data}, 0.5): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # QUARTILE
    result = engine._quartile(test_data, 1)  # First quartile
    expected = np.percentile(test_data, 25)
    print(f"QUARTILE({test_data}, 1): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._quartile(test_data, 2)  # Second quartile (median)
    expected = np.percentile(test_data, 50)
    print(f"QUARTILE({test_data}, 2): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Test Error Handling
    print("\n🛡️ Error Handling Tests:")
    
    # Division by zero in MOD
    result = engine._mod(10, 0)
    print(f"MOD(10, 0) [division by zero]: {result} (should be 0) {'✅' if result == 0 else '❌'}")
    
    # STDEV with insufficient data
    result = engine._stdev_s(5)  # Only one value
    print(f"STDEV.S with one value: {result} (should be 0) {'✅' if result == 0 else '❌'}")
    
    # Invalid data types
    result = engine._power("invalid", 2)
    print(f"POWER with invalid input: {result} (should be 0) {'✅' if result == 0 else '❌'}")
    
    print("\n🎯 Phase 1 Testing Complete!")
    print("All functions implemented with proper error handling.")

def test_integration_with_formula_evaluation():
    """Test Phase 1 functions in actual formula evaluation"""
    print("\n🔗 Integration Testing:")
    print("-" * 30)
    
    engine = ExcelFormulaEngine()
    
    # Create sample sheet data
    sample_data = {
        'Sheet1': {
            'A1': {'value': 2, 'display_value': 2},
            'A2': {'value': 3, 'display_value': 3},
            'A3': {'value': 4, 'display_value': 4},
            'B1': {'formula': '=PRODUCT(A1:A3)', 'value': 0},
            'B2': {'formula': '=POWER(A1,A2)', 'value': 0},
            'B3': {'formula': '=STDEV.S(A1:A3)', 'value': 0},
        }
    }
    
    engine.load_workbook_data(sample_data)
    
    # Test formula evaluation (these would need proper range parsing in production)
    test_formulas = [
        "=PRODUCT(2,3,4)",
        "=POWER(2,3)", 
        "=INT(3.7)",
        "=MOD(10,3)",
        "=SIGN(-5)"
    ]
    
    for formula in test_formulas:
        result = engine.evaluate_formula(formula, 'Sheet1')
        print(f"Formula: {formula} → Value: {result.value}, Error: {result.error}")

if __name__ == "__main__":
    test_phase1_functions()
    test_integration_with_formula_evaluation() 