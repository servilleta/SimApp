#!/usr/bin/env python3
"""
🔍 DIRECT FORMULA EVALUATION TEST
Test formula evaluation directly without file upload system.
"""

import asyncio
from simulation.engine import _safe_excel_eval, SAFE_EVAL_NAMESPACE

def test_direct_formula_evaluation():
    """Test formula evaluation directly"""
    
    print("🚀 STARTING DIRECT FORMULA EVALUATION TEST")
    print("=" * 60)
    
    # Test case 1: Simple addition
    print("\n🔍 TEST 1: Simple addition A1+B1")
    
    # Simulate values that would be available during simulation
    current_iter_values = {
        ("TestSheet", "A1"): 5.0,     # Constant value
        ("TestSheet", "B1"): 10.0,    # Monte Carlo variable value for this iteration
    }
    
    constant_values = {
        ("TestSheet", "A1"): 5.0,
    }
    
    formula = "A1+B1"
    
    try:
        result = _safe_excel_eval(
            formula_string=formula,
            current_eval_sheet="TestSheet",
            all_current_iter_values=current_iter_values,
            safe_eval_globals=SAFE_EVAL_NAMESPACE,
            current_calc_cell_coord="TestSheet!C1",
            constant_values=constant_values
        )
        
        print(f"✅ Formula: {formula}")
        print(f"✅ Values: A1={current_iter_values[('TestSheet', 'A1')]}, B1={current_iter_values[('TestSheet', 'B1')]}")
        print(f"✅ Result: {result} (type: {type(result)})")
        print(f"✅ Expected: 15.0")
        
        if result == 15.0:
            print("✅ SUCCESS: Formula evaluation working correctly!")
        else:
            print(f"❌ ISSUE: Expected 15.0, got {result}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 2: Formula with multiplication
    print("\n🔍 TEST 2: Multiplication (A1+B1)*2")
    
    # Update iter values with the result from previous formula
    current_iter_values[("TestSheet", "C1")] = 15.0  # Result from A1+B1
    
    formula2 = "C1*2"
    
    try:
        result2 = _safe_excel_eval(
            formula_string=formula2,
            current_eval_sheet="TestSheet",
            all_current_iter_values=current_iter_values,
            safe_eval_globals=SAFE_EVAL_NAMESPACE,
            current_calc_cell_coord="TestSheet!D1",
            constant_values=constant_values
        )
        
        print(f"✅ Formula: {formula2}")
        print(f"✅ Values: C1={current_iter_values[('TestSheet', 'C1')]}")
        print(f"✅ Result: {result2} (type: {type(result2)})")
        print(f"✅ Expected: 30.0")
        
        if result2 == 30.0:
            print("✅ SUCCESS: Complex formula evaluation working correctly!")
        else:
            print(f"❌ ISSUE: Expected 30.0, got {result2}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 3: Test with different Monte Carlo values
    print("\n🔍 TEST 3: Testing with different Monte Carlo values")
    
    test_values = [8.0, 10.0, 12.0]  # Different B1 values
    
    for i, b1_value in enumerate(test_values):
        print(f"\n  Iteration {i+1}: B1 = {b1_value}")
        
        # Reset iteration values
        iter_values = {
            ("TestSheet", "A1"): 5.0,
            ("TestSheet", "B1"): b1_value,
        }
        
        try:
            # Step 1: Calculate C1 = A1 + B1
            c1_result = _safe_excel_eval(
                formula_string="A1+B1",
                current_eval_sheet="TestSheet",
                all_current_iter_values=iter_values,
                safe_eval_globals=SAFE_EVAL_NAMESPACE,
                current_calc_cell_coord="TestSheet!C1",
                constant_values=constant_values
            )
            
            iter_values[("TestSheet", "C1")] = c1_result
            
            # Step 2: Calculate D1 = C1 * 2
            d1_result = _safe_excel_eval(
                formula_string="C1*2",
                current_eval_sheet="TestSheet",
                all_current_iter_values=iter_values,
                safe_eval_globals=SAFE_EVAL_NAMESPACE,
                current_calc_cell_coord="TestSheet!D1",
                constant_values=constant_values
            )
            
            expected_c1 = 5.0 + b1_value
            expected_d1 = expected_c1 * 2
            
            print(f"    C1 = A1+B1 = {c1_result} (expected: {expected_c1})")
            print(f"    D1 = C1*2 = {d1_result} (expected: {expected_d1})")
            
            if c1_result == expected_c1 and d1_result == expected_d1:
                print(f"    ✅ Iteration {i+1}: CORRECT")
            else:
                print(f"    ❌ Iteration {i+1}: INCORRECT")
                
        except Exception as e:
            print(f"    ❌ Iteration {i+1}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print("🏁 DIRECT FORMULA EVALUATION TEST COMPLETED")

if __name__ == "__main__":
    test_direct_formula_evaluation() 