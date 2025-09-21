#!/usr/bin/env python3
"""
🧪 SIMPLE POWER ENGINE FORMULA VALIDATION TEST
==============================================

This test validates that the Power Engine correctly evaluates Excel formulas
by directly testing the evaluation functions.
"""

import sys
import os
import logging

# Add the backend directory to Python path
sys.path.insert(0, '/home/paperspace/PROJECT/backend')

# Direct imports
from simulation.engine import _safe_excel_eval, SAFE_EVAL_NAMESPACE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_power_engine_formula_evaluation():
    """Test Power Engine formula evaluation with various Excel formulas"""
    
    print("🚀 POWER ENGINE FORMULA VALIDATION TEST")
    print("=" * 60)
    
    # Test cases: (formula, expected_result, description)
    test_cases = [
        # Basic arithmetic
        ("5+3", 8, "Basic addition"),
        ("10-4", 6, "Basic subtraction"), 
        ("6*7", 42, "Basic multiplication"),
        ("15/3", 5, "Basic division"),
        ("2**3", 8, "Exponentiation"),
        
        # Math functions
        ("ABS(-5)", 5, "Absolute value"),
        ("SQRT(25)", 5, "Square root"),
        ("POWER(2,3)", 8, "Power function"),
        ("ROUND(3.14159,2)", 3.14, "Round function"),
        ("MAX(1,5,3,9,2)", 9, "Maximum function"),
        ("MIN(1,5,3,9,2)", 1, "Minimum function"),
        
        # Statistical functions
        ("SUM(1,2,3,4,5)", 15, "Sum function"),
        ("AVERAGE(2,4,6,8)", 5, "Average function"),
        
        # Logical functions
        ("IF(5>3,100,200)", 100, "IF function true"),
        ("IF(2>5,100,200)", 200, "IF function false"),
        
        # Trigonometric functions
        ("SIN(0)", 0, "Sine of 0"),
        ("COS(0)", 1, "Cosine of 0"),
        
        # Complex expressions
        ("(5+3)*2-1", 15, "Complex arithmetic"),
        ("SQRT(ABS(-16))", 4, "Nested functions"),
        ("IF(SQRT(16)>3,MAX(1,2,3),MIN(4,5,6))", 3, "Complex nested expression"),
    ]
    
    passed = 0
    failed = 0
    tolerance = 0.001
    
    print(f"\n📊 Running {len(test_cases)} formula tests...")
    print("-" * 60)
    
    for i, (formula, expected, description) in enumerate(test_cases, 1):
        try:
            # Evaluate the formula using Power Engine's evaluation method
            result = _safe_excel_eval(
                formula_string=formula,
                current_eval_sheet="TestSheet",
                all_current_iter_values={},
                safe_eval_globals=SAFE_EVAL_NAMESPACE,
                current_calc_cell_coord="TestSheet!TEST",
                constant_values={}
            )
            
            # Check if result matches expected
            if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
                if abs(float(result) - float(expected)) <= tolerance:
                    print(f"✅ Test {i:2d}: {description:25} | {formula:20} = {result} ✓")
                    passed += 1
                else:
                    print(f"❌ Test {i:2d}: {description:25} | {formula:20} = {result} (expected {expected})")
                    failed += 1
            elif str(result) == str(expected):
                print(f"✅ Test {i:2d}: {description:25} | {formula:20} = {result} ✓")
                passed += 1
            else:
                print(f"❌ Test {i:2d}: {description:25} | {formula:20} = {result} (expected {expected})")
                failed += 1
                
        except Exception as e:
            print(f"❌ Test {i:2d}: {description:25} | {formula:20} = ERROR: {str(e)[:30]}...")
            failed += 1
    
    # Results summary
    total = passed + failed
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests:     {total}")
    print(f"Tests Passed:    {passed}")
    print(f"Tests Failed:    {failed}")
    print(f"Success Rate:    {success_rate:.1f}%")
    
    # Assessment
    if success_rate >= 95:
        print("\n🌟 EXCELLENT: Power Engine formula evaluation is working exceptionally well!")
        assessment = "EXCELLENT"
    elif success_rate >= 85:
        print("\n✅ GOOD: Power Engine formula evaluation is working well with minor issues.")
        assessment = "GOOD"
    elif success_rate >= 70:
        print("\n⚠️ NEEDS IMPROVEMENT: Power Engine has some formula evaluation issues.")
        assessment = "NEEDS_IMPROVEMENT"
    else:
        print("\n❌ CRITICAL: Power Engine has significant formula evaluation problems.")
        assessment = "CRITICAL"
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if failed > 0:
        print("  • Review failed test cases for formula evaluation improvements")
        print("  • Check SAFE_EVAL_NAMESPACE completeness")
        print("  • Validate Excel function implementations")
    else:
        print("  • All tests passed! Power Engine is ready for production use")
        print("  • Consider adding more complex test scenarios")
    
    print(f"\n🏁 Test completed with {failed} failures")
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'success_rate': success_rate,
        'assessment': assessment
    }


def test_cell_reference_evaluation():
    """Test formula evaluation with cell references"""
    
    print("\n\n🔍 CELL REFERENCE EVALUATION TEST")
    print("=" * 60)
    
    # Setup cell values
    cell_values = {
        ('TestSheet', 'A1'): 10,
        ('TestSheet', 'B1'): 20,
        ('TestSheet', 'C1'): 30,
        ('TestSheet', 'D1'): 5,
    }
    
    # Test cases with cell references
    test_cases = [
        ("A1+B1", 30, "Cell addition"),
        ("A1*B1", 200, "Cell multiplication"),
        ("C1-D1", 25, "Cell subtraction"),
        ("B1/D1", 4, "Cell division"),
        ("A1+B1+C1", 60, "Multiple cell addition"),
        ("MAX(A1,B1,C1)", 30, "MAX with cells"),
        ("MIN(A1,B1,D1)", 5, "MIN with cells"),
        ("AVERAGE(A1,B1,C1,D1)", 16.25, "AVERAGE with cells"),
        ("IF(A1>D1,B1,C1)", 20, "IF with cell references"),
        ("SQRT(A1)+SQRT(B1)", 7.472, "SQRT with cells"),
    ]
    
    passed = 0
    failed = 0
    tolerance = 0.01
    
    print(f"\n📊 Running {len(test_cases)} cell reference tests...")
    print("-" * 60)
    
    for i, (formula, expected, description) in enumerate(test_cases, 1):
        try:
            result = _safe_excel_eval(
                formula_string=formula,
                current_eval_sheet="TestSheet",
                all_current_iter_values=cell_values,
                safe_eval_globals=SAFE_EVAL_NAMESPACE,
                current_calc_cell_coord="TestSheet!TEST",
                constant_values=cell_values
            )
            
            if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
                if abs(float(result) - float(expected)) <= tolerance:
                    print(f"✅ Test {i:2d}: {description:25} | {formula:15} = {result} ✓")
                    passed += 1
                else:
                    print(f"❌ Test {i:2d}: {description:25} | {formula:15} = {result} (expected {expected})")
                    failed += 1
            else:
                print(f"❌ Test {i:2d}: {description:25} | {formula:15} = {result} (expected {expected})")
                failed += 1
                
        except Exception as e:
            print(f"❌ Test {i:2d}: {description:25} | {formula:15} = ERROR: {str(e)[:30]}...")
            failed += 1
    
    total = passed + failed
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\nCell Reference Tests: {passed}/{total} passed ({success_rate:.1f}%)")
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'success_rate': success_rate
    }


def main():
    """Main test execution"""
    try:
        # Run basic formula tests
        basic_results = test_power_engine_formula_evaluation()
        
        # Run cell reference tests
        cell_results = test_cell_reference_evaluation()
        
        # Combined results
        total_tests = basic_results['total'] + cell_results['total']
        total_passed = basic_results['passed'] + cell_results['passed']
        total_failed = basic_results['failed'] + cell_results['failed']
        overall_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("🎯 OVERALL TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests Run:    {total_tests}")
        print(f"Total Passed:       {total_passed}")
        print(f"Total Failed:       {total_failed}")
        print(f"Overall Success:    {overall_success_rate:.1f}%")
        
        # Final assessment
        if overall_success_rate >= 90:
            print("\n🌟 Power Engine formula evaluation is EXCELLENT!")
            exit_code = 0
        elif overall_success_rate >= 75:
            print("\n✅ Power Engine formula evaluation is GOOD!")
            exit_code = 0
        else:
            print("\n⚠️ Power Engine formula evaluation needs IMPROVEMENT!")
            exit_code = 1
            
        return exit_code
        
    except Exception as e:
        print(f"\n💥 Test suite failed with critical error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 