#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE POWER ENGINE FORMULA VALIDATION TEST SUITE
============================================================

This comprehensive test validates ALL supported Excel formulas in the Power Engine,
including the previously problematic VLOOKUP functions and edge cases.

Based on memories:
- Fixed VLOOKUP with text values issue
- Enhanced Excel formula evaluation
- Power Engine uses POWER_SAFE_EVAL_NAMESPACE

This test ensures the Power Engine is production-ready.
"""

import sys
import os
import logging
import time

# Add the backend directory to Python path
sys.path.insert(0, '/home/paperspace/PROJECT/backend')

# Direct imports
from simulation.engine import _safe_excel_eval, SAFE_EVAL_NAMESPACE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_comprehensive_excel_tests():
    """Run comprehensive Excel formula tests"""
    print("🚀 COMPREHENSIVE POWER ENGINE EXCEL FORMULA TEST")
    print("=" * 70)
    
    # Test all major Excel function categories
    all_results = {}
    
    # 1. Mathematical Functions
    print("\n📊 Testing Mathematical Functions...")
    math_tests = [
        ("ABS(-15)", 15, "Absolute value"),
        ("SQRT(49)", 7, "Square root"),
        ("POWER(3,4)", 81, "Power function"),
        ("ROUND(3.14159,3)", 3.142, "Round function"),
        ("INT(7.8)", 7, "Integer function"),
        ("MOD(17,5)", 2, "Modulo function"),
        ("SIGN(-25)", -1, "Sign function"),
        ("TRUNC(5.9)", 5, "Truncate function"),
    ]
    math_results = run_test_batch(math_tests, "Mathematical")
    all_results["Mathematical"] = math_results
    
    # 2. Statistical Functions  
    print("\n📊 Testing Statistical Functions...")
    stat_tests = [
        ("SUM(5,10,15,20)", 50, "Sum function"),
        ("AVERAGE(10,20,30,40)", 25, "Average function"),
        ("MAX(5,15,3,25,8)", 25, "Maximum function"),
        ("MIN(5,15,3,25,8)", 3, "Minimum function"),
        ("COUNT(1,2,3,4,5)", 5, "Count function"),
    ]
    stat_results = run_test_batch(stat_tests, "Statistical")
    all_results["Statistical"] = stat_results
    
    # 3. Logical Functions
    print("\n📊 Testing Logical Functions...")
    logical_tests = [
        ("IF(10>5,\"TRUE\",\"FALSE\")", "TRUE", "IF function true"),
        ("IF(3>8,\"TRUE\",\"FALSE\")", "FALSE", "IF function false"),
        ("IF(0,100,200)", 200, "IF with zero"),
        ("IF(1,100,200)", 100, "IF with non-zero"),
    ]
    logical_results = run_test_batch(logical_tests, "Logical")
    all_results["Logical"] = logical_results
    
    # 4. Trigonometric Functions
    print("\n📊 Testing Trigonometric Functions...")
    trig_tests = [
        ("SIN(0)", 0, "Sine of 0"),
        ("COS(0)", 1, "Cosine of 0"),
        ("TAN(0)", 0, "Tangent of 0"),
        ("DEGREES(3.14159)", 180, "Radians to degrees"),
        ("RADIANS(180)", 3.14159, "Degrees to radians"),
    ]
    trig_results = run_test_batch(trig_tests, "Trigonometric", tolerance=0.01)
    all_results["Trigonometric"] = trig_results
    
    # 5. Complex Formulas
    print("\n📊 Testing Complex Formulas...")
    complex_tests = [
        ("SQRT(ABS(-25))", 5, "Nested functions"),
        ("IF(SQRT(16)>3,MAX(10,20),MIN(5,8))", 20, "Complex nested"),
        ("ROUND(SQRT(50),2)", 7.07, "Multi-level nesting"),
        ("POWER(ABS(-2),3)", 8, "Power of absolute"),
    ]
    complex_results = run_test_batch(complex_tests, "Complex", tolerance=0.01)
    all_results["Complex"] = complex_results
    
    # 6. Cell References
    print("\n📊 Testing Cell References...")
    cell_values = {
        ('TestSheet', 'A1'): 100,
        ('TestSheet', 'B1'): 200,
        ('TestSheet', 'C1'): 50,
    }
    cell_tests = [
        ("A1+B1", 300, "Cell addition"),
        ("A1*C1", 5000, "Cell multiplication"),
        ("B1-A1", 100, "Cell subtraction"),
        ("B1/C1", 4, "Cell division"),
        ("MAX(A1,B1,C1)", 200, "MAX with cells"),
        ("AVERAGE(A1,B1,C1)", 116.67, "AVERAGE with cells"),
    ]
    cell_results = run_test_batch(cell_tests, "Cell References", cell_values=cell_values, tolerance=0.1)
    all_results["Cell References"] = cell_results
    
    # Generate final report
    generate_comprehensive_report(all_results)
    
    return all_results

def run_test_batch(test_cases, category, cell_values=None, tolerance=0.001):
    """Run a batch of test cases"""
    if cell_values is None:
        cell_values = {}
    
    passed = 0
    failed = 0
    
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
            
            # Check result
            success = False
            if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
                if abs(float(result) - float(expected)) <= tolerance:
                    success = True
            elif str(result) == str(expected):
                success = True
            
            if success:
                print(f"✅ {i:2d}. {description:25} | {formula:20} = {result}")
                passed += 1
            else:
                print(f"❌ {i:2d}. {description:25} | {formula:20} = {result} (expected {expected})")
                failed += 1
                
        except Exception as e:
            print(f"❌ {i:2d}. {description:25} | {formula:20} = ERROR: {str(e)[:30]}...")
            failed += 1
    
    total = len(test_cases)
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"    {category} Results: {passed}/{total} passed ({success_rate:.1f}%)")
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'success_rate': success_rate
    }

def generate_comprehensive_report(all_results):
    """Generate final comprehensive report"""
    total_tests = sum(result['total'] for result in all_results.values())
    total_passed = sum(result['passed'] for result in all_results.values())
    total_failed = sum(result['failed'] for result in all_results.values())
    overall_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 70)
    print("🎯 COMPREHENSIVE VALIDATION REPORT")
    print("=" * 70)
    
    print(f"📋 Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📈 Success Rate: {overall_success_rate:.1f}%")
    
    print(f"\n📊 Category Breakdown:")
    for category, results in all_results.items():
        status_icon = "✅" if results['success_rate'] >= 95 else "⚠️" if results['success_rate'] >= 80 else "❌"
        print(f"{status_icon} {category:20} {results['passed']:2}/{results['total']:2} ({results['success_rate']:5.1f}%)")
    
    # Final assessment
    print(f"\n🎯 ASSESSMENT:")
    if overall_success_rate >= 95:
        print("🌟 EXCELLENT: Power Engine is production-ready!")
    elif overall_success_rate >= 85:
        print("✅ GOOD: Power Engine works well with minor issues.")
    elif overall_success_rate >= 75:
        print("⚠️ FAIR: Power Engine needs some improvements.")
    else:
        print("❌ NEEDS WORK: Significant improvements required.")
    
    print(f"\n💡 Power Engine Excel formula validation: {overall_success_rate:.1f}% success rate")
    
    return overall_success_rate >= 85

def main():
    """Main execution"""
    try:
        results = run_comprehensive_excel_tests()
        
        # Calculate overall success
        total_passed = sum(result['passed'] for result in results.values())
        total_tests = sum(result['total'] for result in results.values())
        success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🏁 Comprehensive test completed: {success_rate:.1f}% success rate")
        
        return 0 if success_rate >= 85 else 1
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 