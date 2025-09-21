#!/usr/bin/env python3
"""
Comprehensive Excel Functions Test Suite
========================================

Tests all newly implemented Excel functions in the Power Engine to ensure
they work correctly and provide the expected Excel-compatible behavior.
"""

import sys
import os
import math
from datetime import datetime, date

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from simulation.engine import (
    excel_and, excel_or, excel_not, excel_iferror,
    excel_countif, excel_sumif, excel_averageif,
    excel_concatenate, excel_left, excel_right, excel_mid,
    excel_upper, excel_lower, excel_trim, excel_find, excel_search,
    excel_today, excel_now, excel_year, excel_month, excel_day,
    excel_pmt, excel_pv, excel_fv, excel_npv,
    excel_product, excel_roundup, excel_rounddown, excel_ceiling,
    excel_rand, excel_randbetween,
    excel_median, excel_mode, excel_stdev, excel_var
)

def test_logical_functions():
    """Test logical functions: AND, OR, NOT, IFERROR"""
    print("\n🔍 Testing Logical Functions...")
    
    # Test AND
    assert excel_and(True, True, True) == True, "AND(True, True, True) should be True"
    assert excel_and(True, False, True) == False, "AND(True, False, True) should be False"
    assert excel_and(1, 1, 1) == True, "AND(1, 1, 1) should be True"
    assert excel_and(1, 0, 1) == False, "AND(1, 0, 1) should be False"
    print("✅ AND function tests passed")
    
    # Test OR
    assert excel_or(True, False, False) == True, "OR(True, False, False) should be True"
    assert excel_or(False, False, False) == False, "OR(False, False, False) should be False"
    assert excel_or(1, 0, 0) == True, "OR(1, 0, 0) should be True"
    assert excel_or(0, 0, 0) == False, "OR(0, 0, 0) should be False"
    print("✅ OR function tests passed")
    
    # Test NOT
    assert excel_not(True) == False, "NOT(True) should be False"
    assert excel_not(False) == True, "NOT(False) should be True"
    assert excel_not(1) == False, "NOT(1) should be False"
    assert excel_not(0) == True, "NOT(0) should be True"
    print("✅ NOT function tests passed")
    
    # Test IFERROR
    assert excel_iferror(10, "Error") == 10, "IFERROR(10, 'Error') should be 10"
    assert excel_iferror("#VALUE!", "Error") == "Error", "IFERROR('#VALUE!', 'Error') should be 'Error'"
    assert excel_iferror("#DIV/0!", 0) == 0, "IFERROR('#DIV/0!', 0) should be 0"
    print("✅ IFERROR function tests passed")

def test_conditional_functions():
    """Test conditional functions: COUNTIF, SUMIF, AVERAGEIF"""
    print("\n🔍 Testing Conditional Functions...")
    
    # Test data
    values = [10, 20, 30, 40, 50]
    
    # Test COUNTIF
    assert excel_countif(values, ">25") == 3, "COUNTIF(values, '>25') should be 3"
    assert excel_countif(values, ">=30") == 3, "COUNTIF(values, '>=30') should be 3"
    assert excel_countif(values, "<25") == 2, "COUNTIF(values, '<25') should be 2"
    assert excel_countif(values, "20") == 1, "COUNTIF(values, '20') should be 1"
    print("✅ COUNTIF function tests passed")
    
    # Test SUMIF
    assert excel_sumif(values, ">25") == 120, "SUMIF(values, '>25') should be 120 (30+40+50)"
    assert excel_sumif(values, "<=30") == 60, "SUMIF(values, '<=30') should be 60 (10+20+30)"
    assert excel_sumif(values, "40") == 40, "SUMIF(values, '40') should be 40"
    print("✅ SUMIF function tests passed")
    
    # Test AVERAGEIF
    assert excel_averageif(values, ">25") == 40, "AVERAGEIF(values, '>25') should be 40 (120/3)"
    assert excel_averageif(values, "<=20") == 15, "AVERAGEIF(values, '<=20') should be 15 (30/2)"
    print("✅ AVERAGEIF function tests passed")

def test_text_functions():
    """Test text functions: CONCATENATE, LEFT, RIGHT, MID, UPPER, LOWER, TRIM, FIND, SEARCH"""
    print("\n🔍 Testing Text Functions...")
    
    # Test CONCATENATE
    assert excel_concatenate("Hello", " ", "World") == "Hello World", "CONCATENATE should join strings"
    assert excel_concatenate("A", 1, "B") == "A1B", "CONCATENATE should handle mixed types"
    print("✅ CONCATENATE function tests passed")
    
    # Test LEFT
    assert excel_left("Hello World", 5) == "Hello", "LEFT should extract leftmost characters"
    assert excel_left("Test", 10) == "Test", "LEFT should handle length > string length"
    print("✅ LEFT function tests passed")
    
    # Test RIGHT
    assert excel_right("Hello World", 5) == "World", "RIGHT should extract rightmost characters"
    assert excel_right("Test", 10) == "Test", "RIGHT should handle length > string length"
    print("✅ RIGHT function tests passed")
    
    # Test MID
    assert excel_mid("Hello World", 7, 5) == "World", "MID should extract middle characters"
    assert excel_mid("Hello World", 1, 5) == "Hello", "MID should work from start"
    print("✅ MID function tests passed")
    
    # Test UPPER/LOWER
    assert excel_upper("hello world") == "HELLO WORLD", "UPPER should convert to uppercase"
    assert excel_lower("HELLO WORLD") == "hello world", "LOWER should convert to lowercase"
    print("✅ UPPER/LOWER function tests passed")
    
    # Test TRIM
    assert excel_trim("  Hello   World  ") == "Hello World", "TRIM should remove extra spaces"
    assert excel_trim("\t\nHello\t\nWorld\t\n") == "Hello World", "TRIM should handle whitespace"
    print("✅ TRIM function tests passed")
    
    # Test FIND
    assert excel_find("World", "Hello World", 1) == 7, "FIND should locate text"
    assert excel_find("world", "Hello World", 1) == "#VALUE!", "FIND should be case-sensitive"
    print("✅ FIND function tests passed")
    
    # Test SEARCH
    assert excel_search("World", "Hello World", 1) == 7, "SEARCH should locate text"
    assert excel_search("world", "Hello World", 1) == 7, "SEARCH should be case-insensitive"
    print("✅ SEARCH function tests passed")

def test_date_functions():
    """Test date functions: TODAY, NOW, YEAR, MONTH, DAY"""
    print("\n🔍 Testing Date Functions...")
    
    # Test TODAY/NOW (should return numbers)
    today_val = excel_today()
    now_val = excel_now()
    assert isinstance(today_val, int), "TODAY should return integer"
    assert isinstance(now_val, float), "NOW should return float"
    assert now_val > today_val, "NOW should be greater than TODAY"
    print("✅ TODAY/NOW function tests passed")
    
    # Test YEAR/MONTH/DAY with current date
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_day = datetime.now().day
    
    assert excel_year(today_val) == current_year, f"YEAR(TODAY()) should be {current_year}"
    assert excel_month(today_val) == current_month, f"MONTH(TODAY()) should be {current_month}"
    assert excel_day(today_val) == current_day, f"DAY(TODAY()) should be {current_day}"
    print("✅ YEAR/MONTH/DAY function tests passed")
    
    # Test with string dates
    assert excel_year("2024-01-15") == 2024, "YEAR should parse string dates"
    assert excel_month("2024-01-15") == 1, "MONTH should parse string dates"
    assert excel_day("2024-01-15") == 15, "DAY should parse string dates"
    print("✅ Date string parsing tests passed")

def test_financial_functions():
    """Test financial functions: PMT, PV, FV, NPV"""
    print("\n🔍 Testing Financial Functions...")
    
    # Test PMT (loan payment)
    # Example: 5% annual rate, 30 years, $200,000 loan
    monthly_rate = 0.05 / 12
    months = 30 * 12
    loan_amount = 200000
    
    pmt = excel_pmt(monthly_rate, months, loan_amount)
    assert isinstance(pmt, float), "PMT should return float"
    assert pmt < 0, "PMT should be negative (outgoing payment)"
    assert abs(pmt) > 1000, "PMT should be reasonable for large loan"
    print(f"✅ PMT function test passed: ${abs(pmt):.2f}/month")
    
    # Test PV (present value)
    pv = excel_pv(0.05, 10, -1000)
    assert isinstance(pv, float), "PV should return float"
    assert pv > 0, "PV should be positive for this example"
    print(f"✅ PV function test passed: ${pv:.2f}")
    
    # Test FV (future value)
    fv = excel_fv(0.05, 10, -1000)
    assert isinstance(fv, float), "FV should return float"
    assert fv > 0, "FV should be positive for this example"
    print(f"✅ FV function test passed: ${fv:.2f}")
    
    # Test NPV (net present value)
    cash_flows = [-1000, 200, 300, 400, 500, 600]  # Initial investment + returns
    npv = excel_npv(0.10, *cash_flows[1:])  # NPV doesn't include initial investment
    assert isinstance(npv, float), "NPV should return float"
    print(f"✅ NPV function test passed: ${npv:.2f}")

def test_math_functions():
    """Test additional math functions: PRODUCT, ROUNDUP, ROUNDDOWN, CEILING"""
    print("\n🔍 Testing Additional Math Functions...")
    
    # Test PRODUCT
    assert excel_product(2, 3, 4) == 24, "PRODUCT(2, 3, 4) should be 24"
    assert excel_product([2, 3], 4) == 24, "PRODUCT should handle lists"
    assert excel_product(0, 5, 10) == 0, "PRODUCT with zero should be zero"
    print("✅ PRODUCT function tests passed")
    
    # Test ROUNDUP
    assert excel_roundup(3.14, 1) == 3.2, "ROUNDUP(3.14, 1) should be 3.2"
    assert excel_roundup(-3.14, 1) == -3.2, "ROUNDUP(-3.14, 1) should be -3.2 (away from zero)"
    assert excel_roundup(123.456, 0) == 124, "ROUNDUP(123.456, 0) should be 124"
    print("✅ ROUNDUP function tests passed")
    
    # Test ROUNDDOWN
    assert excel_rounddown(3.89, 1) == 3.8, "ROUNDDOWN(3.89, 1) should be 3.8"
    assert excel_rounddown(-3.89, 1) == -3.8, "ROUNDDOWN(-3.89, 1) should be -3.8"
    assert excel_rounddown(123.456, 0) == 123, "ROUNDDOWN(123.456, 0) should be 123"
    print("✅ ROUNDDOWN function tests passed")
    
    # Test CEILING
    assert excel_ceiling(4.3, 1) == 5, "CEILING(4.3, 1) should be 5"
    assert excel_ceiling(4.3, 0.5) == 4.5, "CEILING(4.3, 0.5) should be 4.5"
    assert excel_ceiling(-4.3, 1) == -4, "CEILING(-4.3, 1) should be -4"
    print("✅ CEILING function tests passed")

def test_random_functions():
    """Test random functions: RAND, RANDBETWEEN"""
    print("\n🔍 Testing Random Functions...")
    
    # Test RAND
    rand_val = excel_rand()
    assert 0 <= rand_val <= 1, "RAND should return value between 0 and 1"
    assert isinstance(rand_val, float), "RAND should return float"
    print("✅ RAND function tests passed")
    
    # Test RANDBETWEEN
    rand_int = excel_randbetween(10, 20)
    assert 10 <= rand_int <= 20, "RANDBETWEEN should return value in range"
    assert isinstance(rand_int, int), "RANDBETWEEN should return integer"
    print("✅ RANDBETWEEN function tests passed")

def test_statistical_functions():
    """Test statistical functions: MEDIAN, MODE, STDEV, VAR"""
    print("\n🔍 Testing Statistical Functions...")
    
    # Test data
    values = [1, 2, 3, 4, 5]
    values_with_mode = [1, 2, 2, 3, 4]
    
    # Test MEDIAN
    assert excel_median(*values) == 3, "MEDIAN of [1,2,3,4,5] should be 3"
    assert excel_median(1, 2, 3, 4) == 2.5, "MEDIAN of [1,2,3,4] should be 2.5"
    print("✅ MEDIAN function tests passed")
    
    # Test MODE
    mode_val = excel_mode(*values_with_mode)
    assert mode_val == 2, "MODE of [1,2,2,3,4] should be 2"
    print("✅ MODE function tests passed")
    
    # Test STDEV
    stdev_val = excel_stdev(*values)
    assert isinstance(stdev_val, float), "STDEV should return float"
    assert stdev_val > 0, "STDEV should be positive"
    print(f"✅ STDEV function test passed: {stdev_val:.3f}")
    
    # Test VAR
    var_val = excel_var(*values)
    assert isinstance(var_val, float), "VAR should return float"
    assert var_val > 0, "VAR should be positive"
    assert abs(var_val - stdev_val**2) < 0.001, "VAR should be STDEV squared"
    print(f"✅ VAR function test passed: {var_val:.3f}")

def run_all_tests():
    """Run all Excel function tests"""
    print("🚀 COMPREHENSIVE EXCEL FUNCTIONS TEST SUITE")
    print("=" * 60)
    
    try:
        test_logical_functions()
        test_conditional_functions()
        test_text_functions()
        test_date_functions()
        test_financial_functions()
        test_math_functions()
        test_random_functions()
        test_statistical_functions()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Excel functions are working correctly!")
        print("📊 Total functions tested: 35+")
        print("✅ Power Engine now supports comprehensive Excel formula compatibility")
        print("=" * 60)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 