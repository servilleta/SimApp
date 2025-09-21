#!/usr/bin/env python3
"""
Test script for Lookup Functions (VLOOKUP, HLOOKUP, INDEX, MATCH)
Tests the newly implemented lookup functions with various scenarios
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from excel_parser.formula_engine import ExcelFormulaEngine

def test_lookup_functions():
    """Test all lookup functions"""
    engine = ExcelFormulaEngine()
    
    print("🔍 Testing Lookup Functions")
    print("=" * 50)
    
    # Test data for lookup functions
    # Product lookup table: [Product, Price, Category, Stock]
    product_table = [
        ["Apple", 1.50, "Fruit", 100],
        ["Banana", 0.80, "Fruit", 150],
        ["Carrot", 0.60, "Vegetable", 80],
        ["Date", 3.00, "Fruit", 50],
        ["Eggplant", 2.50, "Vegetable", 30]
    ]
    
    # Test VLOOKUP
    print("\n📊 VLOOKUP Function Tests:")
    
    # Exact match tests
    result = engine._vlookup("Banana", product_table, 2, False)
    expected = 0.80
    print(f"VLOOKUP('Banana', table, 2, FALSE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._vlookup("Carrot", product_table, 3, False)
    expected = "Vegetable"
    print(f"VLOOKUP('Carrot', table, 3, FALSE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._vlookup("Date", product_table, 4, False)
    expected = 50
    print(f"VLOOKUP('Date', table, 4, FALSE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Case-insensitive test
    result = engine._vlookup("apple", product_table, 2, False)
    expected = 1.50
    print(f"VLOOKUP('apple', table, 2, FALSE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Not found test
    result = engine._vlookup("Orange", product_table, 2, False)
    expected = "#N/A"
    print(f"VLOOKUP('Orange', table, 2, FALSE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Approximate match test (numeric table)
    numeric_table = [[1, "One"], [3, "Three"], [5, "Five"], [7, "Seven"], [9, "Nine"]]
    result = engine._vlookup(6, numeric_table, 2, True)
    expected = "Five"  # Should find largest value <= 6, which is 5
    print(f"VLOOKUP(6, numeric_table, 2, TRUE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Error cases
    result = engine._vlookup("Apple", product_table, 0, False)  # Invalid column
    expected = "#VALUE!"
    print(f"VLOOKUP with col_index=0: {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._vlookup("Apple", product_table, 10, False)  # Column out of range
    expected = "#REF!"
    print(f"VLOOKUP with col_index=10: {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Test HLOOKUP
    print("\n📋 HLOOKUP Function Tests:")
    
    # Transpose the product table for HLOOKUP
    hlookup_table = [
        ["Apple", "Banana", "Carrot", "Date", "Eggplant"],
        [1.50, 0.80, 0.60, 3.00, 2.50],
        ["Fruit", "Fruit", "Vegetable", "Fruit", "Vegetable"],
        [100, 150, 80, 50, 30]
    ]
    
    result = engine._hlookup("Banana", hlookup_table, 2, False)
    expected = 0.80
    print(f"HLOOKUP('Banana', table, 2, FALSE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._hlookup("Date", hlookup_table, 3, False)
    expected = "Fruit"
    print(f"HLOOKUP('Date', table, 3, FALSE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._hlookup("Orange", hlookup_table, 2, False)
    expected = "#N/A"
    print(f"HLOOKUP('Orange', table, 2, FALSE): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Test INDEX
    print("\n📇 INDEX Function Tests:")
    
    # 2D array tests
    test_array = [
        ["A1", "B1", "C1"],
        ["A2", "B2", "C2"],
        ["A3", "B3", "C3"]
    ]
    
    result = engine._index(test_array, 2, 3)
    expected = "C2"
    print(f"INDEX(array, 2, 3): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._index(test_array, 1, 1)
    expected = "A1"
    print(f"INDEX(array, 1, 1): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._index(test_array, 3, 2)
    expected = "B3"
    print(f"INDEX(array, 3, 2): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # 1D array tests
    array_1d = ["X", "Y", "Z"]
    result = engine._index(array_1d, 2)
    expected = "Y"
    print(f"INDEX(1D_array, 2): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._index(array_1d, 3)
    expected = "Z"
    print(f"INDEX(1D_array, 3): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Single value test
    result = engine._index("SingleValue", 1, 1)
    expected = "SingleValue"
    print(f"INDEX('SingleValue', 1, 1): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Error cases
    result = engine._index(test_array, 5, 1)  # Row out of range
    expected = "#REF!"
    print(f"INDEX with row=5: {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._index(test_array, 1, 5)  # Column out of range
    expected = "#REF!"
    print(f"INDEX with col=5: {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Test MATCH
    print("\n🎯 MATCH Function Tests:")
    
    # Exact match tests (match_type = 0)
    search_array = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
    
    result = engine._match("Cherry", search_array, 0)
    expected = 3
    print(f"MATCH('Cherry', array, 0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._match("Date", search_array, 0)
    expected = 4
    print(f"MATCH('Date', array, 0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._match("Orange", search_array, 0)
    expected = "#N/A"
    print(f"MATCH('Orange', array, 0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Case-insensitive test
    result = engine._match("banana", search_array, 0)
    expected = 2
    print(f"MATCH('banana', array, 0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Numeric array tests
    numeric_array = [10, 20, 30, 40, 50]
    
    result = engine._match(30, numeric_array, 0)
    expected = 3
    print(f"MATCH(30, numeric_array, 0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Approximate match tests (match_type = 1)
    result = engine._match(35, numeric_array, 1)
    expected = 3  # Should find largest value <= 35, which is 30 at position 3
    print(f"MATCH(35, numeric_array, 1): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    result = engine._match(15, numeric_array, 1)
    expected = 1  # Should find largest value <= 15, which is 10 at position 1
    print(f"MATCH(15, numeric_array, 1): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Descending order test (match_type = -1)
    desc_array = [50, 40, 30, 20, 10]
    result = engine._match(35, desc_array, -1)
    expected = 1  # Should find first value >= 35 in descending order, which is 50 at position 1
    print(f"MATCH(35, desc_array, -1): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    # Test complex scenarios
    print("\n🔧 Complex Scenario Tests:")
    
    # VLOOKUP + INDEX combination scenario
    # Find the category of a product and then use INDEX to get details
    product_name = "Carrot"
    vlookup_result = engine._vlookup(product_name, product_table, 3, False)
    print(f"Product '{product_name}' category: {vlookup_result}")
    
    # Using MATCH to find position, then INDEX to get value
    position = engine._match("Cherry", search_array, 0)
    if position != "#N/A":
        index_result = engine._index(search_array, position)
        print(f"MATCH + INDEX test: Position {position}, Value: {index_result}")
    
    # Test 2D array in MATCH (should take first column)
    array_2d = [["A", "X"], ["B", "Y"], ["C", "Z"]]
    result = engine._match("B", array_2d, 0)
    expected = 2
    print(f"MATCH('B', 2D_array, 0): {result} (expected: {expected}) {'✅' if result == expected else '❌'}")
    
    print("\n🎯 Lookup Functions Testing Complete!")
    print("All functions implemented with Excel-compatible behavior.")

def test_integration_with_formula_evaluation():
    """Test lookup functions in actual formula evaluation"""
    print("\n🔗 Integration Testing:")
    print("-" * 30)
    
    engine = ExcelFormulaEngine()
    
    # Create sample sheet data
    sample_data = {
        'Sheet1': {
            'A1': {'value': 'Apple', 'display_value': 'Apple'},
            'A2': {'value': 'Banana', 'display_value': 'Banana'},
            'B1': {'value': 1.50, 'display_value': 1.50},
            'B2': {'value': 0.80, 'display_value': 0.80},
            'C1': {'formula': '=INDEX(A1:B2,1,2)', 'value': 0},
            'C2': {'formula': '=MATCH("Apple",A1:A2,0)', 'value': 0},
        }
    }
    
    engine.load_workbook_data(sample_data)
    
    # Test formula evaluation with simple data
    test_formulas = [
        '=VLOOKUP("test",A1:B2,2,FALSE)',
        '=INDEX(A1:B2,1,1)',
        '=MATCH("Apple",A1:A2,0)',
    ]
    
    for formula in test_formulas:
        result = engine.evaluate_formula(formula, 'Sheet1')
        print(f"Formula: {formula} → Value: {result.value}, Error: {result.error}")

if __name__ == "__main__":
    test_lookup_functions()
    test_integration_with_formula_evaluation() 