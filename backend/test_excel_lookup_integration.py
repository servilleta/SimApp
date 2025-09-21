#!/usr/bin/env python3
"""
Test script for Excel Lookup Function Integration
Tests that Excel formulas with range references work properly when imported
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from excel_parser.formula_engine import ExcelFormulaEngine

def test_excel_lookup_integration():
    """Test Excel lookup formulas with range references"""
    engine = ExcelFormulaEngine()
    
    print("🔗 Testing Excel Lookup Formula Integration")
    print("=" * 60)
    
    # Create realistic Excel sheet data
    sample_data = {
        'Sheet1': {
            # Product lookup table
            'A1': {'value': 'Product', 'display_value': 'Product'},
            'B1': {'value': 'Price', 'display_value': 'Price'},
            'C1': {'value': 'Category', 'display_value': 'Category'},
            'D1': {'value': 'Stock', 'display_value': 'Stock'},
            
            'A2': {'value': 'Apple', 'display_value': 'Apple'},
            'B2': {'value': 1.50, 'display_value': 1.50},
            'C2': {'value': 'Fruit', 'display_value': 'Fruit'},
            'D2': {'value': 100, 'display_value': 100},
            
            'A3': {'value': 'Banana', 'display_value': 'Banana'},
            'B3': {'value': 0.80, 'display_value': 0.80},
            'C3': {'value': 'Fruit', 'display_value': 'Fruit'},
            'D3': {'value': 150, 'display_value': 150},
            
            'A4': {'value': 'Carrot', 'display_value': 'Carrot'},
            'B4': {'value': 0.60, 'display_value': 0.60},
            'C4': {'value': 'Vegetable', 'display_value': 'Vegetable'},
            'D4': {'value': 80, 'display_value': 80},
            
            'A5': {'value': 'Date', 'display_value': 'Date'},
            'B5': {'value': 3.00, 'display_value': 3.00},
            'C5': {'value': 'Fruit', 'display_value': 'Fruit'},
            'D5': {'value': 50, 'display_value': 50},
            
            # Test lookup formulas
            'F1': {'formula': '=VLOOKUP("Banana",A2:D5,2,FALSE)', 'value': 0},
            'F2': {'formula': '=VLOOKUP("Carrot",A2:D5,3,FALSE)', 'value': 0},
            'F3': {'formula': '=INDEX(A2:D5,2,2)', 'value': 0},
            'F4': {'formula': '=MATCH("Carrot",A2:A5,0)', 'value': 0},
            'F5': {'formula': '=VLOOKUP("Apple",A1:D5,4,FALSE)', 'value': 0},
        }
    }
    
    engine.load_workbook_data(sample_data)
    
    print("\n📊 Test Data Loaded:")
    print("Product lookup table (A1:D5):")
    print("| Product | Price | Category  | Stock |")
    print("|---------|-------|-----------|-------|")
    print("| Apple   | 1.50  | Fruit     | 100   |")
    print("| Banana  | 0.80  | Fruit     | 150   |")
    print("| Carrot  | 0.60  | Vegetable | 80    |")
    print("| Date    | 3.00  | Fruit     | 50    |")
    
    print("\n🧪 Testing Excel Lookup Formulas:")
    
    # Test VLOOKUP with range
    formula = '=VLOOKUP("Banana",A2:D5,2,FALSE)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    expected = 0.80
    print(f"F1: {formula}")
    print(f"    Result: {result.value} (expected: {expected}) {'✅' if result.value == expected else '❌'}")
    if result.error:
        print(f"    Error: {result.error}")
    
    # Test VLOOKUP for category
    formula = '=VLOOKUP("Carrot",A2:D5,3,FALSE)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    expected = "Vegetable"
    print(f"F2: {formula}")
    print(f"    Result: {result.value} (expected: {expected}) {'✅' if result.value == expected else '❌'}")
    if result.error:
        print(f"    Error: {result.error}")
    
    # Test INDEX with range
    formula = '=INDEX(A2:D5,2,2)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    expected = 0.80  # Banana's price
    print(f"F3: {formula}")
    print(f"    Result: {result.value} (expected: {expected}) {'✅' if result.value == expected else '❌'}")
    if result.error:
        print(f"    Error: {result.error}")
    
    # Test MATCH with range
    formula = '=MATCH("Carrot",A2:A5,0)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    expected = 3  # Position of Carrot in the range
    print(f"F4: {formula}")
    print(f"    Result: {result.value} (expected: {expected}) {'✅' if result.value == expected else '❌'}")
    if result.error:
        print(f"    Error: {result.error}")
    
    # Test VLOOKUP with header row included
    formula = '=VLOOKUP("Apple",A1:D5,4,FALSE)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    expected = 100  # Apple's stock
    print(f"F5: {formula}")
    print(f"    Result: {result.value} (expected: {expected}) {'✅' if result.value == expected else '❌'}")
    if result.error:
        print(f"    Error: {result.error}")
    
    print("\n🔄 Testing Complex Range Scenarios:")
    
    # Test single cell as range
    formula = '=INDEX(B3:B3,1,1)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    expected = 0.80
    print(f"Single cell range: {formula}")
    print(f"    Result: {result.value} (expected: {expected}) {'✅' if result.value == expected else '❌'}")
    
    # Test larger range
    formula = '=VLOOKUP("Date",A1:D5,2,FALSE)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    expected = 3.00
    print(f"Larger range test: {formula}")
    print(f"    Result: {result.value} (expected: {expected}) {'✅' if result.value == expected else '❌'}")
    
    print("\n🎯 Integration Testing Complete!")
    
    # Count successful tests
    successful_tests = 0
    total_tests = 7
    
    test_formulas = [
        ('=VLOOKUP("Banana",A2:D5,2,FALSE)', 0.80),
        ('=VLOOKUP("Carrot",A2:D5,3,FALSE)', "Vegetable"),
        ('=INDEX(A2:D5,2,2)', 0.80),
        ('=MATCH("Carrot",A2:A5,0)', 3),
        ('=VLOOKUP("Apple",A1:D5,4,FALSE)', 100),
        ('=INDEX(B3:B3,1,1)', 0.80),
        ('=VLOOKUP("Date",A1:D5,2,FALSE)', 3.00),
    ]
    
    for formula, expected in test_formulas:
        result = engine.evaluate_formula(formula, 'Sheet1')
        if result.value == expected:
            successful_tests += 1
    
    print(f"\n📈 Success Rate: {successful_tests}/{total_tests} tests passed ({successful_tests/total_tests*100:.1f}%)")
    
    if successful_tests == total_tests:
        print("🎉 ALL EXCEL LOOKUP FORMULAS WORKING! Your platform is ready for Excel uploads!")
    else:
        print("⚠️  Some tests failed. Excel lookup formulas need more work.")

def test_edge_cases():
    """Test edge cases and error handling"""
    engine = ExcelFormulaEngine()
    
    print("\n🛡️ Testing Edge Cases:")
    
    # Create minimal data
    sample_data = {
        'Sheet1': {
            'A1': {'value': 'Test', 'display_value': 'Test'},
            'B1': {'value': 42, 'display_value': 42},
        }
    }
    
    engine.load_workbook_data(sample_data)
    
    # Test with non-existent range
    formula = '=VLOOKUP("Test",Z1:Z5,1,FALSE)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    print(f"Non-existent range: {result.value} (should handle gracefully)")
    
    # Test with invalid range format
    formula = '=INDEX(A1:B1,1,1)'
    result = engine.evaluate_formula(formula, 'Sheet1')
    expected = 'Test'
    print(f"Valid range format: {result.value} (expected: {expected}) {'✅' if result.value == expected else '❌'}")

if __name__ == "__main__":
    test_excel_lookup_integration()
    test_edge_cases() 