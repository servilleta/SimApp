#!/usr/bin/env python3
"""
Scientific Notation Parsing Test
===============================
Test specifically for scientific notation issues
"""

import sys
sys.path.append('/home/paperspace/PROJECT/backend')

def test_scientific_notation():
    """Test scientific notation parsing specifically"""
    
    print("🧪 SCIENTIFIC NOTATION PARSING TEST")
    print("=" * 50)
    
    try:
        from excel_parser.formula_engine import ExcelFormulaEngine
        formula_engine = ExcelFormulaEngine()
        print("✅ Formula engine loaded")
    except Exception as e:
        print(f"❌ Failed to load formula engine: {e}")
        return False
    
    # Test the tokenizer directly
    print("\n🔍 Testing Enhanced Tokenizer:")
    
    test_expressions = [
        "1e10",      # Standard scientific notation
        "1E10",      # Capital E
        "1e-2",      # Negative exponent
        "1E-2",      # Capital E with negative
        "2.5e3",     # Decimal with scientific
        "3.14E-1",   # More complex
        "1e+5",      # Positive exponent
        "123e0",     # Zero exponent
    ]
    
    for expr in test_expressions:
        print(f"\nExpression: '{expr}'")
        
        # Test tokenizer
        if hasattr(formula_engine, '_tokenize_expression_fixed'):
            tokens = formula_engine._tokenize_expression_fixed(expr)
            print(f"  Tokens: {tokens}")
            
            # Test full evaluation
            result = formula_engine._evaluate_arithmetic_safely(expr)
            print(f"  Result: {result}")
            
            # Expected values
            expected = {
                "1e10": 1e10,
                "1E10": 1E10,
                "1e-2": 1e-2,
                "1E-2": 1E-2,
                "2.5e3": 2.5e3,
                "3.14E-1": 3.14E-1,
                "1e+5": 1e+5,
                "123e0": 123e0
            }
            
            if expr in expected:
                if abs(result - expected[expr]) < 0.0001:
                    print(f"  Status: ✅ CORRECT (expected {expected[expr]})")
                else:
                    print(f"  Status: ❌ WRONG (expected {expected[expr]})")
            else:
                print(f"  Status: ⚠️ NO EXPECTED VALUE")
        else:
            print("  Status: ❌ Fixed tokenizer not found")
    
    # Test mathematical expressions with scientific notation
    print("\n🧮 Testing Scientific Notation in Expressions:")
    
    math_expressions = [
        ("1e3 + 500", 1500.0),         # 1000 + 500
        ("2e2 * 3", 600.0),            # 200 * 3
        ("1e-1 + 0.9", 1.0),           # 0.1 + 0.9
        ("(1e2)/4", 25.0),             # 100 / 4
    ]
    
    for expr, expected in math_expressions:
        print(f"\nExpression: '{expr}'")
        result = formula_engine._evaluate_arithmetic_safely(expr)
        print(f"  Result: {result}")
        print(f"  Expected: {expected}")
        
        if abs(result - expected) < 0.0001:
            print(f"  Status: ✅ CORRECT")
        else:
            print(f"  Status: ❌ WRONG")

def debug_tokenizer():
    """Debug the tokenizer step by step"""
    
    print("\n🔧 DEBUGGING TOKENIZER STEP BY STEP")
    print("=" * 50)
    
    try:
        from excel_parser.formula_engine import ExcelFormulaEngine
        formula_engine = ExcelFormulaEngine()
        
        # Test expression
        test_expr = "1e10"
        print(f"Testing: '{test_expr}'")
        
        # Step 1: Sanitization
        sanitized = formula_engine._sanitize_expression(test_expr)
        print(f"1. Sanitized: '{sanitized}'")
        
        # Step 2: Tokenization
        if hasattr(formula_engine, '_tokenize_expression_fixed'):
            tokens = formula_engine._tokenize_expression_fixed(sanitized)
            print(f"2. Tokens: {tokens}")
            
            # Step 3: Manual parsing test
            if tokens:
                parser_context = {
                    'tokens': tokens,
                    'index': 0,
                    'expression': test_expr
                }
                
                result = formula_engine._parse_expression_fixed(parser_context)
                print(f"3. Parsed result: {result}")
                
                # Check if it's correct
                expected = 1e10
                if abs(result - expected) < 0.0001:
                    print(f"4. Status: ✅ CORRECT (1e10 = {expected})")
                else:
                    print(f"4. Status: ❌ WRONG (expected {expected})")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    test_scientific_notation()
    debug_tokenizer()

if __name__ == "__main__":
    main() 