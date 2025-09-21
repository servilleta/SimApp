#!/usr/bin/env python3
"""
Debug script to test Arrow Engine Enhanced Formula Engine
"""

import sys
import os
sys.path.append('/home/paperspace/PROJECT/backend')

from excel_parser.enhanced_formula_engine import EnhancedFormulaEngine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_formula_engine():
    """Test the Enhanced Formula Engine with simple formulas"""
    
    logger.info("🔍 Testing Enhanced Formula Engine...")
    
    try:
        # Initialize Enhanced Formula Engine
        engine = EnhancedFormulaEngine(
            max_workers=4,
            cache_size=10000
        )
        logger.info("✅ Enhanced Formula Engine initialized")
        
        # Test simple formula
        test_context = {
            'D2': 100.0,
            'D3': 200.0,
            'D4': 300.0,
            'I6': 8677015.24071426,
            'J6': 6219258.214624323
        }
        
        # Test basic division
        formula1 = "=J6/I6"
        logger.info(f"🧪 Testing formula: {formula1}")
        logger.info(f"🧪 Context: {test_context}")
        
        result1 = engine.evaluate_formula(
            formula=formula1,
            sheet_name="Sheet1",
            context=test_context
        )
        logger.info(f"🧪 Result: {result1} (type: {type(result1)})")
        
        # Test with Monte Carlo variables
        formula2 = "=D2+D3+D4"
        logger.info(f"🧪 Testing formula: {formula2}")
        
        result2 = engine.evaluate_formula(
            formula=formula2,
            sheet_name="Sheet1", 
            context=test_context
        )
        logger.info(f"🧪 Result: {result2} (type: {type(result2)})")
        
        # Test complex formula
        formula3 = "=(D2+D3)*J6/I6"
        logger.info(f"🧪 Testing formula: {formula3}")
        
        result3 = engine.evaluate_formula(
            formula=formula3,
            sheet_name="Sheet1",
            context=test_context
        )
        logger.info(f"🧪 Result: {result3} (type: {type(result3)})")
        
        # Check for NaN results
        results = [result1, result2, result3]
        for i, result in enumerate(results, 1):
            if hasattr(result, 'value'):
                value = result.value
            else:
                value = result
                
            if value != value:  # NaN check
                logger.error(f"❌ Formula {i} returned NaN!")
            else:
                logger.info(f"✅ Formula {i} result: {value}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced Formula Engine test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_formula_engine() 