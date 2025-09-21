#!/usr/bin/env python3
"""
TARGETED FIX for Arrow Engine Context Preparation Issue
The Enhanced Formula Engine works perfectly, but ArrowFormulaProcessor has context issues
"""

import sys
sys.path.append('/home/paperspace/PROJECT/backend')

def fix_arrow_context_preparation():
    """
    Fix the context preparation in ArrowFormulaProcessor
    """
    
    # The issue is in the _prepare_context_for_enhanced_engine method
    # It needs to properly format the context for the Enhanced Formula Engine
    
    fix_code = '''
    def _prepare_context_for_enhanced_engine(self, context: Dict[str, float], sheet_name: str) -> Dict[str, float]:
        """
        🔧 FIXED: Prepare context for enhanced formula engine evaluation
        """
        enhanced_context = {}
        
        # Add all context variables directly (no sheet prefixing for variables)
        for key, value in context.items():
            if isinstance(value, (int, float)):
                # Add variable directly (D2, D3, D4, etc.)
                enhanced_context[key] = float(value)
                
                # Also add with sheet prefix if not already prefixed
                if '!' not in key:
                    enhanced_context[f"{sheet_name}!{key}"] = float(value)
        
        logger.info(f"🚀 [ENHANCED-ARROW] Enhanced context prepared: {enhanced_context}")
        return enhanced_context
    '''
    
    print("🔧 CONTEXT PREPARATION FIX:")
    print(fix_code)
    
    # The issue is also in _process_enhanced_result method
    result_fix_code = '''
    def _process_enhanced_result(self, result) -> float:
        """
        🔧 FIXED: Process enhanced formula engine result
        """
        try:
            # Handle FormulaResult object
            if hasattr(result, 'value'):
                value = result.value
                logger.info(f"🚀 [ENHANCED-ARROW] Extracted value from FormulaResult: {value}")
            else:
                value = result
                logger.info(f"🚀 [ENHANCED-ARROW] Using result directly: {value}")
            
            # Convert to float and check for NaN
            if isinstance(value, (int, float)):
                float_value = float(value)
                if float_value != float_value:  # NaN check
                    logger.error(f"❌ [ENHANCED-ARROW] Result is NaN: {value}")
                    return 0.0
                return float_value
            
            # Handle string numbers
            if isinstance(value, str):
                try:
                    float_value = float(value)
                    return float_value
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ [ENHANCED-ARROW] Cannot convert string to float: {value}")
                    return 0.0
            
            # Handle arrays/lists - extract first element
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return float(value[0])
            
            logger.warning(f"⚠️ [ENHANCED-ARROW] Unexpected result type: {type(value)}, value: {value}")
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ [ENHANCED-ARROW] Result processing failed: {e}")
            return 0.0
    '''
    
    print("🔧 RESULT PROCESSING FIX:")
    print(result_fix_code)

if __name__ == "__main__":
    fix_arrow_context_preparation() 