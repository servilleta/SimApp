#!/usr/bin/env python3
"""
Monte Carlo Connection Fix

This script implements fixes for the identified Monte Carlo variable connection issues:
1. Enhanced debugging in Ultra Engine
2. Variable propagation verification
3. NPV/IRR formula debugging
4. Cash flow variation logging

CONFIRMED ROOT CAUSE:
F4, F5, F6 variables are not connected to cash flow formulas (C161:AL161)
causing identical results across all iterations.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MonteCarloConnectionFixer:
    """Implements fixes for Monte Carlo variable connection issues"""
    
    def __init__(self):
        self.fixes_applied = []
        
    def apply_all_fixes(self):
        """Apply all identified fixes"""
        
        print("🔧 APPLYING MONTE CARLO CONNECTION FIXES")
        print("=" * 60)
        
        # Fix 1: Enhanced Ultra Engine debugging
        self._fix_ultra_engine_debugging()
        
        # Fix 2: Variable injection verification
        self._fix_variable_injection_debugging()
        
        # Fix 3: NPV/IRR formula debugging
        self._fix_npv_irr_debugging()
        
        # Fix 4: Cash flow variation logging
        self._fix_cash_flow_logging()
        
        print(f"\n✅ FIXES COMPLETE: {len(self.fixes_applied)} fixes applied")
        print("🚀 Next: Run a simulation to test the enhanced debugging")
        
    def _fix_ultra_engine_debugging(self):
        """Add enhanced debugging to Ultra Engine"""
        
        print("\n🔧 FIX 1: Ultra Engine Debugging Enhancement")
        print("-" * 50)
        
        ultra_engine_file = "backend/simulation/engines/ultra_engine.py"
        
        # Read current file
        try:
            with open(ultra_engine_file, 'r') as f:
                content = f.read()
                
            # Add debug logging to the Monte Carlo iteration loop
            if "MONTE_CARLO_DEBUG_ENHANCED" not in content:
                # Find the iteration loop section
                insertion_point = content.find("for iteration in range(self.iterations):")
                
                if insertion_point != -1:
                    # Insert debugging code
                    debug_code = '''
                # 🔍 MONTE_CARLO_DEBUG_ENHANCED: Log iteration details
                if iteration < 3:  # Log first 3 iterations
                    logger.info(f"🔍 [ULTRA_DEBUG] Iteration {iteration}: Starting with {len(current_values)} values")
                    
                    # Log Monte Carlo variable values
                    mc_vars = {k: v for k, v in current_values.items() if any(var in str(k) for var in ['F4', 'F5', 'F6', 'F7'])}
                    logger.info(f"🎲 [ULTRA_DEBUG] MC Variables: {mc_vars}")
                    
                    # Log cash flow values (first 5)
                    cash_flows = {k: v for k, v in current_values.items() if '161' in str(k)}
                    cash_flow_sample = dict(list(cash_flows.items())[:5])
                    logger.info(f"💰 [ULTRA_DEBUG] Cash Flows (sample): {cash_flow_sample}")
'''
                    
                    # Insert the debug code
                    new_content = content[:insertion_point] + debug_code + content[insertion_point:]
                    
                    with open(ultra_engine_file, 'w') as f:
                        f.write(new_content)
                        
                    print(f"   ✅ Enhanced debugging added to {ultra_engine_file}")
                    self.fixes_applied.append("Ultra Engine debugging")
                else:
                    print(f"   ⚠️  Could not find iteration loop in {ultra_engine_file}")
                    
        except Exception as e:
            print(f"   ❌ Error modifying {ultra_engine_file}: {e}")
            
    def _fix_variable_injection_debugging(self):
        """Add debugging to verify variable injection"""
        
        print("\n🔧 FIX 2: Variable Injection Debugging")
        print("-" * 50)
        
        engine_file = "backend/simulation/engine.py"
        
        try:
            with open(engine_file, 'r') as f:
                content = f.read()
                
            # Add debugging to _safe_excel_eval function
            if "VARIABLE_INJECTION_DEBUG" not in content:
                # Find _safe_excel_eval function
                func_start = content.find("def _safe_excel_eval(")
                
                if func_start != -1:
                    # Find the parameter processing section
                    debug_insertion = content.find("# Convert Excel cell references", func_start)
                    
                    if debug_insertion != -1:
                        debug_code = '''
    # 🔍 VARIABLE_INJECTION_DEBUG: Log variable injection
    if current_calc_cell_coord and ('B12' in current_calc_cell_coord or 'B13' in current_calc_cell_coord):
        logger.info(f"🎯 [VAR_INJECT] Evaluating {current_calc_cell_coord}: {formula_string[:100]}")
        
        # Log available variables
        mc_vars = {k: v for k, v in all_current_iter_values.items() if any(var in str(k) for var in ['F4', 'F5', 'F6'])}
        if mc_vars:
            logger.info(f"🎲 [VAR_INJECT] Available MC vars: {mc_vars}")
        else:
            logger.warning(f"⚠️  [VAR_INJECT] No MC variables found in iteration values!")
            
        # Log constant fallbacks
        if constant_values:
            const_vars = {k: v for k, v in constant_values.items() if any(var in str(k) for var in ['F4', 'F5', 'F6'])}
            if const_vars:
                logger.info(f"📊 [VAR_INJECT] Constant fallbacks: {const_vars}")
'''
                        
                        new_content = content[:debug_insertion] + debug_code + content[debug_insertion:]
                        
                        with open(engine_file, 'w') as f:
                            f.write(new_content)
                            
                        print(f"   ✅ Variable injection debugging added to {engine_file}")
                        self.fixes_applied.append("Variable injection debugging")
                    else:
                        print(f"   ⚠️  Could not find insertion point in {engine_file}")
                        
        except Exception as e:
            print(f"   ❌ Error modifying {engine_file}: {e}")
            
    def _fix_npv_irr_debugging(self):
        """Add debugging to NPV and IRR functions"""
        
        print("\n🔧 FIX 3: NPV/IRR Function Debugging")
        print("-" * 50)
        
        engine_file = "backend/simulation/engine.py"
        
        try:
            with open(engine_file, 'r') as f:
                content = f.read()
                
            # Add debugging to NPV function
            if "NPV_IRR_DEBUG_ENHANCED" not in content:
                # Find NPV function
                npv_func = content.find("def excel_npv(")
                
                if npv_func != -1:
                    # Find function body
                    func_body_start = content.find(":", npv_func) + 1
                    debug_code = '''
    # 🔍 NPV_IRR_DEBUG_ENHANCED: Log NPV inputs
    logger.info(f"📊 [NPV_DEBUG] NPV called with rate={rate}, values={len(values) if hasattr(values, '__len__') else 'scalar'}")
    if hasattr(values, '__len__') and len(values) > 0:
        logger.info(f"💰 [NPV_DEBUG] First 5 values: {values[:5] if len(values) > 5 else values}")
        total_cf = sum(values) if all(isinstance(v, (int, float)) for v in values) else "mixed_types"
        logger.info(f"💰 [NPV_DEBUG] Total cash flow: {total_cf}")
    else:
        logger.info(f"💰 [NPV_DEBUG] Values: {values}")
'''
                    
                    new_content = content[:func_body_start] + debug_code + content[func_body_start:]
                    
                    with open(engine_file, 'w') as f:
                        f.write(new_content)
                        
                    print(f"   ✅ NPV/IRR debugging added to {engine_file}")
                    self.fixes_applied.append("NPV/IRR debugging")
                else:
                    print(f"   ⚠️  Could not find NPV function in {engine_file}")
                    
        except Exception as e:
            print(f"   ❌ Error modifying {engine_file}: {e}")
            
    def _fix_cash_flow_logging(self):
        """Add cash flow variation logging"""
        
        print("\n🔧 FIX 4: Cash Flow Variation Logging")
        print("-" * 50)
        
        ultra_engine_file = "backend/simulation/engines/ultra_engine.py"
        
        try:
            with open(ultra_engine_file, 'r') as f:
                content = f.read()
                
            # Add cash flow logging after formula evaluation
            if "CASH_FLOW_VARIATION_LOG" not in content:
                # Find the end of formula evaluation loop
                eval_end = content.find("current_values[(sheet, cell.upper())] = result")
                
                if eval_end != -1:
                    # Find end of that line
                    line_end = content.find("\n", eval_end)
                    
                    debug_code = '''
                    
                    # 🔍 CASH_FLOW_VARIATION_LOG: Track cash flow changes
                    if iteration < 3 and '161' in cell:  # Log cash flow cells for first 3 iterations
                        logger.info(f"💰 [CASH_FLOW] Iter {iteration}: {sheet}!{cell} = {result}")
'''
                    
                    new_content = content[:line_end] + debug_code + content[line_end:]
                    
                    with open(ultra_engine_file, 'w') as f:
                        f.write(new_content)
                        
                    print(f"   ✅ Cash flow logging added to {ultra_engine_file}")
                    self.fixes_applied.append("Cash flow variation logging")
                else:
                    print(f"   ⚠️  Could not find formula evaluation in {ultra_engine_file}")
                    
        except Exception as e:
            print(f"   ❌ Error modifying {ultra_engine_file}: {e}")

def create_test_simulation_script():
    """Create a script to test the fixes"""
    
    test_script = """#!/usr/bin/env python3
'''
Test script for Monte Carlo connection fixes
Run this after applying the fixes to verify debugging is working.
'''

import requests
import json
import time

def test_enhanced_debugging():
    '''Test the enhanced debugging in a B12 simulation'''
    
    print("🧪 TESTING ENHANCED DEBUGGING")
    print("=" * 50)
    
    # Simulation request for B12 (the problematic target)
    test_request = {
        "file_id": "c9ebace1-dd72-4a9f-92da-62375ee630cd",
        "targets": ["B12"],
        "variables": [
            {
                "name": "F4",
                "sheet_name": "WIZEMICE Likest",
                "min_value": 0.08,
                "most_likely": 0.10,
                "max_value": 0.12
            },
            {
                "name": "F5", 
                "sheet_name": "WIZEMICE Likest",
                "min_value": 0.12,
                "most_likely": 0.15,
                "max_value": 0.18
            }
        ],
        "iterations": 10,
        "engine_type": "ultra"
    }
    
    print("📤 Sending test simulation request...")
    print("   Target: B12 (NPV formula)")
    print("   Variables: F4, F5")
    print("   Iterations: 10 (small test)")
    print("   Engine: Ultra")
    
    print("\\n🔍 Check backend logs for enhanced debugging output:")
    print("   - [ULTRA_DEBUG] iteration details")
    print("   - [VAR_INJECT] variable injection")
    print("   - [NPV_DEBUG] NPV function calls")
    print("   - [CASH_FLOW] cash flow variations")
    
    print("\\n💻 To run this test:")
    print("   curl -X POST http://localhost:8000/api/simulation/run \\\\")
    print('        -H "Content-Type: application/json" \\\\')
    print(f'        -d \\'{json.dumps(test_request, indent=2)}\\'')

if __name__ == "__main__":
    test_enhanced_debugging()
"""
    
    with open("backend/test_monte_carlo_fixes.py", 'w') as f:
        f.write(test_script)
        
    print(f"\n📝 Test script created: backend/test_monte_carlo_fixes.py")

def main():
    """Main execution"""
    fixer = MonteCarloConnectionFixer()
    fixer.apply_all_fixes()
    create_test_simulation_script()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Check backend logs during simulation")
    print("2. Look for the new debug messages:")
    print("   - [ULTRA_DEBUG] - iteration details")
    print("   - [VAR_INJECT] - variable injection")
    print("   - [NPV_DEBUG] - NPV function calls")
    print("   - [CASH_FLOW] - cash flow variations")
    print("3. Run: python3 backend/test_monte_carlo_fixes.py")

if __name__ == "__main__":
    main() 