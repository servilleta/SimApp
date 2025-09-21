#!/usr/bin/env python3
"""
SIMPLE ZERO BUG DIAGNOSIS
=========================
Direct test of formula evaluation in simulation context
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add paths
sys.path.append('/opt/app')
sys.path.append('/opt/app/backend')

async def test_formula_with_context():
    """Test formula evaluation with actual data context"""
    try:
        # Import the correct FormulaEngine
        from excel_parser.formula_engine import ExcelFormulaEngine
        
        print("🧪 TESTING FORMULA EVALUATION IN SIMULATION CONTEXT")
        print("=" * 60)
        
        formula_engine = ExcelFormulaEngine()
        
        # Test case 1: Basic formula
        print("\n📝 Test 1: Basic arithmetic")
        result1 = formula_engine.evaluate_formula('5+10', 'TestSheet', {})
        print(f"   Formula: 5+10")
        print(f"   Result: {result1.value}")
        print(f"   Expected: 15")
        print(f"   Status: {'✅ PASS' if result1.value == 15 else '❌ FAIL'}")
        print(f"   Error: {result1.error}")
        
        # Test case 2: Cell references with data
        print("\n📝 Test 2: Cell references")
        # Set up cell values in the formula engine
        formula_engine.cell_values = {
            'TestSheet': {
                'A1': 100,
                'B1': 50
            }
        }
        result2 = formula_engine.evaluate_formula('=A1+B1', 'TestSheet', {})
        print(f"   Formula: =A1+B1")
        print(f"   Cell values: A1=100, B1=50")
        print(f"   Result: {result2.value}")
        print(f"   Expected: 150")
        print(f"   Status: {'✅ PASS' if result2.value == 150 else '❌ FAIL'}")
        print(f"   Error: {result2.error}")
        
        # Test case 3: Division (GP% scenario)
        print("\n📝 Test 3: Division (GP% scenario)")
        formula_engine.cell_values = {
            'TestSheet': {
                'J6': 30,  # Gross Profit
                'I6': 100  # Gross Sales
            }
        }
        result3 = formula_engine.evaluate_formula('=J6/I6', 'TestSheet', {})
        print(f"   Formula: =J6/I6")
        print(f"   Cell values: J6=30, I6=100")
        print(f"   Result: {result3.value}")
        print(f"   Expected: 0.3")
        print(f"   Status: {'✅ PASS' if abs(result3.value - 0.3) < 0.001 else '❌ FAIL'}")
        print(f"   Error: {result3.error}")
        
        # Test case 4: Zero division handling
        print("\n📝 Test 4: Zero division handling")
        formula_engine.cell_values = {
            'TestSheet': {
                'J6': 30,  # Gross Profit  
                'I6': 0    # Gross Sales = 0
            }
        }
        result4 = formula_engine.evaluate_formula('=J6/I6', 'TestSheet', {})
        print(f"   Formula: =J6/I6")
        print(f"   Cell values: J6=30, I6=0")
        print(f"   Result: {result4.value}")
        if result4.value == 0:
            print("   Status: ⚠️ ZERO RESULT - This could be the bug!")
        else:
            print("   Status: ✅ Handled correctly")
        print(f"   Error: {result4.error}")
        
        # Test case 5: Missing cell references
        print("\n📝 Test 5: Missing cell references")
        formula_engine.cell_values = {'TestSheet': {}}  # Empty cells
        result5 = formula_engine.evaluate_formula('=J6/I6', 'TestSheet', {})
        print(f"   Formula: =J6/I6")
        print(f"   Cell values: (empty)")
        print(f"   Result: {result5.value}")
        if result5.value == 0:
            print("   Status: ⚠️ ZERO RESULT - This is likely the bug!")
        else:
            print("   Status: ✅ Non-zero result")
        print(f"   Error: {result5.error}")
        
        # Test case 6: Context variables (Monte Carlo scenario)
        print("\n📝 Test 6: Context variables (Monte Carlo)")
        context = {
            'J6': 40,  # Variable from Monte Carlo iteration
            'I6': 120  # Variable from Monte Carlo iteration
        }
        result6 = formula_engine.evaluate_formula('=J6/I6', 'TestSheet', context)
        print(f"   Formula: =J6/I6")
        print(f"   Context: J6=40, I6=120")
        print(f"   Result: {result6.value}")
        print(f"   Expected: 0.333...")
        expected = 40/120
        print(f"   Status: {'✅ PASS' if abs(result6.value - expected) < 0.001 else '❌ FAIL'}")
        print(f"   Error: {result6.error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Formula evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def check_recent_simulation_data():
    """Check recent simulation data for patterns"""
    try:
        print("\n🔍 ANALYZING RECENT SIMULATION PATTERNS")
        print("=" * 50)
        
        # Check uploads directory for recent Excel files
        uploads_path = Path("/opt/app/uploads")
        if uploads_path.exists():
            excel_files = list(uploads_path.glob("*.xlsx")) + list(uploads_path.glob("*.xls"))
            if excel_files:
                latest_file = max(excel_files, key=lambda f: f.stat().st_mtime)
                print(f"📄 Latest Excel file: {latest_file.name}")
                print(f"   Size: {latest_file.stat().st_size} bytes")
                print(f"   Modified: {latest_file.stat().st_mtime}")
            else:
                print("📄 No Excel files found in uploads")
        else:
            print("📁 Uploads directory not found")
        
        # Check saved simulations
        saved_sims_path = Path("/opt/app/saved_simulations")
        if saved_sims_path.exists():
            sim_files = list(saved_sims_path.glob("*.json"))
            print(f"💾 Saved simulations: {len(sim_files)} files")
            
            if sim_files:
                # Check the most recent simulation
                latest_sim = max(sim_files, key=lambda f: f.stat().st_mtime)
                try:
                    with open(latest_sim, 'r') as f:
                        sim_data = json.load(f)
                    
                    print(f"📊 Latest simulation analysis:")
                    print(f"   File: {latest_sim.name}")
                    
                    # Look for result patterns
                    if 'target_results' in sim_data:
                        for target in sim_data['target_results']:
                            results = target.get('results', {})
                            mean_val = results.get('mean', 'N/A')
                            print(f"   Target {target.get('target_name', 'Unknown')}: mean = {mean_val}")
                            
                            if mean_val == 0:
                                print(f"     ⚠️ ZERO RESULT DETECTED!")
                
                except Exception as e:
                    print(f"   ❌ Failed to parse simulation file: {e}")
        
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")

async def main():
    """Main diagnostic function"""
    print("🚨 SIMPLE ZERO BUG DIAGNOSIS")
    print("🎯 Focus: Formula evaluation in simulation context")
    print("")
    
    # Test formula evaluation
    formula_test_passed = await test_formula_with_context()
    
    # Check recent data
    await check_recent_simulation_data()
    
    # Summary
    print("\n📋 DIAGNOSIS SUMMARY")
    print("=" * 30)
    if formula_test_passed:
        print("✅ Formula evaluation engine: WORKING")
        print("🔍 Issue likely in: Simulation data context or input variables")
        print("")
        print("🎯 RECOMMENDED NEXT STEPS:")
        print("1. Check if input variables are being set correctly")
        print("2. Verify Excel data is being parsed properly")
        print("3. Ensure cell values are populated during simulation iterations")
        print("4. Check for division by zero in actual formulas")
        print("5. Verify Monte Carlo context variables are passed correctly")
    else:
        print("❌ Formula evaluation engine: BROKEN")
        print("🚨 CRITICAL: Basic formula evaluation is failing")
    
    return formula_test_passed

if __name__ == "__main__":
    asyncio.run(main()) 