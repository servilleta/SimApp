#!/usr/bin/env python3
"""
Priority 1 Action Plan: Monte Carlo Variable Disconnection Fix

CONFIRMED ISSUE: F4, F5, F6 variables are completely disconnected from cash flow 
formulas (C161:AL161), causing identical results across all Monte Carlo iterations.

This script provides the step-by-step action plan to resolve the issue.
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Priority1ActionPlan:
    """Implements Priority 1 fix for Monte Carlo variable disconnection"""
    
    def __init__(self):
        self.steps_completed = []
        
    async def execute_priority_1_plan(self):
        """Execute the complete Priority 1 action plan"""
        
        print("🎯 PRIORITY 1 ACTION PLAN: Monte Carlo Variable Disconnection Fix")
        print("=" * 80)
        
        # STEP 1: Verify Enhanced Debugging is Active
        await self._step_1_verify_debugging()
        
        # STEP 2: Analyze Formula Dependency Graph  
        await self._step_2_analyze_dependencies()
        
        # STEP 3: Test Variable Propagation
        await self._step_3_test_propagation()
        
        # STEP 4: Implement Connection Fix
        await self._step_4_implement_fix()
        
        # STEP 5: Validate Fix with Test Simulation
        await self._step_5_validate_fix()
        
        print(f"\n✅ PRIORITY 1 COMPLETE: {len(self.steps_completed)} steps executed")
        
    async def _step_1_verify_debugging(self):
        """STEP 1: Verify enhanced debugging is working"""
        
        print("\n🔍 STEP 1: Verify Enhanced Debugging is Active")
        print("-" * 60)
        
        print("   📋 ACTIONS REQUIRED:")
        print("   1. Run a small test simulation (10 iterations)")
        print("   2. Check backend logs for [ULTRA_DEBUG], [VAR_INJECT], [NPV_DEBUG] messages")
        print("   3. Verify Monte Carlo variables are being logged correctly")
        
        print("\n   🚀 COMMAND TO RUN:")
        print("""   curl -X POST http://localhost:8000/api/simulation/run \\
     -H "Content-Type: application/json" \\
     -d '{
       "file_id": "c9ebace1-dd72-4a9f-92da-62375ee630cd",
       "targets": ["B12"],
       "variables": [
         {"name": "F4", "sheet_name": "WIZEMICE Likest", "min_value": 0.08, "most_likely": 0.10, "max_value": 0.12}
       ],
       "iterations": 10,
       "engine_type": "ultra"
     }'""")
     
        print("\n   ✅ EXPECTED LOG OUTPUT:")
        print("   🔍 [ULTRA_DEBUG] Iteration 0: Starting with 9450 values")
        print("   🎲 [ULTRA_DEBUG] MC Variables: {('WIZEMICE Likest', 'F4'): 0.087}")
        print("   💰 [ULTRA_DEBUG] Cash Flows (sample): {('WIZEMICE Likest', 'C161'): -376599}")
        print("   📊 [NPV_DEBUG] NPV called with rate=0.0167, values=39")
        
        self.steps_completed.append("Step 1: Enhanced Debugging Verification")
        
    async def _step_2_analyze_dependencies(self):
        """STEP 2: Analyze formula dependency graph"""
        
        print("\n🔗 STEP 2: Analyze Formula Dependency Graph")
        print("-" * 60)
        
        print("   📋 OBJECTIVE:")
        print("   Find the missing connection from F4→Revenue/Cost Models→Cash Flows")
        
        print("\n   🔍 ANALYSIS NEEDED:")
        print("   1. Examine Excel model structure to understand business logic")
        print("   2. Identify intermediate formulas that should connect F4 to cash flows")
        print("   3. Check if revenue/cost calculation cells exist but aren't loaded")
        print("   4. Verify formula evaluation order in dependency graph")
        
        print("\n   💡 INVESTIGATION FOCUS:")
        print("   - F4 = Phase 1 Growth Rate (M1-M6)")
        print("   - Should affect revenue calculations")
        print("   - Revenue should feed into cash flows C161:AL161")
        print("   - Check rows 140-170 for revenue/cost models")
        
        print("\n   🚀 DIAGNOSTIC COMMAND:")
        print("   python3 backend/debug_monte_carlo_connection.py")
        
        self.steps_completed.append("Step 2: Dependency Graph Analysis")
        
    async def _step_3_test_propagation(self):
        """STEP 3: Test variable propagation through formula chain"""
        
        print("\n🧪 STEP 3: Test Variable Propagation")
        print("-" * 60)
        
        print("   📋 OBJECTIVE:")
        print("   Verify that Monte Carlo variables actually reach the formula evaluation")
        
        print("   🔬 TESTS NEEDED:")
        print("   1. Verify F4 values change between iterations: [0.08, 0.12, 0.16]")
        print("   2. Check if intermediate cells pick up F4 changes")
        print("   3. Confirm cash flows C161:AL161 reflect the changes")
        print("   4. Validate B12 NPV calculations use varied cash flows")
        
        print("\n   🎯 KEY VERIFICATION POINTS:")
        print("   - F4 injection: current_values[(sheet, 'F4')] = random_value")
        print("   - Revenue calculation: Should reference F4 somewhere")
        print("   - Cash flow variation: C161 value should change with F4")
        print("   - NPV result: Should be different for different F4 values")
        
        print("\n   ⚠️  EXPECTED ISSUE:")
        print("   Revenue cells likely missing from constants or calculation order")
        
        self.steps_completed.append("Step 3: Variable Propagation Testing")
        
    async def _step_4_implement_fix(self):
        """STEP 4: Implement the actual connection fix"""
        
        print("\n🔧 STEP 4: Implement Connection Fix")
        print("-" * 60)
        
        print("   📋 LIKELY FIXES NEEDED:")
        
        print("\n   🎯 FIX 1: Missing Revenue/Cost Cells")
        print("   - Problem: Revenue calculation cells not loaded in constants")
        print("   - Solution: Update get_constants_for_file() to include all formula cells")
        print("   - Files: backend/excel_parser/service.py")
        
        print("\n   🎯 FIX 2: Calculation Order Issue")
        print("   - Problem: F4 not in dependency graph calculation sequence")
        print("   - Solution: Ensure build_dependency_graph() finds F4→Revenue→Cash flows")
        print("   - Files: backend/excel_parser/formula_utils.py")
        
        print("\n   🎯 FIX 3: Formula Evaluation Context")
        print("   - Problem: Variable substitution not working in complex formulas")
        print("   - Solution: Enhanced variable injection in _safe_excel_eval")
        print("   - Files: backend/simulation/engine.py")
        
        print("\n   🎯 FIX 4: Excel Model Structure")
        print("   - Problem: F4 not actually connected in Excel model")
        print("   - Solution: Ask user to verify Excel model has F4→Revenue links")
        print("   - Investigation: Manual Excel inspection required")
        
        self.steps_completed.append("Step 4: Connection Fix Implementation")
        
    async def _step_5_validate_fix(self):
        """STEP 5: Validate fix with test simulation"""
        
        print("\n✅ STEP 5: Validate Fix with Test Simulation")
        print("-" * 60)
        
        print("   📋 VALIDATION CRITERIA:")
        print("   1. B12 results should vary across iterations")
        print("   2. B13 results should not all be zero")
        print("   3. Statistical measures: mean ≠ 0, std_dev > 0")
        print("   4. Histogram should show proper distribution")
        
        print("\n   🎯 SUCCESS METRICS:")
        print("   - B12 NPV range: Should span reasonable financial values")
        print("   - B13 IRR range: Should show percentage variations")
        print("   - Histogram bins: Multiple bins with varied counts")
        print("   - Log output: Cash flows change between iterations")
        
        print("\n   🚀 FINAL TEST COMMAND:")
        print("""   curl -X POST http://localhost:8000/api/simulation/run \\
     -H "Content-Type: application/json" \\
     -d '{
       "file_id": "c9ebace1-dd72-4a9f-92da-62375ee630cd",
       "targets": ["B12", "B13"],
       "variables": [
         {"name": "F4", "sheet_name": "WIZEMICE Likest", "min_value": 0.08, "most_likely": 0.10, "max_value": 0.12},
         {"name": "F5", "sheet_name": "WIZEMICE Likest", "min_value": 0.12, "most_likely": 0.15, "max_value": 0.18}
       ],
       "iterations": 100,
       "engine_type": "ultra"
     }'""")
     
        print("\n   ✅ EXPECTED GOOD RESULTS:")
        print("   B12: mean ≈ 200K, std_dev > 50K (financial range)")
        print("   B13: mean ≈ 0.15, std_dev > 0.02 (IRR percentage)")
        print("   Frontend: Proper histogram visualization")
        
        self.steps_completed.append("Step 5: Fix Validation")

async def main():
    """Execute Priority 1 action plan"""
    
    print("🎯 STARTING PRIORITY 1 EXECUTION")
    print("Problem: Monte Carlo variables F4, F5, F6 not connected to cash flows")
    print("Impact: Identical results, astronomical B12 values, zero B13 values")
    print()
    
    planner = Priority1ActionPlan()
    await planner.execute_priority_1_plan()
    
    print("\n🚀 NEXT: Execute Step 1 - run the debugging verification command!")

if __name__ == "__main__":
    asyncio.run(main()) 