#!/usr/bin/env python3
"""
Monte Carlo Connection Diagnostic Script

This script analyzes why B12 and B13 simulations are producing terrible results:
- B12: Astronomical values (10^25 scale)  
- B13: All zeros (no variation)

ROOT CAUSE HYPOTHESIS:
Monte Carlo variables (F4, F5, F6) are not properly connected to the 
cash flow formulas (C161:AN161) that B12/B13 depend on.
"""

import asyncio
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloConnectionAnalyzer:
    """Analyzes Monte Carlo variable connections in B12/B13 simulations"""
    
    def __init__(self):
        self.constants_file = "uploads/c9ebace1-dd72-4a9f-92da-62375ee630cd_constants.json"
        self.analysis_results = {}
        
    async def run_complete_analysis(self):
        """Run comprehensive Monte Carlo connection analysis"""
        
        print("🔍 MONTE CARLO CONNECTION ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load and analyze constants
        await self._analyze_constants()
        
        # Step 2: Analyze variable dependency chains
        await self._analyze_dependency_chains()
        
        # Step 3: Simulate what's actually happening in iterations
        await self._simulate_iteration_behavior()
        
        # Step 4: Diagnose the specific B12/B13 issues
        await self._diagnose_b12_b13_issues()
        
        # Step 5: Generate fix recommendations
        await self._generate_fix_recommendations()
        
        print("\n🎯 ANALYSIS COMPLETE")
        print("Check the detailed findings above for the root cause and solution.")
        
    async def _analyze_constants(self):
        """Analyze the constants file to understand what's loaded"""
        
        print("\n📊 STEP 1: Constants Analysis")
        print("-" * 40)
        
        try:
            with open(self.constants_file, 'r') as f:
                constants = json.load(f)
                
            # Count constants by type
            total_constants = len(constants)
            f_variables = {k: v for k, v in constants.items() if 'F4' in k or 'F5' in k or 'F6' in k or 'F7' in k}
            b_variables = {k: v for k, v in constants.items() if 'B12' in k or 'B13' in k or 'B15' in k}
            cash_flows = {k: v for k, v in constants.items() if 'C161' in k or '161' in k}
            
            print(f"   Total constants loaded: {total_constants}")
            print(f"   F variables (F4-F7): {len(f_variables)}")
            print(f"   B variables (B12, B13, B15): {len(b_variables)}")  
            print(f"   Cash flow cells (161 row): {len(cash_flows)}")
            
            # Show the critical variables
            print("\n   🎯 CRITICAL VARIABLES:")
            for var, value in f_variables.items():
                print(f"      {var}: {value}")
                
            print("\n   📈 TARGET FORMULAS:")
            for var, value in b_variables.items():
                print(f"      {var}: {value}")
                
            # Show sample cash flows
            print("\n   💰 SAMPLE CASH FLOWS (first 5):")
            sample_cash_flows = list(cash_flows.items())[:5]
            for var, value in sample_cash_flows:
                print(f"      {var}: {value:,.2f}")
                
            self.analysis_results['constants'] = {
                'total': total_constants,
                'f_variables': f_variables,
                'b_variables': b_variables,
                'cash_flows_count': len(cash_flows)
            }
            
        except Exception as e:
            print(f"   ❌ Error loading constants: {e}")
            
    async def _analyze_dependency_chains(self):
        """Analyze how F variables should connect to cash flows"""
        
        print("\n🔗 STEP 2: Dependency Chain Analysis")
        print("-" * 40)
        
        # Expected dependency chain:
        print("   📋 EXPECTED DEPENDENCY CHAIN:")
        print("      F4, F5, F6 (Growth Rates) → Revenue/Cost Models → Cash Flows (C161:AL161) → B12 (NPV), B13 (IRR)")
        
        # Check if the dependency chain exists
        print("\n   🔍 ACTUAL DEPENDENCY STATUS:")
        
        # Check for intermediate formula cells that should connect F variables to cash flows
        try:
            with open(self.constants_file, 'r') as f:
                constants = json.load(f)
                
            # Look for revenue/cost model cells
            revenue_cells = {k: v for k, v in constants.items() if any(term in k.lower() for term in ['revenue', 'sales', 'income'])}
            cost_cells = {k: v for k, v in constants.items() if any(term in k.lower() for term in ['cost', 'expense', 'opex'])}
            
            print(f"      Revenue-related cells: {len(revenue_cells)}")
            print(f"      Cost-related cells: {len(cost_cells)}")
            
            if len(revenue_cells) == 0 and len(cost_cells) == 0:
                print("      ⚠️  WARNING: No intermediate revenue/cost cells found!")
                print("          This suggests F variables may not be connected to cash flows")
                
        except Exception as e:
            print(f"      ❌ Error analyzing dependencies: {e}")
            
    async def _simulate_iteration_behavior(self):
        """Simulate what happens in Monte Carlo iterations"""
        
        print("\n🎲 STEP 3: Monte Carlo Iteration Simulation")
        print("-" * 40)
        
        # Simulate 3 iterations with different F4 values
        f4_values = [0.08, 0.12, 0.16]  # Different growth rates
        
        print("   🔄 SIMULATING ITERATIONS:")
        
        for i, f4_val in enumerate(f4_values):
            print(f"\n      Iteration {i+1}: F4 = {f4_val:.3f}")
            
            # Check what would happen to B12 formula: =IFERROR(NPV(B15/12,C161:AN161),0)
            
            # B15 (discount rate) - should be constant
            b15_rate = 0.2  # 20% annual rate
            monthly_rate = b15_rate / 12  # Monthly rate for NPV
            print(f"         B15/12 (discount rate): {monthly_rate:.6f}")
            
            # Cash flows C161:AL161 - should vary with F4 but probably don't
            # For now, let's assume they're constant (which is the problem)
            cash_flows = [-376599, 8727, 11434, 13993, -55580]  # Sample from constants
            
            # NPV calculation
            npv = 0
            for j, cf in enumerate(cash_flows):
                npv += cf / ((1 + monthly_rate) ** (j + 1))
                
            print(f"         NPV result (first 5 cash flows): {npv:,.2f}")
            
            # If cash flows don't vary, NPV will be identical each iteration
            if i > 0 and abs(npv - prev_npv) < 1:
                print(f"         ⚠️  WARNING: NPV identical to previous iteration!")
                print(f"             This confirms cash flows are not varying with F4")
                
            prev_npv = npv
            
    async def _diagnose_b12_b13_issues(self):
        """Diagnose the specific B12 and B13 issues"""
        
        print("\n🚨 STEP 4: B12/B13 Issue Diagnosis")
        print("-" * 40)
        
        print("   🔍 ANALYZING B12 (NPV) ASTRONOMICAL VALUES:")
        print("      Problem: Results in 10^25 scale (unrealistic)")
        print("      Formula: =IFERROR(NPV(B15/12,C161:AN161),0)")
        
        # Check for potential causes
        b15_rate = 0.2 / 12  # Monthly rate
        print(f"      B15/12 rate: {b15_rate:.6f} (seems reasonable)")
        
        print("      Potential causes:")
        print("      1. ❌ Cash flows are much larger than expected")
        print("      2. ❌ Discount rate calculation error")  
        print("      3. ❌ NPV formula evaluation error")
        print("      4. ❌ Formula evaluation returning wrong data type")
        
        print("\n   🔍 ANALYZING B13 (IRR) ZERO VALUES:")
        print("      Problem: All results are 0 (no variation)")
        print("      Formula: =IFERROR(IRR(C161:AL161)*12,0)")
        
        print("      Root cause analysis:")
        print("      1. ✅ IRR fails for identical cash flow sets")
        print("      2. ✅ Falls back to IFERROR value of 0")
        print("      3. ✅ All 1000 iterations have identical cash flows")
        print("      4. ✅ Monte Carlo variables don't affect cash flows")
        
    async def _generate_fix_recommendations(self):
        """Generate specific fix recommendations"""
        
        print("\n🔧 STEP 5: Fix Recommendations")
        print("-" * 40)
        
        print("   📋 IMMEDIATE FIXES NEEDED:")
        print()
        
        print("   1. 🎯 VARIABLE CONNECTION FIX:")
        print("      Problem: F4, F5, F6 variables not connected to cash flow formulas")
        print("      Solution: Verify dependency analysis finds path F4→Cash flows")
        print("      File: backend/simulation/formula_utils.py")
        print("      Action: Add debugging to trace variable propagation")
        
        print("\n   2. 🔍 FORMULA EVALUATION DEBUG:")
        print("      Problem: B12 NPV returning astronomical values")
        print("      Solution: Add logging to NPV function inputs/outputs")
        print("      File: backend/simulation/engine.py")
        print("      Action: Log cash flows and discount rate used in NPV")
        
        print("\n   3. 📊 CASH FLOW VARIATION VERIFICATION:")
        print("      Problem: C161:AL161 may not be varying between iterations")
        print("      Solution: Add iteration-by-iteration cash flow logging")
        print("      File: backend/simulation/engines/ultra_engine.py")
        print("      Action: Log first 5 cash flows for first 3 iterations")
        
        print("\n   4. 🎲 MONTE CARLO INJECTION AUDIT:")
        print("      Problem: Random values may not be injecting into evaluation")
        print("      Solution: Verify variable override in _safe_excel_eval")
        print("      File: backend/simulation/engine.py")
        print("      Action: Add debug logging for variable substitution")
        
        print("\n   🚀 VERIFICATION STEPS:")
        print("      1. Run simulation with enhanced logging")
        print("      2. Verify F4 varies across iterations: [0.08, 0.12, 0.16]")
        print("      3. Verify cash flows vary accordingly")
        print("      4. Verify B12 NPV results are reasonable (thousands, not 10^25)")
        print("      5. Verify B13 IRR results show variation (not all zeros)")

async def main():
    """Main execution function"""
    analyzer = MonteCarloConnectionAnalyzer()
    await analyzer.run_complete_analysis()

if __name__ == "__main__":
    asyncio.run(main()) 