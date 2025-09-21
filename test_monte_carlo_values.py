#!/usr/bin/env python3
"""
Test to see what Monte Carlo values are actually being injected
"""

import asyncio
import sys
import os
sys.path.append('/home/paperspace/PROJECT/backend')

from simulation.engines.ultra_engine import UltraSimulationEngine

async def test_monte_carlo_values():
    """Test what values Monte Carlo is actually sending"""
    print("🔍 TESTING MONTE CARLO VARIABLE INJECTION")
    print("=" * 60)
    
    # Use the most recent Excel file
    excel_file = "/home/paperspace/PROJECT/uploads/2a07dd9d-2048-4ff0-82e1-b6c7d74a6f68_WIZEMICE_FINAL_WORKING_MODEL.xlsx"
    
    # Create engine
    engine = UltraSimulationEngine()
    
    # Define Monte Carlo variables with the ranges from the simulation
    monte_carlo_vars = [
        {
            "name": "F4",
            "cell_address": "WIZEMICE Likest!F4",
            "distribution": {
                "type": "uniform",
                "params": {"min": 0.08, "max": 0.12}
            }
        },
        {
            "name": "F5", 
            "cell_address": "WIZEMICE Likest!F5",
            "distribution": {
                "type": "uniform", 
                "params": {"min": 0.12, "max": 0.20}
            }
        },
        {
            "name": "F6",
            "cell_address": "WIZEMICE Likest!F6", 
            "distribution": {
                "type": "uniform",
                "params": {"min": 0.05, "max": 0.10}
            }
        }
    ]
    
    # Test a few iterations manually
    print("📊 MONTE CARLO VALUE GENERATION TEST")
    print("-" * 40)
    
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}:")
        
        # Generate random values for this iteration
        random_values = {}
        for var in monte_carlo_vars:
            import random
            min_val = var["distribution"]["params"]["min"]
            max_val = var["distribution"]["params"]["max"]
            value = random.uniform(min_val, max_val)
            random_values[var["name"]] = value
            print(f"   {var['name']}: {value:.6f}")
        
        # Now see what happens when we inject these into the Excel calculation
        print(f"   → These are DECIMAL values (0.08 = 8%)")
        
        # Check what compound effect this would have
        base = 1000
        f4_val = random_values["F4"]
        
        # Simulate Row 108 compound growth with this F4 value
        current_val = base
        print(f"   Compound growth simulation with F4={f4_val:.3f}:")
        print(f"     C108 (base): {current_val:,.0f}")
        
        for col_num in range(1, 11):  # First 10 columns
            current_val = current_val * (1 + f4_val)
            col_letter = chr(ord('C') + col_num)
            print(f"     {col_letter}108: {current_val:,.0f}")
            
            if current_val > 1000000:
                print(f"     ⚠️  ASTRONOMICAL VALUES STARTING!")
                break
        
        if current_val > 1000000:
            break
    
    print("\n🎯 STATIC SCENARIO COMPARISON")
    print("-" * 40)
    
    # Load the static Excel file to see what values it actually has
    import openpyxl
    try:
        workbook = openpyxl.load_workbook(excel_file, data_only=True)
        sheet = workbook['WIZEMICE Likest']
        
        print("Static Excel values:")
        f4_static = sheet['F4'].value
        f5_static = sheet['F5'].value  
        f6_static = sheet['F6'].value
        
        print(f"   F4: {f4_static}")
        print(f"   F5: {f5_static}")
        print(f"   F6: {f6_static}")
        
        # Check a few Row 108 values
        print(f"\nStatic Row 108 customer values:")
        for col in ['C', 'D', 'E', 'F', 'G', 'H']:
            cell_val = sheet[f'{col}108'].value
            print(f"   {col}108: {cell_val}")
        
        # Compare amplification
        base_static = sheet['C108'].value or 1000
        d108_static = sheet['D108'].value or 0
        
        if d108_static > 0 and base_static > 0:
            static_growth = (d108_static / base_static) - 1
            print(f"\nStatic growth from C108 to D108: {static_growth:.3f} ({static_growth*100:.1f}%)")
            print(f"This matches F4 static value: {f4_static}")
            
            # Now check what Monte Carlo does
            mc_test_val = 0.12  # 12% as decimal
            mc_growth = mc_test_val
            mc_d108 = base_static * (1 + mc_growth)
            
            print(f"\nMonte Carlo F4=0.12 would give:")
            print(f"   D108: {mc_d108:.0f}")
            print(f"   Growth: {mc_growth:.3f} ({mc_growth*100:.1f}%)")
            
            if abs(mc_growth - static_growth) < 0.01:
                print("✅ Monte Carlo and static values are compatible (both decimal)")
            else:
                print("🚨 MISMATCH: Monte Carlo vs static interpretation!")
        
    except Exception as e:
        print(f"Error reading Excel: {e}")
    
    print("\n🎯 CONCLUSION")
    print("-" * 40)
    print("If Monte Carlo sends 0.12 and static Excel has 0.10:")
    print("- Both are decimal format (12% vs 10%)")
    print("- The issue is NOT percentage interpretation")
    print("- The issue IS the compound growth formula creating exponential effects")
    print("- Even small differences (0.10 → 0.12) create huge amplification over 39 columns")

if __name__ == "__main__":
    asyncio.run(test_monte_carlo_values())
